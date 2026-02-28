# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming
# https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable

import torch
from torch import nn

# vLLM imports
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.diffusion.attention.backends.utils.fa import (
    _pad_input,
    flash_attn_func,
    flash_attn_varlen_func,
)
from vllm_omni.diffusion.attention.backends.utils.fa import (
    _unpad_input as _unpad_input_func,
)
from vllm_omni.diffusion.attention.backends.utils.fa import (
    _upad_input as _unpad_input_util,
)
from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin

from .configuration_bailing_moe_v2 import BailingMoeV2Config
from .modeling_utils import build_modality_mask, patch_continuous_features

logger = init_logger(__name__)


class BailingMoeV2RotaryEmbedding3D(nn.Module):
    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.rope_init_type = "default"
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type")) if config.rope_scaling else "default"
        )
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_init_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=None)

        # Initialize inv_freq manually for multimodal case
        # if hasattr(config, "head_dim") and config.head_dim:
        #     dim = config.head_dim
        # else:
        #     dim = config.hidden_size // config.num_attention_heads
        # partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        # rotary_dim = int(dim * partial_rotary_factor)
        # inv_freq = 1.0 / (
        #     config.rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        # )
        # self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for 3D rotary embeddings.

        Args:
            x: Input tensor
            position_ids: Position IDs, shape (3, batch, seq_len) for 3D rope or (batch, seq_len) for 1D

        Returns:
            Tuple of (cos, sin) tensors
        """
        if self.rope_type == "3D" or self.rope_type == "video_rope":
            # 3D rope: position_ids shape is (3, batch, seq_len)
            inv_freq_expanded = (
                self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
            )
            position_ids_expanded = position_ids[:, :, None, :].float()
        else:
            # 1D rope: position_ids shape is (batch, seq_len) or (3, batch, seq_len) with same values
            if len(position_ids.shape) == 3:
                position_ids = position_ids[0]
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            if self.rope_type == "3D" or self.rope_type == "video_rope":
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            else:
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_3d_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    mrope_section=[8, 12, 12],
    unsqueeze_dim=1,
    rope_type="m_rope",
    rotary_half=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).
    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        rope_type (`str`, *optional*, defaults to "m_rope"):
        rotary_half (`bool`, *optional*, defaults to `False`): Keep half or full tensor for later concatenation
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    if rope_type == "3D":  # rename rope_type
        rope_type = "m_rope"

    if rope_type == "m_rope":
        mrope_section = mrope_section * 2
        cos = (
            torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            .unsqueeze(unsqueeze_dim)
            .to(q.device)
        )
        sin = (
            torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
            .unsqueeze(unsqueeze_dim)
            .to(q.device)
        )
    elif rope_type == "video_rope":
        mrope_section = list(mrope_section)
        mrope_section = [mrope_section[0], mrope_section[1] + mrope_section[2]]
        mrope_section = mrope_section * 2
        # adjust t last -> (48, 16, 48, 16)
        mrope_section = mrope_section[::-1]
        index = 0
        result_cos = []
        result_sin = []
        # get x1, y1, x2, y2, ..., t1, t2, ...
        for i, section in enumerate(mrope_section):
            if i % 2 == 0:
                for j in range(section):
                    row = 1 if j % 2 == 0 else 2
                    result_cos.append(cos[row, ..., index : index + 1])
                    result_sin.append(sin[row, ..., index : index + 1])
                    index += 1
            else:
                result_cos.append(cos[0, ..., index : index + section])
                result_sin.append(sin[0, ..., index : index + section])
                index += section
        cos, sin = (
            torch.cat(result_cos, dim=-1).unsqueeze(dim=unsqueeze_dim).to(q.device),
            torch.cat(result_sin, dim=-1).unsqueeze(dim=unsqueeze_dim).to(q.device),
        )
    else:  # vanilla rope for llm
        cos = cos.unsqueeze(unsqueeze_dim).to(q.device)
        sin = sin.unsqueeze(unsqueeze_dim).to(q.device)

    if rotary_half:
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Apply rotary embeddings on the first half
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back to full shape
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def get_t_scale_rope_index(
    config,
    input_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    scale_factor: float = 1.0,
    second_per_grid_ts: torch.Tensor | None = None,
    use_interleaved_frame_timestamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = config.spatial_merge_size
    image_token_id = config.image_patch_token
    video_token_id = config.video_patch_token
    image_start_token_id = config.image_start_token
    video_start_token_id = config.video_start_token
    use_abs_time_pos = second_per_grid_ts is not None

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        if video_grid_thw is not None and use_interleaved_frame_timestamp:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            if image_grid_thw is not None:
                vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
            if video_grid_thw is not None:
                if use_interleaved_frame_timestamp:
                    vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()
                else:
                    vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # body-diagonal symmetry
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = (
                    torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    - (llm_grid_h - 1) // 2
                )
                w_index = (
                    torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    - (llm_grid_w - 1) // 2
                )

                # time dim adjust step size
                if use_abs_time_pos:
                    t_index = t_index * second_per_grid_t * scale_factor
                else:
                    t_index = t_index * scale_factor
                t_index = t_index + text_len + st_idx

                h_index = h_index + t_index
                w_index = w_index + t_index
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                # next text token near last video token position = last t + 1
                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            llm_positions = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
            # generate first token = last t + 1
            mrope_position_deltas.append(llm_positions[0].max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

    return position_ids, mrope_position_deltas


def get_rope_index(
    config,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    use_interleaved_frame_timestamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate 3D rope index for multimodal inputs."""
    spatial_merge_size = config.spatial_merge_size
    image_token_id = config.image_patch_token
    video_token_id = config.video_patch_token
    image_start_token_id = config.image_start_token
    video_start_token_id = config.video_start_token

    use_abs_time_pos = second_per_grid_ts is not None

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        if video_grid_thw is not None and use_interleaved_frame_timestamp:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            if image_grid_thw is not None:
                vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
            if video_grid_thw is not None:
                if use_interleaved_frame_timestamp:
                    vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()
                else:
                    vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                if use_abs_time_pos:
                    time_tensor = expanded_range * second_per_grid_t * config.tokens_per_second
                    time_tensor_long = time_tensor.long()
                else:
                    time_tensor_long = expanded_range.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

    return position_ids, mrope_position_deltas


class BailingMoeV2MLP(nn.Module):
    def __init__(
        self,
        config: BailingMoeV2Config,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BailingMoeV2Gate(nn.Module):
    """MoE routing gate with grouped expert selection."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gating_dim = config.hidden_size

        self.gate = ReplicatedLinear(
            self.gating_dim,
            self.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.routed_scaling_factor = config.routed_scaling_factor
        self.bias_update_coeff = 0.001

        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)

    def update_bias(self, distribution: torch.Tensor):
        """Update expert bias based on load distribution."""
        with torch.no_grad():
            delta_bias = (distribution.mean() - distribution).sign()
        self.expert_bias.data = self.expert_bias.data + self.bias_update_coeff * delta_bias

    def group_limited_topk(self, scores: torch.Tensor):
        """Group-limited top-k selection for expert routing."""
        num_tokens, _ = scores.size()
        # Organize experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask experts based on selected groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits, _ = self.gate(hidden_states.type(torch.float32))

        scores = torch.sigmoid(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class BailingMoeV2SparseMoeBlock(nn.Module):
    """Sparse MoE block with MultiRouter support for multimodal routing."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Setup experts
        self.experts = nn.ModuleList(
            [
                BailingMoeV2MLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.experts.{i}",
                )
                for i in range(self.config.num_experts)
            ]
        )

        self.router_type = self.config.router_type
        if self.router_type == "topN":
            logger.info("Using topN Router")
            self.gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.gate")
        elif self.router_type == "MultiRouter":
            logger.info("Using MultiRouter")
            self.gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.gate")
            self.image_gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.image_gate")
            self.audio_gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.audio_gate")
        else:
            raise ValueError(f"Unsupported router_type: {self.router_type}")

        if isinstance(self.config.num_shared_experts, int) and self.config.num_shared_experts > 0:
            self.shared_experts = BailingMoeV2MLP(
                config=self.config,
                intermediate_size=self.config.moe_intermediate_size * self.config.num_shared_experts,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )

    def forward(self, hidden_states, image_mask, audio_mask):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape

        if self.router_type == "MultiRouter":
            if image_mask is not None:
                if len(image_mask.shape) == 3:
                    assert image_mask.shape[-1] == 1
                elif len(image_mask.shape) == 2:
                    assert image_mask.shape == hidden_states.shape[:2]
                    image_mask = image_mask.unsqueeze(-1)
                else:
                    raise ValueError(f"Unsupported image_mask shape: {image_mask.shape}")

            if audio_mask is not None:
                if len(audio_mask.shape) == 3:
                    assert audio_mask.shape[-1] == 1
                elif len(audio_mask.shape) == 2:
                    assert audio_mask.shape == hidden_states.shape[:2]
                    audio_mask = audio_mask.unsqueeze(-1)
                else:
                    raise ValueError(f"Unsupported audio_mask shape: {audio_mask.shape}")

            if image_mask is not None and audio_mask is not None:
                assert torch.logical_and(image_mask, audio_mask).sum() == 0

            image_topk_idx, image_topk_weight, image_router_logits = self.image_gate(hidden_states)
            audio_topk_idx, audio_topk_weight, audio_router_logits = self.audio_gate(hidden_states)
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)

            if image_mask is not None:
                image_mask = image_mask.view(-1, 1)
                topk_idx = image_topk_idx * image_mask + topk_idx * torch.logical_not(image_mask)
                topk_weight = image_topk_weight * image_mask + topk_weight * torch.logical_not(image_mask)
                router_logits = image_router_logits * image_mask + router_logits * torch.logical_not(image_mask)
            if audio_mask is not None:
                audio_mask = audio_mask.view(-1, 1)
                audio_mask = audio_mask.to(router_logits.device)
                topk_idx = audio_topk_idx * audio_mask + topk_idx * torch.logical_not(audio_mask)
                topk_weight = audio_topk_weight * audio_mask + topk_weight * torch.logical_not(audio_mask)
                router_logits = audio_router_logits * audio_mask + router_logits * torch.logical_not(audio_mask)
        else:
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)

        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)

        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        """MoE inference with individual expert forward passes."""
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class BailingMoeV2Attention(nn.Module):
    """Multi-headed attention using vLLM's Attention layer with 3D RoPE support."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        # Handle TP
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_kv_heads = max(1, self.num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Partial rotary factor
        partial_rotary_factor = config.partial_rotary_factor
        self.rope_dim = int(self.head_dim * partial_rotary_factor)

        # QKV projection
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.dense = RowParallelLinear(
            total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        # We apply vLLM RMSNorm here rather than BailingMoeV2RMSNorm in the original implementation
        # to achieve better perf
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 3D Rotary embeddings for multimodal
        if config.rope_scaling is not None:
            self.rotary_emb = BailingMoeV2RotaryEmbedding3D(config)
            # XXX: Seems to be a bug but working(?)
            # TODO: Test and triage why the original impl applies hardcoded rope type
            # self.rope_scaling = {"type": "mrope", "mrope_section": [8, 12, 12]}
            # self.mrope_section = self.rope_scaling["mrope_section"]
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            self.mrope_section = config.rope_scaling.get("mrope_section", [8, 12, 12])
        else:
            # Fallback to standard rope (shouldn't happen for this model)
            # self.rotary_emb = BailingMoeV2RotaryEmbedding(config)
            # self.rope_type = "default"
            # self.mrope_section = None
            raise ValueError("rope_scaling must not be None")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass for attention with 3D RoPE.

        Args:
            positions: Position IDs, shape (3, batch, seq_len) for 3D rope
            hidden_states: Input hidden states
            attention_mask: Optional attention mask (batch, seq_len)

        Returns:
            Attention output tensor
        """
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head attention
        batch_size, seq_len = hidden_states.shape[:2]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q/K normalization
        orig_q_shape = q.shape
        orig_k_shape = k.shape
        q = self.q_norm(q.view(-1, self.head_dim)).view(orig_q_shape)
        k = self.k_norm(k.view(-1, self.head_dim)).view(orig_k_shape)

        # Apply 3D rotary embeddings (batch_size, seq_len, heads, head_dim)
        # use unsqueeze_dim 2
        cos, sin = self.rotary_emb(hidden_states, positions)
        q, k = apply_3d_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            mrope_section=self.mrope_section,
            unsqueeze_dim=2,
            rope_type=self.rope_type,
            rotary_half=(self.rope_dim < self.head_dim),
        )

        if attention_mask is not None and torch.any(~attention_mask):
            # Unpad input for variable length sequences
            assert attention_mask.ndim == 2, "attention_mask must be 2D (batch_size, seq_len)"
            q_unpad, k_unpad, v_unpad, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
                _unpad_input_util(q, k, v, attention_mask, seq_len, _unpad_input_func)
            )

            attn_output_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=is_causal,
            )
            attn_output = _pad_input(attn_output_unpad, indices_q, batch_size, seq_len)
        else:
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=is_causal,
            )

        # Reshape for output projection
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        output, _ = self.dense(attn_output)

        return output


class BailingMoeV2DecoderLayer(nn.Module):
    """Decoder layer with attention and MoE MLP."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = BailingMoeV2Attention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )

        # MLP or MoE based on layer index
        if config.num_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = BailingMoeV2SparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = BailingMoeV2MLP(
                config=config,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False

        # apply vLLM RMSNorm to replace BailingMoeV2RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for decoder layer.

        Args:
            positions: Position IDs
            hidden_states: Input hidden states
            residual: Residual connection from previous layer
            image_mask: Mask for image tokens (for MultiRouter MoE)
            audio_mask: Mask for audio tokens (for MultiRouter MoE)
            attention_mask: Attention mask for padding (batch, seq_len)

        Returns:
            Tuple of (hidden_states, residual)
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # pre-norm with fused residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # pre-norm with fused residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_moe:
            hidden_states, _ = self.mlp(hidden_states, image_mask, audio_mask)
        else:
            # Dense MLP only takes hidden_states (no routing masks)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class BailingMoeV2Model(nn.Module):
    """BailingMoeV2 Model adapted from:

    Ming repo BailingMoeV2Model
    https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py
    vLLM repo BailingMoeModel
    https://github.com/vllm-project/vllm/blob/7291d1b288558d48508e1a17c37b0aa170332264/vllm/model_executor/models/bailing_moe.py
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        # BailingMoeV2Config
        config = vllm_config.model_config.hf_text_config
        # from typing import cast
        # config = cast(BailingMoeV2Config, config)

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False) and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers with pipeline parallelism support
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BailingMoeV2DecoderLayer(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            # apply vLLM RMSNorm to replace BailingMoeV2RMSNorm
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # 3D Rotary embeddings
        if config.rope_scaling is not None:
            self.rotary_emb = BailingMoeV2RotaryEmbedding3D(config=config)
        else:
            raise ValueError("rope_scaling must not be None")

        # Multimodal config
        # XXX: The original impl hardcoded the following settings to BailingMoeV2Config
        # https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py
        config.spatial_merge_size = 2
        config.tokens_per_second = 2
        self.rope_deltas = None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.embed_tokens = value

    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, vision_token_id):
        """Merge vision embeddings into input embeddings."""
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        n_image_tokens = (input_ids == vision_token_id).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        vision_mask = (input_ids == vision_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(vision_mask, image_embeds)

        return inputs_embeds

    def prompt_wrap_audio(
        self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens
    ):
        """Merge audio embeddings into input embeddings."""
        inputs_embeds = patch_continuous_features(
            input_embeddings=inputs_embeds,
            placeholder_loc_lens=placeholder_audio_loc_lens,
            encoded_feats=audio_embeds,
            encoded_feat_lens=audio_embeds_lengths,
        )
        router_mask_audio = build_modality_mask(placeholder_audio_loc_lens, inputs_embeds.shape[:-1])
        router_mask_audio = router_mask_audio.to(inputs_embeds.device)
        return inputs_embeds, router_mask_audio

    def prompt_wrap_navit(
        self,
        input_ids,
        config,
        query_embeds_image=None,
        query_embeds_video=None,
        query_embeds_audio=None,
        query_embeds_audio_lengths=None,
        placeholder_audio_loc_lens=None,
    ):
        """Merge all multimodal embeddings."""
        inputs_embeds = self.embed_tokens(input_ids)
        vision_mask = None
        audio_mask = None
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None:
            return inputs_embeds, vision_mask, audio_mask

        if query_embeds_image is not None:
            inputs_embeds = self.prompt_wrap_vision(
                input_ids, inputs_embeds, query_embeds_image, config.image_patch_token
            )
        if query_embeds_video is not None:
            inputs_embeds = self.prompt_wrap_vision(
                input_ids, inputs_embeds, query_embeds_video, config.video_patch_token
            )

        image_mask = input_ids == config.image_patch_token
        video_mask = input_ids == config.video_patch_token
        vision_mask = (image_mask + video_mask) > 0
        vision_mask = vision_mask.unsqueeze(-1).to(input_ids.device)

        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids,
                inputs_embeds,
                query_embeds_audio,
                query_embeds_audio_lengths,
                placeholder_audio_loc_lens,
            )

        return inputs_embeds, vision_mask, audio_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_embeds_image: torch.Tensor | None = None,
        query_embeds_video: torch.Tensor | None = None,
        query_embeds_audio: torch.Tensor | None = None,
        query_embeds_audio_lengths: torch.Tensor | None = None,
        placeholder_audio_loc_lens: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_grid_thw_video: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass for BailingMoeV2 model.

        Args:
            input_ids: Input token IDs
            positions: Position IDs (3D for multimodal)
            intermediate_tensors: Intermediate tensors for pipeline parallelism
            inputs_embeds: Pre-computed input embeddings (optional)
            query_embeds_image: Image embeddings
            query_embeds_video: Video embeddings
            query_embeds_audio: Audio embeddings
            query_embeds_audio_lengths: Lengths of audio embeddings
            placeholder_audio_loc_lens: Audio placeholder locations
            image_grid_thw: Image grid dimensions (T, H, W)
            image_grid_thw_video: Video grid dimensions (T, H, W)
            attention_mask: Attention mask for padding (batch, seq_len)

        Returns:
            Hidden states or IntermediateTensors for pipeline parallelism
        """
        if get_pp_group().is_first_rank:
            # Merge multimodal embeddings
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                image_mask = None
                audio_mask = None
            else:
                if (
                    query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None
                ) or input_ids.size(1) == 1:
                    hidden_states = self.embed_tokens(input_ids)
                    image_mask = None
                    audio_mask = None
                else:
                    hidden_states, image_mask, audio_mask = self.prompt_wrap_navit(
                        input_ids,
                        self.config,
                        query_embeds_image,
                        query_embeds_video,
                        query_embeds_audio,
                        query_embeds_audio_lengths,
                        placeholder_audio_loc_lens,
                    )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # Masks are not passed through pipeline
            image_mask = None
            audio_mask = None

        # Pass through decoder layers
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                image_mask=image_mask,
                audio_mask=audio_mask,
                attention_mask=attention_mask,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BailingMoeV2ForCausalLM(nn.Module, CustomProcessMixin):
    """BailingMoeV2 model for causal language modeling, adapted for vLLM.

    Inherits from CustomProcessMixin to support custom preprocessing and postprocessing
    for integration with omni model pipelines.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # BailingMoeV2Config
        config = vllm_config.model_config.hf_text_config
        # from typing import cast
        # config = cast(BailingMoeV2Config, config)
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = BailingMoeV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass for causal LM.

        Returns:
            torch.Tensor: Hidden states from the model.

        Note:
            When used in a unified omni model pipeline, the wrapper can convert
            this output to OmniOutput format using:
                OmniOutput(text_hidden_states=hidden_states,
                          intermediate_tensors=intermediate_tensors)
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        stacked_params_mapping = [
            # BailingMoE uses fused query_key_value
            ("qkv_proj", "query_key_value", None),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embedding inv_freq
            if "rotary_emb.inv_freq" in name:
                continue

            # Remap HF checkpoint names to vLLM parameter names:
            # Router gate: HF stores gate weight directly (e.g. audio_gate.weight)
            #    but vLLM wraps in ReplicatedLinear (audio_gate.gate.weight)
            for router_name in ("gate", "image_gate", "audio_gate"):
                # Only remap .weight, not .expert_bias
                old = f".{router_name}.weight"
                new = f".{router_name}.gate.weight"
                if old in name:
                    name = name.replace(old, new)

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Regular parameter loading
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params
