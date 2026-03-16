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
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul

# vLLM imports
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMoeV2Config

from .modeling_utils import build_modality_mask, patch_continuous_features

logger = init_logger(__name__)


class MingVideoRopeMRotaryEmbedding(MRotaryEmbedding):
    """MRotaryEmbedding with Ming's video_rope cos/sin interleaving.

    Unlike standard mrope which maps contiguous frequency sections to T/H/W,
    video_rope alternates H/W frequencies element-wise in the spatial section
    and places temporal frequencies at the end:
        Standard mrope:  [T T T ... H H H ... W W W ...]
        Video rope:      [H W H W ... H W ... T T T ...]

    Refer to Ming's BailingMoeV2RotaryEmbedding3D
    https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py#L174
    """

    def _remap_video_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap 3D cos/sin to video_rope interleaved layout.

        Args:
            cos, sin: [3, num_tokens, rotary_dim // 2]
        Returns:
            cos, sin: [num_tokens, rotary_dim // 2]

        Refer to Ming's apply_3d_rotary_pos_emb
        https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py#L226
        """
        assert self.mrope_section is not None
        hw_size = self.mrope_section[1] + self.mrope_section[2]

        result_cos = torch.empty_like(cos[0])
        result_sin = torch.empty_like(sin[0])

        # Spatial frequencies: even indices from H (dim 1), odd from W (dim 2)
        result_cos[:, 0:hw_size:2] = cos[1, :, 0:hw_size:2]
        result_cos[:, 1:hw_size:2] = cos[2, :, 1:hw_size:2]
        result_sin[:, 0:hw_size:2] = sin[1, :, 0:hw_size:2]
        result_sin[:, 1:hw_size:2] = sin[2, :, 1:hw_size:2]

        # Temporal frequencies at the end
        result_cos[:, hw_size:] = cos[0, :, hw_size:]
        result_sin[:, hw_size:] = sin[0, :, hw_size:]

        return result_cos, result_sin

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        if positions.ndim == 2:
            cos, sin = self._remap_video_rope(cos, sin)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb.forward_native(key_rot, cos, sin)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # No custom Triton kernel for video_rope; fall back to native for 3D
        # TODO: Consider custom optimization
        if positions.ndim == 2:
            return self.forward_native(positions, query, key, offsets)
        return super().forward_cuda(positions, query, key, offsets)

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)


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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
            logger.debug("Loaded weight %s", name)
        return loaded_params


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
        logits, _ = self.gate(hidden_states)

        logits = logits.float()  # cast for numerical precision
        scores = torch.sigmoid(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class _CachedRoutingFn:
    """Stateful callable that returns pre-computed routing results.

    Used as ``custom_routing_function`` for :class:`FusedMoE` so that
    the multimodal multi-router logic can be computed externally while
    still leveraging the fused expert kernels.
    """

    def __init__(self):
        self._topk_weights: torch.Tensor | None = None
        self._topk_ids: torch.Tensor | None = None

    def cache(self, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        self._topk_weights = topk_weights
        self._topk_ids = topk_ids

    def __call__(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._topk_weights is not None, "call cache() before forward"
        return self._topk_weights.to(torch.float32), self._topk_ids.to(torch.int32)


class BailingMoeV2SparseMoeBlock(nn.Module):
    """Sparse MoE block with MultiRouter support for multimodal routing.

    Keep the custom multi-router gating logic external.
    """

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Routing function for FusedMoE (populated before each forward)
        self._routing_fn = _CachedRoutingFn()

        # Replace nn.ModuleList of individual experts with FusedMoE
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            custom_routing_function=self._routing_fn,
            renormalize=False,  # we handle normalization in the gate
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        # Set expert weight mapping so AutoWeightsLoader can dispatch
        # checkpoint weights (experts.{i}.gate_proj/up_proj/down_proj)
        # into the fused w13/w2 tensors.
        self.experts.expert_mapping = FusedMoE.make_expert_params_mapping(
            self.experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.num_experts,
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
        input_is_2d = hidden_states.ndim == 2
        if input_is_2d:
            hidden_states = hidden_states.unsqueeze(0)

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

        # Cache pre-computed routing for the custom_routing_function
        self._routing_fn.cache(topk_weight, topk_idx)

        # FusedMoE expects 2D hidden_states and router_logits
        hidden_states_2d = hidden_states.view(-1, h)
        y = self.experts(hidden_states_2d, router_logits)
        y = y.view(bsz, seq_len, h)

        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)

        if input_is_2d:
            y = y.squeeze(0)
            router_logits = router_logits.view(bsz, seq_len, -1).squeeze(0)
            topk_idx = topk_idx.view(bsz, seq_len, -1).squeeze(0)
            return y, (router_logits, topk_idx)

        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))


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
        if config.rope_scaling is None:
            raise ValueError("rope_scaling must not be None")

        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        mrope_section = config.rope_scaling.get("mrope_section", [8, 12, 12])

        if rope_type == "video_rope":
            # Ming-specific video_rope with custom H/W interleaving
            self.rotary_emb = MingVideoRopeMRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.rope_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=True,
                dtype=torch.get_default_dtype(),
                mrope_section=mrope_section,
            )
        else:
            # Standard m_rope (rope_type "3D", "default", or None)
            rope_scaling = dict(config.rope_scaling)
            rope_scaling["rope_type"] = "default"  # normalize for get_rope dispatch
            rope_scaling["mrope_section"] = mrope_section
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters={
                    "rope_theta": config.rope_theta,
                    "partial_rotary_factor": config.partial_rotary_factor,
                    **rope_scaling,
                },
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for attention with 3D RoPE.

        Args:
            positions: Position IDs, shape (3, num_tokens) for 3D rope
                or (num_tokens,) for text-only
            hidden_states: Input hidden states, shape (num_tokens, hidden_size)

        Returns:
            Attention output tensor, shape (num_tokens, hidden_size)
        """
        # QKV projection: [num_tokens, hidden_size] -> [num_tokens, q_size + 2*kv_size]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply Q/K normalization (per-head)
        num_tokens = q.shape[0]
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k.view(num_tokens, self.num_kv_heads, self.head_dim)).view(num_tokens, self.kv_size)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # vLLM attention (handles KV cache, paged attention, etc.)
        attn_output = self.attn(q, k, v)

        output, _ = self.dense(attn_output)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # BailingMoE stores fused query_key_value in the checkpoint
            ("qkv_proj", "query_key_value", None),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
            logger.debug("Loaded weight %s", name)
        return loaded_params


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
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Embeddings
        if get_pp_group().is_first_rank or (self.tie_word_embeddings and get_pp_group().is_last_rank):
            self.word_embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.word_embeddings = PPMissingLayer()

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
        return self.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.word_embeddings = value

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
        inputs_embeds = self.word_embeddings(input_ids)
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
                    hidden_states = self.word_embeddings(input_ids)
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

        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.word_embeddings.weight

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
        # Remap HF checkpoint names to vLLM parameter names:
        # Router gate: HF stores gate weight directly (e.g. audio_gate.weight)
        #    but vLLM wraps in ReplicatedLinear (audio_gate.gate.weight)
        mapper = WeightsMapper(
            orig_to_new_substr={f".{r}.weight": f".{r}.gate.weight" for r in ("gate", "image_gate", "audio_gate")}
        )
        # Stacked param remapping (query_key_value → qkv_proj, gate/up_proj →
        # gate_up_proj) is handled by BailingMoeV2Attention.load_weights and
        # BailingMoeV2MLP.load_weights respectively, which AutoWeightsLoader
        # dispatches to automatically.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
