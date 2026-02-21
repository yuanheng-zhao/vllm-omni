# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
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

"""Configuration for Ming-flash-omni-2.0 model"""

import os
from typing import Any

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BailingMoeV2Config(PretrainedConfig):
    model_type = "bailing_moe_v2"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,
        use_qk_norm=False,
        use_bias=True,
        rms_norm_eps=1e-05,
        norm_head=False,
        tie_word_embeddings=False,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=81920,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        router_type="topN",
        _attn_implementation="flash_attention_2",
        use_interleaved_frame_timestamp=True,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm  # unused

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor
        self.router_type = router_type
        self.use_interleaved_frame_timestamp = use_interleaved_frame_timestamp
        self._attn_implementation = _attn_implementation

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3VLMoeVisionConfig(PretrainedConfig):
    """Configuration class for Qwen3 MoE Vision Transformer"""

    model_type = "qwen3_moe_vit"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class WhisperEncoderConfig(PretrainedConfig):
    """Configuration class for Whisper audio encoder"""

    model_type = "whisper_encoder"

    def __init__(
        self,
        whisper_encoder_config: dict[str, Any] | None = None,
        ds_kernel_size=3,
        ds_stride=2,
        norm_query_embeds=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper_encoder_config = whisper_encoder_config or {}
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride
        self.norm_query_embeds = norm_query_embeds


class BailingMM2Config(PretrainedConfig):
    model_type = "bailingmm_moe_v2_lite"

    def __init__(
        self,
        mlp_depth=1,
        llm_config: BailingMoeV2Config | None = None,
        vision_config: Qwen3VLMoeVisionConfig | None = None,
        audio_config: WhisperEncoderConfig | None = None,
        **kwargs,
    ):
        self.audio_config = WhisperEncoderConfig(**audio_config) if isinstance(audio_config, dict) else audio_config
        self.vision_config = (
            Qwen3VLMoeVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.llm_config = BailingMoeV2Config(**llm_config) if isinstance(llm_config, dict) else llm_config
        self.mlp_depth = mlp_depth
        super().__init__(**kwargs)


class MingFlashOmniThinkerConfig(BailingMM2Config):
    # Override model type
    model_type = "ming_flash_omni_thinker"


class MingFlashOmniConfig(PretrainedConfig):
    """Configuration class for unified Ming-flash-omni-2.0 model"""

    model_type = "ming_flash_omni"

    def __init__(
        self,
        thinker_config: MingFlashOmniThinkerConfig | None = None,
        image_gen_config: dict[str, Any] | None = None,
        talker_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(thinker_config, dict):
            self.thinker_config = MingFlashOmniThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config or MingFlashOmniThinkerConfig()

        # Image generation config (for future implementation)
        self.image_gen_config = image_gen_config

        # Talker config (for future implementation)
        self.talker_config = talker_config
