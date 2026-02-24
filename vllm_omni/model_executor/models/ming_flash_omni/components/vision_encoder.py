# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
# Adapted from Ming repository qwen3_moe_vit.py
# https://github.com/inclusionAI/Ming

"""Vision encoder for Ming-flash-omni-2.0.

Reuses vLLM's Qwen3Omni_VisionTransformer which is architecturally identical
to Ming's Qwen3MoeVisionTransformer, with weight name mapping handled in
load_weights().

Key differences between Ming's HF checkpoint and vLLM's implementation:
  - Ming: ``merger.norm`` → vLLM: ``merger.ln_q``
  - Ming: ``merger.linear_fc1`` → vLLM: ``merger.mlp.0``
  - Ming: ``merger.linear_fc2`` → vLLM: ``merger.mlp.2``
  - Ming: ``deepstack_merger_list`` → vLLM: ``merger_list``
  - Ming uses ``num_position_embeddings`` config → vLLM expects ``image_size``
    and ``apply_vit_abs_pos_embed``
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
)
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


def _adapt_vision_config(vision_config):
    """Adapt Ming's Qwen3VLMoeVisionConfig to be compatible with vLLM's
    Qwen3Omni_VisionTransformer expectations.

    Ming uses ``num_position_embeddings`` (e.g. 2304 = 48^2) while vLLM
    expects ``image_size`` and ``apply_vit_abs_pos_embed``.
    """
    if not hasattr(vision_config, "image_size") or vision_config.image_size is None:
        if hasattr(vision_config, "num_position_embeddings") and vision_config.num_position_embeddings:
            import math

            num_grid = int(math.sqrt(vision_config.num_position_embeddings))
            vision_config.image_size = num_grid * vision_config.patch_size
        else:
            vision_config.image_size = vision_config.patch_size * 14  # fallback

    if not hasattr(vision_config, "apply_vit_abs_pos_embed"):
        # Ming always uses nn.Embedding for pos_embed
        vision_config.apply_vit_abs_pos_embed = True

    return vision_config


class MingVisionEncoder(nn.Module):
    """**Wrapper** around vLLM's Qwen3Omni_VisionTransformer for Ming.

    Handles config adaptation and weight name remapping so that Ming's HF
    checkpoint weights can be loaded directly into vLLM's TP-aware ViT.
    """

    # Weight name mapping from Ming HF checkpoint names to vLLM parameter names.
    # Applied during load_weights() to translate checkpoint keys.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "deepstack_merger_list.": "merger_list.",
            "merger.norm.": "merger.ln_q.",
            "merger.linear_fc1.": "merger.mlp.0.",
            "merger.linear_fc2.": "merger.mlp.2.",
        }
    )

    def __init__(
        self,
        vision_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        adapted_config = _adapt_vision_config(vision_config)
        norm_eps = 1e-6
        self.encoder = Qwen3Omni_VisionTransformer(
            vision_config=adapted_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.image_emb_dim = vision_config.out_hidden_size
        self.use_deepstack = (
            hasattr(vision_config, "deepstack_visual_indexes") and vision_config.deepstack_visual_indexes is not None
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run vision encoder.

        Args:
            pixel_values: Flattened pixel values.
            grid_thw: [num_images, 3] tensor of (t, h, w) grid sizes.

        Returns:
            If deepstack is enabled, returns concatenated multi-scale features
            along the feature dim: [seq_len, hidden_size * (1 + num_deepstack)].
            Otherwise returns [seq_len, hidden_size].
        """
        return self.encoder(pixel_values, grid_thw=grid_thw)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        remapped_weights = self.hf_to_vllm_mapper.apply(weights)
        return self.encoder.load_weights(remapped_weights)
