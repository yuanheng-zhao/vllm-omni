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

"""Ming-flash-omni-2.0 unified model (thinker + imagegen + talker)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

from .configuration_ming_flash_omni import (
    MingFlashOmniConfig,
    MingFlashOmniThinkerConfig,
)

logger = init_logger(__name__)


# TODO: Register multimodal processor when implementing multimodal support
# @MULTIMODAL_REGISTRY.register_processor(
#     MingFlashOmniMultiModalProcessor,
#     info=MingFlashOmniProcessingInfo,
#     dummy_inputs=MingFlashOmniDummyInputsBuilder,
# )
class MingFlashOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """
    Unified Ming-flash-omni-2.0 model combining thinker, imagegen, and talker.

    Architecture:
    - Thinker (Stage 0): Multimodal understanding (text + image + video + audio) → text generation
    - Image Generator (Stage 1a): Text embeddings → PIL Image [NOT IMPLEMENTED YET]
    - Talker (Stage 1b): Text embeddings → Audio waveform (TTS) [NOT IMPLEMENTED YET]

    Usage:
        Set `model_stage` in vllm_config to one of: "thinker", "imagegen", "talker"

    Supports Multi-Dimensional RoPE (MRoPE) for 3D position encoding in multimodal contexts.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config: MingFlashOmniConfig = vllm_config.model_config.hf_config

        # Keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        self.config = config

        # Get thinker config
        thinker_config: MingFlashOmniThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config

        # Determine model stage
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            # Initialize thinker model (multimodal processing + text generation)
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config.llm_config, architectures=["MingFlashOmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                architectures=["MingFlashOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.imagegen = None
            self.talker = None

        elif self.model_stage == "imagegen":
            # TODO: Implement image generator stage
            raise NotImplementedError(
                "Image generation stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        elif self.model_stage == "talker":
            # TODO: Implement talker (TTS) stage
            raise NotImplementedError(
                "Talker (TTS) stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'imagegen', 'talker'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else lambda: None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches,
        attn_metadata: AttentionMetadata,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward to active stage."""
        return self.model.forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample next tokens from logits."""
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        raise NotImplementedError("sample method not available on current stage")

    def get_mrope_input_positions(self, *args, **kwargs):
        """Get MRoPE input positions - delegates to active stage model."""
        if hasattr(self.model, "get_mrope_input_positions"):
            return self.model.get_mrope_input_positions(*args, **kwargs)
        raise NotImplementedError("get_mrope_input_positions not available on current stage")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for all components of the Ming model.

        Weights are expected to have prefixes:
        - thinker.* for thinker stage
        - imagegen.* for image generator stage
        - talker.* for talker stage
        """
        loaded_weights = set()
        thinker_weights = []
        imagegen_weights = []
        talker_weights = []

        # Separate weights by component prefix
        for name, value in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name, value))
            elif name.startswith("imagegen."):
                imagegen_weights.append((name, value))
            elif name.startswith("talker."):
                talker_weights.append((name, value))
            else:
                # Weights without prefix go to thinker by default
                thinker_weights.append((name, value))

        # Load thinker weights if available
        if self.model_stage == "thinker" and thinker_weights:
            # Remove "thinker." prefix before loading
            thinker_weights_stripped = [
                (name.replace("thinker.", "", 1) if name.startswith("thinker.") else name, value)
                for name, value in thinker_weights
            ]
            thinker_loaded = self.thinker.load_weights(thinker_weights_stripped)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # TODO: Load imagegen weights when implemented
        # TODO: Load talker weights when implemented

        return loaded_weights

    def get_mm_mapping(self) -> str | None:
        """Get multimodal mapping configuration."""
        # TODO: Implement when multimodal support is added
        return None

    @property
    def sampler(self):
        """Get sampler from active model."""
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return None
