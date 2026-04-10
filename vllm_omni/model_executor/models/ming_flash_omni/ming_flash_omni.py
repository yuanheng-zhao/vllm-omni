# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
# Adapted from Ming repository modeling_bailingmm2.py
# https://github.com/inclusionAI/Ming
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

"""Ming-flash-omni-2.0 unified model (thinker + talker + imagegen).

For multi-stage deployment, both thinker and talker stages use this unified
wrapper with model_stage='thinker' or 'talker'.  For standalone TTS, use
MingFlashOmniTalkerForConditionalGeneration directly.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.ming_flash_omni import (
    MingFlashOmniConfig,
    MingFlashOmniThinkerConfig,
)

from .ming_flash_omni_thinker import (
    MingFlashOmniThinkerDummyInputsBuilder,
    MingFlashOmniThinkerMultiModalProcessor,
    MingFlashOmniThinkerProcessingInfo,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MingFlashOmniThinkerMultiModalProcessor,
    info=MingFlashOmniThinkerProcessingInfo,
    dummy_inputs=MingFlashOmniThinkerDummyInputsBuilder,
)
class MingFlashOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """Unified Ming-flash-omni-2.0 model (thinker + talker + future imagegen).

    For multi-stage deployment, both thinker and talker stages use this class
    with model_stage='thinker' or 'talker'.  For standalone TTS, use
    MingFlashOmniTalkerForConditionalGeneration directly.
    """

    supports_multimodal = True
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            if isinstance(config, MingFlashOmniConfig):
                thinker_config: MingFlashOmniThinkerConfig = config.thinker_config
            else:
                thinker_config = config  # BailingMM2Config is the thinker config

            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["MingFlashOmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                architectures=["MingFlashOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.make_empty_intermediate_tensors = self.thinker.make_empty_intermediate_tensors

        elif self.model_stage == "talker":
            # The talker sub-model is self-contained: it resolves its own
            # MingFlashOmniTalkerConfig from talker/config.json and handles
            # its own safetensors loading.
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                architectures=["MingFlashOmniTalkerForConditionalGeneration"],
            )
            self.model = self.talker
            self.make_empty_intermediate_tensors = self.talker.make_empty_intermediate_tensors
            # Propagate weight loading attributes from the talker sub-model.
            self.allow_patterns_overrides = self.talker.allow_patterns_overrides
            self.fall_back_to_pt_during_load = self.talker.fall_back_to_pt_during_load

        elif self.model_stage == "imagegen":
            # TODO: Implement image generator stage
            raise NotImplementedError(
                "Image generation stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'talker', 'imagegen'."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        return self.model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        return None

    def get_mrope_input_positions(self, *args, **kwargs):
        if hasattr(self.model, "get_mrope_input_positions"):
            return self.model.get_mrope_input_positions(*args, **kwargs)
        raise NotImplementedError("get_mrope_input_positions not available on current stage")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "thinker":
            # Strip optional "thinker." prefix so the sub-model sees bare keys.
            stripped = ((name.removeprefix("thinker."), value) for name, value in weights)
            thinker_loaded = self.thinker.load_weights(stripped)
            return add_prefix_to_loaded_weights(thinker_loaded, "thinker")

        elif self.model_stage == "talker":
            # The talker's load_weights is self-contained: when standalone
            # it reads talker/model*.safetensors directly via
            # _iter_talker_safetensors(), bypassing the incoming generator.
            talker_loaded = self.talker.load_weights(weights)
            return add_prefix_to_loaded_weights(talker_loaded, "talker")

        return set()

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="thinker.language_model",
            connector=["thinker.linear_proj.", "thinker.linear_proj_audio."],
            tower_model=["thinker.vision.", "thinker.audio."],
        )

    @property
    def sampler(self):
        return getattr(self.model, "sampler", None)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs):
        return self.model.embed_multimodal(**kwargs)
