# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming model components - ported from Ming repository."""

from .audio_encoder import WhisperAudioEncoder
from .modeling_bailing_moe_v2 import (
    BailingMoeV2ForCausalLM,
    BailingMoeV2Model,
    get_rope_index,
)
from .modeling_utils import (
    Transpose,
    build_modality_mask,
    compute_placeholder_loc_lens,
)
from .projectors import AudioProjector, VisionProjector
from .vision_encoder import MingVisionEncoder

__all__ = [
    # Models
    "BailingMoeV2ForCausalLM",
    "BailingMoeV2Model",
    "MingVisionEncoder",
    "WhisperAudioEncoder",
    # Projectors
    "VisionProjector",
    "AudioProjector",
    # Utilities
    "Transpose",
    "compute_placeholder_loc_lens",
    "build_modality_mask",
    "get_rope_index",
]
