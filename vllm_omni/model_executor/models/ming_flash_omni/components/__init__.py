# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming model components - ported from Ming repository."""

from .audio_encoder import WhisperAudioEncoder
from .configuration_bailing_moe_v2 import BailingMoeV2Config
from .modeling_bailing_moe_v2 import (
    BailingMoeV2ForCausalLM,
    BailingMoeV2Model,
    get_rope_index,
)
from .modeling_utils import (
    Transpose,
    build_modality_mask,
    encode_audio_segments,
    patch_continuous_features,
    unwrap_feats,
    wrap_feats,
)
from .projectors import AudioProjector, VisionProjector
from .vision_encoder import MingVisionEncoder

__all__ = [
    # Config
    "BailingMoeV2Config",
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
    "patch_continuous_features",
    "build_modality_mask",
    "encode_audio_segments",
    "unwrap_feats",
    "wrap_feats",
    "get_rope_index",
]
