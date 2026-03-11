# SPDX-License-Identifier: Apache-2.0
# Re-export from canonical location for backward compatibility.
# The actual config classes now live in vllm_omni.transformers_utils.configs.ming_flash_omni
# so that AutoConfig.register() runs at import time, consistent with other models.

from vllm_omni.transformers_utils.configs.ming_flash_omni import (  # noqa: F401
    BailingMM2Config,
    BailingMoeV2Config,
    MingFlashOmniConfig,
    MingFlashOmniThinkerConfig,
    Qwen3VLMoeVisionConfig,
    WhisperEncoderConfig,
)

__all__ = [
    "BailingMM2Config",
    "BailingMoeV2Config",
    "MingFlashOmniConfig",
    "MingFlashOmniThinkerConfig",
    "Qwen3VLMoeVisionConfig",
    "WhisperEncoderConfig",
]
