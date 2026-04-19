# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image model integration for vLLM-Omni.

Importing this package as a side effect registers the aux-stage adapters
owned by the Qwen-Image pipeline (currently: VAE decode). The
:class:`StageAuxProc` subprocess imports
``vllm_omni.model_executor.models.<model_arch>`` during startup so the
adapter-registry lookup for ``(vae, qwen_image, decode)`` resolves.
"""

from vllm_omni.model_executor.models.qwen_image.vae_decode_adapter import (
    QwenImageVaeDecodeAdapter,
)
from vllm_omni.stages.aux import register_adapter

# Register the VAE-decode adapter at import time. ``register_adapter`` is
# idempotent so repeated imports (e.g. across subprocess fork + engine
# startup) are harmless.
register_adapter(
    module_kind="vae",
    model_arch="qwen_image",
    op="decode",
    adapter_cls=QwenImageVaeDecodeAdapter,
)

__all__ = ["QwenImageVaeDecodeAdapter"]
