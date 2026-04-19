# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy ``StageVAEClient`` — thin compat shim over :class:`StageAuxClient`.

The split-VAE demo originally carried its own stage client with a
``stage_type = "diffusion"`` masquerade (so the orchestrator's existing
diffusion branch would route it unchanged) and a hard-coded
``decode_qwen_image`` RPC. Both concerns now live in the generic
aux-stage runtime:

- Routing: the orchestrator recognizes ``stage_type == "aux"`` natively
  (see :mod:`vllm_omni.engine.orchestrator`), so no masquerade is
  required.
- Model-specific math: handled by
  :class:`vllm_omni.model_executor.models.qwen_image.QwenImageVaeDecodeAdapter`
  resolved via the adapter registry with
  ``(module_kind="vae", model_arch="qwen_image", op="decode")``.

This shim is kept so code that still imports
``vllm_omni.diffusion.stage_vae_client.StageVAEClient`` (or that
instantiates it via the legacy ``stage_type: vae`` YAML path) keeps
working. New code should use :class:`StageAuxClient` directly.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.stages.aux import StageAuxClient

logger = init_logger(__name__)


class StageVAEClient(StageAuxClient):
    """Back-compat alias for :class:`StageAuxClient`.

    Accepts the original ``(model, vae_subfolder, torch_dtype, device)``
    keyword set and synthesizes the aux-stage dispatch triple
    ``(vae, qwen_image, decode)`` that preserves the split-VAE demo's
    behavior.
    """

    def __init__(
        self,
        model: str,
        vae_subfolder: str = "vae",
        torch_dtype: str = "bfloat16",
        device: str = "cuda:0",
        stage_init_timeout: int = 600,
        metadata: Any = None,
    ) -> None:
        logger.debug("[StageVAEClient] legacy shim constructor — delegating to StageAuxClient(vae/qwen_image/decode)")
        super().__init__(
            module_kind="vae",
            model_arch="qwen_image",
            op="decode",
            model=model,
            device=device,
            engine_args={
                "vae_subfolder": vae_subfolder,
                "torch_dtype": torch_dtype,
            },
            stage_init_timeout=stage_init_timeout,
            metadata=metadata,
        )
