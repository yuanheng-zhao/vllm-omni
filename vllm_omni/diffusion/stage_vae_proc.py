# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy re-exports for the VAE-stage subprocess.

The generalized aux-stage runtime in :mod:`vllm_omni.stages.aux`
supersedes the original ``StageVAEProc`` class. This module is retained
so older imports (``from vllm_omni.diffusion.stage_vae_proc import
spawn_vae_proc``) keep working; every symbol forwards to the aux
equivalent.
"""

from __future__ import annotations

from vllm_omni.stages.aux.proc import (
    StageAuxProc as StageVAEProc,  # noqa: F401 — re-export
)
from vllm_omni.stages.aux.proc import (
    complete_aux_handshake as complete_vae_handshake,  # noqa: F401
)
from vllm_omni.stages.aux.proc import (
    spawn_aux_proc,
)


def spawn_vae_proc(
    model: str,
    vae_subfolder: str = "vae",
    torch_dtype: str = "bfloat16",
    device: str = "cuda:0",
    handshake_address: str | None = None,
    request_address: str | None = None,
    response_address: str | None = None,
):
    """Legacy shim — delegates to :func:`spawn_aux_proc`."""
    return spawn_aux_proc(
        module_kind="vae",
        model_arch="qwen_image",
        op="decode",
        model=model,
        device=device,
        engine_args={
            "vae_subfolder": vae_subfolder,
            "torch_dtype": torch_dtype,
        },
        handshake_address=handshake_address,
        request_address=request_address,
        response_address=response_address,
    )


__all__ = [
    "StageVAEProc",
    "complete_vae_handshake",
    "spawn_vae_proc",
]
