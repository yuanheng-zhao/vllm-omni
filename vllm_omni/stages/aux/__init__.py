# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary-module stage family for vLLM-Omni.

An "aux stage" hosts one non-autoregressive, non-denoising PyTorch
module (VAE encode / decode, text encoder, CLIP / SigLIP, audio codec,
refiner, upscaler, ...). It is a peer of the ``LLM`` and ``DIFFUSION``
stage types, routed by the orchestrator through the same non-LLM path.

Runtime split:

- :class:`AuxAdapter` — per-``(module_kind, model_arch, op)`` adapter
  that owns tensor-shape math (unpack, tile, denormalize). Lives next
  to the model's pipeline code.
- :class:`AuxModuleRunner` — model-agnostic runner; resolves the
  adapter, loads the module, dispatches requests.
- :class:`StageAuxProc` — subprocess host.
- :class:`StageAuxClient` — orchestrator-facing client.
"""

from vllm_omni.stages.aux.adapter import (
    AuxAdapter,
    AuxAdapterKey,
    AuxAdapterResult,
    get_adapter,
    iter_adapter_keys,
    register_adapter,
)
from vllm_omni.stages.aux.client import StageAuxClient
from vllm_omni.stages.aux.proc import (
    StageAuxProc,
    complete_aux_handshake,
    spawn_aux_proc,
)
from vllm_omni.stages.aux.runner import AuxModuleRunner

__all__ = [
    "AuxAdapter",
    "AuxAdapterKey",
    "AuxAdapterResult",
    "AuxModuleRunner",
    "StageAuxClient",
    "StageAuxProc",
    "complete_aux_handshake",
    "get_adapter",
    "iter_adapter_keys",
    "register_adapter",
    "spawn_aux_proc",
]
