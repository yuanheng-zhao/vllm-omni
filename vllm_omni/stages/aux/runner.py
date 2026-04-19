# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-agnostic runner for aux stages.

The runner is the subprocess-side counterpart of
:class:`~vllm_omni.stages.aux.adapter.AuxAdapter`. It resolves the
adapter for a given ``(module_kind, model_arch, op)`` triple, loads the
underlying PyTorch module once at startup, and dispatches each incoming
request through the adapter's ``pre_transform / forward / post_transform``
pipeline.

Historically the split-VAE demo kept Qwen-Image-specific unpack math
inside the runner (``VAEModelRunner._unpack_qwen_image_latents``). That
is now the adapter's job; the runner has no ``if model_arch ==`` or
``if op ==`` branches.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.stages.aux.adapter import (
    AuxAdapter,
    AuxAdapterResult,
    get_adapter,
)
from vllm_omni.stages.bridge import StageBridgePayload

logger = init_logger(__name__)


class AuxModuleRunner:
    """Hosts one aux module and dispatches adapter-driven requests.

    A runner is bound to a single ``(module_kind, model_arch, op)`` at
    construction. Future extensions could let one runner serve multiple
    ops on the same module (encode + decode on one VAE weight) but the
    first iteration keeps it one-runner-per-op for simplicity.
    """

    def __init__(
        self,
        module_kind: str,
        model_arch: str,
        op: str,
        *,
        model: str,
        device: torch.device,
        engine_args: dict[str, Any],
    ) -> None:
        self._module_kind = module_kind
        self._model_arch = model_arch
        self._op = op
        self._model = model
        self._device = device
        self._engine_args = dict(engine_args)
        self._adapter: AuxAdapter | None = None
        self._module: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def module_kind(self) -> str:
        return self._module_kind

    @property
    def model_arch(self) -> str:
        return self._model_arch

    @property
    def op(self) -> str:
        return self._op

    @property
    def schema_in(self) -> str:
        if self._adapter is None:
            raise RuntimeError("AuxModuleRunner.initialize() not called yet")
        return self._adapter.schema_in

    @property
    def schema_out(self) -> str | None:
        if self._adapter is None:
            raise RuntimeError("AuxModuleRunner.initialize() not called yet")
        return self._adapter.schema_out

    @property
    def final_output_type(self) -> str | None:
        if self._adapter is None:
            return None
        return self._adapter.final_output_type

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Resolve the adapter and load the underlying module."""
        adapter_cls = get_adapter(self._module_kind, self._model_arch, self._op)
        adapter = adapter_cls(**self._engine_args)
        adapter.bind_device(self._device)
        logger.info(
            "[AuxModuleRunner] loading %s/%s/%s from %s on %s",
            self._module_kind,
            self._model_arch,
            self._op,
            self._model,
            self._device,
        )
        self._module = adapter.load_module(self._model, self._device)
        self._adapter = adapter

    def close(self) -> None:
        self._adapter = None
        self._module = None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handle(self, payload_dict: dict[str, Any]) -> dict[str, Any]:
        """Run one request.

        ``payload_dict`` is the on-the-wire dict form of a
        :class:`StageBridgePayload` (as produced by
        :meth:`StageBridgePayload.to_dict`). Returns a dict suitable
        for ZMQ transport back to :class:`StageAuxClient` — see
        :class:`StageAuxProc` for the message shape.
        """
        if self._adapter is None or self._module is None:
            raise RuntimeError("AuxModuleRunner not initialized")

        payload = StageBridgePayload.from_dict(payload_dict)
        module_input = self._adapter.pre_transform(payload)
        with torch.no_grad():
            module_output = self._adapter.forward(module_input)
        result: AuxAdapterResult = self._adapter.post_transform(module_output)
        return self._encode_result(result)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_result(result: AuxAdapterResult) -> dict[str, Any]:
        """Convert an :class:`AuxAdapterResult` to a wire-form dict."""
        if result.final_tensor is not None:
            return {
                "kind": "final",
                "tensor": result.final_tensor.detach().cpu(),
                "extras": dict(result.extras or {}),
            }
        assert result.payload is not None  # invariant: __post_init__
        bp = result.payload
        cpu_tensor = bp.tensor.detach().cpu() if bp.tensor is not None else None
        return {
            "kind": "payload",
            "payload": {
                "schema": bp.schema,
                "tensor": cpu_tensor,
                "extras": dict(bp.extras),
            },
            "extras": dict(result.extras or {}),
        }
