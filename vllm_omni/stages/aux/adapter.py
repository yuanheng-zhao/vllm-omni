# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-(model, op) adapters for aux stages.

An :class:`AuxAdapter` owns everything that is model-specific inside an
aux subprocess: loading the PyTorch module, unpack / tile / denormalize
transforms on the input, and any post-processing of the module output
before it is either returned as the final stage output or repacked into
a :class:`~vllm_omni.stages.bridge.StageBridgePayload` for a downstream
stage.

The :class:`AuxModuleRunner` (see :mod:`vllm_omni.stages.aux.runner`) is
deliberately model-agnostic — it asks the registry for the adapter
matching the stage's ``(module_kind, model_arch, op)`` triple and
dispatches through it. Adding support for a new model means registering
a new adapter class next to the model's pipeline code; no changes to
the runner, the subprocess, or the orchestrator are required.

Registration happens at import time of the model package (typically
from ``model_executor/models/<model>/__init__.py``).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.stages.bridge import StageBridgePayload

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Adapter key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxAdapterKey:
    """Lookup key for :class:`AuxAdapter` registrations.

    ``module_kind`` is the coarse category (``"vae"``, ``"text_encoder"``,
    ``"audio_codec"``, ...). ``model_arch`` is the model-specific key
    used to pick up normalization constants or packing conventions
    (e.g. ``"qwen_image"``, ``"flux2"``). ``op`` is the operation the
    aux stage performs (``"decode"``, ``"encode"``, ``"embed"``).
    """

    module_kind: str
    model_arch: str
    op: str

    def __str__(self) -> str:  # pragma: no cover - diag helper
        return f"{self.module_kind}:{self.model_arch}:{self.op}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[AuxAdapterKey, type[AuxAdapter]] = {}


def register_adapter(
    module_kind: str,
    model_arch: str,
    op: str,
    adapter_cls: type[AuxAdapter],
) -> None:
    """Register an adapter class for a ``(module_kind, model_arch, op)``.

    Idempotent when the same class is registered twice. Raises if a
    different class is registered for an existing key (that would be a
    silent override and is almost always a bug).
    """
    key = AuxAdapterKey(module_kind=module_kind, model_arch=model_arch, op=op)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not adapter_cls:
        raise ValueError(
            f"AuxAdapter for {key} already registered as {existing.__name__}; "
            f"attempted to re-register with {adapter_cls.__name__}"
        )
    _REGISTRY[key] = adapter_cls
    logger.debug("Registered aux adapter %s -> %s", key, adapter_cls.__name__)


def get_adapter(module_kind: str, model_arch: str, op: str) -> type[AuxAdapter]:
    """Look up the adapter class for a ``(module_kind, model_arch, op)``.

    Raises :class:`LookupError` with a helpful list of registered keys
    on miss.
    """
    key = AuxAdapterKey(module_kind=module_kind, model_arch=model_arch, op=op)
    cls = _REGISTRY.get(key)
    if cls is None:
        known = sorted(str(k) for k in _REGISTRY)
        raise LookupError(f"No AuxAdapter registered for {key}. Known adapters: {known or '(none)'}")
    return cls


def iter_adapter_keys() -> list[AuxAdapterKey]:
    """Return a snapshot of every registered adapter key (for diagnostics)."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class AuxAdapter(abc.ABC):
    """Abstract base for per-(model, op) aux-stage adapters.

    Subclasses override :meth:`load_module`, :meth:`pre_transform`,
    :meth:`forward`, and :meth:`post_transform`. The runner's call
    order is:

    1. ``adapter = adapter_cls(**engine_args)`` — adapter constructs with
       engine args (torch_dtype, vae_subfolder, etc.).
    2. ``module = adapter.load_module(model, device)`` — loads the
       underlying torch.nn.Module on the given device.
    3. For each request::

           x = adapter.pre_transform(payload)
           y = adapter.forward(x)
           result = adapter.post_transform(y)

    The adapter is the only model-specific code path; the runner,
    subprocess, and orchestrator are oblivious to what ``module`` does.

    Attributes:
        schema_in: Bridge-payload schema this adapter consumes
            (e.g. ``"qwen_image.vae.latents.v1"``). The runner uses
            this to validate inputs and to advertise the default
            ``custom_process_input_func`` for this stage.
        schema_out: Optional schema for the output payload. ``None`` if
            the adapter produces a final user-visible tensor (e.g.
            decoded image) rather than a payload for a downstream
            stage.
        final_output_type: Hint for :class:`OmniRequestOutput` when the
            adapter produces a final output (``"image"``, ``"audio"``,
            ``"video"``).
    """

    schema_in: str = ""
    schema_out: str | None = None
    final_output_type: str | None = None

    def __init__(self, **engine_args: Any) -> None:
        self._engine_args = dict(engine_args)
        self._device: torch.device | None = None

    # -- loading --------------------------------------------------------

    @abc.abstractmethod
    def load_module(self, model: str, device: torch.device) -> torch.nn.Module:
        """Instantiate the underlying torch.nn.Module.

        Implementations may read any engine-arg they were constructed
        with (e.g. ``vae_subfolder``, ``torch_dtype``) from
        ``self._engine_args``. Expected to return a module in eval
        mode, already moved to ``device``.
        """

    # -- request lifecycle ---------------------------------------------

    @abc.abstractmethod
    def pre_transform(self, payload: StageBridgePayload) -> Any:
        """Convert a bridge payload into the module's input.

        For a VAE decode adapter this is typically: move to device,
        unpack packed latents, denormalize. For a text encoder adapter
        it would be: tokenize, batch, build attention mask.
        """

    @abc.abstractmethod
    def forward(self, module_input: Any) -> Any:
        """Call ``self.module(module_input)`` and return the output.

        Kept as a method (rather than just invoking the module
        directly) so adapters can gate with ``torch.no_grad()``, pick
        the right forward flavor (``decode`` vs ``encode``), or apply
        tiling / patch-parallel reduction transparently.
        """

    @abc.abstractmethod
    def post_transform(self, module_output: Any) -> AuxAdapterResult:
        """Pack the module output into a runner-understood result.

        The result selects between "this is the final user-visible
        output" and "this is a bridge payload for the next stage".
        """

    # -- shared utilities ----------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("AuxAdapter used before load_module(); no device bound")
        return self._device

    def bind_device(self, device: torch.device) -> None:
        self._device = device

    def resolve_torch_dtype(self, default: str = "bfloat16") -> torch.dtype:
        raw = self._engine_args.get("torch_dtype", default)
        if isinstance(raw, torch.dtype):
            return raw
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        key = str(raw).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported torch_dtype {raw!r}; got {list(mapping)}")
        return mapping[key]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AuxAdapterResult:
    """What an adapter returns to the runner.

    Exactly one of ``final_tensor`` / ``payload`` should be set.

    Attributes:
        final_tensor: When the adapter produces user-visible output
            (e.g. a decoded image tensor). The runner packs it into
            :class:`OmniRequestOutput` using ``final_output_type``.
        payload: When the adapter feeds a downstream stage. The runner
            writes the payload via
            :func:`vllm_omni.stages.bridge.write_bridge_payload`.
        extras: Optional free-form dict merged into the output's
            ``custom_output`` (e.g. timing breakdowns). Kept small.
    """

    final_tensor: torch.Tensor | None = None
    payload: StageBridgePayload | None = None
    extras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.final_tensor is None) == (self.payload is None):
            raise ValueError(
                "AuxAdapterResult must set exactly one of final_tensor or payload; "
                f"got final_tensor={self.final_tensor is not None}, "
                f"payload={self.payload is not None}"
            )
