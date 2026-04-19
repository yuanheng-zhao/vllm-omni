# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-agnostic bridge-payload layer for multi-stage pipelines.

A ``StageBridgePayload`` is a typed, schema-tagged carrier used to move a
tensor (plus small typed metadata) from one stage to another. Unlike the
ad-hoc ``custom_output["remote_vae_payload"]`` dict used by the initial
split-VAE demo, bridge payloads are producer-agnostic: the same reader
helper can pick them up regardless of whether the upstream stage emitted
an :class:`~vllm_omni.outputs.OmniRequestOutput` (diffusion / aux stage)
or an LLM-style output with ``additional_information``.

The intent is to give every pipeline a single narrow-waist contract for
"payload that bridges two stages" so that aux stages (VAE encode/decode,
text encoders, audio codecs, ...) can be wired after an LLM stage just
as easily as after a diffusion stage.

Schema naming convention: ``<module_kind>.<op>.<version>`` (e.g.
``vae.latents.v1``) or model-scoped when the latent-space normalization
is model-specific (e.g. ``qwen_image.vae.latents.v1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


@dataclass
class StageBridgePayload:
    """A typed, schema-tagged tensor handoff between two stages.

    Attributes:
        schema: Fully-qualified schema name, used by the downstream
            reader to filter payloads. Must be registered via
            :func:`register_schema` at import time of the producing
            module so version skew is detectable at startup.
        tensor: The primary tensor (may be ``None`` for schemas that
            only carry metadata).
        extras: Small, schema-specific typed metadata (ints / strs /
            small lists). Anything large should ride in ``tensor``.
    """

    schema: str
    tensor: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # --- transport helpers ---
    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for ZMQ / msgpack transport."""
        return {
            "schema": self.schema,
            "tensor": self.tensor,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageBridgePayload:
        if "schema" not in data:
            raise ValueError(f"bridge payload dict missing 'schema': {list(data)}")
        return cls(
            schema=data["schema"],
            tensor=data.get("tensor"),
            extras=dict(data.get("extras") or {}),
        )

    # --- convenience ---
    def get(self, key: str, default: Any = None) -> Any:
        """Read a key from ``extras``."""
        return self.extras.get(key, default)


# ---------------------------------------------------------------------------
# Schema registry (narrow-waist version check)
# ---------------------------------------------------------------------------


_SCHEMAS: dict[str, dict[str, Any]] = {}


def register_schema(
    schema: str,
    *,
    description: str = "",
    required_extras: tuple[str, ...] = (),
) -> None:
    """Register a bridge-payload schema.

    Idempotent: re-registering a schema with identical metadata is a
    no-op. Re-registering with different metadata raises so producer
    and consumer modules cannot silently disagree on a schema shape.
    """
    spec = {"description": description, "required_extras": tuple(required_extras)}
    existing = _SCHEMAS.get(schema)
    if existing is not None and existing != spec:
        raise ValueError(f"bridge schema {schema!r} re-registered with different metadata: {existing} vs {spec}")
    _SCHEMAS[schema] = spec


def is_registered(schema: str) -> bool:
    return schema in _SCHEMAS


def _validate(payload: StageBridgePayload) -> None:
    spec = _SCHEMAS.get(payload.schema)
    if spec is None:
        # Unknown schemas are a warning, not a hard error, so that
        # development-time additions don't explode production before the
        # producer side has been taught to register.
        logger.debug("bridge payload schema %r is not registered", payload.schema)
        return
    missing = [k for k in spec["required_extras"] if k not in payload.extras]
    if missing:
        raise ValueError(f"bridge payload {payload.schema!r} missing required extras: {missing}")


# ---------------------------------------------------------------------------
# Storage conventions
# ---------------------------------------------------------------------------
#
# Two physical slots, one logical list per output object:
#
#   * OmniRequestOutput.custom_output["bridge_payloads"] -> list[dict]
#       (produced by diffusion / aux stages)
#   * EngineCoreOutputs.additional_information["bridge_payloads"] -> list[dict]
#       (produced by LLM stages — reader ready, producer wiring deferred)
#
# Each entry is ``StageBridgePayload.to_dict()``.
#
# The helpers below hide the slot split from callers.

_SLOT_KEY = "bridge_payloads"


def _get_non_llm_slot(output: Any) -> tuple[dict[str, Any], bool]:
    """Return ``(custom_output_dict, is_omni_request_output)``.

    The dict is a live reference — mutating it mutates the output.
    """
    # OmniRequestOutput has a property-backed custom_output with a setter.
    custom = getattr(output, "custom_output", None)
    if custom is None:
        custom = {}
        try:
            output.custom_output = custom
        except AttributeError:
            pass
    return custom, True


def _get_llm_slot(output: Any) -> dict[str, Any] | None:
    """Return the LLM-side additional_information dict, or None."""
    addl = getattr(output, "additional_information", None)
    if isinstance(addl, dict):
        return addl
    return None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_bridge_payload(output: Any, payload: StageBridgePayload) -> None:
    """Append ``payload`` to the appropriate slot on ``output``.

    Supports:

    - :class:`~vllm_omni.outputs.OmniRequestOutput` (diffusion / aux):
      stored under ``custom_output[_SLOT_KEY]``.
    - :class:`~vllm_omni.diffusion.data.DiffusionOutput`: same, via its
      ``custom_output`` field. Pipelines emitting from
      :meth:`_decode_latents` typically build a ``DiffusionOutput``
      directly; this helper makes the mutation explicit.
    - Any object exposing an ``additional_information: dict``
      (LLM-stage path): stored under
      ``additional_information[_SLOT_KEY]``. Reader support is the
      primary motivation; writer support is a convenience for unified
      MLLMs that want to emit latents.
    """
    _validate(payload)
    serialized = payload.to_dict()

    llm_slot = _get_llm_slot(output)
    if llm_slot is not None:
        payloads = llm_slot.setdefault(_SLOT_KEY, [])
        payloads.append(serialized)
        return

    slot, _ = _get_non_llm_slot(output)
    payloads = slot.setdefault(_SLOT_KEY, [])
    payloads.append(serialized)
    # Assign back: custom_output may be a property-backed setter on
    # OmniRequestOutput. setdefault has already mutated the underlying
    # dict, but on some wrappers the getter returns a copy; write back
    # defensively.
    try:
        output.custom_output = slot
    except AttributeError:
        pass


def build_custom_output_with_payload(
    payload: StageBridgePayload,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a new ``custom_output`` dict with ``payload`` attached.

    Convenience for pipelines that build a fresh
    :class:`~vllm_omni.diffusion.data.DiffusionOutput` per step and want
    to stash a payload on it inline.
    """
    _validate(payload)
    out = dict(base or {})
    out.setdefault(_SLOT_KEY, []).append(payload.to_dict())
    return out


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _iter_payloads_on_output(output: Any) -> list[dict[str, Any]]:
    """Extract the raw bridge-payload dicts from a single output object.

    Returns an empty list if none are present — absent slot is not an
    error at this layer; the caller's ``strict`` controls that.
    """
    # LLM-side slot
    addl = _get_llm_slot(output)
    if addl is not None:
        llm_bp = addl.get(_SLOT_KEY)
        if llm_bp:
            return list(llm_bp)

    # Non-LLM slot
    custom = getattr(output, "custom_output", None)
    if isinstance(custom, dict):
        bp = custom.get(_SLOT_KEY)
        if bp:
            return list(bp)

    return []


def read_bridge_payloads(
    stage_list: list[Any],
    schema: str,
    source_ids: list[int],
    *,
    strict: bool = True,
) -> list[StageBridgePayload]:
    """Collect bridge payloads matching ``schema`` from upstream stages.

    Walks ``stage_list[src].engine_outputs`` for each ``src`` in
    ``source_ids`` and returns every payload whose ``schema`` field
    matches.

    When no matching payload is found and ``strict`` is true, raises
    ``RuntimeError`` with context about the missing schema and sources.
    """
    if not source_ids:
        raise ValueError("read_bridge_payloads requires a non-empty source_ids list")

    out: list[StageBridgePayload] = []
    for src_id in source_ids:
        try:
            client = stage_list[src_id]
        except (IndexError, KeyError) as exc:
            raise RuntimeError(f"unknown source stage id {src_id!r}") from exc
        outputs = getattr(client, "engine_outputs", None)
        if not outputs:
            continue
        for output in outputs:
            for raw in _iter_payloads_on_output(output):
                if raw.get("schema") == schema:
                    out.append(StageBridgePayload.from_dict(raw))

    if strict and not out:
        raise RuntimeError(
            f"No bridge payload with schema {schema!r} found in stages {source_ids}. "
            "Did the upstream stage emit a StageBridgePayload via write_bridge_payload "
            "or build_custom_output_with_payload?"
        )
    return out


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


_LEGACY_REMOTE_VAE_KEY = "remote_vae_payload"
_LEGACY_REMOTE_VAE_SCHEMA = "qwen_image.vae.latents.v1"


def read_bridge_payloads_with_legacy_fallback(
    stage_list: list[Any],
    schema: str,
    source_ids: list[int],
    *,
    legacy_key: str = _LEGACY_REMOTE_VAE_KEY,
    legacy_schema: str = _LEGACY_REMOTE_VAE_SCHEMA,
    strict: bool = True,
) -> list[StageBridgePayload]:
    """Like :func:`read_bridge_payloads`, with a one-release fallback.

    Promotes any legacy ``custom_output[legacy_key]`` dict on upstream
    outputs to a ``StageBridgePayload`` with ``legacy_schema`` before
    matching. This lets the Qwen-Image demo's ``remote_vae_payload``
    path keep working while producers migrate to the explicit bridge.
    """
    collected = read_bridge_payloads(stage_list, schema, source_ids, strict=False)
    if collected:
        return collected
    if schema != legacy_schema:
        if strict:
            raise RuntimeError(f"No bridge payload with schema {schema!r} found in stages {source_ids}")
        return []

    # Try legacy key promotion
    promoted: list[StageBridgePayload] = []
    for src_id in source_ids:
        client = stage_list[src_id]
        for output in getattr(client, "engine_outputs", None) or []:
            custom = getattr(output, "custom_output", None)
            if not isinstance(custom, dict):
                continue
            legacy = custom.get(legacy_key)
            if not isinstance(legacy, dict):
                continue
            promoted.append(
                StageBridgePayload(
                    schema=legacy_schema,
                    tensor=legacy.get("packed_latents"),
                    extras={k: v for k, v in legacy.items() if k != "packed_latents"},
                )
            )

    if strict and not promoted:
        raise RuntimeError(
            f"No bridge payload with schema {schema!r} (legacy key {legacy_key!r}) found in stages {source_ids}"
        )
    return promoted
