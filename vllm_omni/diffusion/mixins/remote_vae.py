# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin for diffusion pipelines that route VAE decode to an aux stage.

When ``od_config.remote_vae`` is True, ``_decode_latents`` should short-
circuit and emit the packed latents + geometry via a
:class:`~vllm_omni.stages.bridge.StageBridgePayload` instead of calling
the in-process VAE. The downstream stage (a generic aux stage running
:class:`~vllm_omni.stages.aux.StageAuxProc` with the
``QwenImageVaeDecodeAdapter``) picks this up through the orchestrator's
engine_outputs plumbing.

Bridge schema: ``qwen_image.vae.latents.v1``. The legacy
``custom_output["remote_vae_payload"]`` shape is still emitted for
one release so consumers can migrate; both paths carry the same bytes.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.stages.bridge import (
    StageBridgePayload,
    build_custom_output_with_payload,
)

# Schema the QwenImageVaeDecodeAdapter consumes.
_BRIDGE_SCHEMA = "qwen_image.vae.latents.v1"


class RemoteVaeMixin:
    """Opt-in remote-VAE emission for diffusion pipelines.

    Pipelines inherit this mixin, set ``self._remote_vae`` from
    ``od_config.remote_vae`` in ``__init__``, and call
    ``_emit_remote_vae_output(...)`` from ``_decode_latents`` when the flag
    is set. ``_build_remote_vae_payload`` can be overridden if a pipeline
    needs extra geometry / keys beyond the default Qwen-Image packing;
    whatever it returns is surfaced on the bridge payload as ``extras``.
    """

    _remote_vae: bool = False
    vae_scale_factor: int

    def _build_remote_vae_payload(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        """Return the legacy dict shape.

        Kept for subclass overrides; the bridge-payload path derives its
        tensor + extras from the same values so both consumers see the
        same bytes during the one-release migration window.
        """
        return {
            "packed_latents": latents.detach().cpu(),
            "height": int(height),
            "width": int(width),
            "vae_scale_factor": int(self.vae_scale_factor),
        }

    def _build_remote_vae_bridge_payload(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> StageBridgePayload:
        """Return the :class:`StageBridgePayload` the aux adapter consumes."""
        return StageBridgePayload(
            schema=_BRIDGE_SCHEMA,
            tensor=latents.detach().cpu(),
            extras={
                "height": int(height),
                "width": int(width),
                "vae_scale_factor": int(self.vae_scale_factor),
            },
        )

    def _emit_remote_vae_output(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> DiffusionOutput:
        # Emit both the bridge payload (primary) and the legacy key
        # (compat). ``read_bridge_payloads_with_legacy_fallback`` on the
        # reader side promotes the legacy key if the bridge payload is
        # missing, so this double-write window is safe in both
        # directions.
        bridge_payload = self._build_remote_vae_bridge_payload(latents, height, width)
        custom_output = build_custom_output_with_payload(bridge_payload)
        custom_output["remote_vae_payload"] = self._build_remote_vae_payload(latents, height, width)
        return DiffusionOutput(
            output=None,
            custom_output=custom_output,
            stage_durations=getattr(self, "stage_durations", None) or {},
        )
