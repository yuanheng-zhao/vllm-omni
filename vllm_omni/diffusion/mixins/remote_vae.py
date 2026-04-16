# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin for diffusion pipelines that route VAE decode to a separate stage.

When ``od_config.remote_vae`` is True, ``_decode_latents`` should short-
circuit and emit the packed latents + geometry via
``DiffusionOutput.custom_output["remote_vae_payload"]`` instead of calling
the in-process VAE.  The downstream stage (``StageVAEProc``) picks this
up through the orchestrator's engine_outputs plumbing.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput


class RemoteVaeMixin:
    """Opt-in remote-VAE emission for diffusion pipelines (RFC #2089).

    Pipelines inherit this mixin, set ``self._remote_vae`` from
    ``od_config.remote_vae`` in ``__init__``, and call
    ``_emit_remote_vae_output(...)`` from ``_decode_latents`` when the flag
    is set.  ``_build_remote_vae_payload`` can be overridden if a pipeline
    needs extra geometry/keys beyond the default Qwen-Image packing.
    """

    _remote_vae: bool = False
    vae_scale_factor: int

    def _build_remote_vae_payload(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        return {
            "packed_latents": latents.detach().cpu(),
            "height": int(height),
            "width": int(width),
            "vae_scale_factor": int(self.vae_scale_factor),
        }

    def _emit_remote_vae_output(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> DiffusionOutput:
        return DiffusionOutput(
            output=None,
            custom_output={"remote_vae_payload": self._build_remote_vae_payload(latents, height, width)},
            stage_durations=getattr(self, "stage_durations", None) or {},
        )
