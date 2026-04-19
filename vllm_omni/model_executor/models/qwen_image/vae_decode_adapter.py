# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image VAE-decode adapter for the generic aux-stage runtime.

This module hosts :class:`QwenImageVaeDecodeAdapter`, an :class:`AuxAdapter`
implementation that wraps :class:`DistributedAutoencoderKLQwenImage` and
moves the unpack / denormalize math out of ``VAEModelRunner`` and into
the adapter. It lives next to the model's pipeline code so the aux
runtime stays model-agnostic.

Registration happens at model import time, via
:func:`vllm_omni.stages.aux.register_adapter` called from the
``qwen_image`` package ``__init__``.

Bridge schema: ``qwen_image.vae.latents.v1`` (module_kind=vae,
model_arch=qwen_image, op=decode). Required extras: ``height``,
``width``, ``vae_scale_factor``.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.stages.aux import AuxAdapter, AuxAdapterResult
from vllm_omni.stages.bridge import (
    StageBridgePayload,
    is_registered,
    register_schema,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Schema registration
# ---------------------------------------------------------------------------

_SCHEMA_IN = "qwen_image.vae.latents.v1"

if not is_registered(_SCHEMA_IN):
    register_schema(
        _SCHEMA_IN,
        description=(
            "Qwen-Image packed VAE latents handed off from the diffusion "
            "stage to a VAE-decode aux stage. Tensor: packed latents "
            "(batch, seq, channels). Required extras encode the geometry "
            "needed to unpack them back to a 5-D volume."
        ),
        required_extras=("height", "width", "vae_scale_factor"),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QwenImageVaeDecodeAdapter(AuxAdapter):
    """Unpacks + denormalizes + VAE-decodes Qwen-Image latents.

    The adapter owns every model-specific detail:

    - Loading :class:`DistributedAutoencoderKLQwenImage` from the model
      repo (honors ``vae_subfolder`` / ``torch_dtype`` engine args).
    - Caching the ``latents_mean`` / ``latents_std`` buffers on device.
    - Packing ``(B, seq, C)`` latents into the 5-D VAE input shape.
    - Returning the decoded image tensor as a final output so the
      :class:`StageAuxClient` can turn it into PIL images.

    The runner, proc, and client never touch any of the above.
    """

    schema_in = _SCHEMA_IN
    schema_out = None
    final_output_type = "image"

    def __init__(self, **engine_args: Any) -> None:
        super().__init__(**engine_args)
        self._vae: torch.nn.Module | None = None
        self._latents_mean: torch.Tensor | None = None
        self._latents_std: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_module(self, model: str, device: torch.device) -> torch.nn.Module:
        # Late import so the registry entry is cheap (no diffusers on the
        # import path of callers that only list available adapters).
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
            DistributedAutoencoderKLQwenImage,
        )

        dtype = self.resolve_torch_dtype(default="bfloat16")
        vae_subfolder = self._engine_args.get("vae_subfolder", "vae")
        logger.info(
            "[QwenImageVaeDecodeAdapter] loading VAE from %s (subfolder=%s, dtype=%s)",
            model,
            vae_subfolder,
            dtype,
        )
        vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model,
            subfolder=vae_subfolder,
            torch_dtype=dtype,
        )
        vae = vae.to(device).eval()
        self._vae = vae

        # Cache normalization constants as device tensors so pre_transform
        # does not rebuild them per request.
        cfg = vae.config
        self._latents_mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(device, dtype=vae.dtype)
        self._latents_std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(device, dtype=vae.dtype)
        return vae

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def pre_transform(self, payload: StageBridgePayload) -> torch.Tensor:
        """Move packed latents to device, unpack, and denormalize."""
        assert self._vae is not None, "pre_transform called before load_module"
        packed = payload.tensor
        if packed is None:
            raise ValueError(f"{self.schema_in}: payload.tensor is required (packed latents)")
        height = int(payload.get("height"))
        width = int(payload.get("width"))
        vae_scale_factor = int(payload.get("vae_scale_factor"))

        latents = packed.to(device=self.device, dtype=self._vae.dtype)
        latents = self._unpack_qwen_image_latents(latents, height, width, vae_scale_factor)
        assert self._latents_mean is not None and self._latents_std is not None
        latents = latents / (1.0 / self._latents_std) + self._latents_mean
        return latents

    def forward(self, module_input: torch.Tensor) -> torch.Tensor:
        assert self._vae is not None
        # [B, C, T=1, H, W]
        return self._vae.decode(module_input, return_dict=False)[0]

    def post_transform(self, module_output: torch.Tensor) -> AuxAdapterResult:
        # Collapse the singleton time axis so downstream code sees [B, C, H, W].
        image = module_output[:, :, 0]
        return AuxAdapterResult(final_tensor=image)

    # ------------------------------------------------------------------
    # Tensor-shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_qwen_image_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        """Reverse the pipeline's pack: ``(B, seq, C) -> (B, C, 1, H', W')``.

        Mirrors ``pipeline_qwen_image._unpack_latents`` and the legacy
        ``VAEModelRunner._unpack_qwen_image_latents``.
        """
        batch_size, _, channels = latents.shape
        h = height // (vae_scale_factor * 2)
        w = width // (vae_scale_factor * 2)
        latents = latents.view(batch_size, h, w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, h * 2, w * 2)
        return latents
