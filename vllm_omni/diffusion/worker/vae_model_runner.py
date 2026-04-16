# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal model runner hosting a VAE for a dedicated VAE stage.

This is the AUX-stage runner for the split-VAE demo (#2089).  Unlike
``DiffusionModelRunner`` it has no scheduler, no KV cache, no continuous
batching — a VAE stage processes one decode / encode call per request.

The runner owns the unpack+denormalize math so the upstream diffusion stage
only needs to emit raw packed latents + request-level metadata (height,
width, output_type).  Keeping that math next to the VAE weights means the
diffusion stage does not need to import VAE config to produce its payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class VAEModelRunner:
    """Hosts a single VAE module.  Dispatches encode/decode calls.

    The runner is deliberately thin: it is not a scheduler.  It takes one
    request at a time and runs either ``vae_decode_qwen_image`` (packed
    latents → image tensor) or ``vae_encode`` (pixels → latents).
    """

    def __init__(self, vae: torch.nn.Module, device: torch.device) -> None:
        self._vae = vae
        self._device = device
        # Cache normalization constants as tensors on device so we do not
        # rebuild them per request.
        cfg = vae.config
        self._latents_mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(device, dtype=vae.dtype)
        self._latents_std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(device, dtype=vae.dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._vae.dtype

    # ------------------------------------------------------------------
    # Qwen-Image specific: unpack packed latents → spatial 5D tensor.
    # Ported from pipeline_qwen_image._unpack_latents; kept here so the
    # VAE stage is self-contained.
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_qwen_image_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        h = height // (vae_scale_factor * 2)
        w = width // (vae_scale_factor * 2)
        latents = latents.view(batch_size, h, w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, h * 2, w * 2)
        return latents

    def decode_qwen_image(
        self,
        packed_latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        """Unpack, denormalize, and VAE-decode Qwen-Image latents."""
        latents = packed_latents.to(device=self._device, dtype=self._vae.dtype)
        latents = self._unpack_qwen_image_latents(latents, height, width, vae_scale_factor)
        latents = latents / (1.0 / self._latents_std) + self._latents_mean
        with torch.no_grad():
            image = self._vae.decode(latents, return_dict=False)[0]
        # [B, C, T=1, H, W] → [B, C, H, W]
        return image[:, :, 0]

    def handle(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch table for RPC calls from ``StageVAEProc``."""
        if method == "decode_qwen_image":
            image = self.decode_qwen_image(
                payload["latents"],
                int(payload["height"]),
                int(payload["width"]),
                int(payload["vae_scale_factor"]),
            )
            return {"image": image.cpu()}
        raise ValueError(f"VAEModelRunner: unsupported method {method!r}")
