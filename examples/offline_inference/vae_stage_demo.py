# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone demo for the split-VAE stage (RFC #2089).

Spawns a ``StageVAEProc`` subprocess via ``StageVAEClient`` and sends it
packed Qwen-Image latents; writes the decoded PNG to disk.

This does NOT go through the Omni engine or orchestrator — the full
stage-pipeline integration (extract_stage_metadata / initialize_vae_stage
/ stage_engine_startup routing) is still pending. The purpose here is to
prove the VAE-as-subprocess pattern end-to-end on its own.

Two input modes:

    --latents-file foo.pt      load pre-captured packed latents saved from
                               a prior run of pipeline_qwen_image with
                               `engine_args.remote_vae: true`
    (default)                  generate random latents shaped for Qwen-Image
                               at the requested height/width — output will
                               be noise, but proves the stage is wired up

Usage:

    python examples/offline_inference/vae_stage_demo.py \\
        --model Qwen/Qwen-Image --height 1024 --width 1024 \\
        --output vae_stage_demo.png
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.stage_vae_client import StageVAEClient


def _mock_packed_latents(
    height: int,
    width: int,
    vae_scale_factor: int,
    z_dim: int = 16,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reproduce the packed-latent shape emitted by pipeline_qwen_image.

    See ``QwenImagePipeline._pack_latents``: the DiT works on tensors
    shaped ``[B, (H/vs/2) * (W/vs/2), z_dim * 4]``.
    """
    h = height // (vae_scale_factor * 2)
    w = width // (vae_scale_factor * 2)
    return torch.randn(batch_size, h * w, z_dim * 4, dtype=dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen-Image")
    p.add_argument("--vae-subfolder", default="vae")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--vae-scale-factor", type=int, default=8)
    p.add_argument(
        "--latents-file",
        type=str,
        default=None,
        help="Optional .pt file containing a dict with keys "
        "'packed_latents', 'height', 'width', 'vae_scale_factor' "
        "(as emitted by pipeline_qwen_image when remote_vae=True).",
    )
    p.add_argument("--output", default="vae_stage_demo.png")
    p.add_argument("--stage-init-timeout", type=int, default=600)
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if args.latents_file:
        blob = torch.load(args.latents_file, map_location="cpu")
        packed = blob["packed_latents"]
        height = int(blob["height"])
        width = int(blob["width"])
        vae_scale_factor = int(blob["vae_scale_factor"])
        print(f"[demo] loaded packed latents {tuple(packed.shape)} from {args.latents_file}")
    else:
        height, width = args.height, args.width
        vae_scale_factor = args.vae_scale_factor
        packed = _mock_packed_latents(
            height,
            width,
            vae_scale_factor,
            dtype=getattr(torch, args.torch_dtype),
        )
        print(
            f"[demo] generated random packed latents {tuple(packed.shape)} "
            f"(output will be noise — this only proves the stage is wired up)"
        )

    print("[demo] spawning StageVAEProc subprocess...")
    t_spawn = time.perf_counter()
    client = StageVAEClient(
        model=args.model,
        vae_subfolder=args.vae_subfolder,
        torch_dtype=args.torch_dtype,
        device=args.device,
        stage_init_timeout=args.stage_init_timeout,
    )
    print(f"[demo] stage ready in {time.perf_counter() - t_spawn:.2f}s")

    try:
        t_decode = time.perf_counter()
        image = await client.decode_qwen_image(
            latents=packed,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )
        print(
            f"[demo] decode done in {time.perf_counter() - t_decode:.2f}s, "
            f"image tensor {tuple(image.shape)} dtype={image.dtype}"
        )
    finally:
        client.shutdown()

    image = image.float().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    image = (image * 255).round().to(torch.uint8)
    # [B, C, H, W] -> first sample HWC
    img0 = image[0].permute(1, 2, 0).cpu().numpy()
    try:
        from PIL import Image

        Image.fromarray(img0).save(args.output)
        print(f"[demo] wrote {args.output}")
    except ImportError:
        out_pt = Path(args.output).with_suffix(".pt")
        torch.save(image.cpu(), out_pt)
        print(f"[demo] PIL unavailable; saved raw tensor to {out_pt}")


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
