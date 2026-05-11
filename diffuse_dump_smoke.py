# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-prompt diffusion smoke driver for vllm-omni#3256 dump bisection.

This is **not** an automated pytest. It is the in-process / HTTP driver invoked
manually after starting a vllm-omni serve with ``VLLM_OMNI_DIFFUSE_DEBUG_DIR``
pointing at a fresh dump directory. It sends a small fixed batch of image
generations whose shapes match the historical dump set (``req_000_hw512x512``
warmup-style request, ``req_001_hw576x768`` real request), so that the diff
harness can compare against the pre-rebase dumps in
``/tmp/dump_before-rebase-*``.

Usage:

    # 1. start the server in a separate shell, e.g.:
    #    VLLM_OMNI_DIFFUSE_DEEP_DUMP=1 \
    #    VLLM_OMNI_DIFFUSE_DEBUG_DIR=/tmp/dump_after-rebase-fixed \
    #    python -m vllm_omni.entrypoints.cli.main serve Qwen/Qwen-Image-2512 \
    #        --omni --port 8093 --num-gpus 1 --stage-init-timeout 1500
    # 2. run the smoke driver:
    python tests/e2e/accuracy/diffuse_dump_smoke.py \
        --base-url http://127.0.0.1:8093 \
        --model Qwen/Qwen-Image-2512 \
        --num-inference-steps 2 --seed 42

The driver does not capture the returned image — it relies on the dump hooks
inside ``pipeline_qwen_image.py`` / ``cfg_parallel.py`` /
``qwen_image_transformer.py`` for tensor capture. We intentionally do not
instantiate the pipeline in-process here: the regression we are bisecting only
shows up through the real engine path, including the executor / scheduler /
attention backend, so we drive it via the OpenAI-compatible HTTP endpoint the
production smoke test uses.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

import requests

# The two-request pattern matches the existing dump trail
# (``req_000_hw512x512_steps1`` warmup-style + ``req_001_hw576x768_steps2`` real).
# Keeping the same prompt/shape/seed combination lets us reuse the pre-rebase
# dumps in /tmp/dump_before-rebase-* without re-collecting them.
PROMPT_SHORT = "A cat."
PROMPT_LONG = (
    "A close-up portrait of a calico cat sitting on an antique wooden desk "
    "next to an open notebook, soft window light, shallow depth of field, "
    "photorealistic, 35mm film grain, warm tones."
)
NEGATIVE_PROMPT = "blurry, low quality, deformed"


def _generate(
    base_url: str,
    *,
    model: str,
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    seed: int,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "response_format": "b64_json",
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/v1/images/generations",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    # Touch the b64 body so the server actually completes the round-trip and
    # the dump for this request is flushed before we send the next one.
    if body.get("data"):
        b = body["data"][0].get("b64_json")
        if b:
            _ = base64.b64decode(b)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-inference-steps",
        type=int,
        default=2,
        help="Used for the second (real) request; the first is a 1-step warmup.",
    )
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the 512x512 1-step request (req_000). Use only when the comparison set was collected without it.",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--prompt", default=None, help="Override the default long prompt for the real request.")
    args = ap.parse_args()

    prompt_long = args.prompt if args.prompt is not None else PROMPT_LONG

    # req_000_hw512x512_steps1 — small warmup-style request, matches existing dump set.
    if not args.skip_warmup:
        print(f"[smoke] req_000 hw=512x512 steps=1 seed={args.seed}", flush=True)
        _generate(
            args.base_url,
            model=args.model,
            prompt=PROMPT_SHORT,
            width=512,
            height=512,
            num_inference_steps=1,
            true_cfg_scale=args.true_cfg_scale,
            seed=args.seed,
            timeout=args.timeout,
        )

    # req_001_hw576x768_steps{N} — the actual regression-sensitive shape.
    print(f"[smoke] req_001 hw=576x768 steps={args.num_inference_steps} seed={args.seed}", flush=True)
    body = _generate(
        args.base_url,
        model=args.model,
        prompt=prompt_long,
        width=768,
        height=576,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        timeout=args.timeout,
    )

    # Cheap sanity: confirm we got a non-empty image payload back.
    n_bytes = 0
    if body.get("data"):
        b = body["data"][0].get("b64_json") or ""
        n_bytes = len(base64.b64decode(b))
    print(json.dumps({"ok": True, "image_bytes": n_bytes}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
