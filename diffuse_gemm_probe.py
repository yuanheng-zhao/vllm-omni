# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe whether op variant or M-padding alters the cuBLAS algorithm for the
temb.linear_1 small-GEMM (vllm-omni#3256).

Phase A (this script, single-env): enumerate variants; group bit-identical
outputs into equivalence classes. If everything ends up in one class, this
env's cuBLAS heuristic is locked for the (M,N,K,dtype) and the only escape
is a custom kernel (cuBLAS-Lt or Triton). If we see >1 class, every class
is a candidate that might match the BEFORE env; phase B is to re-run this
script in the BEFORE venv and compare ``--save`` artifacts cross-env.

Variants probed for ``(M, N=3072, K=256, bf16)``:
    - F.linear, addmm, matmul, einsum, bmm-with-batch1
    - M ∈ {1, 2, 4, 8, 16, 32}; for M>1 only row 0 is compared (the value
      we'd actually use after stripping padding)
    - both x.contiguous() and a non-contig (sliced) variant of x

Inputs are generated CPU-side with a fixed seed so two venvs see the same
bytes. Weights also CPU-seeded; not the real Qwen-Image weights — for the
algorithm-selection question, only the shape/dtype matters, not the values.

Usage:
    # In each venv:
    python diffuse_gemm_probe.py --save /tmp/probe_after.pt
    python diffuse_gemm_probe.py --save /tmp/probe_before.pt

    # Then cross-compare:
    python diffuse_gemm_probe.py --compare /tmp/probe_before.pt /tmp/probe_after.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
SEED = 0xC0FFEE


def _make_inputs(max_M: int, K: int, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn((max_M, K), dtype=torch.float32, generator=g).to(DTYPE).cuda()
    w = torch.randn((N, K), dtype=torch.float32, generator=g).to(DTYPE).cuda()
    b = torch.randn((N,), dtype=torch.float32, generator=g).to(DTYPE).cuda()
    return x, w, b


def _row0(t: torch.Tensor) -> torch.Tensor:
    return t[:1].contiguous().clone()


def _run_variants(K: int = 256, N: int = 3072, max_M: int = 32) -> OrderedDict[str, torch.Tensor]:
    x_full, w, b = _make_inputs(max_M, K, N)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()

    Ms = [1, 2, 4, 8, 16, 32]
    for M in Ms:
        x = x_full[:M].contiguous()

        out[f"M{M:02d}_F.linear"] = _row0(F.linear(x, w, b))
        out[f"M{M:02d}_addmm"] = _row0(torch.addmm(b, x, w.t()))
        out[f"M{M:02d}_matmul_plus_b"] = _row0(x @ w.t() + b)
        out[f"M{M:02d}_einsum_plus_b"] = _row0(torch.einsum("mk,nk->mn", x, w) + b)
        out[f"M{M:02d}_bmm_plus_b"] = _row0(torch.bmm(x.unsqueeze(0), w.t().unsqueeze(0)).squeeze(0) + b)

        # Non-contiguous x via stride trick (wider buffer, sliced view).
        x_wide = torch.empty((M, K + 8), dtype=DTYPE, device="cuda")
        x_wide[:, :K] = x
        x_nc = x_wide[:, :K]
        out[f"M{M:02d}_F.linear_xnc"] = _row0(F.linear(x_nc, w, b))

    # fp32 reference (not bf16 — sanity only)
    x32 = x_full[:1].float()
    w32 = w.float()
    b32 = b.float()
    ref_fp32_bf16 = (x32 @ w32.t() + b32).to(DTYPE).contiguous().clone()
    out["ref_fp32_then_cast"] = ref_fp32_bf16

    torch.cuda.synchronize()
    return out


def _equivalence_classes(variants: OrderedDict[str, torch.Tensor]) -> list[list[str]]:
    classes: list[list[str]] = []
    for name, t in variants.items():
        placed = False
        for cls in classes:
            ref = variants[cls[0]]
            if t.shape == ref.shape and torch.equal(t, ref):
                cls.append(name)
                placed = True
                break
        if not placed:
            classes.append([name])
    return classes


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, int]:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    return float(diff.max()), float(diff.mean()), int((diff > 0).sum())


def _cmd_run(args: argparse.Namespace) -> int:
    print(f"# torch={torch.__version__}  cuda={torch.version.cuda}")
    print(f"# device={torch.cuda.get_device_name(0)}")
    print(f"# probe shape: K={args.K} N={args.N}")
    variants = _run_variants(K=args.K, N=args.N)

    classes = _equivalence_classes(variants)

    print("\n=== Equivalence classes within this env (bit-identical outputs) ===")
    print(f"  ({len(variants)} variants → {len(classes)} classes)")
    for i, cls in enumerate(classes):
        print(f"\n  Class {i} ({len(cls)} members):")
        for m in cls:
            print(f"    {m}")

    base = variants["M01_F.linear"]
    ref = variants["ref_fp32_then_cast"]
    print("\n=== Per-variant diff vs M01_F.linear (this env's baseline) ===")
    print(f"{'variant':32s} {'max|Δbase|':>12s} {'max|Δfp32|':>12s} {'nonzero|Δbase|':>14s}")
    for name, t in variants.items():
        d_base = _diff_stats(t, base)
        d_ref = _diff_stats(t, ref)
        print(f"{name:32s} {d_base[0]:12.6e} {d_ref[0]:12.6e} {d_base[2]:14d}")

    if args.save:
        payload = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "K": args.K,
            "N": args.N,
            "variants": {k: v.cpu() for k, v in variants.items()},
            "classes": classes,
        }
        torch.save(payload, args.save)
        print(f"\n[saved] {args.save}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    pa = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    pb = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    print(f"A: torch={pa['torch_version']}  cuda={pa['cuda_version']}  dev={pa['device_name']}")
    print(f"B: torch={pb['torch_version']}  cuda={pb['cuda_version']}  dev={pb['device_name']}")

    keys = sorted(set(pa["variants"].keys()) & set(pb["variants"].keys()))
    print("\n=== Cross-env per-variant diff (A vs B) ===")
    print(f"{'variant':32s} {'max|ΔAB|':>12s} {'mean|ΔAB|':>12s} {'identical':>10s}")
    matches = []
    for k in keys:
        a = pa["variants"][k]
        b = pb["variants"][k]
        if a.shape != b.shape:
            print(f"{k:32s} (shape mismatch)")
            continue
        d = _diff_stats(a, b)
        identical = bool(torch.equal(a, b))
        if identical:
            matches.append(k)
        print(f"{k:32s} {d[0]:12.6e} {d[1]:12.6e} {str(identical):>10s}")

    print(f"\n=== Cross-env bit-identical variants ({len(matches)}) ===")
    for m in matches:
        print(f"  {m}")
    if not matches:
        print("  (none — no user-space variant matches BEFORE bit-exactly)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", default=None, help="Save variant outputs to this path for cross-env compare.")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None, help="Compare two saved files cross-env.")
    ap.add_argument("--K", type=int, default=256, help="K dim (default 256 = linear_1; try 3072 for linear_2).")
    ap.add_argument("--N", type=int, default=3072, help="N dim (default 3072).")
    args = ap.parse_args()

    if args.compare:
        return _cmd_compare(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
