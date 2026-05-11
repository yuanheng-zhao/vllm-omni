# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff harness for vllm-omni#3256 dump bisection.

Walks two ``VLLM_OMNI_DIFFUSE_DEBUG_DIR`` trees produced by
``debug_dump.py`` and reports per-tag tensor divergences. Output is sorted
two ways:

    1. By ``max_abs_diff`` descending (top divergences) — to pick the worst
       offender at a glance.
    2. By per-request ``seq`` ascending (first divergent boundary) — to
       localize *where* the divergence first appears in the pipeline.

Multiple tensor keys per dump file are supported (each ``debug_dump`` call
saves one ``.pt`` with possibly several tensor kwargs); each key is a
separate diff row. Tags occurring multiple times in the same request (e.g.
two transformer occurrences per step) are disambiguated as ``#occ0``,
``#occ1``, … in tag-order — same convention the analysis docs use.

Usage:

    python tests/e2e/accuracy/diffuse_dump_diff.py \\
        --good /tmp/dump_before-rebase-dd0fa025 \\
        --bad  /tmp/dump_after-rebase-fixed \\
        --req 1 --rtol 1e-4 --atol 1e-4 \\
        --include 'req_001_*/S4_*' \\
        --include 'req_001_*/S6_proj_out' \\
        --include 'req_001_*/S9_*' \\
        --top 50

``--include`` patterns are glob-matched against the human-readable tag path
``req_NNN[_label]/<tag>[#occN]``. Repeat to OR multiple patterns. ``--exclude``
also accepts globs and wins over ``--include``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from dataclasses import dataclass

import torch


@dataclass
class _Row:
    req_dir: str
    seq_a: int
    seq_b: int
    tag: str
    occ: int
    key: str
    file_a: str | None
    file_b: str | None
    shape: tuple[int, ...] | None
    dtype_a: str | None
    dtype_b: str | None
    max_abs: float
    mean_abs: float
    cos: float
    oot: int
    allclose: bool
    note: str = ""

    @property
    def path(self) -> str:
        return f"{self.req_dir}/{self.tag}#occ{self.occ}"

    @property
    def first_seq(self) -> int:
        # for "first divergent boundary": prefer the lower of the two seq ids.
        if self.seq_a >= 0 and self.seq_b >= 0:
            return min(self.seq_a, self.seq_b)
        return max(self.seq_a, self.seq_b)


def _list_req_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith("req_") and os.path.isdir(os.path.join(root, name)):
            out.append(name)
    return out


def _list_dump_files(req_dir: str) -> list[str]:
    if not os.path.isdir(req_dir):
        return []
    return sorted(f for f in os.listdir(req_dir) if f.endswith(".pt") and "__" in f)


def _parse_filename(fname: str) -> tuple[int, str] | None:
    # ``<seq>__<tag>.pt`` where ``<seq>`` is zero-padded.
    base = fname[:-3] if fname.endswith(".pt") else fname
    if "__" not in base:
        return None
    seq_str, tag = base.split("__", 1)
    try:
        seq = int(seq_str)
    except ValueError:
        return None
    return seq, tag


def _load_tensor_dict(fpath: str) -> dict[str, torch.Tensor]:
    payload = torch.load(fpath, map_location="cpu", weights_only=False)
    tensors_field = payload.get("tensors", {}) if isinstance(payload, dict) else {}
    out: dict[str, torch.Tensor] = {}
    for k, v in tensors_field.items():
        if isinstance(v, dict) and "data" in v and isinstance(v["data"], torch.Tensor):
            out[k] = v["data"]
        elif isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _load_orig_dtypes(fpath: str) -> dict[str, str]:
    try:
        payload = torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    tensors_field = payload.get("tensors", {}) if isinstance(payload, dict) else {}
    out: dict[str, str] = {}
    for k, v in tensors_field.items():
        if isinstance(v, dict):
            d = v.get("orig_dtype")
            if isinstance(d, str):
                out[k] = d
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).reshape(-1)
    b32 = b.detach().to(torch.float32).reshape(-1)
    na = float(a32.norm())
    nb = float(b32.norm())
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a32, b32) / (na * nb))


def _diff_pair(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, int, bool]:
    if a is None or b is None:
        return float("inf"), float("inf"), 0.0, -1, False
    if a.shape != b.shape:
        return float("inf"), float("inf"), 0.0, -1, False
    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    diff = (a32 - b32).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    cos = _cosine(a32, b32)
    tol = atol + rtol * b32.abs()
    oot = int((diff > tol).sum())
    allclose = bool(torch.allclose(a32, b32, rtol=rtol, atol=atol))
    return max_abs, mean_abs, cos, oot, allclose


def _matches(path: str, includes: list[str], excludes: list[str]) -> bool:
    if includes and not any(fnmatch.fnmatch(path, p) for p in includes):
        return False
    if excludes and any(fnmatch.fnmatch(path, p) for p in excludes):
        return False
    return True


def _index_request(root: str, req_dir: str) -> dict[tuple[str, int], tuple[int, str]]:
    """Map (tag, occ) -> (seq, file) inside ``<root>/<req_dir>/``."""
    full = os.path.join(root, req_dir)
    files = _list_dump_files(full)
    occ_counter: dict[str, int] = {}
    out: dict[tuple[str, int], tuple[int, str]] = {}
    for fname in files:
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        seq, tag = parsed
        occ = occ_counter.get(tag, 0)
        occ_counter[tag] = occ + 1
        out[(tag, occ)] = (seq, os.path.join(full, fname))
    return out


def _format_row(row: _Row) -> str:
    parts = [row.path, "::", row.key]
    if row.note:
        parts.append(row.note)
    if row.shape is not None:
        parts.append(f"shape={list(row.shape)}")
    if row.dtype_a or row.dtype_b:
        parts.append(f"dtype_a={row.dtype_a} dtype_b={row.dtype_b}")
    parts.append(f"max_abs={row.max_abs:.3e}")
    parts.append(f"mean_abs={row.mean_abs:.3e}")
    parts.append(f"cos={row.cos:.6f}")
    parts.append(f"oot={row.oot}")
    parts.append(f"allclose={row.allclose}")
    return "  " + "  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good", required=True, help="Reference dump root (e.g. before-rebase).")
    ap.add_argument("--bad", required=True, help="Comparison dump root (e.g. after-rebase[+fix]).")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--top", type=int, default=50, help="How many rows to print in the 'top divergences' section.")
    ap.add_argument(
        "--first",
        type=int,
        default=20,
        help="How many out-of-tolerance rows to print, sorted by seq id ascending, "
        "to show where divergence first appears and how it propagates.",
    )
    ap.add_argument(
        "--req", type=int, default=None, help="Restrict to req_<N>* directories (e.g. --req 1 -> req_001*)."
    )
    ap.add_argument(
        "--include", action="append", default=[], help="Glob over tag path (e.g. 'req_001_*/S4_*'). Repeatable; ORed."
    )
    ap.add_argument(
        "--exclude", action="append", default=[], help="Glob over tag path. Repeatable; wins over --include."
    )
    ap.add_argument(
        "--print-allclose",
        action="store_true",
        help="Also list rows where allclose=True (default hides them in 'top').",
    )
    args = ap.parse_args()

    good_dirs = _list_req_dirs(args.good)
    bad_dirs = _list_req_dirs(args.bad)
    union = sorted(set(good_dirs) | set(bad_dirs))
    if args.req is not None:
        prefix = f"req_{args.req:03d}"
        union = [d for d in union if d.startswith(prefix)]

    rows: list[_Row] = []

    for req_dir in union:
        idx_a = _index_request(args.good, req_dir)
        idx_b = _index_request(args.bad, req_dir)
        keys = sorted(set(idx_a.keys()) | set(idx_b.keys()), key=lambda kv: (kv[0], kv[1]))

        for tag, occ in keys:
            a_seq, a_file = idx_a.get((tag, occ), (-1, None))
            b_seq, b_file = idx_b.get((tag, occ), (-1, None))
            tag_path = f"{req_dir}/{tag}#occ{occ}"
            if not _matches(tag_path, args.include, args.exclude):
                continue

            if a_file is None or b_file is None:
                rows.append(
                    _Row(
                        req_dir=req_dir,
                        seq_a=a_seq,
                        seq_b=b_seq,
                        tag=tag,
                        occ=occ,
                        key="-",
                        file_a=a_file,
                        file_b=b_file,
                        shape=None,
                        dtype_a=None,
                        dtype_b=None,
                        max_abs=float("inf"),
                        mean_abs=float("inf"),
                        cos=0.0,
                        oot=-1,
                        allclose=False,
                        note=f"MISSING in {'good' if a_file is None else 'bad'}",
                    )
                )
                continue

            try:
                ta = _load_tensor_dict(a_file)
                tb = _load_tensor_dict(b_file)
            except Exception as e:
                rows.append(
                    _Row(
                        req_dir=req_dir,
                        seq_a=a_seq,
                        seq_b=b_seq,
                        tag=tag,
                        occ=occ,
                        key="-",
                        file_a=a_file,
                        file_b=b_file,
                        shape=None,
                        dtype_a=None,
                        dtype_b=None,
                        max_abs=float("inf"),
                        mean_abs=float("inf"),
                        cos=0.0,
                        oot=-1,
                        allclose=False,
                        note=f"LOAD_ERROR: {e!r}",
                    )
                )
                continue

            keys_t = sorted(set(ta.keys()) | set(tb.keys()))
            if not keys_t:
                continue
            dtype_meta_a = _load_orig_dtypes(a_file)
            dtype_meta_b = _load_orig_dtypes(b_file)

            for k in keys_t:
                a_t = ta.get(k)
                b_t = tb.get(k)
                if a_t is None or b_t is None:
                    rows.append(
                        _Row(
                            req_dir=req_dir,
                            seq_a=a_seq,
                            seq_b=b_seq,
                            tag=tag,
                            occ=occ,
                            key=k,
                            file_a=a_file,
                            file_b=b_file,
                            shape=None,
                            dtype_a=dtype_meta_a.get(k),
                            dtype_b=dtype_meta_b.get(k),
                            max_abs=float("inf"),
                            mean_abs=float("inf"),
                            cos=0.0,
                            oot=-1,
                            allclose=False,
                            note=f"key MISSING in {'good' if a_t is None else 'bad'}",
                        )
                    )
                    continue
                if a_t.shape != b_t.shape:
                    rows.append(
                        _Row(
                            req_dir=req_dir,
                            seq_a=a_seq,
                            seq_b=b_seq,
                            tag=tag,
                            occ=occ,
                            key=k,
                            file_a=a_file,
                            file_b=b_file,
                            shape=None,
                            dtype_a=dtype_meta_a.get(k),
                            dtype_b=dtype_meta_b.get(k),
                            max_abs=float("inf"),
                            mean_abs=float("inf"),
                            cos=0.0,
                            oot=-1,
                            allclose=False,
                            note=f"SHAPE_MISMATCH a={list(a_t.shape)} b={list(b_t.shape)}",
                        )
                    )
                    continue
                max_abs, mean_abs, cos, oot, allclose = _diff_pair(a_t, b_t, args.rtol, args.atol)
                rows.append(
                    _Row(
                        req_dir=req_dir,
                        seq_a=a_seq,
                        seq_b=b_seq,
                        tag=tag,
                        occ=occ,
                        key=k,
                        file_a=a_file,
                        file_b=b_file,
                        shape=tuple(a_t.shape),
                        dtype_a=dtype_meta_a.get(k) or str(a_t.dtype),
                        dtype_b=dtype_meta_b.get(k) or str(b_t.dtype),
                        max_abs=max_abs,
                        mean_abs=mean_abs,
                        cos=cos,
                        oot=oot,
                        allclose=allclose,
                    )
                )

    if not rows:
        print("(no overlapping tags after include/exclude filters)")
        return 0

    print()
    print("=== Top divergences (sorted by max_abs_diff desc) ===")
    visible = rows if args.print_allclose else [r for r in rows if not r.allclose]
    visible_sorted = sorted(
        visible, key=lambda r: (-(r.max_abs if r.max_abs != float("inf") else 1e308), r.req_dir, r.first_seq)
    )
    for r in visible_sorted[: args.top]:
        print(_format_row(r))

    print()
    print(f"=== First {args.first} divergent rows (by seq id asc) ===")
    diverged = [r for r in rows if not r.allclose and not r.note.startswith("MISSING")]
    if diverged:
        diverged_sorted = sorted(diverged, key=lambda r: (r.req_dir, r.first_seq))
        for r in diverged_sorted[: args.first]:
            print(_format_row(r))
    else:
        print("  (none — all overlapping tags within tolerance)")

    n_total = len(rows)
    n_close = sum(1 for r in rows if r.allclose)
    n_missing = sum(1 for r in rows if r.note.startswith("MISSING"))
    print()
    print(
        f"=== Summary === total={n_total}  allclose={n_close}  "
        f"missing={n_missing}  diverged={n_total - n_close - n_missing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
