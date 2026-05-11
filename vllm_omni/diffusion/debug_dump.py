# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in tensor dump utility for diffusion-pipeline accuracy bisection.

Activated only when ``VLLM_OMNI_DIFFUSE_DEBUG_DIR`` is set. Otherwise every
call is an immediate no-op (one ``os.environ.get`` lookup).

Set ``VLLM_OMNI_DIFFUSE_DEEP_DUMP=1`` to additionally enable inside-transformer
dump points (block / temb / proj_out).

Per-request partitioning:
    The pipeline calls ``mark_request_start()`` at the top of each request.
    That bumps the request counter and switches the active write directory
    to ``<root>/req_<NNN>/``. This keeps engine warmup / profile passes from
    sharing occurrence indices with the actual generation request.

Usage:
    from vllm_omni.diffusion.debug_dump import debug_dump, deep_dump_enabled

    debug_dump("S1_latents_init", latents=latents)
    debug_dump("S2_timesteps", timesteps=timesteps, mu=mu)

Files are written as ``<root>/req_<NNN>/<seq>__<tag>.pt`` where ``<seq>`` is
a zero-padded counter that resets per request. A line per call is also
appended to ``<root>/req_<NNN>/stats.jsonl``.

Multi-rank: only rank 0 (or non-distributed runs) writes anything. If you
want to compare CFG-parallel runs, pin ``cfg_parallel_size=1`` for the diff
run so positive/negative branches both materialize on rank 0.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()
_COUNTER = 0
_REQ_INDEX = -1  # bumped to 0 on first mark_request_start()
_INITIALIZED = False
_ROOT: str | None = None
_DIR: str | None = None
_DEEP: bool = False


def _maybe_init() -> bool:
    global _INITIALIZED, _ROOT, _DIR, _DEEP
    if _INITIALIZED:
        return _ROOT is not None
    _INITIALIZED = True
    d = os.environ.get("VLLM_OMNI_DIFFUSE_DEBUG_DIR")
    if not d:
        return False
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return False
    _ROOT = d
    # Until the first mark_request_start(), dumps land in <root>/req_pre/
    # so any work done before request boundaries is still captured but
    # cleanly separated from real requests.
    _DIR = os.path.join(_ROOT, "req_pre")
    try:
        os.makedirs(_DIR, exist_ok=True)
    except OSError:
        pass
    _DEEP = os.environ.get("VLLM_OMNI_DIFFUSE_DEEP_DUMP", "") not in ("", "0", "false", "False")
    return True


def _is_rank_zero() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def deep_dump_enabled() -> bool:
    """Whether deep (in-transformer) dumps are enabled."""
    if not _maybe_init():
        return False
    return _DEEP


def mark_request_start(label: str | None = None) -> None:
    """Begin a new request partition.

    Bumps the internal request index, resets the per-tag occurrence counter,
    and switches the active write directory to ``<root>/req_<NNN>[_label]/``.
    No-op when the dump directory is not configured.
    """
    if not _maybe_init():
        return
    if not _is_rank_zero():
        return
    global _REQ_INDEX, _DIR, _COUNTER
    with _LOCK:
        _REQ_INDEX += 1
        suffix = f"_{label}" if label else ""
        _DIR = os.path.join(_ROOT, f"req_{_REQ_INDEX:03d}{suffix}")
        _COUNTER = 0
        try:
            os.makedirs(_DIR, exist_ok=True)
        except OSError:
            pass


def _summarize(t: Any) -> dict[str, Any]:
    import torch

    if not isinstance(t, torch.Tensor):
        return {"type": type(t).__name__, "value": _scalarize(t)}
    tt = t.detach()
    info: dict[str, Any] = {
        "type": "tensor",
        "shape": list(tt.shape),
        "dtype": str(tt.dtype),
        "device": str(tt.device),
    }
    if tt.numel() == 0:
        return info
    try:
        ttf = tt.to("cpu", torch.float32)
        info.update(
            {
                "mean": float(ttf.mean()),
                "std": float(ttf.std()) if tt.numel() > 1 else 0.0,
                "min": float(ttf.min()),
                "max": float(ttf.max()),
                "abs_mean": float(ttf.abs().mean()),
                "abs_max": float(ttf.abs().max()),
            }
        )
    except Exception as e:
        info["stat_error"] = repr(e)
    return info


def _scalarize(v: Any) -> Any:
    """Turn small non-tensor metadata into JSON-friendly values."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_scalarize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _scalarize(x) for k, x in v.items()}
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return {"ndarray_shape": list(v.shape), "ndarray_dtype": str(v.dtype)}
    except Exception:
        pass
    return repr(v)


def _debug_dump_inner(tag: str, **kwargs: Any) -> None:
    """Save tensors + metadata to the configured dump directory.

    No-op when ``VLLM_OMNI_DIFFUSE_DEBUG_DIR`` is unset, or when called from
    a non-rank-0 process. Tensors are cloned to CPU fp32 before saving so
    cross-config diffs are dtype-agnostic; original dtype is recorded in the
    stats line.

    Wrapped by :func:`debug_dump` with ``torch.compiler.disable`` so Dynamo
    treats it as an opaque external call: avoids both the graph break at
    every ``float(tensor.mean())`` in :func:`_summarize` and the recompile-
    limit hit caused by Dynamo specializing on each unique ``tag`` string
    when called from inside a compiled transformer block.
    """
    if not _maybe_init():
        return
    if not _is_rank_zero():
        return
    _debug_dump_body(tag, kwargs)


def _debug_dump_body_impl(tag: str, kwargs: dict[str, Any]) -> None:
    import torch

    global _COUNTER
    with _LOCK:
        seq = _COUNTER
        _COUNTER += 1
        cur_dir = _DIR
        cur_req = _REQ_INDEX

    safe_tag = tag.replace("/", "_").replace(" ", "_")
    fname = f"{seq:05d}__{safe_tag}.pt"
    fpath = os.path.join(cur_dir, fname)

    payload_tensors: dict[str, Any] = {}
    payload_meta: dict[str, Any] = {}
    stats: dict[str, Any] = {"req": cur_req, "seq": seq, "tag": tag, "file": fname, "items": {}}

    for k, v in kwargs.items():
        stats["items"][k] = _summarize(v)
        if isinstance(v, torch.Tensor):
            payload_tensors[k] = {
                "data": v.detach().to("cpu", torch.float32).clone(),
                "orig_dtype": str(v.dtype),
                "orig_device": str(v.device),
                "shape": list(v.shape),
            }
        else:
            payload_meta[k] = _scalarize(v)

    try:
        torch.save(
            {"tag": tag, "seq": seq, "req": cur_req, "tensors": payload_tensors, "meta": payload_meta},
            fpath,
        )
    except Exception as e:
        stats["save_error"] = repr(e)

    try:
        with open(os.path.join(cur_dir, "stats.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    except OSError:
        pass


# Bind the Dynamo-disabled body and the public entry point. We disable the
# whole ``debug_dump`` (not just the body) so Dynamo never specializes on
# unique ``tag`` strings, which previously triggered the recompile_limit=8
# warning when called from inside a compiled transformer block.
# ``torch.compiler.disable`` is the modern entry point (torch 2.1+); fall
# back to a no-op wrap if unavailable.
try:
    import torch as _torch_for_disable

    _debug_dump_body = _torch_for_disable.compiler.disable(_debug_dump_body_impl)
    debug_dump = _torch_for_disable.compiler.disable(_debug_dump_inner)
except (ImportError, AttributeError):
    _debug_dump_body = _debug_dump_body_impl
    debug_dump = _debug_dump_inner


def reset_counter() -> None:
    """Reset the per-process sequence counter without changing the directory.

    Prefer ``mark_request_start()`` instead — it both bumps the request
    partition and resets the counter.
    """
    global _COUNTER
    with _LOCK:
        _COUNTER = 0
