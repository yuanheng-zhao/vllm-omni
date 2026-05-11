# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# XXX: temporary trials

from __future__ import annotations

import torch
import torch.nn as nn

_PATCH_ATTR = "_vllm_omni_bias_outside_patched"


@torch.compiler.disable  # type: ignore[misc]
def bias_outside_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def patch_linear_bias_outside(linear: nn.Module) -> nn.Module:
    if getattr(linear, _PATCH_ATTR, False):
        return linear
    if not hasattr(linear, "weight"):
        raise TypeError(
            f"patch_linear_bias_outside expects a module with a `.weight` attribute, got {type(linear).__name__}"
        )

    return_bias = bool(getattr(linear, "return_bias", True))
    skip_bias_add = bool(getattr(linear, "skip_bias_add", False))

    def _forward(x: torch.Tensor):
        weight: torch.Tensor = linear.weight  # type: ignore[assignment]
        out = x @ weight.t()
        bias_param = getattr(linear, "bias", None)
        if not skip_bias_add and isinstance(bias_param, torch.Tensor):
            out = out + bias_param
        if return_bias:
            output_bias = bias_param if skip_bias_add else None
            return out, output_bias
        return out

    setattr(linear, _PATCH_ATTR, True)
    linear.forward = _forward  # type: ignore[method-assign]
    return linear
