# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# XXX: temporary tials

from __future__ import annotations

import torch
from vllm import envs as _vllm_envs
from vllm.model_executor.layers.layernorm import RMSNorm as _VLLMRMSNorm


class Bf16CastRMSNorm(_VLLMRMSNorm):
    """Drop-in for ``vllm.RMSNorm`` that replays 0.19.x's bf16-cast numerics
    on 0.20.0 for the no-residual case.
    """

    def _bf16_cast_no_residual(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        weight = self.weight.data
        x32 = x.to(torch.float32)
        if self.variance_size_override is None:
            x_var = x32
        else:
            x_var = x32[..., : self.variance_size_override]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(variance + self.variance_epsilon)
        return (normed.to(weight.dtype) * weight).to(orig_dtype)

    def _can_downgrade(self, residual: torch.Tensor | None) -> bool:
        return residual is None and not _vllm_envs.VLLM_BATCH_INVARIANT and self.has_weight

    def forward_cuda(  # type: ignore[override]
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if self._can_downgrade(residual):
            return self._bf16_cast_no_residual(x)
        return super().forward_cuda(x, residual)

    def forward_native(  # type: ignore[override]
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if self._can_downgrade(residual):
            return self._bf16_cast_no_residual(x)
        return super().forward_native(x, residual)
