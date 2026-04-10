# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Standalone Rotary Position Embedding to replace x_transformers dependency.
# Compatible with the API used by Ming's DiT and Aggregator modules.


import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, scale=1.0):
    """Apply rotary position embeddings to tensor t.

    Compatible with x_transformers.x_transformers.apply_rotary_pos_emb.
    Args:
        t: tensor of shape (..., seq_len, dim)
        freqs: frequencies of shape (seq_len, dim) or broadcastable
        scale: optional xpos scale (unused in Ming, kept for API compat)
    """
    rot_dim = freqs.shape[-1]
    t_left, t_right = t[..., :rot_dim], t[..., rot_dim:]
    # Compute RoPE in float32 for precision, then cast back.
    orig_dtype = t_left.dtype
    t_left = t_left.float()
    t_transformed = ((t_left * freqs.cos() + rotate_half(t_left) * freqs.sin()) * scale).to(orig_dtype)
    return torch.cat((t_transformed, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding compatible with x_transformers.RotaryEmbedding.

    Provides `forward_from_seq_len(seq_len)` returning (freqs, None) tuple
    matching the API used by DiT and Aggregator.
    """

    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        return freqs, None
