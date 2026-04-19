# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/aggregator.py

import torch
import torch.nn as nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import DiTBlock, FinalLayer


class Aggregator(nn.Module):
    """Maps generated audio latent patches back to LLM embedding space."""

    def __init__(
        self,
        in_channels=64,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_input_dim=896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def forward(self, x, mask=None):
        x = self.x_embedder(x)
        cls_embed = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        x = torch.cat([cls_embed, x], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1]
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        x = x[:, :1, :]
        return x
