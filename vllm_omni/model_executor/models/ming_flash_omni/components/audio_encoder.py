# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming repository modeling_whisper_encoder.py
# https://github.com/inclusionAI/Ming

"""Whisper-based audio encoder for Ming-flash-omni-2.0.

Ming uses a modified OpenAI Whisper encoder that removes the fixed 30-second
padding, allowing variable-length audio inputs. The architecture is:

    mel_spectrogram → Conv1d(n_mels, n_state, k=3) → GELU
                    → Conv1d(n_state, n_state, k=3, s=2) → GELU
                    → add positional_embedding (truncated to actual length)
                    → N × ResidualAttentionBlock
                    → LayerNorm

The encoder outputs are then projected by the audio projector (see projectors.py).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention compatible with Whisper's weight format."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head

        q = q.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)
        k = k.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)
        v = v.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(n_batch, n_ctx, n_state)
        return self.out(attn_output)


class ResidualAttentionBlock(nn.Module):
    """Whisper-style residual attention block."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperAudioEncoder(nn.Module):
    """Vendored Whisper audio encoder for Ming.

    This is a self-contained re-implementation of OpenAI Whisper's AudioEncoder
    that does NOT depend on the ``whisper`` package. It matches the original
    weight format exactly so HF checkpoints load without remapping.

    The key modification from standard Whisper: positional embeddings are
    truncated to match the actual input length instead of padding to 30 seconds.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 15000,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)
        self.audio_emb_dim = n_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with variable-length support.

        Args:
            x: [B, T, n_mels] mel spectrogram features.

        Returns:
            [B, T', n_state] encoded audio features, where T' = (T - 3 + 2) // 2 + 1
            after the two Conv1d layers.
        """
        x = x.transpose(1, 2)  # [B, n_mels, T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [B, T', n_state]

        # Truncate positional embedding to actual sequence length
        positional_embedding = self.positional_embedding[: x.shape[1], :]
        assert x.shape[1:] == positional_embedding.shape, (
            f"incorrect audio shape: x={x.shape[1:]}, pos_emb={positional_embedding.shape}"
        )
        x = (x + positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HF checkpoint.

        Whisper weights use the naming convention:
            blocks.N.attn.query.weight
            blocks.N.attn.key.weight
            blocks.N.attn.value.weight
            blocks.N.attn.out.weight
            blocks.N.attn_ln.weight/bias
            blocks.N.mlp.0.weight/bias  (linear)
            blocks.N.mlp.2.weight/bias  (linear)
            blocks.N.mlp_ln.weight/bias
            conv1.weight/bias
            conv2.weight/bias
            positional_embedding
            ln_post.weight/bias
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                logger.warning("Skipping unknown audio encoder weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
