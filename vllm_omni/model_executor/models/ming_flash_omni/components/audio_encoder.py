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

Enhanced with features from Qwen3-TTS WhisperEncoder:
- Flash attention support with automatic fallback for faster inference
- Dtype-safe linear and conv layers for mixed precision inference
"""

import operator
from collections.abc import Iterable
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN, flash_attn_varlen_func

logger = init_logger(__name__)


# vllm_omni/model_executor/models/qwen3_tts/tokenizer_25hz/vq/whisper_encoder.py
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# vllm_omni/model_executor/models/qwen3_tts/tokenizer_25hz/vq/whisper_encoder.py
class Conv1d(nn.Conv1d):
    """Conv1d with automatic dtype casting for mixed precision inference."""

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


# vllm_omni/model_executor/models/qwen3_tts/tokenizer_25hz/vq/whisper_encoder.py
class Linear(nn.Linear):
    """Linear layer with automatic dtype casting for mixed precision inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with packed sequence support via cu_seqlens.

    This implementation uses flash_attn_varlen_func for efficient variable-length
    attention without padding, adapted from Qwen3-TTS WhisperEncoder.
    """

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        if use_flash_attn and not HAS_FLASH_ATTN:
            logger.warning("flash-attn is not available. Fallback to manual PyTorch version")
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with packed sequence support.

        Args:
            x: [total_tokens, n_state] packed sequence
            cu_seqlens: [num_seqs + 1] cumulative sequence lengths, e.g. [0, len1, len1+len2, ...]

        Returns:
            [total_tokens, n_state] attention output
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head

        # Reshape for multi-head attention: [T, H, D]
        q = q.view(n_ctx, self.n_head, head_dim)
        k = k.view(n_ctx, self.n_head, head_dim)
        v = v.view(n_ctx, self.n_head, head_dim)

        # Try flash attention varlen if available and dtype is supported
        if self.use_flash_attn and cu_seqlens is not None and q.dtype in [torch.float16, torch.bfloat16]:
            try:
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                attn_output = flash_attn_varlen_func(
                    q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
                )
            except Exception as e:
                logger.warning(f"Flash attention varlen failed ({e}), falling back to manual")
                self.use_flash_attn = False  # Disable for future calls
                attn_output = self._manual_attention(q, k, v, cu_seqlens)
        else:
            attn_output = self._manual_attention(q, k, v, cu_seqlens)

        # Reshape back: [T, H, D] -> [T, H*D]
        attn_output = attn_output.contiguous().view(n_ctx, n_state)
        return self.out(attn_output)

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Manual attention for variable-length sequences (fallback)."""
        _, n_head, head_dim = q.shape
        scale = head_dim**-0.5

        # Unpack sequences and pad to max length
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)

        # Create padded tensors
        q_padded = torch.zeros(batch_size, max_seqlen, n_head, head_dim, dtype=q.dtype, device=q.device)
        k_padded = torch.zeros_like(q_padded)
        v_padded = torch.zeros_like(q_padded)

        # Fill with actual sequences
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_len = seqlens[i]
            q_padded[i, :seq_len] = q[start_idx:end_idx]
            k_padded[i, :seq_len] = k[start_idx:end_idx]
            v_padded[i, :seq_len] = v[start_idx:end_idx]

        # Transpose for attention: [B, H, T, D]
        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        # Create attention mask for variable lengths
        attn_mask = torch.arange(max_seqlen, device=q.device)[None, :] < torch.tensor(seqlens, device=q.device)[:, None]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -torch.finfo(q.dtype).max)

        # Compute attention
        attn_scores = torch.matmul(q_padded, k_padded.transpose(-2, -1)) * scale
        attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_padded)

        # Transpose back: [B, H, T, D] -> [B, T, H, D]
        context = context.transpose(1, 2).contiguous()

        # Pack back into single tensor
        output_packed = torch.cat([context[i, : seqlens[i]] for i in range(batch_size)], dim=0)

        return output_packed


class ResidualAttentionBlock(nn.Module):
    """Whisper-style residual attention block with packed sequence support.

    vllm_omni/model_executor/models/qwen3_tts/tokenizer_25hz/vq/whisper_encoder.py
    whisper/model.py
    """

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, use_flash_attn=use_flash_attn)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperAudioEncoder(nn.Module):
    """Whisper audio encoder for Ming with packed sequence support.

    Adapted from
    whisper/model.py
    vllm_omni/model_executor/models/qwen3_tts/tokenizer_25hz/vq/whisper_encoder.py
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 15000,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, use_flash_attn=use_flash_attn) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)
        self.audio_emb_dim = n_state

        # Configuration
        self.n_layer = n_layer
        self.n_mels = n_mels
        self.use_flash_attn = use_flash_attn

    def forward(
        self,
        x_list: list[torch.Tensor],
        audio_lens: list[int],
    ) -> torch.Tensor:
        """Forward pass with packed sequence format for variable-length inputs.

        This method processes a list of variable-length audio features without
        padding, concatenating them into a packed format for efficient processing.

        Args:
            x_list: List of [n_mels, T_i] mel spectrogram features for each audio
            audio_lens: List of original audio lengths in frames

        Returns:
            [total_T', n_state] packed encoded audio features, where
            total_T' is the sum of all encoded sequence lengths
        """
        # Process each audio through conv layers and add positional embeddings
        encoded_list = []
        encoded_lens = []

        for mel_spec in x_list:
            # mel_spec: [n_mels, T] - process through conv layers
            x = mel_spec.unsqueeze(0)  # [1, n_mels, T]
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.squeeze(0).transpose(0, 1)  # [T', n_state]

            # Add positional embedding
            seq_len = x.shape[0]
            positional_embedding = self.positional_embedding[:seq_len, :]
            x = (x + positional_embedding).to(x.dtype)

            encoded_list.append(x)
            encoded_lens.append(seq_len)

        # Concatenate all sequences into packed format
        x_packed = torch.cat(encoded_list, dim=0)  # [total_T', n_state]

        # Compute cumulative sequence lengths for attention
        cu_seqlens = list(accumulate(encoded_lens, func=operator.add, initial=0))
        cu_seqlens = torch.tensor(cu_seqlens, device=x_packed.device, dtype=torch.int32)

        # Process through transformer blocks with packed sequences
        for block in self.blocks:
            x_packed = block(x_packed, cu_seqlens=cu_seqlens)

        x_packed = self.ln_post(x_packed)
        return x_packed

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HF checkpoint.

        TODO: clean the following reference notes
        "audio.blocks.9.attn.key.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.out.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.out.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.query.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.query.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.value.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn.value.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn_ln.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.attn_ln.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp.0.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp.0.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp.2.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp.2.weight": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp_ln.bias": "model-00001-of-00042.safetensors",
        "audio.blocks.9.mlp_ln.weight": "model-00001-of-00042.safetensors",
        "audio.conv1.bias": "model-00001-of-00042.safetensors",
        "audio.conv1.weight": "model-00001-of-00042.safetensors",
        "audio.conv2.bias": "model-00001-of-00042.safetensors",
        "audio.conv2.weight": "model-00001-of-00042.safetensors",
        "audio.ln_post.bias": "model-00001-of-00042.safetensors",
        "audio.ln_post.weight": "model-00001-of-00042.safetensors",
        "audio.positional_embedding": "model-00001-of-00042.safetensors",
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
