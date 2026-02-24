# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from Ming repository modeling_bailingmm2.py
# https://github.com/inclusionAI/Ming

"""Modality projectors for Ming-flash-omni-2.0.

Two projectors map encoder outputs into the LLM's embedding space:

1. **Vision projector** (``linear_proj``): MLP that projects vision encoder
   output (``out_hidden_size``) to LLM hidden size. Depth controlled by
   ``mlp_depth`` config.

2. **Audio projector** (``linear_proj_audio``): Conv1d downsampling followed
   by MLP layers. The Conv1d reduces the temporal resolution by
   ``ds_stride``, then MLP layers project to LLM hidden size. The audio
   projector wraps the result with Transpose layers to handle the
   channel-last ↔ channel-first conversion around Conv1d.

Architecture from modeling_bailingmm2.py::

    # Vision projector
    linear_proj = Sequential(
        Linear(vision_dim, llm_dim),
        *[GELU(), Linear(llm_dim, llm_dim)] * (mlp_depth - 1)
    )

    # Audio projector
    linear_proj_audio = Sequential(
        Conv1d(audio_dim, llm_dim, k=ds_kernel_size, s=ds_stride, p=ds_kernel_size//2),
        Transpose(-1, -2),
        *[GELU(), Linear(llm_dim, llm_dim)] * (mlp_depth - 1),
        Transpose(-1, -2),
    )
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .modeling_utils import Transpose

logger = init_logger(__name__)


class VisionProjector(nn.Module):
    """MLP projector from vision encoder output to LLM hidden space.

    Args:
        vision_dim: Vision encoder output dimension (out_hidden_size).
        llm_dim: LLM hidden dimension.
        mlp_depth: Number of linear layers (>= 1).
    """

    def __init__(self, vision_dim: int, llm_dim: int, mlp_depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features.

        Args:
            x: [seq_len, vision_dim] or [B, seq_len, vision_dim]

        Returns:
            Projected features with last dim = llm_dim.
        """
        return self.proj(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                logger.warning("Skipping unknown vision projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class AudioProjector(nn.Module):
    """Conv1d downsampling + MLP projector for audio features.

    The projection pipeline:
        1. Conv1d: [B, audio_dim, T] → [B, llm_dim, T']  (temporal downsampling)
        2. Transpose to [B, T', llm_dim]
        3. Optional MLP layers: GELU + Linear (repeated mlp_depth-1 times)
        4. Transpose back to [B, llm_dim, T'] (for compatibility with wrap_feats)

    Args:
        audio_dim: Audio encoder output dimension (n_state).
        llm_dim: LLM hidden dimension.
        ds_kernel_size: Conv1d kernel size for downsampling.
        ds_stride: Conv1d stride for downsampling.
        mlp_depth: Total number of projection layers (>= 1).
    """

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        ds_kernel_size: int = 3,
        ds_stride: int = 2,
        mlp_depth: int = 1,
    ):
        super().__init__()
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride

        layers: list[nn.Module] = [
            nn.Conv1d(
                audio_dim,
                llm_dim,
                kernel_size=ds_kernel_size,
                stride=ds_stride,
                padding=ds_kernel_size // 2,
            ),
            Transpose(-1, -2),
        ]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        layers.append(Transpose(-1, -2))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features with temporal downsampling.

        Args:
            x: [B, T, audio_dim] audio encoder output (channel-last).

        Returns:
            [B, T', llm_dim] projected features (channel-last), where
            T' = (T - ds_kernel_size + 2*(ds_kernel_size//2)) // ds_stride + 1.
        """
        # Conv1d expects [B, C, T], so transpose input
        x = x.transpose(-1, -2)  # [B, audio_dim, T]
        x = self.proj(x)  # ends with Transpose(-1, -2) → [B, T', llm_dim] → transposed
        # The final Transpose in proj converts back, so output is [B, llm_dim, T']
        # For external callers, transpose to channel-last
        return x.transpose(-1, -2)  # [B, T', llm_dim]

    def compute_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Compute output sequence length after Conv1d downsampling.

        This accounts for both the Whisper encoder's internal Conv1d
        and this projector's Conv1d.

        Args:
            input_length: Original mel spectrogram lengths.

        Returns:
            Output lengths after both convolutions.
        """
        # Whisper encoder conv: kernel=3, stride=2, padding=1
        length = (input_length - 3 + 2 * 1) // 2 + 1
        # Audio projector conv
        length = (length - self.ds_kernel_size + 2 * (self.ds_kernel_size // 2)) // self.ds_stride + 1
        return length

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                logger.warning("Skipping unknown audio projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
