# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming repository modeling_utils.py
# https://github.com/inclusionAI/Ming

"""Utility functions for Ming-flash-omni multimodal embedding injection."""

import torch
import torch.nn as nn


class Transpose(nn.Module):
    """Transpose two dimensions of a tensor. Used in audio projection pipelines."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)
