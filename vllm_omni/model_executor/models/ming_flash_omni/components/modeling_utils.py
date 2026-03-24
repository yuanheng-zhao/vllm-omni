# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming repository modeling_utils.py
# https://github.com/inclusionAI/Ming

"""Utility functions for Ming-flash-omni multimodal embedding injection."""

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class Transpose(nn.Module):
    """Transpose two dimensions of a tensor. Used in audio projection pipelines."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


def build_modality_mask(placeholder_loc_lens: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Build a boolean mask indicating positions occupied by a specific modality.

    Args:
        placeholder_loc_lens: [B, N, 2] each 2-tuple is (start, length)
        shape: Target shape for the mask (typically batch_size x seq_len)

    Returns:
        Boolean mask of given shape, True at modality positions.
    """
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(placeholder_loc_lens.shape[0]):
        for j in range(placeholder_loc_lens.shape[1]):
            start: int = int(placeholder_loc_lens[i, j, 0].item())
            length: int = int(placeholder_loc_lens[i, j, 1].item())
            if length <= 0:
                break
            mask[i, start : start + length] = True
    return mask


def compute_placeholder_loc_lens(
    input_ids: torch.Tensor,
    patch_token_id: int,
) -> torch.Tensor:
    """Compute placeholder (start, length) pairs by finding contiguous runs
    of `patch_token_id` in `input_ids`.

    This mirrors what the original BailingMM2Processor does at preprocessing
    time, but computed on-the-fly from the already-expanded input_ids so
    that the vLLM prompt-expansion pipeline does not need to produce this
    tensor explicitly.

    Args:
        input_ids: [B, T] token IDs (after placeholder expansion).
        patch_token_id: The token ID of the patch token (e.g. <audioPatch>).

    Returns:
        [B, N, 2] int64 tensor where each 2-tuple is (start, length) of a
        contiguous run of `patch_token_id`. `N is the maximum number of
        segments across the batch; shorter samples are zero-padded.
    """
    batch_size = input_ids.size(0)
    all_loc_lens: list[list[tuple[int, int]]] = []
    max_segments = 0

    for i in range(batch_size):
        ids = input_ids[i]
        mask = ids == patch_token_id
        segments: list[tuple[int, int]] = []

        if mask.any():
            # Find boundaries of contiguous runs.
            # Pad with False at both ends to detect edges cleanly.
            padded = torch.cat(
                [
                    torch.tensor([False], device=mask.device),
                    mask,
                    torch.tensor([False], device=mask.device),
                ]
            )
            diff = padded[1:].int() - padded[:-1].int()
            starts = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
            ends = (diff == -1).nonzero(as_tuple=False).squeeze(-1)

            for s, e in zip(starts.tolist(), ends.tolist()):
                segments.append((s, e - s))

        all_loc_lens.append(segments)
        max_segments = max(max_segments, len(segments))

    # Build [B, N, 2] tensor, zero-padded for samples with fewer segments.
    if max_segments == 0:
        return torch.zeros(batch_size, 0, 2, dtype=torch.long, device=input_ids.device)

    result = torch.zeros(batch_size, max_segments, 2, dtype=torch.long, device=input_ids.device)
    for i, segments in enumerate(all_loc_lens):
        for j, (start, length) in enumerate(segments):
            result[i, j, 0] = start
            result[i, j, 1] = length

    return result
