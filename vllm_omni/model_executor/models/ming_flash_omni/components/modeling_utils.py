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


def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
) -> torch.Tensor:
    """Patch continuous features (e.g. audio embeddings) into input embeddings
    at placeholder positions, handling variable-length features.

    Args:
        input_embeddings: [B, T, D] input token embeddings
        placeholder_loc_lens: [B, N, 2] each 2-tuple is (start, length) of a placeholder
        encoded_feats: [B, L1+L2+...+LN, D] concatenated encoded features
        encoded_feat_lens: [B, N] length of each encoded feature segment

    Returns:
        Modified input_embeddings with features patched in at placeholder positions.
    """
    batch_size = input_embeddings.size(0)
    for i in range(batch_size):
        audio_feat_start = 0
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                logger.warning(
                    "Feature length (%d) > placeholder length (%d). Truncating feature to avoid errors.",
                    feat_len,
                    placeholder_len,
                )
                feat_len = placeholder_len
            target_len = min(feat_len, placeholder_len)
            input_embeddings[i, placeholder_start : placeholder_start + target_len] = encoded_feats[
                i, audio_feat_start : audio_feat_start + target_len
            ]
            audio_feat_start += real_feat_len
    return input_embeddings


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


def unwrap_feats(
    feats: torch.Tensor,
    feats_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unwrap concatenated feature segments into individual padded segments.

    Input features are in "wrapped" format where features from multiple audio
    segments are concatenated in a single row. This function separates them.

    Args:
        feats: [B, L1+L2+...+LN, ...] concatenated features
        feats_lengths: [B, N] lengths of each segment within each batch item

    Returns:
        Tuple of (feat_segs_batch, feat_seg_lengths) where:
        - feat_segs_batch: [total_segs, max_seg_len, ...] padded segments
        - feat_seg_lengths: [total_segs] actual length of each segment
    """
    feat_segs = []
    feat_seg_lengths = []
    for i in range(feats_lengths.shape[0]):
        feat_index = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            feat_segs.append(feats[i, feat_index : feat_index + feat_len])
            feat_seg_lengths.append(feat_len)
            feat_index += feat_len
    feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, True).to(feats.device)
    feat_seg_lengths = torch.tensor(feat_seg_lengths, dtype=torch.long, device=feats.device)
    return feat_segs_batch, feat_seg_lengths


def wrap_feats(
    feat_segs: torch.Tensor,
    feats_lengths: torch.Tensor,
    feats_seg_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrap individual feature segments back into concatenated format.

    Inverse operation of unwrap_feats().

    Args:
        feat_segs: [total_segs, max_seg_len, ...] padded segments
        feats_lengths: [B, N] original segment layout
        feats_seg_lengths: [total_segs] actual lengths (optional)

    Returns:
        Tuple of (feats, feats_locs, feats_new_lengths) where:
        - feats: [B, max_total_len, ...] concatenated features per batch
        - feats_locs: [B, N] start offsets of each segment
        - feats_new_lengths: [B, N] actual lengths of each segment
    """
    feat_idx = 0
    feats_buffer = []
    feats_locs_buffer = []
    feats_lengths_buffer = []
    for i in range(feats_lengths.shape[0]):
        feat_buffer = []
        feat_locs_buffer = []
        feat_lengths_buffer = []
        feat_total_len = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            if feats_seg_lengths is not None:
                feat_len = feats_seg_lengths[feat_idx].item()
            feat_buffer.append(feat_segs[feat_idx, :feat_len])
            feat_locs_buffer.append(feat_total_len)
            feat_lengths_buffer.append(feat_len)
            feat_idx += 1
            feat_total_len += feat_len
        feats_buffer.append(torch.cat(feat_buffer))
        feats_locs_buffer.append(torch.tensor(feat_locs_buffer, dtype=torch.long))
        feats_lengths_buffer.append(torch.tensor(feat_lengths_buffer, dtype=torch.long))
    feats = torch.nn.utils.rnn.pad_sequence(feats_buffer, True).to(feat_segs.device)
    feats_locs = torch.nn.utils.rnn.pad_sequence(feats_locs_buffer, True).to(feats_lengths.device)
    feats_new_lengths = torch.nn.utils.rnn.pad_sequence(feats_lengths_buffer, True).to(feats_lengths.device)
    return feats, feats_locs, feats_new_lengths


def encode_audio_segments(
    encoder: torch.nn.Module,
    proj_layer: torch.nn.Module,
    wav_feats: torch.Tensor,
    wav_feats_lengths: torch.Tensor,
    audio_config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode audio segments using the Whisper encoder and projection layer.

    Audio features arrive in "wrapped" format (multiple segments concatenated
    per batch row). This function:
      1. Unwraps segments into individual rows
      2. Runs the Whisper encoder on each segment
      3. Projects through the audio projection layer
      4. Wraps results back into batch format

    Args:
        encoder: WhisperAudioEncoder instance.
        proj_layer: Audio projection module (Conv1d + MLP).
        wav_feats: [B, L_total, n_mels] wrapped mel features.
        wav_feats_lengths: [B, N] lengths of each segment.
        audio_config: WhisperEncoderConfig with ds_kernel_size, ds_stride.

    Returns:
        Tuple of (audio_feats_proj, audio_feats, audio_feats_lengths):
        - audio_feats_proj: [B, L_proj, D] projected features in wrapped format
        - audio_feats: [B, L_enc, D] encoder features in wrapped format
        - audio_feats_lengths: [B, N] output lengths per segment
    """
    # Unwrap: separate concatenated segments into individual rows
    feat_segs_batch, feat_seg_lengths = unwrap_feats(wav_feats, wav_feats_lengths)

    # Encode with Whisper
    audio_feats_seg = encoder(feat_segs_batch)

    # Project: Conv1d expects [B, C, T] format
    audio_feats_seg_proj = proj_layer(audio_feats_seg.transpose(-1, -2)).transpose(-1, -2)

    # Compute output lengths after Whisper conv and projector conv
    feat_seg_lengths = feat_seg_lengths.to(feat_segs_batch.device)
    # Whisper encoder conv: kernel=3, stride=2, padding=1
    audio_feat_seg_lengths = (feat_seg_lengths - 3 + 2 * 1) // 2 + 1
    # Projector conv
    audio_feat_seg_lengths = (
        audio_feat_seg_lengths - audio_config.ds_kernel_size + 2 * (audio_config.ds_kernel_size // 2)
    ) // audio_config.ds_stride + 1

    # Wrap back into batch format
    audio_feats, _, audio_feats_lengths = wrap_feats(audio_feats_seg, wav_feats_lengths, audio_feat_seg_lengths)
    audio_feats_proj, _, audio_feats_lengths2 = wrap_feats(
        audio_feats_seg_proj, wav_feats_lengths, audio_feat_seg_lengths
    )
    assert torch.all(audio_feats_lengths == audio_feats_lengths2)

    return audio_feats_proj, audio_feats, audio_feats_lengths
