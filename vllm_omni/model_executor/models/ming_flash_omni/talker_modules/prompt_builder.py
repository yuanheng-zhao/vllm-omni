# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

_MUSIC_TAGS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


def _looks_like_music_prompt(text: str) -> bool:
    return all(tag in text for tag in _MUSIC_TAGS)


def build_tts_input(
    *,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    text: str,
    prompt: str,
    spk_emb: list[torch.Tensor] | None = None,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_wav_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, input_ids) for one TTS segment.

    Args:
        tokenizer: HF tokenizer
        embed_tokens: The LLM's input-embedding module
        device: Device to place the returned tensors on.
        dtype: dtype for the returned `inputs_embeds`.
        text: Text to synthesize.
        prompt: System-level instruction prompt prepended to the user turn.
        spk_emb: Optional list of speaker embeddings already projected into
            LLM hidden dim; each is injected at a `<|vision_start|>` slot.
        instruction: Optional free-form instruction
        prompt_text: Reference text for zero-shot voice cloning.
        prompt_wav_emb: Reference-wav embeddings to inject.
    """
    spk_emb_prompt: list[int] = []
    if spk_emb is not None:
        for i in range(len(spk_emb)):
            spk_emb_prompt.extend(
                tokenizer.encode(f"  speaker_{i + 1}:")
                + tokenizer.encode("<|vision_start|>")
                + tokenizer.encode("<|vision_pad|>")
                + tokenizer.encode("<|vision_end|>\n")
            )

    instruction_prompt: list[int] = []
    if instruction is not None:
        instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|im_end|>")

    prompt_text_token: list[int] = []
    prompt_latent_token: list[int] = []
    if prompt_wav_emb is not None and prompt_text is not None:
        prompt_text_token = tokenizer.encode(prompt_text)
        prompt_latent_token = tokenizer.encode("<audioPatch>") * prompt_wav_emb.size(1)

    prompt2 = [] if _looks_like_music_prompt(text) else tokenizer.encode(" Text input:\n")

    input_part = (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + spk_emb_prompt
        + prompt2
        + prompt_text_token
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_token
    )

    input_ids = torch.tensor(input_part, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embed_tokens(input_ids).to(device=device, dtype=dtype)

    # inject speaker embeddings
    if spk_emb is not None:
        spk_token_id = tokenizer.encode("<|vision_start|>")
        assert len(spk_token_id) == 1, "<|vision_start|> must tokenize to a single id"
        spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
        assert len(spk_indices) > 0, "expected at least one <|vision_start|> slot"
        for i, se in enumerate(spk_emb):
            inputs_embeds[0, spk_indices[i] + 1] = se

    # inject prompt-wav embeddings after <audio>
    if prompt_wav_emb is not None and prompt_text is not None:
        audio_token_id = tokenizer.encode("<audio>")
        assert len(audio_token_id) == 1, "<audio> must tokenize to a single id"
        audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
        assert len(audio_indices) > 0, "expected at least one <audio> slot"
        start = audio_indices[0] + 1
        inputs_embeds[0, start : start + prompt_wav_emb.size(1), :] = prompt_wav_emb[0]

    return inputs_embeds, input_ids
