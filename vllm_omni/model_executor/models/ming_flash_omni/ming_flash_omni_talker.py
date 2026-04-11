# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming repository modeling_bailing_talker.py
# https://github.com/inclusionAI/Ming
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ming-flash-omni-2.0 talker (TTS) stage model."""

from __future__ import annotations

import glob as glob_module
import os
import re
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model, StaticCache
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingFlashOmniTalkerConfig

from .audio_vae import AudioVAE, AudioVAEConfig
from .talker_modules.aggregator import Aggregator
from .talker_modules.cfm import CFM, get_epss_timesteps
from .talker_modules.cfm_graph_executor import CFMGraphExecutorPool
from .talker_modules.dit import DiT

logger = init_logger(__name__)


class MingFlashOmniTalkerForConditionalGeneration(nn.Module, CustomProcessMixin):
    """Ming-flash-omni-2.0 talker stage: text → audio waveform.

    Uses Qwen2 LLM + CFM (Conditional Flow Matching with DiT) + Aggregator
    in an autoregressive loop to produce continuous audio latents, then
    AudioVAE decodes latents to waveforms.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        print(f" >>> config : {type(config)}")

        model_path = vllm_config.model_config.model
        self._model_path = model_path
        talker_dir = (
            os.path.join(model_path, "talker") if os.path.isdir(os.path.join(model_path, "talker")) else model_path
        )
        self._talker_dir = talker_dir

        # When used standalone (model_arch=MingFlashOmniTalkerForConditionalGeneration),
        # the root hf_config may be BailingMM2Config (thinker-only).  Resolve the
        # talker config from talker/config.json in that case.
        if not isinstance(config, MingFlashOmniTalkerConfig):
            config = self._resolve_talker_config(config, talker_dir, model_path)

        self.config = config

        self._standalone = prefix == "" or prefix == "talker"
        if self._standalone:
            self.allow_patterns_overrides = ["talker/model*.safetensors"]
            self.fall_back_to_pt_during_load = False

        # LLM
        llm_config = self._resolve_llm_config(config, talker_dir, model_path)
        llm_config._attn_implementation = "sdpa"
        self.llm_config = llm_config
        self.hidden_size = llm_config.hidden_size
        self.latent_dim = config.latent_dim
        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size
        self.cfg_strength = config.cfg_strength

        # Sub-modules
        self.model = Qwen2Model(llm_config)
        self.cfm = CFM(
            DiT(llm_input_dim=self.hidden_size, **config.flowmodel),
            steps=config.steps,
        )
        self.aggregator = Aggregator(llm_input_dim=self.hidden_size, **config.aggregator)
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.hidden_size, bias=True)  # CAMPPlus 192-dim → hidden

        # AudioVAE
        self.audio_vae, self._vae_weight_source = self._init_audio_vae(
            config,
            talker_dir,
            model_path,
        )

        # Tokenizer (loaded lazily from talker/llm/ subdirectory)
        self._tokenizer = None

        # CFM Graph executor (initialized lazily on first forward)
        self._sampler_pool: CFMGraphExecutorPool | None = None
        self._use_cuda_graphs = not vllm_config.model_config.enforce_eager

    @staticmethod
    def _resolve_talker_config(
        config,
        talker_dir: str,
        model_path: str,
    ) -> MingFlashOmniTalkerConfig:
        """Resolve MingFlashOmniTalkerConfig when the root config is not one.

        This happens in standalone TTS mode where hf_config is BailingMM2Config.
        """
        # If the root config wraps a talker_config, use it
        talker_config = getattr(config, "talker_config", None)
        if isinstance(talker_config, MingFlashOmniTalkerConfig):
            return talker_config

        # Try loading from talker/config.json
        if os.path.isdir(talker_dir):
            try:
                resolved = MingFlashOmniTalkerConfig.from_pretrained(talker_dir)
                logger.info("Resolved talker config from %s", talker_dir)
                return resolved
            except Exception:
                pass

        # HF hub fallback
        try:
            resolved = MingFlashOmniTalkerConfig.from_pretrained(
                model_path,
                subfolder="talker",
                trust_remote_code=True,
            )
            logger.info("Resolved talker config from %s/talker (HF hub)", model_path)
            return resolved
        except Exception as e:
            raise ValueError(
                f"Cannot resolve MingFlashOmniTalkerConfig. The root config "
                f"is {type(config).__name__}, and talker/config.json was not "
                f"found at {talker_dir} or via HF hub: {e}"
            ) from e

    @staticmethod
    def _resolve_llm_config(
        config: MingFlashOmniTalkerConfig,
        talker_dir: str,
        model_path: str,
    ) -> Qwen2Config:
        """Resolve the Qwen2 LLM config for the talker backbone."""
        if config.llm_config is not None:
            if isinstance(config.llm_config, dict):
                return Qwen2Config(**config.llm_config)
            return config.llm_config

        # Try local talker/llm directory
        llm_dir = os.path.join(talker_dir, "llm")
        if os.path.isdir(llm_dir):
            return Qwen2Config.from_pretrained(llm_dir)

        # HF hub fallback
        for subfolder in ("talker/llm", "llm"):
            try:
                return Qwen2Config.from_pretrained(
                    model_path,
                    subfolder=subfolder,
                    trust_remote_code=True,
                )
            except Exception:
                continue

        raise ValueError(
            f"Cannot find talker LLM config at {llm_dir}. "
            "Either provide llm_config in MingFlashOmniTalkerConfig or "
            "ensure the model path contains talker/llm/config.json."
        )

    @staticmethod
    def _init_audio_vae(
        config: MingFlashOmniTalkerConfig,
        talker_dir: str,
        model_path: str,
    ) -> tuple[AudioVAE | None, str | tuple[str, str] | None]:
        """Initialize AudioVAE and return (vae, weight_source).

        weight_source is either a local directory path (str) or an
        (repo_id, subfolder) tuple for HF hub downloads, or None.
        """
        vae_path = config.audio_vae_path or os.path.join(talker_dir, "vae")

        # Try local directory first
        if os.path.isdir(vae_path):
            try:
                vae_config = AudioVAEConfig.from_pretrained(vae_path)
                vae = AudioVAE(vae_config)
                logger.info("Initialized AudioVAE from %s (sr=%d)", vae_path, vae_config.sample_rate)
                return vae, vae_path
            except Exception as e:
                logger.warning("Failed to initialize AudioVAE from %s: %s", vae_path, e)
                return None, None

        # HF hub fallback
        for subfolder in ("talker/vae", "vae"):
            try:
                vae_config = AudioVAEConfig.from_pretrained(
                    model_path,
                    subfolder=subfolder,
                    trust_remote_code=True,
                )
                vae = AudioVAE(vae_config)
                logger.info("Initialized AudioVAE from %s/%s (sr=%d)", model_path, subfolder, vae_config.sample_rate)
                return vae, (model_path, subfolder)
            except Exception:
                continue

        logger.info("AudioVAE not found at %s; waveform decoding unavailable", vae_path)
        return None, None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # Try local dirs first (talker/llm, talker, model root).
            candidates = [os.path.join(self._talker_dir, "llm"), self._talker_dir, self._model_path]
            for path in candidates:
                if os.path.isdir(path):
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                        print(f" >>> tokenizer: local dir {path}")
                        break
                    except Exception:
                        continue

            if self._tokenizer is None:
                # HF repo-id fallback: talker/llm is the canonical tokenizer location.
                for subfolder in ("talker/llm", "llm"):
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            self._model_path, subfolder=subfolder, trust_remote_code=True
                        )
                        print(f" >>> tokenizer: repo id subfolder {subfolder}")
                        break
                    except Exception:
                        continue

            if self._tokenizer is None:
                # Last resort: try the raw model_path/tokenizer auto resolution.
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
                print(" >>> tokenizer: raw model_path/tokenizer auto resolution")
        return self._tokenizer

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _get_sampler_pool(self):
        """Lazily initialize CFM graph executor pool."""
        if self._sampler_pool is None:
            if self._use_cuda_graphs and self.device.type == "cuda":
                self._sampler_pool = CFMGraphExecutorPool(
                    self.config, self.cfm, self.aggregator, self.stop_head, pool_size=1
                )
            else:
                self._sampler_pool = None
        return self._sampler_pool

    def _cfm_sample_step(self, last_hidden_state, his_lat, cfg=None, sigma=0.25, temperature=0.0):
        if cfg is None:
            cfg = self.cfg_strength
        """Run one CFM sampling step: LLM hidden → audio latent + next embedding + stop."""
        sampler_pool = self._get_sampler_pool()
        if sampler_pool is not None:
            return sampler_pool.execute(last_hidden_state, his_lat, cfg, sigma, temperature)

        # Fallback: direct computation without CUDA graphs
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.patch_size, z_dim), device=last_hidden_state.device, dtype=last_hidden_state.dtype
        )
        t = get_epss_timesteps(self.config.steps, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        sde_rnd = torch.randn(
            (self.config.steps, *randn_tensor.shape), device=last_hidden_state.device, dtype=last_hidden_state.dtype
        )
        sde_args = torch.tensor(
            [cfg, sigma, temperature], device=last_hidden_state.device, dtype=last_hidden_state.dtype
        )

        gen_lat = self.cfm.sample(last_hidden_state, his_lat, randn_tensor, t, sde_args, sde_rnd)
        inputs_embeds = self.aggregator(gen_lat)
        stop_out = self.stop_head(last_hidden_state[:, -1, :]).softmax(dim=-1)
        return gen_lat, inputs_embeds, stop_out

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _segment_text(self, text: str, max_length: int = 50) -> list[str]:
        """Lightweight sentence segmentation aligned with original Ming flow."""
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        segments: list[str] = []
        buffer: list[str] = []
        boundaries = set("！？。，!?")

        def flush_buffer() -> None:
            if not buffer:
                return
            chunk = "".join(buffer).strip()
            if chunk.startswith("，"):
                chunk = chunk[1:].strip()
            if chunk:
                segments.append(chunk)
            buffer.clear()

        for ch in normalized:
            buffer.append(ch)
            should_flush = False
            if ch == "\n":
                should_flush = len(buffer) >= 8
            elif ch in boundaries:
                should_flush = len(buffer) >= 8
            elif len(buffer) >= max_length:
                should_flush = True

            if should_flush:
                flush_buffer()

        flush_buffer()
        return segments if segments else [normalized]

    @staticmethod
    def _trim_trailing_silence(
        waveform: torch.Tensor,
        sample_rate: int,
        sil_th: float = 1e-3,
        tail_silence_s: float = 0.3,
    ) -> torch.Tensor:
        """Trim low-energy tail while keeping a short trailing silence."""
        if waveform.numel() == 0:
            return waveform

        original_dim = waveform.dim()
        if original_dim == 3:
            speech = waveform[:, 0, :]
        elif original_dim == 2:
            speech = waveform
        else:
            return waveform

        frame_step = int(sample_rate * 0.1)
        frame_size = int(sample_rate * 0.1)
        if speech.shape[-1] < frame_size:
            keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
            trimmed = speech[..., :keep]
        else:
            num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
            cur_len = (num_frame - 1) * frame_step + frame_size
            speech = speech[..., :cur_len]
            spe_frames = speech.unfold(-1, frame_size, frame_step)
            scores = spe_frames.abs().mean(dim=-1)
            scores = scores.mean(dim=list(range(scores.dim() - 1)))
            idx = scores.shape[0] - 1
            while idx >= 0 and scores[idx] <= sil_th:
                idx -= 1
            if idx < 0:
                keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
                trimmed = speech[..., :keep]
            else:
                non_sil_len = idx * frame_step + frame_size + int(tail_silence_s * sample_rate)
                non_sil_len = min(non_sil_len, speech.shape[-1])
                trimmed = speech[..., :non_sil_len]

        if original_dim == 3:
            return trimmed.unsqueeze(1)
        return trimmed

    def _duration_capped_steps(self, text_len: int, requested_max_steps: int) -> int:
        """Apply original Ming duration heuristic as an upper bound on decode steps."""
        if self.audio_vae is None:
            return requested_max_steps

        sample_rate = float(self.audio_vae.config.sample_rate)
        vae_patch_size = float(getattr(self.audio_vae.config, "patch_size", 4))
        hop_size = float(getattr(self.audio_vae.decoder, "hop_length", 320))
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps

        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    @staticmethod
    def _silence_holder(
        speech: torch.Tensor,
        sample_rate: int,
        sil_cache: dict | None = None,
        last_chunk: bool = True,
        sil_th: float = 1e-3,
        last_sil: float = 0.3,
    ) -> tuple[torch.Tensor, dict]:
        """Ming-style silence holder used by streaming decode."""
        if speech.numel() == 0:
            return speech, sil_cache or {"holder": [], "buffer": []}

        frame_step = int(sample_rate * 0.1)
        frame_size = int(sample_rate * 0.1)
        if sil_cache is None:
            sil_cache = {"holder": [], "buffer": []}

        if sil_cache["buffer"]:
            speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
            sil_cache["buffer"] = []

        if speech.shape[-1] < frame_size:
            sil_cache["buffer"].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
                return speech[..., : int(last_sil * sample_rate)], sil_cache
            return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        if speech.shape[-1] > cur_len:
            sil_cache["buffer"].append(speech[..., cur_len:])
            speech = speech[..., :cur_len]

        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim() - 1)))
        idx = scores.shape[0] - 1
        while idx >= 0 and scores[idx] <= sil_th:
            idx -= 1

        if idx < 0:
            sil_cache["holder"].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
                return speech[..., : int(last_sil * sample_rate)], sil_cache
            return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

        non_sil_len = idx * frame_step + frame_size
        if last_chunk:
            non_sil_len += int(last_sil * sample_rate)
        non_sil_len = min(non_sil_len, speech.shape[-1])
        speech_out = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
        sil_cache["holder"] = []
        if non_sil_len < speech.shape[-1]:
            sil_cache["holder"].append(speech[..., non_sil_len:])
        return speech_out, sil_cache

    @torch.no_grad()
    def generate_audio(
        self,
        inputs_embeds: torch.Tensor,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 10,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
    ) -> list[torch.Tensor]:
        if cfg is None:
            cfg = self.cfg_strength
        """Autoregressive generation loop: LLM + CFM → audio latents.

        Returns list of generated latent tensors, each (1, patch_size, latent_dim).
        """
        # Initialize latent history
        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim, device=self.device, dtype=self.dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat

        max_cache_len = 2048
        if use_static_cache:
            past_key_values = StaticCache(
                config=self.llm_config,
                max_batch_size=1,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            past_key_values = None

        prefill_len = inputs_embeds.shape[1]
        all_latents = []

        for step in range(min(max_steps, max_cache_len - prefill_len)):
            if step == 0 or not use_static_cache:
                outputs = self.model(
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                outputs = self.model(
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    cache_position=cache_position,
                )

            last_hs = outputs.last_hidden_state[:, -1:, :]
            gen_lat, inputs_embeds, stop_out = self._cfm_sample_step(last_hs, his_lat, cfg, sigma, temperature)

            # Update latent history
            if self.his_patch_size == self.patch_size:
                his_lat = gen_lat
            elif self.his_patch_size > self.patch_size:
                his_lat = torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
            else:
                raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

            all_latents.append(gen_lat)

            stop_prob = stop_out.cpu()[0, 1].item()
            if step % 50 == 0 or step < 5:
                logger.info(
                    "step=%d stop_prob=%.4f hs_norm=%.4f lat_norm=%.4f emb_norm=%.4f",
                    step,
                    stop_prob,
                    last_hs.float().norm().item(),
                    gen_lat.float().norm().item(),
                    inputs_embeds.float().norm().item(),
                )

            if step > min_new_token and stop_prob > 0.5:
                logger.info("Stopping at step %d with stop_prob=%.4f", step, stop_prob)
                break

        return all_latents

    def decode_latents_to_waveform(self, latents: list[torch.Tensor], stream_decode: bool = True) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self.audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

        if stream_decode:
            sr = int(self.audio_vae.config.sample_rate)
            vae_cache = {"past_key_values": None, "stream_state": (None, None, None)}
            sil_cache = None
            wav_chunks: list[torch.Tensor] = []
            for i, lat in enumerate(latents):
                last_chunk = i == (len(latents) - 1)
                speech, stream_state, past_key_values = self.audio_vae.decode(
                    lat,
                    past_key_values=vae_cache["past_key_values"],
                    use_cache=True,
                    stream_state=vae_cache["stream_state"],
                    last_chunk=last_chunk,
                )
                vae_cache = {"past_key_values": past_key_values, "stream_state": stream_state}
                speech_chunk = speech[0].detach().float()
                speech_chunk, sil_cache = self._silence_holder(
                    speech_chunk,
                    sr,
                    sil_cache=sil_cache,
                    last_chunk=last_chunk,
                )
                if speech_chunk.numel() > 0:
                    wav_chunks.append(speech_chunk)

            if not wav_chunks:
                return torch.zeros((1, 1, 0), device=self.device, dtype=self.dtype)
            waveform = torch.cat(wav_chunks, dim=-1).unsqueeze(0)
            return waveform

        all_lat = torch.cat(latents, dim=1)  # (1, total_patches * patch_size, latent_dim)
        waveform, _, _ = self.audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True
        )
        return waveform  # (1, 1, T_wav)

    def build_tts_input(
        self,
        text: str,
        prompt: str = "Please generate speech based on the following description.\n",
        spk_emb: list[torch.Tensor] | None = None,
        instruction: str | None = None,
        prompt_text: str | None = None,
        prompt_wav_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input embeddings for TTS generation.

        Returns (inputs_embeds, input_ids) for the generation loop.
        """
        tokenizer = self.tokenizer

        # Speaker embedding prompt tokens
        spk_emb_prompt = []
        if spk_emb is not None:
            for i, se in enumerate(spk_emb):
                spk_emb_prompt.extend(
                    tokenizer.encode(f"  speaker_{i + 1}:")
                    + tokenizer.encode("<|vision_start|>")
                    + tokenizer.encode("<|vision_pad|>")
                    + tokenizer.encode("<|vision_end|>\n")
                )

        # Instruction tokens
        instruction_prompt = []
        if instruction is not None:
            instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|im_end|>")

        # Zero-shot prompt tokens
        prompt_text_token = []
        prompt_latent_token = []
        if prompt_wav_emb is not None and prompt_text is not None:
            prompt_text_token = tokenizer.encode(prompt_text)
            prompt_latent_token = tokenizer.encode("<audioPatch>") * prompt_wav_emb.size(1)

        # Text input prefix
        prompt2 = tokenizer.encode(" Text input:\n")
        if (
            "Genre: " in text
            and "Mood: " in text
            and "Instrument: " in text
            and "Theme: " in text
            and "Duration: " in text
        ):
            prompt2 = []

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

        input_ids = torch.tensor(input_part, dtype=torch.long).unsqueeze(0).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(self.device, dtype=torch.bfloat16)

        # Inject speaker embeddings at <|vision_start|> positions
        if spk_emb is not None:
            spk_token_id = tokenizer.encode("<|vision_start|>")
            assert len(spk_token_id) == 1
            spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
            assert len(spk_indices) > 0
            for i, se in enumerate(spk_emb):
                inputs_embeds[0, spk_indices[i] + 1] = se

        # Inject prompt wav embeddings after <audio> token
        if prompt_wav_emb is not None and prompt_text is not None:
            audio_token_id = tokenizer.encode("<audio>")
            assert len(audio_token_id) == 1
            audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
            assert len(audio_indices) > 0
            inputs_embeds[0, audio_indices[0] + 1 : audio_indices[0] + 1 + prompt_wav_emb.size(1), :] = prompt_wav_emb[
                0
            ]

        return inputs_embeds, input_ids

    def get_dummy_runtime_additional_information(
        self,
        num_reqs: int,
    ) -> list[dict[str, object]]:
        """Provide dummy inputs for the warmup/profiling dummy run.

        Without this, the dummy run passes empty text and 1D input_ids,
        producing 2D inputs_embeds that the transformers Qwen2Model
        cannot handle (it expects 3D batch×seq×hidden).  Providing a
        short text string causes build_tts_input to tokenize it and
        produce correctly-shaped 3D embeddings.  max_steps=1 keeps the
        AR loop to a single iteration so profiling is fast.
        """
        info: dict[str, object] = {
            "text": "dummy",
            "use_zero_spk_emb": True,
            "max_steps": 1,
        }
        return [info for _ in range(num_reqs)]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        # Talker does not produce text logits; generation is handled
        # entirely inside forward() via the CFM sampling loop.
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        # Not applicable — the talker's autoregressive loop is custom
        # (LLM + CFM), not standard token sampling.
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict] | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass: run TTS generation and return audio output.

        For the talker, the full autoregressive generation loop is executed
        inside forward() since it's not standard AR token generation.

        runtime_additional_information is a list of per-request dicts
        populated by the GPUGenerationModelRunner from
        OmniTokensPrompt.additional_information.
        """
        # GPUGenerationModelRunner passes per-request info as a list;
        # take the first entry (batch_size=1).
        if runtime_additional_information and len(runtime_additional_information) > 0:
            additional_info = runtime_additional_information[0] or {}
        else:
            additional_info = {}

        text = additional_info.get("text", "")
        prompt = additional_info.get("prompt", "Please generate speech based on the following description.\n")
        spk_emb = additional_info.get("spk_emb", None)
        use_zero_spk_emb = additional_info.get("use_zero_spk_emb", False)
        instruction = additional_info.get("instruction", None)
        prompt_text = additional_info.get("prompt_text", None)
        prompt_wav_lat = additional_info.get("prompt_wav_lat", None)
        prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
        cfg = additional_info.get("cfg", self.cfg_strength)
        sigma = additional_info.get("sigma", 0.25)
        temperature = additional_info.get("temperature", 0.0)
        max_steps = int(additional_info.get("max_steps", additional_info.get("max_decode_steps", 20)))
        max_text_length = int(additional_info.get("max_text_length", 50))
        use_static_cache = bool(additional_info.get("use_static_cache", True))
        stream_decode = bool(additional_info.get("stream_decode", True))

        if inputs_embeds is None:
            # Process speaker embedding through spk_head once and reuse across fragments.
            processed_spk_emb = None
            if spk_emb is not None:
                if isinstance(spk_emb, list):
                    processed_spk_emb = [self.spk_head(se.to(device=self.device, dtype=self.dtype)) for se in spk_emb]
                else:
                    processed_spk_emb = [self.spk_head(spk_emb.to(device=self.device, dtype=self.dtype))]
            elif use_zero_spk_emb:
                processed_spk_emb = [torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)]

            text_segments = self._segment_text(text, max_length=max_text_length) if text else []
            if not text_segments:
                # Fallback: use input_ids directly. vLLM passes 1D input_ids (num_tokens,),
                # while Qwen2Model expects (batch, seq_len, hidden_size).
                inputs_embeds = self.model.get_input_embeddings()(input_ids.to(self.device)).unsqueeze(0)
                text_segments = [""]

            all_segment_latents: list[torch.Tensor] = []
            for text_segment in text_segments:
                if text_segment:
                    inputs_embeds, _ = self.build_tts_input(
                        text=text_segment,
                        prompt=prompt,
                        spk_emb=processed_spk_emb,
                        instruction=instruction,
                        prompt_text=prompt_text,
                        prompt_wav_emb=prompt_wav_emb,
                    )
                effective_max_steps = self._duration_capped_steps(len(text_segment), max_steps)
                segment_latents = self.generate_audio(
                    inputs_embeds=inputs_embeds,
                    prompt_wav_lat=prompt_wav_lat,
                    max_steps=effective_max_steps,
                    cfg=cfg,
                    sigma=sigma,
                    temperature=temperature,
                    use_static_cache=use_static_cache,
                )
                all_segment_latents.extend(segment_latents)
                prompt_wav_lat = None
                prompt_wav_emb = None
            latents = all_segment_latents
        else:
            latents = self.generate_audio(
                inputs_embeds=inputs_embeds,
                prompt_wav_lat=prompt_wav_lat,
                max_steps=max_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                use_static_cache=use_static_cache,
            )

        # Decode to waveform if AudioVAE is available
        multimodal_outputs = {}
        if latents and self.audio_vae is not None:
            waveform = self.decode_latents_to_waveform(latents, stream_decode=stream_decode)
            if not stream_decode:
                waveform = self._trim_trailing_silence(waveform, int(self.audio_vae.config.sample_rate))
            multimodal_outputs["audio"] = waveform.detach().float().cpu()
            multimodal_outputs["sr"] = torch.tensor(self.audio_vae.config.sample_rate)
        elif latents:
            # Return raw latents if no AudioVAE
            all_lat = torch.cat(latents, dim=1)
            multimodal_outputs["audio_latents"] = all_lat.detach().float().cpu()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=multimodal_outputs,
        )

    def _iter_talker_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from talker/model*.safetensors.

        Upstream ``_prepare_weights`` only sets ``use_safetensors=True`` for
        the exact glob ``"*.safetensors"``, not subdirectory patterns like
        ``"talker/model*.safetensors"``.  Loading .safetensors with
        ``pt_weights_iterator`` (torch.load) crashes, so we read them
        directly via safetensors.torch.load_file.
        """
        from safetensors.torch import load_file

        model_path = self._model_path

        # Try local path first
        for candidate in (os.path.join(model_path, "talker"), model_path):
            sf_files = sorted(glob_module.glob(os.path.join(candidate, "model*.safetensors")))
            if sf_files:
                for sf_path in sf_files:
                    yield from load_file(sf_path, device="cpu").items()
                return

        # HF hub fallback: download only the talker checkpoint files
        model_root = download_weights_from_hf_specific(
            model_path,
            self.vllm_config.load_config.download_dir,
            allow_patterns=["talker/model*.safetensors"],
        )
        talker_dir = os.path.join(model_root, "talker")
        sf_files = sorted(glob_module.glob(os.path.join(talker_dir, "model*.safetensors")))
        if not sf_files:
            raise RuntimeError(f"No talker safetensors found under {model_root}. Expected talker/model*.safetensors.")
        for sf_path in sf_files:
            yield from load_file(sf_path, device="cpu").items()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all talker components.

        The talker's HF checkpoint (talker/model.safetensors) stores weights
        with prefixes matching this module's submodule names directly:
        model.*, cfm.*, aggregator.*, stop_head.*, spk_head.*.

        AudioVAE weights live in a separate file (talker/vae/model.safetensors)
        and are loaded separately via _load_vae_weights().
        """
        # When standalone, bypass the default loader's weight iterator
        # (which would try pt_weights_iterator on .safetensors files)
        # and load directly from talker/model*.safetensors.
        if self._standalone:
            weights = self._iter_talker_safetensors()

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["audio_vae."],  # loaded separately
            skip_substrs=["rotary_embed.inv_freq"],  # non-persistent buffer
        )
        loaded = loader.load_weights(weights)
        logger.info("Loaded %d talker weights from checkpoint", len(loaded))

        # Load AudioVAE weights from separate safetensors file
        if self.audio_vae is not None and self._vae_weight_source is not None:
            loaded.update(self._load_vae_weights())

        return loaded

    def _load_vae_weights(self) -> set[str]:
        """Load AudioVAE weights from talker/vae/model.safetensors."""
        if self.audio_vae is None or self._vae_weight_source is None:
            return set()

        # Resolve safetensors file paths from the weight source
        safetensors_files: list[str] = []
        source = self._vae_weight_source
        if isinstance(source, str):
            # Local directory path
            safetensors_files = sorted(glob_module.glob(os.path.join(source, "*.safetensors")))
        elif isinstance(source, tuple):
            # (repo_id, subfolder) for HF hub
            repo_id, subfolder = source
            for filename in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
                try:
                    cached = cached_file(repo_id, filename, subfolder=subfolder)
                except Exception:
                    cached = None
                if cached is not None:
                    safetensors_files.append(cached)
                    break

        if not safetensors_files:
            logger.warning("No AudioVAE safetensors files found for source=%s", source)
            return set()

        from safetensors.torch import load_file

        vae_state_keys = set(self.audio_vae.state_dict().keys())
        vae_loader = AutoWeightsLoader(self.audio_vae)
        loaded: set[str] = set()
        for sf_path in safetensors_files:
            file_weights = load_file(sf_path, device="cpu")
            matched_weights = ((name, tensor) for name, tensor in file_weights.items() if name in vae_state_keys)
            loaded.update(f"audio_vae.{name}" for name in vae_loader.load_weights(matched_weights))

        logger.info("Loaded %d AudioVAE weights from %s", len(loaded), source)
        return loaded

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors | None:
        return None
