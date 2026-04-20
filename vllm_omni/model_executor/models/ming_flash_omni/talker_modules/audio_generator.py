# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch
from transformers import StaticCache
from vllm.logger import init_logger

from .audio_postprocess import silence_holder, trim_trailing_silence
from .cfm import CFMGraphExecutorPool, get_epss_timesteps

if TYPE_CHECKING:
    from transformers import Qwen2Config, Qwen2Model

    from ..audio_vae import AudioVAE
    from .aggregator import Aggregator
    from .cfm import CFM

logger = init_logger(__name__)


class MingAudioGenerator:
    """Generator driving prefill -> AR decode -> VAE decode
    for a single TTS request. The generator is stateless across requests.
    """

    def __init__(
        self,
        config,
        llm_config: Qwen2Config,
        model: Qwen2Model,
        cfm: CFM,
        aggregator: Aggregator,
        stop_head: torch.nn.Module,
        audio_vae: AudioVAE | None,
        patch_size: int,
        his_patch_size: int,
        latent_dim: int,
        cfg_strength: float,
        use_cuda_graphs: bool,
    ) -> None:
        self._config = config
        self._llm_config = llm_config
        self._model = model
        self._cfm = cfm
        self._aggregator = aggregator
        self._stop_head = stop_head
        self._audio_vae = audio_vae

        self.patch_size = patch_size
        self.his_patch_size = his_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength

        self._use_cuda_graphs = use_cuda_graphs

    @cached_property
    def _sampler_pool(self) -> CFMGraphExecutorPool | None:
        device = next(self._model.parameters()).device
        if self._use_cuda_graphs and device.type == "cuda":
            return CFMGraphExecutorPool(self._config, self._cfm, self._aggregator, self._stop_head, pool_size=1)
        return None

    def duration_capped_steps(self, text_len: int, requested_max_steps: int) -> int:
        """Apply the original Ming duration heuristic as a cap on decode steps."""
        if self._audio_vae is None:
            return requested_max_steps

        sample_rate = float(self._audio_vae.config.sample_rate)
        vae_patch_size = float(getattr(self._audio_vae.config, "patch_size", 4))
        hop_size = float(getattr(self._audio_vae.decoder, "hop_length", 320))
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps

        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    @torch.no_grad()
    def generate_latents(
        self,
        inputs_embeds: torch.Tensor,
        *,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 10,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
    ) -> list[torch.Tensor]:
        """Autoregressive LLM + CFM sampling loop"""
        if cfg is None:
            cfg = self.cfg_strength
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        his_lat = self._init_his_lat(prompt_wav_lat, device, dtype)
        past_key_values, max_cache_len = self._init_kv_cache(use_static_cache, device, dtype)
        prefill_len = inputs_embeds.shape[1]
        all_latents: list[torch.Tensor] = []

        for step in range(min(max_steps, max_cache_len - prefill_len)):
            last_hs = self.llm_step(
                inputs_embeds,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=use_static_cache,
            )
            gen_lat, inputs_embeds, stop_out = self.cfm_sample_step(
                last_hs, his_lat, cfg=cfg, sigma=sigma, temperature=temperature
            )
            his_lat = self._update_his_lat(his_lat, gen_lat)
            all_latents.append(gen_lat)

            stop_prob = stop_out.cpu()[0, 1].item()
            self._log_step(step, stop_prob, last_hs, gen_lat, inputs_embeds)

            if step > min_new_token and stop_prob > 0.5:
                logger.info("Stopping at step %d with stop_prob=%.4f", step, stop_prob)
                break

        return all_latents

    def cfm_sample_step(
        self,
        last_hidden_state: torch.Tensor,
        his_lat: torch.Tensor,
        *,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one CFM sampling step.

        This is the CFM one-shot sampling step with CUDA-graph fast path.
        """
        if cfg is None:
            cfg = self.cfg_strength

        if self._sampler_pool is not None:
            return self._sampler_pool.execute(last_hidden_state, his_lat, cfg, sigma, temperature)

        bat_size, _, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.patch_size, z_dim),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        t = get_epss_timesteps(self._config.steps, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        sde_rnd = torch.randn(
            (self._config.steps, *randn_tensor.shape),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        sde_args = torch.tensor(
            [cfg, sigma, temperature],
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        gen_lat = self._cfm.sample(last_hidden_state, his_lat, randn_tensor, t, sde_args, sde_rnd)
        inputs_embeds = self._aggregator(gen_lat)
        stop_out = self._stop_head(last_hidden_state[:, -1, :]).softmax(dim=-1)

        return gen_lat, inputs_embeds, stop_out

    def decode_to_waveform(self, latents: list[torch.Tensor], stream_decode: bool = True) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self._audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

        if stream_decode:
            return self._stream_decode(latents)

        all_lat = torch.cat(latents, dim=1)
        waveform, _, _ = self._audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True
        )
        return waveform

    def llm_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        step: int,
        past_key_values: StaticCache | None,
        use_static_cache: bool,
    ) -> torch.Tensor:
        if step == 0 or not use_static_cache:
            outputs = self._model(
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
            outputs = self._model(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                cache_position=cache_position,
            )
        return outputs.last_hidden_state[:, -1:, :]

    def _init_his_lat(
        self, prompt_wav_lat: torch.Tensor | None, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim, device=device, dtype=dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat
        return his_lat

    def _init_kv_cache(
        self, use_static_cache: bool, device: torch.device, dtype: torch.dtype
    ) -> tuple[StaticCache | None, int]:
        max_cache_len = 2048
        if not use_static_cache:
            return None, max_cache_len
        cache = StaticCache(
            config=self._llm_config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        return cache, max_cache_len

    def _update_his_lat(self, his_lat: torch.Tensor, gen_lat: torch.Tensor) -> torch.Tensor:
        if self.his_patch_size == self.patch_size:
            return gen_lat
        if self.his_patch_size > self.patch_size:
            return torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
        raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

    @staticmethod
    def _log_step(
        step: int,
        stop_prob: float,
        last_hs: torch.Tensor,
        gen_lat: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        if step % 50 == 0 or step < 5:
            logger.info(
                "step=%d stop_prob=%.4f hs_norm=%.4f lat_norm=%.4f emb_norm=%.4f",
                step,
                stop_prob,
                last_hs.float().norm().item(),
                gen_lat.float().norm().item(),
                inputs_embeds.float().norm().item(),
            )

    # VAE streaming decode
    def _stream_decode(self, latents: list[torch.Tensor]) -> torch.Tensor:
        sr = int(self._audio_vae.config.sample_rate)
        vae_cache = {"past_key_values": None, "stream_state": (None, None, None)}
        sil_cache: dict | None = None
        wav_chunks: list[torch.Tensor] = []

        for i, lat in enumerate(latents):
            last_chunk = i == (len(latents) - 1)
            speech, stream_state, past_key_values = self._audio_vae.decode(
                lat,
                past_key_values=vae_cache["past_key_values"],
                use_cache=True,
                stream_state=vae_cache["stream_state"],
                last_chunk=last_chunk,
            )
            vae_cache = {"past_key_values": past_key_values, "stream_state": stream_state}
            speech_chunk = speech[0].detach().float()
            speech_chunk, sil_cache = silence_holder(
                speech_chunk,
                sr,
                sil_cache=sil_cache,
                last_chunk=last_chunk,
            )
            if speech_chunk.numel() > 0:
                wav_chunks.append(speech_chunk)

        if not wav_chunks:
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)
        return torch.cat(wav_chunks, dim=-1).unsqueeze(0)

    # Post-decode helper
    def trim_trailing_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._audio_vae is None:
            return waveform
        return trim_trailing_silence(waveform, int(self._audio_vae.config.sample_rate))
