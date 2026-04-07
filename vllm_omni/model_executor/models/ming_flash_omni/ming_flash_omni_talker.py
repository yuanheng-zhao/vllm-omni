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

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model, StaticCache
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
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

        config: MingFlashOmniTalkerConfig = vllm_config.model_config.hf_config
        self.config = config

        # Resolve model path.
        # The HF repo layout is:
        #   <model_root>/talker/config.json        (BailingTalkerConfig)
        #   <model_root>/talker/model.safetensors   (talker LLM + CFM + aggregator + heads)
        #   <model_root>/talker/llm/config.json     (Qwen2Config for talker LLM)
        #   <model_root>/talker/llm/tokenizer.json  (tokenizer)
        #   <model_root>/talker/vae/config.json     (AudioVAEConfig)
        #   <model_root>/talker/vae/model.safetensors
        #   <model_root>/talker/campplus.onnx
        model_path = vllm_config.model_config.model
        # Detect whether model_path already points to the talker subdirectory
        talker_dir = model_path
        if os.path.isdir(os.path.join(model_path, "talker")):
            talker_dir = os.path.join(model_path, "talker")
        self._talker_dir = talker_dir

        # Load LLM config from talker/llm/ subdirectory or from config
        if config.llm_config is not None:
            if isinstance(config.llm_config, dict):
                llm_config = Qwen2Config(**config.llm_config)
            else:
                llm_config = config.llm_config
        else:
            llm_config_path = os.path.join(talker_dir, "llm")
            if os.path.isdir(llm_config_path):
                llm_config = Qwen2Config.from_pretrained(llm_config_path)
            else:
                raise ValueError(
                    f"Cannot find talker LLM config at {llm_config_path}. "
                    "Either provide llm_config in MingFlashOmniTalkerConfig or "
                    "ensure the model path contains talker/llm/config.json."
                )

        llm_config._attn_implementation = "sdpa"
        self.llm_config = llm_config
        self.hidden_size = llm_config.hidden_size
        self.latent_dim = config.latent_dim
        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size
        self.cfg_strength = config.cfg_strength

        # Qwen2 LLM backbone (using HuggingFace Qwen2Model with StaticCache)
        self.model = Qwen2Model(llm_config)

        # CFM (Conditional Flow Matching) with DiT backbone
        self.cfm = CFM(
            DiT(
                llm_input_dim=self.hidden_size,
                **config.flowmodel,
            ),
            steps=config.steps,
        )

        # Aggregator: maps generated latent back to LLM embedding space
        self.aggregator = Aggregator(
            llm_input_dim=self.hidden_size,
            **config.aggregator,
        )

        # Stop prediction head
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)

        # Speaker embedding projection (192-dim CAMPPlus → hidden_size)
        self.spk_head = nn.Linear(192, self.hidden_size, bias=True)

        # AudioVAE for latent → waveform decoding (loaded from talker/vae/ subdir)
        self.audio_vae: AudioVAE | None = None
        vae_path = config.audio_vae_path or os.path.join(talker_dir, "vae")
        if os.path.isdir(vae_path):
            try:
                vae_config = AudioVAEConfig.from_pretrained(vae_path)
                self.audio_vae = AudioVAE(vae_config)
                self._vae_path = vae_path
                logger.info("Initialized AudioVAE from %s (sample_rate=%d)", vae_path, vae_config.sample_rate)
            except Exception as e:
                logger.warning("Failed to initialize AudioVAE from %s: %s", vae_path, e)
                self._vae_path = None
        else:
            logger.info("AudioVAE path %s not found; waveform decoding will be unavailable", vae_path)
            self._vae_path = None

        # Tokenizer (loaded lazily from talker/llm/ subdirectory)
        self._tokenizer = None

        # CFM Graph executor (initialized lazily on first forward)
        self._sampler_pool: CFMGraphExecutorPool | None = None
        self._use_cuda_graphs = not vllm_config.model_config.enforce_eager

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # Try talker/llm/ first, then talker/, then model root
            candidates = [
                os.path.join(self._talker_dir, "llm"),
                self._talker_dir,
            ]
            for path in candidates:
                if os.path.isdir(path) and os.path.isfile(os.path.join(path, "tokenizer_config.json")):
                    self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    break
            if self._tokenizer is None:
                # Last resort: try the talker dir itself
                self._tokenizer = AutoTokenizer.from_pretrained(self._talker_dir, trust_remote_code=True)
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
        past_key_values = StaticCache(
            config=self.llm_config, max_batch_size=1, max_cache_len=max_cache_len, device=self.device, dtype=self.dtype
        )

        prefill_len = inputs_embeds.shape[1]
        all_latents = []

        for step in range(min(max_steps, max_cache_len - prefill_len)):
            if step == 0:
                # Prefill
                outputs = self.model(
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                # Decode step
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

            gen_lat, inputs_embeds, stop_out = self._cfm_sample_step(
                outputs.last_hidden_state[:, -1:, :], his_lat, cfg, sigma, temperature
            )

            # Update latent history
            if self.his_patch_size == self.patch_size:
                his_lat = gen_lat
            elif self.his_patch_size > self.patch_size:
                his_lat = torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
            else:
                raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

            all_latents.append(gen_lat)

            if step > min_new_token and stop_out.cpu()[0, 1] > 0.5:
                break

        return all_latents

    def decode_latents_to_waveform(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self.audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

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
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(self.device)

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
        instruction = additional_info.get("instruction", None)
        prompt_text = additional_info.get("prompt_text", None)
        prompt_wav_lat = additional_info.get("prompt_wav_lat", None)
        prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
        cfg = additional_info.get("cfg", self.cfg_strength)
        sigma = additional_info.get("sigma", 0.25)
        temperature = additional_info.get("temperature", 0.0)

        # Build input embeddings if text is provided
        if text and inputs_embeds is None:
            # Process speaker embedding through spk_head
            processed_spk_emb = None
            if spk_emb is not None:
                if isinstance(spk_emb, list):
                    processed_spk_emb = [self.spk_head(se.to(device=self.device, dtype=self.dtype)) for se in spk_emb]
                else:
                    processed_spk_emb = [self.spk_head(spk_emb.to(device=self.device, dtype=self.dtype))]

            inputs_embeds, _ = self.build_tts_input(
                text=text,
                prompt=prompt,
                spk_emb=processed_spk_emb,
                instruction=instruction,
                prompt_text=prompt_text,
                prompt_wav_emb=prompt_wav_emb,
            )

        if inputs_embeds is None:
            # Fallback: use input_ids directly
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(self.device))

        # Run autoregressive generation
        latents = self.generate_audio(
            inputs_embeds=inputs_embeds,
            prompt_wav_lat=prompt_wav_lat,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        )

        # Decode to waveform if AudioVAE is available
        multimodal_outputs = {}
        if latents and self.audio_vae is not None:
            waveform = self.decode_latents_to_waveform(latents)
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all talker components.

        The talker's HF checkpoint (talker/model.safetensors) stores weights
        with prefixes matching this module's submodule names directly:
        model.*, cfm.*, aggregator.*, stop_head.*, spk_head.*.

        Note: We use manual named_parameters() matching instead of vLLM's
        AutoWeightsLoader because the talker LLM is a HuggingFace Qwen2Model
        (not vLLM's), so stacked-param merging (qkv_proj, gate_up_proj) does
        not apply.

        AudioVAE weights live in a separate file (talker/vae/model.safetensors)
        and are loaded via _load_vae_weights().
        """
        loaded = set()

        # Build param lookup excluding audio_vae (loaded separately)
        param_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())
        talker_params = {k: v for k, v in {**param_dict, **buffer_dict}.items() if not k.startswith("audio_vae.")}

        for name, tensor in weights:
            if name in talker_params:
                talker_params[name].data.copy_(tensor)
                loaded.add(name)

        logger.info("Loaded %d talker weights from checkpoint", len(loaded))

        # Load AudioVAE weights from separate safetensors file
        if self.audio_vae is not None and self._vae_path is not None:
            vae_loaded = self._load_vae_weights()
            loaded.update(vae_loaded)

        return loaded

    def _load_vae_weights(self) -> set[str]:
        """Load AudioVAE weights from talker/vae/model.safetensors."""
        import glob as glob_module

        loaded = set()
        if self.audio_vae is None or self._vae_path is None:
            return loaded

        safetensors_files = sorted(glob_module.glob(os.path.join(self._vae_path, "*.safetensors")))
        if not safetensors_files:
            logger.warning("No safetensors files found in %s", self._vae_path)
            return loaded

        from safetensors.torch import load_file

        vae_state_dict = self.audio_vae.state_dict()
        for sf_path in safetensors_files:
            file_weights = load_file(sf_path, device="cpu")
            for name, tensor in file_weights.items():
                if name in vae_state_dict:
                    vae_state_dict[name].copy_(tensor)
                    loaded.add(f"audio_vae.{name}")

        logger.info("Loaded %d AudioVAE weights from %s", len(loaded), self._vae_path)
        return loaded

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors | None:
        return None
