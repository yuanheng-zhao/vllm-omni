# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.

"""Ming-flash-omni-2.0 Thinker stage implementation (multimodal understanding)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .components import (
    AudioProjector,
    BailingMoeV2ForCausalLM,
    MingVisionEncoder,
    VisionProjector,
    WhisperAudioEncoder,
    get_rope_index,
)
from .configuration_ming_flash_omni import MingFlashOmniThinkerConfig

logger = init_logger(__name__)


class MingFlashOmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """
    Ming Thinker stage: Multimodal understanding → text generation.

    Components:
    - Vision: MingVisionEncoder (wraps Qwen3Omni_VisionTransformer)
    - Audio: WhisperAudioEncoder (audio inputs)
    - LLM: BailingMoeV2ForCausalLM (100B total, 6B active MoE)

    The Thinker stage processes multimodal inputs (text, image, video, audio) and
    generates text responses. It also captures intermediate embeddings for potential
    downstream stages (image generation, audio generation).

    Supports Multi-Dimensional RoPE (MRoPE) for 3D position encoding in multimodal contexts.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # Get the thinker config
        config = vllm_config.model_config.hf_config

        if hasattr(config, "llm_config"):
            # If initialized with MingFlashOmniThinkerConfig
            # TODO: Maybe adding a type check / casting
            thinker_config: MingFlashOmniThinkerConfig = config
            llm_config = thinker_config.llm_config
        else:
            # If initialized directly with LLM config (from unified model)
            llm_config = config
            thinker_config = None  # type: ignore

        self.config = llm_config
        self.thinker_config = thinker_config
        self.have_multimodal_outputs = True

        # Initialize LLM as a component (not inherit)
        llm_vllm_config = vllm_config.with_hf_config(llm_config)
        self.llm = BailingMoeV2ForCausalLM(vllm_config=llm_vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Initialize vision encoder if configured
        self.vision = None
        self.linear_proj = None
        if thinker_config and thinker_config.vision_config:
            self.vision = MingVisionEncoder(
                vision_config=thinker_config.vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision"),
            )
            self.linear_proj = VisionProjector(
                vision_dim=self.vision.image_emb_dim,
                llm_dim=llm_config.hidden_size,
                mlp_depth=getattr(thinker_config, "mlp_depth", 2),
            )
            logger.info("Initialized MingVisionEncoder and VisionProjector")

        # Initialize audio encoder if configured
        self.audio = None
        self.linear_proj_audio = None
        if thinker_config and thinker_config.audio_config:
            audio_cfg = thinker_config.audio_config
            self.audio = WhisperAudioEncoder(
                n_mels=getattr(audio_cfg, "n_mels", 128),
                n_ctx=getattr(audio_cfg, "n_ctx", 15000),
                n_state=getattr(audio_cfg, "n_state", 1280),
                n_head=getattr(audio_cfg, "n_head", 20),
                n_layer=getattr(audio_cfg, "n_layer", 32),
                use_flash_attn=True,
            )
            self.linear_proj_audio = AudioProjector(
                audio_dim=self.audio.audio_emb_dim,
                llm_dim=llm_config.hidden_size,
                ds_kernel_size=getattr(audio_cfg, "ds_kernel_size", 3),
                ds_stride=getattr(audio_cfg, "ds_stride", 2),
                mlp_depth=getattr(thinker_config, "mlp_depth", 1),
            )
            logger.info("Initialized WhisperAudioEncoder and AudioProjector")

        logger.info(
            f"MingFlashOmniThinker initialized with: "
            f"vision={'yes' if self.vision else 'no'}, "
            f"audio={'yes' if self.audio else 'no'}"
        )

        # Expose interfaces from LLM
        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

    def extract_image_feature(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Extract and project image features.

        Args:
            pixel_values: Flattened pixel values from vision processor.
            grid_thw: [num_images, 3] tensor of (t, h, w) grid dimensions.

        Returns:
            [seq_len, hidden_size] L2-normalized image embeddings.
        """
        if self.vision is None:
            raise ValueError("Vision encoder not initialized")

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)

        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def extract_audio_feature(
        self, audio_feats: torch.Tensor, audio_feats_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and project audio features.

        Args:
            audio_feats: [B, n_mels, T] mel spectrogram features.
            audio_feats_lengths: [B] original audio lengths in frames.

        Returns:
            Tuple of:
                - audio_embeds: [B, T', hidden_size] projected audio embeddings
                - audio_embeds_lengths: [B] lengths after projection
        """
        if self.audio is None:
            raise ValueError("Audio encoder not initialized")

        # Convert to list format expected by WhisperAudioEncoder
        x_list = [audio_feats[i, :, : audio_feats_lengths[i]] for i in range(audio_feats.size(0))]
        audio_lens = audio_feats_lengths.tolist()

        # Encode audio (returns packed format [total_T', n_state])
        audio_embeds_packed = self.audio(x_list, audio_lens)

        # Project audio features
        # Need to split packed format back into list, project each, then repack
        # For simplicity, we'll work with the full tensor
        # TODO: Optimize by keeping packed format through projection
        audio_embeds_list = []
        offset = 0
        audio_embeds_lengths = []
        for audio_len in audio_lens:
            # Calculate encoded length after Whisper's Conv1d (stride=2)
            encoded_len = (audio_len - 3 + 2 * 1) // 2 + 1
            audio_segment = audio_embeds_packed[offset : offset + encoded_len]
            offset += encoded_len

            # Project this segment
            audio_segment_proj = self.linear_proj_audio(audio_segment.unsqueeze(0))
            audio_embeds_list.append(audio_segment_proj.squeeze(0))

            # Calculate final length after audio projector Conv1d
            final_len = self.linear_proj_audio.compute_output_length(torch.tensor([audio_len])).item()
            audio_embeds_lengths.append(int(final_len))

        # Stack into batch format [B, max_T', hidden_size]
        # Pad to max length
        max_len = max(audio_embeds_lengths)
        audio_embeds = torch.zeros(
            len(audio_embeds_list),
            max_len,
            audio_embeds_list[0].size(-1),
            dtype=audio_embeds_packed.dtype,
            device=audio_embeds_packed.device,
        )
        for i, emb in enumerate(audio_embeds_list):
            audio_embeds[i, : emb.size(0)] = emb

        audio_embeds_lengths = torch.tensor(audio_embeds_lengths, dtype=torch.long, device=audio_embeds.device)

        # Apply L2 normalization if configured
        if (
            self.thinker_config
            and self.thinker_config.audio_config
            and getattr(self.thinker_config.audio_config, "norm_query_embeds", False)
        ):
            audio_embeds = F.normalize(audio_embeds, dim=2)

        return audio_embeds.to(audio_feats.dtype), audio_embeds_lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        """
        Forward pass with multimodal inputs.

        This method follows vLLM's standard interface for omni models.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Token positions (3D for MRoPE)
            intermediate_tensors: Intermediate tensors from previous pipeline stages
            inputs_embeds: Optional pre-computed input embeddings
            **kwargs: Additional multimodal arguments:
                - pixel_values: Image pixel values
                - image_grid_thw: Image grid dimensions
                - video_grid_thw: Video grid dimensions
                - audio_feats: Audio mel spectrogram features
                - audio_feats_lengths: Audio feature lengths
                - placeholder_audio_loc_lens: Audio placeholder locations

        Returns:
            OmniOutput with text hidden states and multimodal outputs
        """
        # Extract multimodal data from kwargs
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        audio_feats = kwargs.get("audio_feats")
        audio_feats_lengths = kwargs.get("audio_feats_lengths")
        placeholder_audio_loc_lens = kwargs.get("placeholder_audio_loc_lens")

        # Process images
        query_embeds_image = None
        query_embeds_video = None
        if pixel_values is not None and self.vision is not None:
            # Combine image and video grids if both present
            if image_grid_thw is not None and video_grid_thw is not None:
                grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
            elif image_grid_thw is not None:
                grid_thw = image_grid_thw
            elif video_grid_thw is not None:
                grid_thw = video_grid_thw
            else:
                grid_thw = None

            if grid_thw is not None:
                image_embeds = self.extract_image_feature(pixel_values, grid_thw)

                # Split back into image and video if needed
                if image_grid_thw is not None and video_grid_thw is not None:
                    # Calculate split point based on actual feature lengths
                    # This is a simplification; actual split logic may differ
                    query_embeds_image = image_embeds  # TODO: proper split
                    query_embeds_video = None
                elif video_grid_thw is not None:
                    query_embeds_video = image_embeds
                else:
                    query_embeds_image = image_embeds

        # Process audio
        query_embeds_audio = None
        query_embeds_audio_lengths = None
        if audio_feats is not None and self.audio is not None:
            query_embeds_audio, query_embeds_audio_lengths = self.extract_audio_feature(
                audio_feats, audio_feats_lengths
            )

        # Forward through LLM with multimodal embeddings
        # The LLM's forward method will call BailingMoeV2Model.prompt_wrap_navit()
        # to merge embeddings internally
        hidden_states = self.llm.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_embeds_image=query_embeds_image,
            query_embeds_video=query_embeds_video,
            query_embeds_audio=query_embeds_audio,
            query_embeds_audio_lengths=query_embeds_audio_lengths,
            placeholder_audio_loc_lens=placeholder_audio_loc_lens,
            image_grid_thw=image_grid_thw,
            image_grid_thw_video=video_grid_thw,
        )

        # Capture embeddings for downstream stages
        multimodal_outputs = {
            "final_hidden_states": hidden_states,
        }

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        return self.llm.compute_logits(hidden_states, sampling_metadata)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Sample next tokens from logits."""
        return self.llm.sample(logits, sampling_metadata)

    @property
    def sampler(self):
        """Get sampler from LLM."""
        return self.llm.sampler

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        *,
        hf_config: PretrainedConfig | None = None,
        image_grid_thw: list[list[int]] | torch.Tensor | None = None,
        video_grid_thw: list[list[int]] | torch.Tensor | None = None,
        second_per_grid_ts: list[float] | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Get MRoPE input positions and delta value for Ming-flash-omni.

        Computes 3D position indices (T, H, W) for multimodal inputs including
        images, videos, and audio, then returns position tensor and position delta.

        Args:
            input_tokens: List of input token IDs
            mm_features: Multimodal feature specifications (optional, for vLLM integration)
            hf_config: HuggingFace config (uses self.config if not provided)
            image_grid_thw: Image grid dimensions [(T, H, W), ...] or tensor of shape (num_images, 3)
            video_grid_thw: Video grid dimensions [(T, H, W), ...] or tensor of shape (num_videos, 3)
            second_per_grid_ts: Seconds per temporal grid for videos
            audio_feature_lengths: Audio feature sequence lengths

        Returns:
            Tuple of (position_ids, mrope_position_delta):
                - position_ids: Shape (3, seq_len) with [T, H, W] position indices
                - mrope_position_delta: Scalar offset for position expansion
        """
        # Use model's config if not provided
        if hf_config is None:
            if self.thinker_config is not None:
                llm_config = self.thinker_config.llm_config
            else:
                llm_config = self.config
        else:
            # Extract LLM config from thinker config if needed
            if hasattr(hf_config, "llm_config"):
                llm_config = hf_config.llm_config
            else:
                llm_config = hf_config

        # Gather multimodal metadata from mm_features if provided
        if mm_features is not None:
            kwargs = MultiModalFeatureSpec.gather_kwargs(
                mm_features,
                {
                    "image_grid_thw",
                    "video_grid_thw",
                    "second_per_grid_ts",
                    "audio_feature_lengths",
                },
            )
            image_grid_thw = kwargs.get("image_grid_thw", image_grid_thw or [])
            video_grid_thw = kwargs.get("video_grid_thw", video_grid_thw or [])
            second_per_grid_ts = kwargs.get("second_per_grid_ts", second_per_grid_ts)
            audio_feature_lengths = kwargs.get("audio_feature_lengths", audio_feature_lengths)

        # Convert to tensors if needed
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, list):
                image_grid_thw = torch.tensor(image_grid_thw) if image_grid_thw else None
        if video_grid_thw is not None:
            if isinstance(video_grid_thw, list):
                video_grid_thw = torch.tensor(video_grid_thw) if video_grid_thw else None

        # Convert input_tokens to tensor
        input_ids_tensor = torch.tensor([input_tokens], dtype=torch.long)

        # Call get_rope_index to compute 3D positions
        position_ids, mrope_position_deltas = get_rope_index(
            config=llm_config,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,  # Will use default (all ones)
            second_per_grid_ts=second_per_grid_ts,
            use_interleaved_frame_timestamp=getattr(llm_config, "use_interleaved_frame_timestamp", True),
        )

        # Remove batch dimension for vLLM (expects shape [3, seq_len])
        position_ids = position_ids.squeeze(1)  # [3, 1, seq_len] -> [3, seq_len]
        mrope_position_delta = mrope_position_deltas[0, 0].item()  # Extract scalar

        return position_ids, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for thinker model (LLM + encoders + projectors).

        Routes weights to appropriate submodules based on name prefixes:
        - model.* → LLM (BailingMoeV2ForCausalLM)
        - vision.* → Vision encoder (MingVisionEncoder)
        - audio.* → Audio encoder (WhisperAudioEncoder)
        - linear_proj.* → Vision projector (VisionProjector)
        - linear_proj_audio.* → Audio projector (AudioProjector)

        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint.

        Returns:
            Set of successfully loaded parameter names.
        """
        loaded_params = set()

        # Separate weights by component prefix
        llm_weights = []
        vision_weights = []
        audio_weights = []
        vision_proj_weights = []
        audio_proj_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model."):
                llm_weights.append((name, loaded_weight))
            elif name.startswith("vision."):
                vision_weights.append((name[len("vision.") :], loaded_weight))
            elif name.startswith("audio."):
                audio_weights.append((name[len("audio.") :], loaded_weight))
            elif name.startswith("linear_proj_audio."):
                audio_proj_weights.append((name[len("linear_proj_audio.") :], loaded_weight))
            elif name.startswith("linear_proj."):
                vision_proj_weights.append((name[len("linear_proj.") :], loaded_weight))
            else:
                logger.warning(f"Unrecognized weight prefix in thinker: {name}")

        # Load LLM weights
        if llm_weights:
            logger.info(f"Loading {len(llm_weights)} LLM weights")
            llm_loaded = self.llm.load_weights(llm_weights)
            loaded_params.update(llm_loaded)

        # Load vision encoder weights
        if self.vision and vision_weights:
            logger.info(f"Loading {len(vision_weights)} vision encoder weights")
            vision_loaded = self.vision.load_weights(vision_weights)
            loaded_params.update([f"vision.{n}" for n in vision_loaded])

        # Load audio encoder weights
        if self.audio and audio_weights:
            logger.info(f"Loading {len(audio_weights)} audio encoder weights")
            audio_loaded = self.audio.load_weights(audio_weights)
            loaded_params.update([f"audio.{n}" for n in audio_loaded])

        # Load vision projector weights
        if self.linear_proj and vision_proj_weights:
            logger.info(f"Loading {len(vision_proj_weights)} vision projector weights")
            vision_proj_loaded = self.linear_proj.load_weights(vision_proj_weights)
            loaded_params.update([f"linear_proj.{n}" for n in vision_proj_loaded])

        # Load audio projector weights
        if self.linear_proj_audio and audio_proj_weights:
            logger.info(f"Loading {len(audio_proj_weights)} audio projector weights")
            audio_proj_loaded = self.linear_proj_audio.load_weights(audio_proj_weights)
            loaded_params.update([f"linear_proj_audio.{n}" for n in audio_proj_loaded])

        return loaded_params
