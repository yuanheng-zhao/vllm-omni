# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.

"""Ming-flash-omni-2.0 Thinker stage implementation (multimodal understanding)."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config, MingFlashOmniThinkerConfig
from vllm_omni.transformers_utils.processors.ming import MingFlashOmniProcessor, MingWhisperFeatureExtractor

from .components import (
    AudioProjector,
    BailingMoeV2ForCausalLM,
    MingVisionEncoder,
    VisionProjector,
    WhisperAudioEncoder,
    get_rope_index,
)

logger = init_logger(__name__)


# === Multimodal Processor Classes === #


class MingFlashOmniThinkerProcessingInfo(Qwen2_5_VLProcessingInfo):
    """Processing info for Ming-flash-omni Thinker stage.

    Provides access to HuggingFace config, processor, and tokenizer for
    multimodal input processing.
    """

    def get_hf_config(self) -> BailingMM2Config:
        """Get the HuggingFace configuration for Ming-flash-omni Thinker."""
        return self.ctx.get_hf_config(BailingMM2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(MingFlashOmniProcessor, **kwargs)

    def get_target_channels(self) -> int:
        """Return target audio channels (mono).

        See `_normalize_audio_tensor` in vllm_omni/transformers_utils/processors/ming.py
        """
        return 1

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """Get supported multimodal limits.

        Returns:
            Dict with None (unlimited) for image, video, and audio.
        """
        return {"image": None, "video": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        mm_counts = mm_counts or {}
        requested_modalities = {m for m, c in mm_counts.items() if c > 0}
        mm_max_tokens: dict[str, int] = {}

        if requested_modalities & {"image", "video"}:
            vl_tokens = Qwen2_5_VLProcessingInfo.get_mm_max_tokens_per_item(
                self,
                seq_len=seq_len,
                mm_counts=mm_counts,
            )
            mm_max_tokens.update({m: vl_tokens[m] for m in ["image", "video"] if m in requested_modalities})

        if "audio" in requested_modalities:
            # TODO: HARDCODED for now
            mm_max_tokens["audio"] = 3000

        return mm_max_tokens

    def get_feature_extractor(self, **kwargs: object) -> MingWhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.audio_processor  # type: ignore
        assert isinstance(feature_extractor, MingWhisperFeatureExtractor)

        return feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class MingFlashOmniThinkerDummyInputsBuilder(BaseDummyInputsBuilder[MingFlashOmniThinkerProcessingInfo]):
    """Dummy inputs builder for profiling Ming-flash-omni Thinker."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with multimodal placeholders.

        Args:
            mm_counts: Dict with counts for each modality (image, video, audio).

        Returns:
            String with placeholder tokens for each modality.
        """
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        if hf_processor is not None:
            image_token = getattr(hf_processor, "image_token", "<IMAGE>")
            video_token = getattr(hf_processor, "video_token", "<VIDEO>")
            audio_token = getattr(hf_processor, "audio_token", "<AUDIO>")
        else:
            image_token = "<IMAGE>"
            video_token = "<VIDEO>"
            audio_token = "<AUDIO>"

        return image_token * num_images + video_token * num_videos + audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        """Generate dummy multimodal data for profiling.

        Args:
            seq_len: Sequence length.
            mm_counts: Dict with counts for each modality.

        Returns:
            Dict with dummy image, video, and audio data.
        """
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        # Default dimensions for dummy data
        image_width, image_height = 448, 448
        video_width, video_height = 448, 448
        num_frames = 8
        audio_duration = 3.0  # seconds
        sample_rate = 16000

        audio_length = int(audio_duration * sample_rate)

        mm_data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=num_images,
            ),
            "video": self._get_dummy_videos(
                width=video_width,
                height=video_height,
                num_frames=num_frames,
                num_videos=num_videos,
            ),
            "audio": [(np.random.randn(audio_length).astype(np.float32), sample_rate) for _ in range(num_audios)],
        }

        return mm_data


class MingFlashOmniThinkerMultiModalProcessor(BaseMultiModalProcessor[MingFlashOmniThinkerProcessingInfo]):
    """Multimodal processor for Ming-flash-omni Thinker stage.

    Handles preprocessing of image, video, and audio inputs, and expands
    placeholder tokens to the correct number of patch tokens.
    """

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Generate prompt updates for multimodal placeholders.

        This method defines how placeholder tokens (e.g., <IMAGE>, <VIDEO>, <AUDIO>)
        are replaced with the actual patch tokens based on the processed multimodal data.
        Only patch tokens (<imagePatch>, <framePatch>, <audioPatch>) receive multimodal
        embeddings; delimiter tokens (<image>, </image>, etc.) use regular text embeddings.

        Args:
            mm_items: Parsed multimodal data items.
            hf_processor_mm_kwargs: Kwargs for HF processor.
            out_mm_kwargs: Output multimodal kwargs with processed features.

        Returns:
            List of PromptUpdate objects defining token replacements.
        """
        from vllm_omni.transformers_utils.processors.ming import (
            PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
            PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
            PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
        )

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _get_placeholder_ids(token_str: str) -> list[int] | None:
            """Get token IDs for a placeholder string.

            First tries vocab lookup (fast path for single special tokens).
            Falls back to tokenizer.encode() which handles multi-token cases
            and tokens stored with SentencePiece prefixes (e.g. '▁<IMAGE>').
            """
            single_id = vocab.get(token_str)
            if single_id is not None:
                return [single_id]
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            return ids if ids else None

        # High-level placeholder token IDs (targets for replacement)
        image_placeholder_ids = _get_placeholder_ids(PLACEHOLDER_IMAGE_TOKEN_IN_TEXT)
        video_placeholder_ids = _get_placeholder_ids(PLACEHOLDER_VIDEO_TOKEN_IN_TEXT)
        audio_placeholder_ids = _get_placeholder_ids(PLACEHOLDER_AUDIO_TOKEN_IN_TEXT)

        # Low-level patch/delimiter token IDs (used in replacement sequences)
        image_start_token = vocab.get("<image>", vocab.get("<|image|>", None))
        image_patch_token = vocab.get("<imagePatch>", None)
        image_end_token = vocab.get("</image>", vocab.get("<|/image|>", None))

        video_start_token = vocab.get("<video>", vocab.get("<|video|>", None))
        frame_patch_token = vocab.get("<framePatch>", None)
        video_end_token = vocab.get("</video>", vocab.get("<|/video|>", None))

        audio_start_token = vocab.get("<audio>", vocab.get("<|audio|>", None))
        audio_patch_token = vocab.get("<audioPatch>", None)
        audio_end_token = vocab.get("</audio>", vocab.get("<|/audio|>", None))

        # Get config for spatial merge size
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config
        spatial_merge_size = vision_config.spatial_merge_size if vision_config else 2

        out_mm_data = out_mm_kwargs.get_data()

        def get_replacement_image(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for an image."""
            grid_thw = out_mm_data.get("image_grid_thw", None)
            if grid_thw is None:
                logger.warning("image_grid_thw not found in output, using default 256 patches")
                num_patches = 256
            else:
                if isinstance(grid_thw, torch.Tensor):
                    thw = grid_thw[item_idx]
                    num_patches = int(thw.prod().item()) // (spatial_merge_size**2)
                else:
                    thw = grid_thw[item_idx]
                    num_patches = (thw[0] * thw[1] * thw[2]) // (spatial_merge_size**2)

            # Build token sequence: <image> <imagePatch>*N </image>
            tokens: list[int] = []
            if image_start_token is not None:
                tokens.append(image_start_token)
            if image_patch_token is not None:
                tokens.extend([image_patch_token] * num_patches)
            if image_end_token is not None:
                tokens.append(image_end_token)

            # Only <imagePatch> tokens receive multimodal embeddings
            if image_patch_token is not None:
                return PromptUpdateDetails.select_token_id(tokens, image_patch_token)
            return PromptUpdateDetails.from_seq(tokens)

        def get_replacement_video(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for a video."""
            grid_thw = out_mm_data.get("video_grid_thw", None)
            if grid_thw is None:
                logger.warning("video_grid_thw not found in output, using default 512 patches")
                num_patches = 512
            else:
                if isinstance(grid_thw, torch.Tensor):
                    thw = grid_thw[item_idx]
                    num_patches = int(thw.prod().item()) // (spatial_merge_size**2)
                else:
                    thw = grid_thw[item_idx]
                    num_patches = (thw[0] * thw[1] * thw[2]) // (spatial_merge_size**2)

            # Build token sequence: <video> <framePatch>*N </video>
            tokens: list[int] = []
            if video_start_token is not None:
                tokens.append(video_start_token)
            if frame_patch_token is not None:
                tokens.extend([frame_patch_token] * num_patches)
            if video_end_token is not None:
                tokens.append(video_end_token)

            # Only <framePatch> tokens receive multimodal embeddings
            if frame_patch_token is not None:
                return PromptUpdateDetails.select_token_id(tokens, frame_patch_token)
            return PromptUpdateDetails.from_seq(tokens)

        def get_replacement_audio(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for an audio."""
            encoder_feats_lengths = out_mm_data.get("encoder_feats_lengths", None)
            if encoder_feats_lengths is None:
                logger.warning("encoder_feats_lengths not found in output, using default 100 patches")
                num_patches = 100
            else:
                if isinstance(encoder_feats_lengths, torch.Tensor):
                    num_patches = int(encoder_feats_lengths[item_idx].item())
                else:
                    num_patches = encoder_feats_lengths[item_idx]

            # Build token sequence: <audio> <audioPatch>*N </audio>
            tokens: list[int] = []
            if audio_start_token is not None:
                tokens.append(audio_start_token)
            if audio_patch_token is not None:
                tokens.extend([audio_patch_token] * num_patches)
            if audio_end_token is not None:
                tokens.append(audio_end_token)

            # Only <audioPatch> tokens receive multimodal embeddings
            if audio_patch_token is not None:
                return PromptUpdateDetails.select_token_id(tokens, audio_patch_token)
            return PromptUpdateDetails.from_seq(tokens)

        # Build prompt updates
        updates: list[PromptUpdate] = []

        # Image replacement
        if "image" in mm_items and mm_items.get_items("image", ImageProcessorItems):
            if image_placeholder_ids is not None:
                updates.append(
                    PromptReplacement(
                        modality="image",
                        target=image_placeholder_ids,
                        replacement=get_replacement_image,
                    )
                )

        # Video replacement
        if "video" in mm_items and mm_items.get_items("video", VideoProcessorItems):
            if video_placeholder_ids is not None:
                updates.append(
                    PromptReplacement(
                        modality="video",
                        target=video_placeholder_ids,
                        replacement=get_replacement_video,
                    )
                )

        # Audio replacement
        if "audio" in mm_items and mm_items.get_items("audio", AudioProcessorItems):
            if audio_placeholder_ids is not None:
                updates.append(
                    PromptReplacement(
                        modality="audio",
                        target=audio_placeholder_ids,
                        replacement=get_replacement_audio,
                    )
                )

        return updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Get multimodal field configurations.

        Defines how each multimodal field should be sliced into per-item chunks.
        Qwen2VL-style image processing concatenates patches from all images into a
        flat tensor, so pixel_values must use ``flat_from_sizes`` (not ``batched``).

        Args:
            hf_inputs: Output from HuggingFace processor.
            hf_processor_mm_kwargs: Kwargs used for HF processor.

        Returns:
            Dict mapping field names to their configurations.
        """
        config: dict[str, MultiModalFieldConfig] = {}

        # Image fields — pixel_values is flat (concatenated patches from all images)
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        if "pixel_values" in hf_inputs:
            image_sizes = image_grid_thw.prod(-1)
            config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image",
                image_sizes,
            )
        if "image_grid_thw" in hf_inputs:
            config["image_grid_thw"] = MultiModalFieldConfig.batched("image")

        # Video fields — same flat layout as images
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        if "pixel_values_videos" in hf_inputs:
            video_sizes = video_grid_thw.prod(-1)
            config["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video",
                video_sizes,
            )
        if "video_grid_thw" in hf_inputs:
            config["video_grid_thw"] = MultiModalFieldConfig.batched("video")

        # Audio fields
        if "audio_feats" in hf_inputs:
            config["audio_feats"] = MultiModalFieldConfig.batched("audio")
        if "audio_feats_lengths" in hf_inputs:
            config["audio_feats_lengths"] = MultiModalFieldConfig.batched("audio")
        if "encoder_feats_lengths" in hf_inputs:
            config["encoder_feats_lengths"] = MultiModalFieldConfig.batched("audio")
        if "placeholder_audio_loc_lens" in hf_inputs:
            config["placeholder_audio_loc_lens"] = MultiModalFieldConfig.batched("audio")

        return config

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call sub-processors for multimodal inputs and tokenize.

        We call the image/audio sub-processors directly (instead of going
        through ``MingFlashOmniProcessor.__call__``) so that the high-level
        placeholder tokens (``<IMAGE>``, ``<VIDEO>``, ``<AUDIO>``) remain
        **unexpanded** in the tokenized output.  vLLM's
        ``_apply_prompt_updates`` will expand them via the
        ``PromptReplacement`` objects returned by ``_get_prompt_updates``.

        Args:
            prompt: Text prompt with placeholders.
            mm_data: Multimodal data (images, videos, audios).
            mm_kwargs: Multimodal processing kwargs.
            tok_kwargs: Tokenizer kwargs.

        Returns:
            BatchFeature with processed inputs.
        """
        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()

        data: dict[str, object] = {}

        # Process images (pixel values + grid_thw only, no text expansion)
        images = mm_data.get("images", None)
        if images is not None and hf_processor.image_processor is not None:
            image_outputs = hf_processor.image_processor(
                images=images,
                videos=None,
                return_tensors="pt",
            )
            data.update(image_outputs)

        # Process videos
        videos = mm_data.get("videos", None)
        if videos is not None and hf_processor.image_processor is not None:
            video_outputs = hf_processor.image_processor(
                images=None,
                videos=videos,
                return_tensors="pt",
            )
            # Rename keys to distinguish from images
            if "pixel_values" in video_outputs:
                video_outputs["pixel_values_videos"] = video_outputs.pop("pixel_values")
            if "image_grid_thw" in video_outputs:
                video_outputs["video_grid_thw"] = video_outputs.pop("image_grid_thw")
            data.update(video_outputs)

        # Process audio
        audios = mm_data.get("audios", None)
        if audios is not None and hf_processor.audio_processor is not None:
            # vLLM's AudioProcessorItems provides raw numpy arrays (already
            # resampled).  MingWhisperAudioProcessor expects (waveform, sr)
            # tuples, so wrap them with the target sample rate.
            target_sr = hf_processor.audio_processor.sampling_rate
            audio_tuples = [(a, target_sr) if not isinstance(a, tuple) else a for a in audios]

            audio_outputs = hf_processor.audio_processor(
                audio_tuples,
                return_tensors="pt",
            )
            data.update(audio_outputs)

        # Tokenize text with placeholders still intact
        text_outputs = tokenizer(prompt, return_tensors="pt", **tok_kwargs)
        data.update(text_outputs)

        return BatchFeature(data=data)


# === Main Model Class === #


@MULTIMODAL_REGISTRY.register_processor(
    MingFlashOmniThinkerMultiModalProcessor,
    info=MingFlashOmniThinkerProcessingInfo,
    dummy_inputs=MingFlashOmniThinkerDummyInputsBuilder,
)
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
        logger.info(f" >>> config: {type(config)}")

        if hasattr(config, "llm_config"):
            # If initialized with MingFlashOmniThinkerConfig
            # TODO: Maybe adding a type check / casting
            thinker_config: MingFlashOmniThinkerConfig = config
            llm_config = thinker_config.llm_config
        else:
            # If initialized directly with LLM config (from unified model)
            llm_config = config
            thinker_config = None  # type: ignore

        logger.info(f" >>> thinker_config: {type(thinker_config)}; llm_config: {type(llm_config)}")

        self.config = llm_config
        self.thinker_config = thinker_config
        self.have_multimodal_outputs = True

        # Initialize LLM as a component (not inherit)
        llm_vllm_config = vllm_config.with_hf_config(llm_config)
        self.llm = BailingMoeV2ForCausalLM(vllm_config=llm_vllm_config, prefix=maybe_prefix(prefix, "llm"))

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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)

        # When deepstack is enabled, the vision encoder returns concatenated
        # multi-scale features: [seq_len, out_hidden_size * (1 + num_deepstack)].
        # The original Ming repo discards deepstack features and only projects
        # the main features (first out_hidden_size dims) through linear_proj.
        if self.vision.use_deepstack:
            image_embeds = image_embeds[:, : self.vision.image_emb_dim]

        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def extract_audio_feature(
        self, audio_feats: torch.Tensor, audio_feats_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and project audio features.

        Args:
            audio_feats: [B, L_total, n_mels] wrapped mel features — multiple audio
                clips per batch item are concatenated along the time dimension
                (time-first, as produced by MingWhisperFeatureExtractor).
            audio_feats_lengths: [B, N] lengths of each audio clip per batch item.
                N is the max number of clips per item; zero-padded entries are skipped.

        Returns:
            Tuple of:
                - audio_embeds: [total_clips, T'_max, hidden_size] projected audio
                  embeddings, padded to the longest projected clip length.
                - audio_embeds_lengths: [total_clips] lengths after projection.
        """
        if self.audio is None:
            raise ValueError("Audio encoder not initialized")

        # Unwrap packed [B, L_total, n_mels] into a list of [n_mels, T_i] tensors,
        # one per audio clip, as expected by WhisperAudioEncoder.
        x_list: list[torch.Tensor] = []
        audio_lens: list[int] = []
        for i in range(audio_feats_lengths.shape[0]):
            feat_index = 0
            for j in range(audio_feats_lengths.shape[1]):
                feat_len = int(audio_feats_lengths[i, j].item())
                if feat_len == 0:
                    break
                # audio_feats[i] is [L_total, n_mels]; slice one clip and
                # transpose to [n_mels, T] for Conv1d.
                mel_seg = audio_feats[i, feat_index : feat_index + feat_len].transpose(0, 1)
                x_list.append(mel_seg)
                audio_lens.append(feat_len)
                feat_index += feat_len

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
            # WhisperAudioEncoder: conv1 (stride=1, no length change) +
            # conv2 (kernel=3, stride=2, padding=1) → (T-1)//2 + 1
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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Encode multimodal inputs and return per-item 2D embedding tensors.

        Called by the vLLM v1 model runner before the forward pass.
        Returns a tuple of 2D tensors (one per multimodal item) which are then
        merged into input_ids embeddings via embed_input_ids.
        """
        embeddings: list[torch.Tensor] = []

        # The vision encoder applies a 2×2 spatial merger, so token count per
        # image/video is T*H*W // spatial_merge_size².
        spatial_merge_size = 2
        if self.thinker_config and self.thinker_config.vision_config:
            spatial_merge_size = getattr(self.thinker_config.vision_config, "spatial_merge_size", 2)
        spatial_merge_unit = spatial_merge_size**2

        # --- Images ---
        pixel_values: torch.Tensor | None = kwargs.get("pixel_values")  # type: ignore[assignment]
        image_grid_thw: torch.Tensor | None = kwargs.get("image_grid_thw")  # type: ignore[assignment]
        if pixel_values is not None and image_grid_thw is not None and self.vision is not None:
            image_embeds = self.extract_image_feature(pixel_values, image_grid_thw)
            # Split flat [total_tokens, D] into per-image tensors.
            # After the 2×2 merger, each image has T*H*W // spatial_merge_unit tokens.
            sizes = (image_grid_thw.prod(dim=-1) // spatial_merge_unit).tolist()
            for emb in image_embeds.split([int(s) for s in sizes], dim=0):
                embeddings.append(emb)

        # --- Videos ---
        pixel_values_videos: torch.Tensor | None = kwargs.get("pixel_values_videos")  # type: ignore[assignment]
        video_grid_thw: torch.Tensor | None = kwargs.get("video_grid_thw")  # type: ignore[assignment]
        if pixel_values_videos is not None and video_grid_thw is not None and self.vision is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, video_grid_thw)
            sizes = (video_grid_thw.prod(dim=-1) // spatial_merge_unit).tolist()
            for emb in video_embeds.split([int(s) for s in sizes], dim=0):
                embeddings.append(emb)

        # --- Audio ---
        audio_feats: torch.Tensor | None = kwargs.get("audio_feats")  # type: ignore[assignment]
        audio_feats_lengths: torch.Tensor | None = kwargs.get("audio_feats_lengths")  # type: ignore[assignment]
        if audio_feats is not None and audio_feats_lengths is not None and self.audio is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths)
            for i in range(audio_embeds.size(0)):
                length = int(audio_embeds_lengths[i].item())
                embeddings.append(audio_embeds[i, :length, :])

        return tuple(embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input token IDs and scatter multimodal embeddings."""
        inputs_embeds = self.llm.model.word_embeddings(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None, "is_multimodal mask required when multimodal_embeddings provided"
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

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
                llm_weights.append((name[len("model.") :], loaded_weight))
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
            loaded_params.update([f"llm.{n}" for n in llm_loaded])

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
