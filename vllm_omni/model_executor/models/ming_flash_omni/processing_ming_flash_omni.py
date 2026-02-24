# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from Ming-flash-omni 2.0 original implementation

"""
Simplified BailingMM2 Processor for vLLM-omni.

This is a lightweight adaptation of the original Ming processing_bailingmm2.py
for use with vLLM's multimodal infrastructure. It provides the essential
processing capabilities needed for the Thinker stage.
"""

from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

# Token constants (from original Ming implementation)
DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"

DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"

# High-level placeholders in user prompts
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<IMAGE>"
PLACEHOLDER_VIDEO_TOKEN_IN_TEXT = "<VIDEO>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<AUDIO>"

# Chat template constants (from Ming)
USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
SYSTEM_PROMPT_NOTHINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
SYSTEM_PROMPT_THINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking on"


class MingFlashOmniProcessor(ProcessorMixin):
    """
    Simplified processor for Ming-flash-omni Thinker stage.

    This processor combines image, audio, and text processing for the Ming model.
    It's adapted from the original BailingMM2Processor to work with vLLM's
    multimodal infrastructure.

    Args:
        image_processor: Image processor for handling images and videos
        audio_processor: Audio processor for handling audio inputs
        tokenizer: Tokenizer for text processing
        image_token: High-level image placeholder token (default: "<IMAGE>")
        video_token: High-level video placeholder token (default: "<VIDEO>")
        audio_token: High-level audio placeholder token (default: "<AUDIO>")
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    optional_attributes = []

    # These are used by the ProcessorMixin base class
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        tokenizer=None,
        image_token: str = PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
        video_token: str = PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
        audio_token: str = PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

        # High-level placeholder tokens (used in user prompts)
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token

        super().__init__(image_processor, audio_processor, tokenizer)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: Any | None = None,
        videos: Any | None = None,
        audios: tuple[np.ndarray, int] | list[tuple[np.ndarray, int]] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process multimodal inputs for Ming-flash-omni.

        Args:
            text: Text input(s) with placeholder tokens
            images: Image input(s)
            videos: Video input(s)
            audios: Audio input(s) as (waveform, sample_rate) tuples
            **kwargs: Additional arguments passed to sub-processors

        Returns:
            BatchFeature with processed inputs
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("Text must be a string or list of strings")

        data = {}

        # Process images
        if images is not None and self.image_processor is not None:
            image_outputs = self.image_processor(
                images=images,
                videos=None,
                return_tensors="pt",
                **kwargs.get("images_kwargs", {}),
            )
            data.update(image_outputs)

            # Expand image tokens in text based on actual patch counts
            if "image_grid_thw" in image_outputs:
                text = self._expand_image_tokens(text, image_outputs["image_grid_thw"])

        # Process videos
        if videos is not None and self.image_processor is not None:
            video_outputs = self.image_processor(
                images=None,
                videos=videos,
                return_tensors="pt",
                **kwargs.get("videos_kwargs", {}),
            )
            # Rename keys for videos
            if "pixel_values" in video_outputs:
                video_outputs["pixel_values_videos"] = video_outputs.pop("pixel_values")
            if "image_grid_thw" in video_outputs:
                video_outputs["video_grid_thw"] = video_outputs.pop("image_grid_thw")
            data.update(video_outputs)

            # Expand video tokens in text
            if "video_grid_thw" in video_outputs:
                text = self._expand_video_tokens(text, video_outputs["video_grid_thw"])

        # Process audio
        if audios is not None and self.audio_processor is not None:
            audio_outputs = self.audio_processor(
                audios,
                return_tensors="pt",
                **kwargs.get("audio_kwargs", {}),
            )
            data.update(audio_outputs)

            # Expand audio tokens in text
            if "audio_feats_lengths" in audio_outputs:
                text = self._expand_audio_tokens(text, audio_outputs["audio_feats_lengths"])

        # Tokenize the (now expanded) text
        text_outputs = self.tokenizer(
            text,
            return_tensors="pt",
            **kwargs.get("text_kwargs", {}),
        )
        data.update(text_outputs)

        return BatchFeature(data=data)

    def _expand_image_tokens(
        self,
        text: list[str],
        image_grid_thw: torch.Tensor,
        special_token: str = PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    ) -> list[str]:
        """
        Expand high-level <IMAGE> tokens to actual patch tokens.

        Replaces: <IMAGE> → <image><imagePatch>*N</image>
        where N is calculated from image_grid_thw.
        """
        prompt_strings = []
        image_index = 0

        # Calculate number of patches per image
        # grid_thw format: [num_images, 3] with (t, h, w)
        # After spatial merge: patches = (t * h * w) / (merge_size^2)
        merge_size = 2  # From config, typically 2
        num_patches_per_image = torch.prod(image_grid_thw, dim=1) // (merge_size**2)

        for sample in text:
            num_images = sample.count(special_token)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    num_patches = int(num_patches_per_image[i].item())
                    # Format: <image><imagePatch>*N</image>
                    img_text = (
                        DEFAULT_IM_START_TOKEN + (DEFAULT_IMAGE_PATCH_TOKEN * num_patches) + DEFAULT_IM_END_TOKEN + "\n"
                    )
                    sample = sample.replace(special_token, img_text, 1)
            image_index += num_images
            prompt_strings.append(sample)

        return prompt_strings

    def _expand_video_tokens(
        self,
        text: list[str],
        video_grid_thw: torch.Tensor,
        special_token: str = PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
    ) -> list[str]:
        """
        Expand high-level <VIDEO> tokens to actual frame patch tokens.

        Replaces: <VIDEO> → <video><framePatch>*N</video>
        where N is calculated from video_grid_thw.
        """
        prompt_strings = []
        video_index = 0

        # Calculate number of patches per video
        merge_size = 2
        num_patches_per_video = torch.prod(video_grid_thw, dim=1) // (merge_size**2)

        for sample in text:
            num_videos = sample.count(special_token)
            if num_videos > 0:
                for i in range(video_index, num_videos + video_index):
                    num_patches = int(num_patches_per_video[i].item())
                    # Format: <video><framePatch>*N</video>
                    video_text = (
                        DEFAULT_VID_START_TOKEN
                        + (DEFAULT_FRAME_PATCH_TOKEN * num_patches)
                        + DEFAULT_VID_END_TOKEN
                        + "\n"
                    )
                    sample = sample.replace(special_token, video_text, 1)
            video_index += num_videos
            prompt_strings.append(sample)

        return prompt_strings

    def _expand_audio_tokens(
        self,
        text: list[str],
        audio_feats_lengths: torch.Tensor,
        special_token: str = PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    ) -> list[str]:
        """
        Expand high-level <AUDIO> tokens to actual audio patch tokens.

        Replaces: <AUDIO> → <audio><audioPatch>*N</audio>
        where N is the audio feature length.
        """
        prompt_strings = []

        for sample, audio_feats_length_tensor in zip(text, audio_feats_lengths):
            for audio_feats_length in audio_feats_length_tensor:
                num_patches = int(audio_feats_length.item())
                # Format: <audio><audioPatch>*N</audio>
                audio_text = DEFAULT_AU_START_TOKEN + (DEFAULT_AUDIO_PATCH_TOKEN * num_patches) + DEFAULT_AU_END_TOKEN
                if special_token in sample:
                    sample = sample.replace(special_token, audio_text, 1)
                else:
                    # If no placeholder, append audio at the end
                    sample = sample + audio_text + "\n"
            prompt_strings.append(sample)

        return prompt_strings

    def apply_system_template(
        self,
        sys_prompt_exp: str | None = None,
        use_cot_system_prompt: bool = False,
    ) -> str:
        """
        Apply Ming's system prompt template.

        Args:
            sys_prompt_exp: Optional custom system prompt to replace default
            use_cot_system_prompt: Whether to enable detailed thinking mode

        Returns:
            Formatted system prompt string
        """
        if use_cot_system_prompt:
            sys_prompt = SYSTEM_PROMPT_THINK
        else:
            sys_prompt = SYSTEM_PROMPT_NOTHINK

        if sys_prompt_exp is not None:
            # Replace the default friendly assistant message with custom prompt
            sys_prompt = sys_prompt.replace("你是一个友好的AI助手。", sys_prompt_exp)

        return sys_prompt

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        sys_prompt_exp: str | None = None,
        use_cot_system_prompt: bool = False,
        **kwargs,
    ) -> str:
        """
        Apply Ming's chat template to format multi-turn conversations.

        This formats conversations using Ming's role-based format:
        <role>SYSTEM</role>...<eos>
        <role>HUMAN</role>...<eos>
        <role>ASSISTANT</role>...<eos>
        <role>HUMAN</role>...<eos>
        <role>ASSISTANT</role>

        Args:
            conversation: List of message dicts with 'role' and 'content' keys
                Each message should have:
                - role: "HUMAN" or "ASSISTANT"
                - content: Either a string or list of content dicts
                    Content dict format: {"type": "text/image/video/audio", ...}
            sys_prompt_exp: Optional custom system prompt
            use_cot_system_prompt: Whether to enable detailed thinking mode
            **kwargs: Additional arguments (unused)

        Returns:
            Formatted conversation string ready for tokenization

        Example:
            >>> conversation = [
            ...     {
            ...         "role": "HUMAN",
            ...         "content": [
            ...             {"type": "image", "image": image_obj},
            ...             {"type": "text", "text": "What's in this image?"}
            ...         ]
            ...     },
            ...     {
            ...         "role": "ASSISTANT",
            ...         "content": [{"type": "text", "text": "I see a cat."}]
            ...     },
            ...     {
            ...         "role": "HUMAN",
            ...         "content": [{"type": "text", "text": "What color is it?"}]
            ...     }
            ... ]
            >>> text = processor.apply_chat_template(conversation)
        """
        text = ""

        # Add system prompt
        sys_prompt = self.apply_system_template(sys_prompt_exp, use_cot_system_prompt)
        if self.tokenizer is not None:
            text = sys_prompt + self.tokenizer.eos_token
        else:
            text = sys_prompt + "</s>"  # fallback

        # Process each message in the conversation
        for idx, message in enumerate(conversation):
            assert message["role"] in ["HUMAN", "ASSISTANT"], (
                f"Invalid role: {message['role']}. Must be 'HUMAN' or 'ASSISTANT'"
            )

            # Last message must be from HUMAN (expecting assistant response)
            if idx == len(conversation) - 1:
                assert message["role"] == "HUMAN", "Last message in conversation must be from HUMAN"

            # Add role prefix
            if message["role"] == "HUMAN":
                text += USER_PREFIX
            elif message["role"] == "ASSISTANT":
                text += ASSISTANT_PREFIX

            # Handle message content
            content = message["content"]

            # Support both string content and structured content list
            if isinstance(content, str):
                # Simple text-only message
                text += content
            elif isinstance(content, list):
                # Structured content with multimodal elements
                # Count existing placeholders in the content
                content_str = str(content)
                image_counts = content_str.count("<image>")
                video_counts = content_str.count("<video>")
                audio_counts = content_str.count("<audio>")

                for content_item in content:
                    content_type = content_item.get("type", "text")

                    if content_type == "image":
                        # Add image placeholder if not already in content
                        image_data = content_item.get("image")
                        if image_data is not None:
                            # Determine number of images
                            from PIL import Image as PILImage

                            num_images = 1 if isinstance(image_data, (str, PILImage.Image)) else len(image_data)

                            if image_counts < num_images:
                                # Add missing placeholders
                                image_placeholder = (PLACEHOLDER_IMAGE_TOKEN_IN_TEXT + "\n") * (
                                    num_images - image_counts
                                )
                                text += image_placeholder.rstrip("\n")
                                image_counts = num_images

                    elif content_type == "video":
                        # Video placeholder
                        assert video_counts <= 1, "Video count must be at most 1 per message!"
                        if video_counts == 0:
                            text += PLACEHOLDER_VIDEO_TOKEN_IN_TEXT
                            video_counts = 1

                    elif content_type == "audio":
                        # Add audio placeholder if not already in content
                        audio_data = content_item.get("audio")
                        if audio_data is not None:
                            num_audios = 1 if isinstance(audio_data, str) else len(audio_data)

                            if audio_counts < num_audios:
                                audio_placeholder = (PLACEHOLDER_AUDIO_TOKEN_IN_TEXT + "\n") * (
                                    num_audios - audio_counts
                                )
                                text += audio_placeholder.rstrip("\n")
                                audio_counts = num_audios

                    elif content_type == "text":
                        # Add text content
                        text += content_item.get("text", "")

            # Add EOS token after each message except the last one
            if self.tokenizer is not None:
                text += self.tokenizer.eos_token
            else:
                text += "</s>"

        # Add assistant prefix for the expected response
        text += ASSISTANT_PREFIX

        return text

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """Get model input names from all sub-processors."""
        tokenizer_input_names = self.tokenizer.model_input_names if self.tokenizer else []
        image_processor_input_names = self.image_processor.model_input_names if self.image_processor else []
        audio_processor_input_names = self.audio_processor.model_input_names if self.audio_processor else []

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_processor_input_names))
