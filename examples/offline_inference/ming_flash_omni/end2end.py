# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference example for Ming-flash-omni 2.0 (Thinker stage).
"""

import os
from typing import NamedTuple

import librosa
import numpy as np
from PIL import Image
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# ── Ming chat-template constants ──────────────────────────────────────────────
SYSTEM_PROMPT_NOTHINK = "你是一个友好的AI助手。\n\ndetailed thinking off"
SYSTEM_PROMPT_THINK = "你是一个友好的AI助手。\n\ndetailed thinking on"
EOS_TOKEN = "<|role_end|>"

IMAGE_TOKEN = "<IMAGE>"
VIDEO_TOKEN = "<VIDEO>"
AUDIO_TOKEN = "<AUDIO>"


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def build_prompt(user_text: str, think: bool = False) -> str:
    """Build a Ming chat prompt with role tags."""
    system = SYSTEM_PROMPT_THINK if think else SYSTEM_PROMPT_NOTHINK
    return f"<role>SYSTEM</role>{system}{EOS_TOKEN}<role>HUMAN</role>{user_text}{EOS_TOKEN}<role>ASSISTANT</role>"


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        question = "请详细介绍鹦鹉的生活习性。"
    return QueryResult(
        inputs={"prompt": build_prompt(question)},
        limit_mm_per_prompt={},
    )


def get_image_query(
    question: str = None,
    image_path: str | None = None,
) -> QueryResult:
    if question is None:
        question = "Describe this image in detail."
    prompt = build_prompt(f"{IMAGE_TOKEN}{question}")

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data},
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(
    question: str = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "Please recognize the language of this speech and transcribe it. Format: oral."
    prompt = build_prompt(f"{AUDIO_TOKEN}{question}")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_video_query(
    question: str = None,
    video_path: str | None = None,
    num_frames: int = 16,
) -> QueryResult:
    if question is None:
        question = "Describe what is happening in this video."
    prompt = build_prompt(f"{VIDEO_TOKEN}{question}")

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"video": video_frames},
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_mixed_modalities_query(
    image_path: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """Mixed image + audio understanding."""
    question = "Describe the image and transcribe the audio."
    prompt = build_prompt(f"{IMAGE_TOKEN}{AUDIO_TOKEN}{question}")

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        sig, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (sig.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data, "audio": audio_data},
        },
        limit_mm_per_prompt={"image": 1, "audio": 1},
    )


def get_reasoning_query(
    question: str = None,
    image_path: str | None = None,
) -> QueryResult:
    if question is None:
        question = "What is the sum of all prime numbers less than 50?"
    # Enable thinking mode for step-by-step reasoning
    prompt = build_prompt(question, think=True)

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
        prompt = build_prompt(f"{IMAGE_TOKEN}{question}", think=True)
        return QueryResult(
            inputs={
                "prompt": prompt,
                "multi_modal_data": {"image": image_data},
            },
            limit_mm_per_prompt={"image": 1},
        )

    return QueryResult(
        inputs={"prompt": prompt},
        limit_mm_per_prompt={},
    )


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_mixed_modalities": get_mixed_modalities_query,
    "reasoning": get_reasoning_query,
}


def main(args):
    model_name = "Jonathan1909/Ming-flash-omni-2.0"

    query_func = query_map[args.query_type]
    if args.query_type == "use_image":
        query_result = query_func(image_path=args.image_path)
    elif args.query_type == "use_audio":
        query_result = query_func(audio_path=args.audio_path, sampling_rate=args.sampling_rate)
    elif args.query_type == "use_video":
        query_result = query_func(video_path=args.video_path, num_frames=args.num_frames)
    elif args.query_type == "use_mixed_modalities":
        query_result = query_func(
            image_path=args.image_path,
            audio_path=args.audio_path,
            sampling_rate=args.sampling_rate,
        )
    elif args.query_type == "reasoning":
        query_result = query_func(image_path=args.image_path)
    else:
        query_result = query_func()

    # Initialize Omni (with thinker-only stage config)
    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    # Thinker sampling params
    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
        seed=SEED,
        detokenize=True,
    )

    prompts = [query_result.inputs for _ in range(args.num_prompts)]

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    print(f"Query type: {args.query_type}")
    print(f"Number of prompts: {len(prompts)}")
    print("-" * 60)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni.generate(prompts, [sampling_params]):
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                print(f"\n[Request {request_id}]")
                print(f"Prompt: {output.prompt!r}")
                print(f"Output: {text_output}")

                # Save to file
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"Prompt:\n{output.prompt}\n\n")
                    f.write(f"Output:\n{text_output.strip()}\n")
                print(f"Saved to {out_txt}")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(description="Ming-flash-omni 2.0 offline inference example")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs YAML file.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from video.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities (comma-separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_ming",
        help="Output directory for results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
