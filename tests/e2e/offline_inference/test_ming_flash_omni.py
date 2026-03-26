"""
E2E offline tests for Ming-flash-omni-2.0 model (Thinker stage).
Tests multimodal understanding with text, image, audio, video, and mixed inputs.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test

models = ["Jonathan1909/Ming-flash-omni-2.0"]

# Ming-specific chat-template
SYSTEM_PROMPT = "你是一个友好的AI助手。\n\ndetailed thinking off"
EOS_TOKEN = "<|role_end|>"
IMAGE_TOKEN = "<IMAGE>"
VIDEO_TOKEN = "<VIDEO>"
AUDIO_TOKEN = "<AUDIO>"


def build_prompt(user_text: str) -> str:
    """Build a Ming chat prompt."""
    return (
        f"<role>SYSTEM</role>{SYSTEM_PROMPT}{EOS_TOKEN}<role>HUMAN</role>{user_text}{EOS_TOKEN}<role>ASSISTANT</role>"
    )


def build_omni_input(prompt_text, multi_modal_data=None, modalities=None):
    """Build a single Omni input dict for Ming."""
    input_dict = {"prompt": prompt_text}
    if multi_modal_data:
        input_dict["multi_modal_data"] = multi_modal_data
    if modalities:
        input_dict["modalities"] = modalities
    return input_dict


def assert_text_output(outputs):
    """Assert that thinker stage produced text output."""
    for stage_output in outputs:
        if getattr(stage_output, "final_output_type", None) == "text":
            text = stage_output.request_output[0].outputs[0].text
            assert text is not None and len(text) > 0, "No text output generated"
            print(f"Generated text: {text}")
            return text
    raise AssertionError("No text stage output found")


def get_eager_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "ming_flash_omni_ci.yaml"),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
            },
        },
    )
    return path


stage_configs = [get_eager_config()]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner) -> None:
    """
    Test text-only input processing and text output generation.
    Input Modal: text
    Output Modal: text
    """
    prompt = build_prompt("请详细介绍鹦鹉的生活习性。")
    inputs = [build_omni_input(prompt, modalities=["text"])]
    outputs = omni_runner.generate(inputs)
    assert_text_output(outputs)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner) -> None:
    """
    Test image understanding with text output.
    Input Modal: image + text
    Output Modal: text
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    prompt = build_prompt(f"{IMAGE_TOKEN}Describe this image briefly.")
    inputs = [
        build_omni_input(
            prompt,
            multi_modal_data={"image": image},
            modalities=["text"],
        )
    ]
    outputs = omni_runner.generate(inputs)
    assert_text_output(outputs)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner) -> None:
    """
    Test audio understanding with text output.
    Input Modal: audio + text
    Output Modal: text
    """
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    prompt = build_prompt(f"{AUDIO_TOKEN}Please recognize the language of this speech and transcribe it. Format: oral.")
    inputs = [
        build_omni_input(
            prompt,
            multi_modal_data={"audio": (audio, 16000)},
            modalities=["text"],
        )
    ]
    outputs = omni_runner.generate(inputs)
    assert_text_output(outputs)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner) -> None:
    """
    Test video understanding with text output.
    Input Modal: video + text
    Output Modal: text
    """
    video = generate_synthetic_video(224, 224, 30)["np_array"]
    prompt = build_prompt(f"{VIDEO_TOKEN}Describe what is happening in this video.")
    inputs = [
        build_omni_input(
            prompt,
            multi_modal_data={"video": video},
            modalities=["text"],
        )
    ]
    outputs = omni_runner.generate(inputs)
    assert_text_output(outputs)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mixed_to_text(omni_runner) -> None:
    """
    Test mixed modality input (image + audio) with text output.
    Input Modal: image + audio + text
    Output Modal: text
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    prompt = build_prompt(f"{IMAGE_TOKEN}{AUDIO_TOKEN}Describe the image and transcribe the audio.")
    inputs = [
        build_omni_input(
            prompt,
            multi_modal_data={"image": image, "audio": (audio, 16000)},
            modalities=["text"],
        )
    ]
    outputs = omni_runner.generate(inputs)
    assert_text_output(outputs)
