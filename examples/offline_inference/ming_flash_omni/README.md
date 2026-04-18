# Ming-flash-omni 2.0

[Ming-flash-omni-2.0](https://github.com/inclusionAI/Ming) is an omni-modal model supporting text, image, video, and audio understanding, with text and speech outputs.

vLLM-Omni supports two deployment modes:

| Mode | Stage config | Output |
|------|-------------|--------|
| Thinker only (multimodal understanding) | `ming_flash_omni_thinker.yaml` (default `--omni`) | Text |
| Thinker + Talker (omni-speech) | `ming_flash_omni.yaml` | Text + Audio |

For standalone TTS (talker only), see [`examples/offline_inference/ming_flash_omni_tts/`](../ming_flash_omni_tts/).

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

The default `--omni` flag runs thinker only.  For omni-speech, pass the two-stage config explicitly:

```bash
--stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml
```

## Run examples

### Text-only
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type text
```

#### Reasoning (Thinking Mode)

Reasoning (Thinking) mode is enabled via applying "detailed thinking on" when building the system prompt template (in `apply_chat_template`).

In the end2end example, a default problem for thinking mode is provided, as referred to the example usage of Ming's cookbook;
To utilize it, you have to download the example figure from https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/figures/cases/3_0.png

```bash
python examples/offline_inference/ming_flash_omni/end2end.py -q reasoning --image-path ./3_0.png
```

### Image understanding
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_image

# With a local image
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_image --image-path /path/to/image.jpg
```

### Audio understanding
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_audio

# With a local audio file
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_audio --audio-path /path/to/audio.wav
```

### Video understanding
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_video

# With a local video and custom frame count
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_video --video-path /path/to/video.mp4 --num-frames 16
```

### Mixed modalities (image + audio)
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_mixed_modalities \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

If media file paths are not provided, the script uses built-in default assets.

### Omni-speech (thinker + talker)

To enable spoken output, use the two-stage config and request `audio` (or `text,audio`) modalities.
The thinker processes your multimodal input, generates text, then the talker synthesises the response as speech.

**Audio-only output** (speech response, no text):
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type text \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities audio \
    --output-dir output_ming_omni_speech
```

**Both text and audio output**:
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type use_audio \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities text,audio \
    --output-dir output_ming_omni_speech
```

Generated `.wav` files are saved to `--output-dir` (default `output_ming`), one per request.

The stage config allocates thinker on GPUs 0–3 and talker on GPU 3 by default. Adjust `devices` in the YAML to match your hardware.

### Modality control

| `--modalities` | Thinker output | Talker | Saved files |
|---------------|----------------|--------|-------------|
| `text` (default) | Text | Not run | `<id>.txt` |
| `audio` | Text (internal) | Runs | `<id>.wav` |
| `text,audio` | Text | Runs | `<id>.txt` + `<id>.wav` |

### Custom stage config
```bash
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_image \
    --stage-configs-path /path/to/your_config.yaml
```

## Online serving

For online serving via the OpenAI-compatible API, see [examples/online_serving/ming_flash_omni/README.md](../../online_serving/ming_flash_omni/README.md).
