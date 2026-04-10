# Ming-flash-omni Standalone TTS (Online Serving)

This directory contains online e2e examples for **Ming-flash-omni-2.0 standalone talker deployment**.

Server uses:

- `model`: `Jonathan1909/Ming-flash-omni-2.0`
- `stage config`: `vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml`

## Launch the Server

```bash
# from repo root
bash examples/online_serving/ming_flash_omni_tts/run_server.sh
```

Equivalent manual command:

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni
```

## Send TTS Request

### Python client

```bash
python examples/online_serving/ming_flash_omni_tts/speech_client.py \
    --text "我们当迎着阳光辛勤耕作，去摘取，去制作，去品尝，去馈赠。" \
    --output ming_online.wav
```

### curl

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Jonathan1909/Ming-flash-omni-2.0",
        "input": "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？",
        "response_format": "wav"
    }' --output ming_online.wav
```

## Notes

- This is the **online serving** counterpart of `examples/offline_inference/ming_flash_omni_tts/`.
- For cookbook-style fine-grained talker controls (`instruction`, `max_decode_steps`, `cfg`, `sigma`, etc.), use the offline e2e example where `additional_information` is set explicitly.
