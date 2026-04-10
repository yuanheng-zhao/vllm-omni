#!/bin/bash
# Launch vLLM-Omni server for Ming-flash-omni-2.0 standalone talker (TTS).
#
# Usage:
#   ./run_server.sh
#   MODEL=/path/to/local/model ./run_server.sh
#   PORT=8091 ./run_server.sh

set -e

MODEL="${MODEL:-Jonathan1909/Ming-flash-omni-2.0}"
PORT="${PORT:-8091}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml}"

echo "Starting Ming standalone TTS server with model: $MODEL"
echo "Stage config: $STAGE_CONFIG"

vllm serve "$MODEL" \
    --stage-configs-path "$STAGE_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
