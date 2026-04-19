# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for the Qwen-Image split-VAE topology.

Reads the upstream diffusion stage's bridge payloads (schema
``qwen_image.vae.latents.v1``) and forwards them as the prompt for the
downstream VAE-decode aux stage. Falls back to the legacy
``custom_output["remote_vae_payload"]`` shape during the migration
window so older diffusion-stage builds keep working with new aux
clients.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.stages.bridge import (
    StageBridgePayload,
    read_bridge_payloads_with_legacy_fallback,
)

logger = init_logger(__name__)

_VAE_BRIDGE_SCHEMA = "qwen_image.vae.latents.v1"


def latents_from_diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Materialize the VAE-stage prompts from the upstream diffusion outputs.

    The aux stage's ``StageAuxClient._normalize_prompt`` accepts either a
    :class:`StageBridgePayload` or its dict form, so we hand the dict
    form through. Each upstream output yields one prompt.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    payloads: list[StageBridgePayload] = read_bridge_payloads_with_legacy_fallback(
        stage_list,
        schema=_VAE_BRIDGE_SCHEMA,
        source_ids=list(engine_input_source),
        strict=True,
    )
    return [payload.to_dict() for payload in payloads]
