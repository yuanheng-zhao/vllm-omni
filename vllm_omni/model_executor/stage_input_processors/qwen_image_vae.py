# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for the Qwen-Image split-VAE topology (#2089).

Reads the upstream diffusion stage's ``OmniRequestOutput.custom_output``,
where the pipeline placed a ``remote_vae_payload`` dict containing the
packed latents and geometry, and emits it verbatim as the prompt for the
VAE stage.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def latents_from_diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    src = stage_list[source_stage_id]
    if src.engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[dict[str, Any]] = []
    for out in src.engine_outputs:
        payload = (out.custom_output or {}).get("remote_vae_payload")
        if payload is None:
            raise RuntimeError(
                f"Stage-{source_stage_id} output has no 'remote_vae_payload' "
                "in custom_output — is the upstream pipeline running with "
                "remote_vae enabled?"
            )
        vae_inputs.append(payload)
    return vae_inputs
