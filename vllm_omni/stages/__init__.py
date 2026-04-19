# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-cutting stage infrastructure for vLLM-Omni.

This package hosts stage-type-agnostic primitives used by both the
orchestrator and concrete stage subprocesses:

- :mod:`vllm_omni.stages.bridge` — ``StageBridgePayload`` and the
  producer-agnostic read/write helpers that let a downstream stage
  consume a typed, schema-tagged payload regardless of whether the
  producer was an LLM, diffusion, or aux stage.
- :mod:`vllm_omni.stages.aux` — ``AuxAdapter`` base / registry and the
  generalized ``StageAuxProc`` / ``StageAuxClient`` runtime stage.
"""
