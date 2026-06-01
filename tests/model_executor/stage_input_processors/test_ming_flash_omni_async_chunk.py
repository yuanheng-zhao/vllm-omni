# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
from __future__ import annotations

import torch

from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
    thinker2talker_async_chunk,
)

# Sentences long enough (>12 / >8 tokens) to cross the min-token gate.
S1 = "The quick brown fox jumps over the lazy dog today."
S2 = "Now the second sentence here is also quite long indeed."


class _FakeTransferManager:
    """Minimal stand-in: the producer stores its per-request state here."""


class _FakeRequest:
    def __init__(self, request_id="r0", finished=False, additional_information=None):
        self.external_req_id = request_id
        self._finished = finished
        self.additional_information = additional_information or {}

    def is_finished(self):
        return self._finished


def _step(tm, req, text, is_finished=False):
    """Drive one producer step with cumulative `text`."""
    return thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"text": text},
        request=req,
        is_finished=is_finished,
    )


def _text(out):
    return out.kv_metadata["text"]


def _info(out):
    return out.kv_metadata["additional_information"]


def test_no_flush_mid_sentence():
    tm = _FakeTransferManager()
    req = _FakeRequest()
    # No terminator yet -> back-pressure (None).
    assert _step(tm, req, "The quick brown fox jumps over") is None
    assert _step(tm, req, "The quick brown fox jumps over the lazy") is None


def test_min_token_gate_holds_short_sentence():
    tm = _FakeTransferManager()
    req = _FakeRequest()
    # Terminator present but < 12 tokens -> gate holds, no flush.
    assert _step(tm, req, "Hello world. How are you") is None


def test_one_emit_per_completed_sentence():
    tm = _FakeTransferManager()
    req = _FakeRequest()

    # First completed sentence flushes; the growing tail is held back.
    out = _step(tm, req, S1 + " Now the second")
    assert out is not None
    assert _text(out) == S1
    assert _info(out)["text"] == S1
    assert bool(out.meta.finished.item()) is False

    # Same completed sentence, tail still growing -> nothing new.
    assert _step(tm, req, S1 + " " + S2[:20]) is None

    # Second sentence completes -> only the new sentence is emitted.
    out2 = _step(tm, req, S1 + " " + S2 + " Tail")
    assert out2 is not None
    assert _text(out2) == S2


def test_final_flush_of_trailing_partial_on_finish():
    tm = _FakeTransferManager()
    req = _FakeRequest()

    out = _step(tm, req, S1 + " Now the second")
    assert out is not None and _text(out) == S1

    # Finish: the held trailing fragment is flushed even without a terminator,
    # and the chunk is marked finished.
    out_final = _step(tm, req, S1 + " Now the second", is_finished=True)
    assert out_final is not None
    assert _text(out_final) == "Now the second"
    assert bool(out_final.meta.finished.item()) is True


def test_finish_with_no_output_emits_terminal_marker():
    tm = _FakeTransferManager()
    req = _FakeRequest()
    out = _step(tm, req, "", is_finished=True)
    assert out is not None
    assert out.kv_metadata is None
    assert bool(out.meta.finished.item()) is True


def test_voice_metadata_preserved_on_every_chunk():
    tm = _FakeTransferManager()
    req = _FakeRequest(
        additional_information={
            "voice_name": "Custom",
            "prompt_text": "ref",
            "max_text_length": 40,
        }
    )
    out = _step(tm, req, S1 + " more text following along here")
    assert out is not None
    info = _info(out)
    assert info["ming_task"] == "omni"
    assert info["voice_name"] == "Custom"
    assert info["prompt_text"] == "ref"
    assert info["max_text_length"] == 40


def test_spk_emb_list_is_coerced_to_tensor():
    tm = _FakeTransferManager()
    req = _FakeRequest(additional_information={"spk_emb": [0.1, 0.2, 0.3]})
    out = _step(tm, req, S1 + " more text following along here")
    assert out is not None
    spk = _info(out)["spk_emb"]
    assert isinstance(spk, torch.Tensor)
    assert spk.shape == (1, 3)


def test_multiple_completed_sentences_flush_together():
    tm = _FakeTransferManager()
    req = _FakeRequest()
    # Two sentences complete between steps -> both emitted in one chunk.
    out = _step(tm, req, S1 + " " + S2 + " Tail")
    assert out is not None
    assert _text(out) == S1 + " " + S2


def test_per_request_isolation():
    tm = _FakeTransferManager()
    a = _FakeRequest(request_id="a")
    b = _FakeRequest(request_id="b")
    # a flushes S1, holds the trailing fragment.
    assert _step(tm, a, S1 + " trailing fragment a") is not None
    # b is independent: its first completed sentence still flushes.
    out_b = _step(tm, b, S2 + " trailing fragment b")
    assert out_b is not None and _text(out_b) == S2
    # a does not re-flush S1.
    assert _step(tm, a, S1 + " trailing fragment a") is None
