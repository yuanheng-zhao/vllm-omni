# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrator-facing client for auxiliary-module stages.

``StageAuxClient`` is the generalized counterpart to the split-VAE demo's
``StageVAEClient``. It owns the ZMQ transport to a :class:`StageAuxProc`
subprocess, translates orchestrator-level ``add_request`` calls into
bridge-payload RPCs, and surfaces results back to the orchestrator as
:class:`OmniRequestOutput` objects.

The important difference from ``StageVAEClient`` is that **nothing here
is model-specific**: prompt normalization is purely "shape the
:class:`StageBridgePayload` the next stage consumes", result
post-processing is schema-dispatched off the adapter's declared
``schema_out`` / ``final_output_type``, and the runtime routing hint
:attr:`stage_type` is the first-class value ``"aux"`` rather than a
masquerade as ``"diffusion"``.

Note on orchestrator routing: until the orchestrator gains explicit
``stage_type == "aux"`` branches (tracked separately), the VAE
back-compat shim (:class:`vllm_omni.diffusion.stage_vae_client.StageVAEClient`)
keeps the ``"diffusion"`` dispatch key and wraps this class.
"""

from __future__ import annotations

import asyncio
import threading
from multiprocessing.process import BaseProcess
from typing import Any

import torch
import zmq
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.stages.aux.proc import (
    complete_aux_handshake,
    spawn_aux_proc,
)
from vllm_omni.stages.bridge import (
    StageBridgePayload,
    build_custom_output_with_payload,
)

logger = init_logger(__name__)


class StageAuxClient:
    """Drive a :class:`StageAuxProc` subprocess from the Orchestrator.

    Exposes the subset of :class:`StageDiffusionClient` that the
    orchestrator polls:

    - :attr:`stage_type` — routing hint.
    - :meth:`add_request_async` — push one request to the subprocess.
    - :meth:`get_diffusion_output_nowait` — drain one completed result.
    - :meth:`set_engine_outputs` / :attr:`engine_outputs` — expose a
      recent-results list for downstream bridge-payload readers.
    - :meth:`shutdown` — teardown the subprocess + sockets.

    The aux client is stateless per-request: it forwards a
    :class:`StageBridgePayload` to the proc, waits for a single response,
    and either packs that response into an :class:`OmniRequestOutput`
    (when the adapter declared a ``final_output_type``) or re-publishes
    it as a bridge payload for a downstream stage.
    """

    # First-class routing key. The orchestrator treats ``"aux"`` the
    # same as ``"diffusion"`` for non-LLM scheduling; only the branch
    # gates differ.
    stage_type: str = "aux"

    def __init__(
        self,
        *,
        module_kind: str,
        model_arch: str,
        op: str,
        model: str,
        device: str = "cuda:0",
        engine_args: dict[str, Any] | None = None,
        stage_init_timeout: int = 600,
        metadata: Any = None,
    ) -> None:
        self._module_kind = module_kind
        self._model_arch = model_arch
        self._op = op
        self._device = device

        proc, handshake, request_addr, response_addr = spawn_aux_proc(
            module_kind=module_kind,
            model_arch=model_arch,
            op=op,
            model=model,
            device=device,
            engine_args=engine_args,
        )
        complete_aux_handshake(proc, handshake, stage_init_timeout)

        self._proc: BaseProcess = proc
        self._owns_process = True
        self._ctx = zmq.Context()
        self._req = self._ctx.socket(zmq.PUSH)
        self._req.connect(request_addr)
        self._resp = self._ctx.socket(zmq.PULL)
        self._resp.connect(response_addr)
        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()

        # Orchestrator-facing output queue, drained via
        # get_diffusion_output_nowait(). Populated from the same thread
        # that polls (no async reader task — unlike the legacy VAE
        # client's demo path, the aux client is orchestrator-only).
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._pull_lock = threading.Lock()
        self._engine_outputs: list[OmniRequestOutput] = []
        self._shutting_down = False

        # Metadata the orchestrator reads off the stage client. Engine
        # startup passes the stage_metadata record here; standalone use
        # (tests) can pass None for sensible defaults.
        if metadata is not None:
            self.stage_id = metadata.stage_id
            self.final_output = metadata.final_output
            self.final_output_type = metadata.final_output_type or "image"
            self.default_sampling_params = metadata.default_sampling_params
            self.custom_process_input_func = metadata.custom_process_input_func
            self.engine_input_source = metadata.engine_input_source
        else:
            self.stage_id = 0
            self.final_output = True
            self.final_output_type = "image"
            self.default_sampling_params = None
            self.custom_process_input_func = None
            self.engine_input_source = []

        logger.info(
            "[StageAuxClient] Stage-%s ready (%s/%s/%s, device=%s)",
            self.stage_id,
            module_kind,
            model_arch,
            op,
            device,
        )

    # ------------------------------------------------------------------
    # Orchestrator-facing API
    # ------------------------------------------------------------------

    async def add_request_async(
        self,
        request_id: str,
        prompt: Any,
        sampling_params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        payload = self._normalize_prompt(prompt)
        msg = {
            "type": "add_request",
            "request_id": request_id,
            "payload": payload.to_dict(),
        }
        self._req.send(self._encoder.encode(msg))

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        # Aux stages don't exploit cross-prompt batching at this layer;
        # fan out as independent requests keyed by the same request_id.
        for p in prompts:
            await self.add_request_async(request_id, p, sampling_params, kv_sender_info)

    # ------------------------------------------------------------------
    # Prompt normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_prompt(prompt: Any) -> StageBridgePayload:
        """Coerce the orchestrator's prompt into a :class:`StageBridgePayload`.

        The custom_process_input_func for an aux stage is expected to
        produce one of:

        - a :class:`StageBridgePayload` directly
        - the dict form (``StageBridgePayload.to_dict()``)
        - a list wrapping one of the above

        Anything else is a programming error in the caller's pipeline
        wiring and raises early with a descriptive message.
        """
        if isinstance(prompt, list):
            if not prompt:
                raise ValueError("Empty prompt list for aux stage")
            prompt = prompt[0]
        if isinstance(prompt, StageBridgePayload):
            return prompt
        if isinstance(prompt, dict):
            if "schema" in prompt:
                return StageBridgePayload.from_dict(prompt)
            raise ValueError(
                "Aux stage prompt dict missing 'schema' — "
                "produce it via StageBridgePayload.to_dict() in the "
                "stage_input_processor."
            )
        raise TypeError(
            f"Aux stage expects a StageBridgePayload (or its dict form) as prompt, got {type(prompt).__name__}"
        )

    # ------------------------------------------------------------------
    # Output pipeline (orchestrator polls this)
    # ------------------------------------------------------------------

    def set_engine_outputs(self, outputs: list[OmniRequestOutput]) -> None:
        """Store recent outputs so a downstream stage's reader can see them."""
        self._engine_outputs = list(outputs)

    @property
    def engine_outputs(self) -> list[OmniRequestOutput]:
        return self._engine_outputs

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        """Non-blocking drain. Method name preserved for orch compatibility."""
        self._drain_responses()
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            if not self._shutting_down and self._owns_process and self._proc is not None and not self._proc.is_alive():
                exitcode = self._proc.exitcode
                self._drain_responses()
                try:
                    return self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                if exitcode is not None and exitcode > 128:
                    # Exit by signal (SIGTERM/SIGINT) — graceful shutdown.
                    self._shutting_down = True
                    return None
                raise RuntimeError(f"StageAuxProc died unexpectedly (exit code {exitcode})")
            return None

    def _drain_responses(self) -> None:
        with self._pull_lock:
            while True:
                try:
                    raw = self._resp.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                msg = self._decoder.decode(raw)
                mtype = msg.get("type")
                rid = msg.get("request_id", "")

                if mtype == "error":
                    logger.error(
                        "[StageAuxClient] stage-%s error for %s: %s",
                        self.stage_id,
                        rid,
                        msg.get("error"),
                    )
                    # Post an empty result so the orchestrator's state
                    # machine can still advance the failing request.
                    self._output_queue.put_nowait(
                        OmniRequestOutput.from_diffusion(
                            request_id=rid,
                            images=[],
                            final_output_type=self.final_output_type,
                        )
                    )
                    continue

                self._publish_result(rid, msg)

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _publish_result(self, request_id: str, msg: dict[str, Any]) -> None:
        """Convert a proc reply into an :class:`OmniRequestOutput`."""
        kind = msg.get("kind")
        extras = msg.get("extras") or {}
        if kind == "final":
            output = self._pack_final(request_id, msg, extras)
        elif kind == "payload":
            output = self._pack_payload(request_id, msg, extras)
        else:
            logger.error(
                "[StageAuxClient] unexpected result kind %r for %s",
                kind,
                request_id,
            )
            output = OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                final_output_type=self.final_output_type,
            )
        self._output_queue.put_nowait(output)

    def _pack_final(
        self,
        request_id: str,
        msg: dict[str, Any],
        extras: dict[str, Any],
    ) -> OmniRequestOutput:
        tensor: torch.Tensor | None = msg.get("tensor")
        final_type = self.final_output_type or "image"
        if final_type == "image" and tensor is not None:
            images = self._tensor_to_pil(tensor)
            return OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=images,
                final_output_type=final_type,
                custom_output=dict(extras) if extras else None,
            )
        # Non-image finals: surface the raw tensor via custom_output so
        # downstream glue (audio encoding, video packing, ...) can pick
        # it up without the aux client having to grow codec knowledge.
        custom: dict[str, Any] = {"final_tensor": tensor}
        if extras:
            custom.update(extras)
        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            final_output_type=final_type,
            custom_output=custom,
        )

    def _pack_payload(
        self,
        request_id: str,
        msg: dict[str, Any],
        extras: dict[str, Any],
    ) -> OmniRequestOutput:
        payload_dict = msg.get("payload") or {}
        payload = StageBridgePayload.from_dict(payload_dict)
        custom = build_custom_output_with_payload(payload)
        if extras:
            custom.update(extras)
        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            final_output_type=self.final_output_type or "latents",
            custom_output=custom,
        )

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> list:
        """Convert ``[B, C, H, W]`` in ``[-1, 1]`` to PIL images.

        Kept identical to the VAE client's legacy helper so both paths
        produce byte-identical outputs. Adapters that want to hand back
        already-PIL images should instead put them in ``extras`` and
        return a final tensor of shape ``(0,)``.
        """
        from PIL import Image

        image = image.detach().float().clamp(-1, 1)
        image = ((image + 1.0) / 2.0 * 255).round().to(torch.uint8)
        pil_images = []
        for i in range(image.shape[0]):
            arr = image[i].permute(1, 2, 0).cpu().numpy()
            pil_images.append(Image.fromarray(arr))
        return pil_images

    # ------------------------------------------------------------------
    # Orchestrator lifecycle stubs
    # ------------------------------------------------------------------

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # Aux subprocess is stateless per-request; abort is a no-op.
        return

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        logger.debug("[StageAuxClient] collective_rpc %s ignored (aux stage)", method)
        return None

    def get_kv_sender_info(self) -> dict[str, Any]:
        return {}

    def shutdown(self) -> None:
        self._shutting_down = True
        try:
            self._req.send(self._encoder.encode({"type": "shutdown"}), flags=zmq.NOBLOCK)
        except Exception:
            pass
        try:
            self._req.close()
            self._resp.close()
            self._ctx.term()
        except Exception:
            pass
        try:
            if self._proc is not None:
                self._proc.join(timeout=5)
                if self._proc.is_alive():
                    self._proc.terminate()
        except Exception as e:
            logger.warning("Error shutting down aux proc: %s", e)

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.shutdown()
        except Exception:
            pass
