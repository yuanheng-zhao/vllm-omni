# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrator-facing client for the dedicated VAE stage (#2089).

Mirrors the subset of :class:`StageDiffusionClient` that the Orchestrator
calls — ``stage_type`` presents as ``"diffusion"`` so the existing
routing branches in ``orchestrator.py`` handle it unchanged; the
subprocess it drives is a ``StageVAEProc`` (no denoiser, no scheduler).

It accepts either a ``remote_vae_payload`` dict as the per-request prompt
(produced by the ``latents_from_diffusion`` stage input processor) or the
legacy ``decode_qwen_image`` RPC used by the standalone demo.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from multiprocessing.process import BaseProcess
from typing import Any

import torch
import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.stage_vae_proc import (
    complete_vae_handshake,
    spawn_vae_proc,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class StageVAEClient:
    """Drives a ``StageVAEProc`` subprocess from the Orchestrator.

    Exposes the subset of :class:`StageDiffusionClient` the orchestrator
    relies on, so the VAE stage slots into the existing "diffusion" next-
    stage branch in ``_forward_to_next_stage``.
    """

    # Orchestrator dispatch key — reuse the diffusion branch rather than
    # proliferating `stage_type == "vae"` special cases through the
    # orchestrator.  The config layer still carries StageType.VAE for
    # pipeline.yaml typing; this attr is only the runtime routing hint.
    stage_type: str = "diffusion"

    def __init__(
        self,
        model: str,
        vae_subfolder: str = "vae",
        torch_dtype: str = "bfloat16",
        device: str = "cuda:0",
        stage_init_timeout: int = 600,
        metadata: Any = None,
    ) -> None:
        proc, handshake, request_addr, response_addr = spawn_vae_proc(
            model=model,
            vae_subfolder=vae_subfolder,
            torch_dtype=torch_dtype,
            device=device,
        )
        complete_vae_handshake(proc, handshake, stage_init_timeout)

        self._proc: BaseProcess = proc
        self._owns_process = True
        self._ctx = zmq.Context()
        self._req = self._ctx.socket(zmq.PUSH)
        self._req.connect(request_addr)
        self._resp = self._ctx.socket(zmq.PULL)
        self._resp.connect(response_addr)
        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()

        # Legacy demo path: fire-and-wait decode_qwen_image, served off an
        # asyncio reader loop using futures indexed by request_id.  This
        # coexists with the orchestrator queue below.
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

        # Orchestrator-facing output queue, drained via
        # get_diffusion_output_nowait().  Populated from a background
        # thread so polling is non-blocking regardless of event loop state.
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._pull_lock = threading.Lock()
        self._engine_outputs: list[OmniRequestOutput] = []
        self._shutting_down = False

        # Metadata fields the orchestrator reads off the stage client.
        if metadata is not None:
            self.stage_id = metadata.stage_id
            self.final_output = metadata.final_output
            self.final_output_type = metadata.final_output_type or "image"
            self.default_sampling_params = metadata.default_sampling_params
            self.custom_process_input_func = metadata.custom_process_input_func
            self.engine_input_source = metadata.engine_input_source
        else:
            # Standalone-demo defaults — decode_qwen_image path only.
            self.stage_id = 0
            self.final_output = True
            self.final_output_type = "image"
            self.default_sampling_params = None
            self.custom_process_input_func = None
            self.engine_input_source = []

        logger.info(
            "[StageVAEClient] Stage-%s ready (device=%s)",
            self.stage_id,
            device,
        )

    # ------------------------------------------------------------------
    # Legacy demo API (decode_qwen_image RPC)
    # ------------------------------------------------------------------
    async def _ensure_reader(self) -> None:
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        """Legacy demo reader — drives pending futures for decode_qwen_image.

        Runs the blocking ZMQ recv in the default executor so the event loop
        stays responsive.  Do NOT start this when running under the
        orchestrator: get_diffusion_output_nowait drives the same socket via
        NOBLOCK recv and concurrent recvs on one ZMQ socket are unsafe.
        """
        loop = asyncio.get_running_loop()
        while True:
            raw = await loop.run_in_executor(None, self._resp.recv)
            msg = self._decoder.decode(raw)
            rid = msg.get("request_id")
            fut = self._pending.pop(rid, None) if rid else None
            if fut is not None and not fut.done():
                if msg.get("type") == "error":
                    fut.set_exception(RuntimeError(msg.get("error", "unknown VAE error")))
                else:
                    fut.set_result(msg)

    async def decode_qwen_image(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
        request_id: str | None = None,
    ) -> torch.Tensor:
        """Legacy demo API: fire-and-wait single-request decode."""
        await self._ensure_reader()
        rid = request_id or uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        payload = {
            "type": "decode_qwen_image",
            "request_id": rid,
            "latents": latents.detach().cpu(),
            "height": int(height),
            "width": int(width),
            "vae_scale_factor": int(vae_scale_factor),
        }
        self._req.send(self._encoder.encode(payload))
        reply = await fut
        return reply["image"]

    # ------------------------------------------------------------------
    # Orchestrator-facing API (matches StageDiffusionClient subset)
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
            **payload,
        }
        self._req.send(self._encoder.encode(msg))

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        # The VAE stage has no cross-prompt batching advantage; fan out
        # as independent requests keyed by the same request_id so the
        # orchestrator receives one OmniRequestOutput per prompt. Here we
        # assume a single prompt group per request_id (the typical
        # text-to-image path emits batch_size=1 per prompt).
        for p in prompts:
            await self.add_request_async(request_id, p, sampling_params, kv_sender_info)

    @staticmethod
    def _normalize_prompt(prompt: Any) -> dict[str, Any]:
        """Extract the payload the VAE subprocess expects."""
        if isinstance(prompt, list):
            if not prompt:
                raise ValueError("Empty prompt list for VAE stage")
            prompt = prompt[0]
        if not isinstance(prompt, dict):
            raise TypeError(f"VAE stage expects a remote_vae_payload dict as prompt, got {type(prompt).__name__}")
        required = ("packed_latents", "height", "width", "vae_scale_factor")
        missing = [k for k in required if k not in prompt]
        if missing:
            raise ValueError(f"VAE prompt missing keys: {missing}")
        return {
            "latents": prompt["packed_latents"],
            "height": int(prompt["height"]),
            "width": int(prompt["width"]),
            "vae_scale_factor": int(prompt["vae_scale_factor"]),
        }

    def set_engine_outputs(self, outputs: list[OmniRequestOutput]) -> None:
        # VAE is the final stage in the #2089 demo; nothing downstream
        # consumes engine_outputs, but the orchestrator still calls this.
        self._engine_outputs = list(outputs)

    @property
    def engine_outputs(self) -> list[OmniRequestOutput]:
        return self._engine_outputs

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        """Non-blocking drain of completed VAE decodes."""
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
                    self._shutting_down = True
                    return None
                raise RuntimeError(f"StageVAEProc died unexpectedly (exit code {exitcode})")
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

                # Legacy demo path: complete the awaiting future instead
                # of posting to the orchestrator queue.
                if rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        if mtype == "error":
                            fut.set_exception(RuntimeError(msg.get("error", "VAE error")))
                        else:
                            fut.set_result(msg)
                    continue

                if mtype == "error":
                    logger.error("[StageVAEClient] stage-%s error for %s: %s", self.stage_id, rid, msg.get("error"))
                    # Publish an empty result so the orchestrator can
                    # progress the request state machine.
                    self._output_queue.put_nowait(
                        OmniRequestOutput.from_diffusion(
                            request_id=rid,
                            images=[],
                            final_output_type=self.final_output_type,
                        )
                    )
                    continue

                image_tensor = msg.get("image")
                images = self._tensor_to_pil(image_tensor) if image_tensor is not None else []
                self._output_queue.put_nowait(
                    OmniRequestOutput.from_diffusion(
                        request_id=rid,
                        images=images,
                        final_output_type=self.final_output_type,
                    )
                )

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> list:
        from PIL import Image

        # [B, C, H, W] in [-1, 1]
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
        # The VAE subprocess is stateless per-request; aborts are no-ops.
        return

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        logger.debug("[StageVAEClient] collective_rpc %s ignored (vae stage)", method)
        return None

    def get_kv_sender_info(self) -> dict[str, Any]:
        return {}

    def shutdown(self) -> None:
        self._shutting_down = True
        try:
            self._req.send(self._encoder.encode({"type": "shutdown"}), flags=zmq.NOBLOCK)
        except Exception:
            pass
        if self._reader_task is not None:
            self._reader_task.cancel()
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
            logger.warning("Error shutting down VAE proc: %s", e)

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.shutdown()
        except Exception:
            pass
