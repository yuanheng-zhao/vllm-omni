# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess entry point for the dedicated VAE stage (#2089).

Modeled on ``stage_diffusion_proc.py`` but without a DiffusionEngine —
the VAE stage has no scheduler, no KV cache, no continuous batching.
One request in, one decoded image out.

Message protocol (msgpack over ZMQ PULL/PUSH):

    request   {"type": "decode_qwen_image",
               "request_id": str,
               "latents": torch.Tensor (packed, cpu),
               "height": int, "width": int,
               "vae_scale_factor": int}
    response  {"type": "result", "request_id": str,
               "image": torch.Tensor (cpu, [B, C, H, W])}
    request   {"type": "shutdown"}      → break loop

The scope is intentionally narrow so the demo proves the subprocess
pattern end-to-end; future methods (encode, batched decode, tiling) add
cases to ``_dispatch``.
"""

from __future__ import annotations

import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.process import BaseProcess
from typing import Any

import msgspec
import torch
import zmq
import zmq.asyncio
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.utils import shutdown

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)

logger = init_logger(__name__)


class StageVAEProc:
    """Loads a VAE module and serves encode/decode RPCs over ZMQ."""

    def __init__(
        self,
        model: str,
        vae_subfolder: str = "vae",
        torch_dtype: str = "bfloat16",
        device: str = "cuda:0",
    ) -> None:
        self._model = model
        self._vae_subfolder = vae_subfolder
        self._torch_dtype = torch_dtype
        self._device = torch.device(device)
        self._runner = None  # type: ignore[assignment]
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Load the VAE module and wrap it in ``VAEModelRunner``."""
        # Late import so the subprocess doesn't pull diffusers at module load.
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
            DistributedAutoencoderKLQwenImage,
        )
        from vllm_omni.diffusion.worker.vae_model_runner import VAEModelRunner

        dtype = getattr(torch, self._torch_dtype)
        logger.info(
            "[StageVAEProc] loading VAE from %s (subfolder=%s, dtype=%s, device=%s)",
            self._model,
            self._vae_subfolder,
            self._torch_dtype,
            self._device,
        )
        vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            self._model,
            subfolder=self._vae_subfolder,
            torch_dtype=dtype,
        )
        vae = vae.to(self._device).eval()
        self._runner = VAEModelRunner(vae, self._device)
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("[StageVAEProc] VAE loaded and ready")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Route an RPC message to the runner in the blocking thread pool."""
        method = msg["type"]
        # Orchestrator path: incoming "add_request" carries the same
        # fields as the legacy "decode_qwen_image" RPC.  Normalize.
        if method == "add_request":
            method = "decode_qwen_image"
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._runner.handle,
            method,
            msg,
        )
        return {"type": "result", "request_id": msg.get("request_id"), **result}

    # ------------------------------------------------------------------
    # ZMQ loop
    # ------------------------------------------------------------------
    async def run_loop(self, request_address: str, response_address: str) -> None:
        ctx = zmq.asyncio.Context()
        request_socket = ctx.socket(zmq.PULL)
        request_socket.bind(request_address)
        response_socket = ctx.socket(zmq.PUSH)
        response_socket.bind(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()
        tasks: dict[str, asyncio.Task] = {}

        async def _handle(msg: dict[str, Any]) -> None:
            rid = msg.get("request_id", "")
            try:
                reply = await self._dispatch(msg)
                await response_socket.send(encoder.encode(reply))
            except Exception as e:
                logger.exception("VAE request %s failed: %s", rid, e)
                await response_socket.send(encoder.encode({"type": "error", "request_id": rid, "error": str(e)}))
            finally:
                if rid:
                    tasks.pop(rid, None)

        try:
            while True:
                raw = await request_socket.recv()
                msg = decoder.decode(raw)
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "abort":
                    continue
                rid = msg.get("request_id") or f"__anon_{len(tasks)}"
                tasks[rid] = asyncio.create_task(_handle(msg))
        finally:
            for t in tasks.values():
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)
            request_socket.close()
            response_socket.close()
            ctx.term()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.warning("Error shutting down VAE executor: %s", e)

    # ------------------------------------------------------------------
    # Subprocess entry point
    # ------------------------------------------------------------------
    @classmethod
    def run_vae_proc(
        cls,
        model: str,
        vae_subfolder: str,
        torch_dtype: str,
        device: str,
        handshake_address: str,
        request_address: str,
        response_address: str,
    ) -> None:
        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, vae_subfolder, torch_dtype, device)
        try:
            proc.initialize()

            handshake_ctx = zmq.Context()
            handshake_socket = handshake_ctx.socket(zmq.DEALER)
            handshake_socket.connect(handshake_address)
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            handshake_socket.close()
            handshake_ctx.term()

            asyncio.run(proc.run_loop(request_address, response_address))
        except SystemExit:
            logger.debug("StageVAEProc exiting.")
            raise
        except Exception:
            logger.exception("StageVAEProc encountered a fatal error.")
            raise
        finally:
            proc.close()


# -- free functions mirroring stage_diffusion_proc.py -----------------------


def spawn_vae_proc(
    model: str,
    vae_subfolder: str = "vae",
    torch_dtype: str = "bfloat16",
    device: str = "cuda:0",
    handshake_address: str | None = None,
    request_address: str | None = None,
    response_address: str | None = None,
) -> tuple[BaseProcess, str, str, str]:
    """Spawn a StageVAEProc subprocess.

    Returns ``(proc, handshake_address, request_address, response_address)``.
    """
    handshake_address = handshake_address or get_open_zmq_ipc_path()
    request_address = request_address or get_open_zmq_ipc_path()
    response_address = response_address or get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageVAEProc.run_vae_proc,
        name="StageVAEProc",
        kwargs={
            "model": model,
            "vae_subfolder": vae_subfolder,
            "torch_dtype": torch_dtype,
            "device": device,
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageVAEProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageVAEProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_vae_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    """Wait for the VAE subprocess to signal READY."""
    try:
        _perform_vae_handshake(proc, handshake_address, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_vae_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as sock:
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)
        timeout_ms = handshake_timeout * 1000
        while True:
            events = dict(poller.poll(timeout=timeout_ms))
            if not events:
                raise TimeoutError(
                    f"Timed out waiting for READY from StageVAEProc after "
                    f"{handshake_timeout}s (VAE model load too slow?)."
                )
            if sock in events:
                _ident, raw = sock.recv_multipart()
                msg = msgspec.msgpack.decode(raw)
                if msg.get("status") == "READY":
                    return
                raise RuntimeError(f"Expected READY, got: {msg}")
            if proc.exitcode is not None:
                raise RuntimeError(f"StageVAEProc died during handshake (exit code {proc.exitcode})")
