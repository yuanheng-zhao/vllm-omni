# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess host for aux stages.

:class:`StageAuxProc` is the generalized counterpart to the original
``StageVAEProc``. It runs in its own process, resolves the
``(module_kind, model_arch, op)`` adapter, loads the backing
``torch.nn.Module`` via :class:`AuxModuleRunner`, and serves ZMQ RPCs.

Message protocol (msgpack over ZMQ PULL/PUSH):

    request
        ``{"type": "add_request",
           "request_id": str,
           "payload": <StageBridgePayload.to_dict()>}``
    response
        ``{"type": "result",
           "request_id": str,
           "kind": "final" | "payload",
           "tensor": torch.Tensor | None,        # for kind=="final"
           "payload": {...StageBridgePayload}|{},# for kind=="payload"
           "extras": dict}``
    request
        ``{"type": "shutdown"}``  → break loop

The wire shape is deliberately narrow: the subprocess never sees any
``custom_output`` field or stage-specific RPC name. Everything that was
model-specific (unpack / denormalize / final-vs-payload selection)
lives inside the adapter; the proc is just a transport + dispatcher.
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
from vllm_omni.stages.aux.runner import AuxModuleRunner

logger = init_logger(__name__)


class StageAuxProc:
    """Subprocess that hosts one :class:`AuxModuleRunner`.

    A single proc drives a single ``(module_kind, model_arch, op)``.
    Parallelism over independent requests is achieved via a
    :class:`ThreadPoolExecutor` so that the ZMQ asyncio loop stays
    responsive while ``adapter.forward()`` runs on the GPU; batching
    across requests is an adapter-internal concern and not yet
    implemented at this layer.
    """

    def __init__(
        self,
        *,
        module_kind: str,
        model_arch: str,
        op: str,
        model: str,
        device: str = "cuda:0",
        engine_args: dict[str, Any] | None = None,
    ) -> None:
        self._module_kind = module_kind
        self._model_arch = model_arch
        self._op = op
        self._model = model
        self._device = torch.device(device)
        self._engine_args = dict(engine_args or {})
        self._runner: AuxModuleRunner | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Import the model package so adapters register, then load."""
        # Importing the model package triggers its adapter registrations.
        # We don't know the exact import path without the arch registry,
        # so rely on the fact that vllm_omni.model_executor.models.<arch>
        # is imported as part of the orchestrator / engine startup, and
        # defer to the runner's registry lookup.
        self._runner = AuxModuleRunner(
            module_kind=self._module_kind,
            model_arch=self._model_arch,
            op=self._op,
            model=self._model,
            device=self._device,
            engine_args=self._engine_args,
        )
        logger.info(
            "[StageAuxProc] initializing %s/%s/%s (model=%s, device=%s)",
            self._module_kind,
            self._model_arch,
            self._op,
            self._model,
            self._device,
        )
        self._runner.initialize()
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("[StageAuxProc] ready")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Error shutting down aux executor: %s", e)
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Error closing aux runner: %s", e)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Route one ``add_request`` message through the runner."""
        assert self._runner is not None, "StageAuxProc not initialized"
        payload_dict = msg.get("payload")
        if not isinstance(payload_dict, dict):
            raise ValueError(f"aux request missing 'payload' dict (got {type(payload_dict).__name__})")
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            self._executor,
            self._runner.handle,
            payload_dict,
        )
        return {
            "type": "result",
            "request_id": msg.get("request_id", ""),
            **encoded,
        }

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
                logger.exception("aux request %s failed: %s", rid, e)
                await response_socket.send(encoder.encode({"type": "error", "request_id": rid, "error": str(e)}))
            finally:
                if rid:
                    tasks.pop(rid, None)

        try:
            while True:
                raw = await request_socket.recv()
                msg = decoder.decode(raw)
                mtype = msg.get("type")
                if mtype == "shutdown":
                    break
                if mtype == "abort":
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
    # Subprocess entry point
    # ------------------------------------------------------------------

    @classmethod
    def run_aux_proc(
        cls,
        *,
        module_kind: str,
        model_arch: str,
        op: str,
        model: str,
        device: str,
        engine_args: dict[str, Any],
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

        proc = cls(
            module_kind=module_kind,
            model_arch=model_arch,
            op=op,
            model=model,
            device=device,
            engine_args=engine_args,
        )
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
            logger.debug("StageAuxProc exiting.")
            raise
        except Exception:
            logger.exception("StageAuxProc encountered a fatal error.")
            raise
        finally:
            proc.close()


# ---------------------------------------------------------------------------
# Spawn / handshake helpers
# ---------------------------------------------------------------------------


def spawn_aux_proc(
    *,
    module_kind: str,
    model_arch: str,
    op: str,
    model: str,
    device: str = "cuda:0",
    engine_args: dict[str, Any] | None = None,
    handshake_address: str | None = None,
    request_address: str | None = None,
    response_address: str | None = None,
) -> tuple[BaseProcess, str, str, str]:
    """Spawn a :class:`StageAuxProc` subprocess.

    Returns ``(proc, handshake_address, request_address, response_address)``.
    """
    handshake_address = handshake_address or get_open_zmq_ipc_path()
    request_address = request_address or get_open_zmq_ipc_path()
    response_address = response_address or get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageAuxProc.run_aux_proc,
        name=f"StageAuxProc[{module_kind}/{model_arch}/{op}]",
        kwargs={
            "module_kind": module_kind,
            "model_arch": model_arch,
            "op": op,
            "model": model,
            "device": device,
            "engine_args": dict(engine_args or {}),
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageAuxProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageAuxProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_aux_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    """Wait for the aux subprocess to signal READY."""
    try:
        _perform_aux_handshake(proc, handshake_address, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_aux_handshake(
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
                    f"Timed out waiting for READY from StageAuxProc after {handshake_timeout}s (module load too slow?)."
                )
            if sock in events:
                _ident, raw = sock.recv_multipart()
                msg = msgspec.msgpack.decode(raw)
                if msg.get("status") == "READY":
                    return
                raise RuntimeError(f"Expected READY, got: {msg}")
            if proc.exitcode is not None:
                raise RuntimeError(f"StageAuxProc died during handshake (exit code {proc.exitcode})")
