# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading utilities for diffusion models.

This module provides mutual-exclusion CPU offloading between DiT and encoders.
When enable_cpu_offload is enabled:
- Text encoders run on GPU while DiT is on CPU
- DiT runs on GPU while encoders are offloaded to CPU

This allows running large models on limited GPU memory.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class SequentialOffloader:
    """Sequential offloader: DiT and encoders take turns on GPU.

    Uses PyTorch's forward pre-hooks to automatically swap models:
    - Before encoder runs: move DiT modules to CPU, move encoder to GPU
    - Before DiT runs: move encoders to CPU, move active DiT to GPU

    This ensures only one large model group is on GPU at a time.
    """

    def __init__(
        self,
        dits: list[nn.Module],
        encoders: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
    ):
        assert all(isinstance(m, nn.Module) for m in dits), "All dits must be nn.Module"
        assert all(isinstance(m, nn.Module) for m in encoders), "All encoders must be nn.Module"
        self.dits = dits
        self.encoders = encoders
        self.device = device
        self.pin_memory = pin_memory
        self._handles: list = []

    def _to_cpu(self, module: nn.Module) -> None:
        """Move module to CPU with optional memory pinning."""
        # Skip if already on CPU
        try:
            param = next(module.parameters())
            if param.device.type == "cpu":
                return
        except StopIteration:
            return

        previous_device = param.device
        module.to("cpu", non_blocking=True)

        # Release allocator blocks when tensors leave the GPU.
        if previous_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.pin_memory:
            for p in module.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    def _to_gpu(self, module: nn.Module) -> None:
        """Move module to GPU."""
        # Skip if already on target device
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        module.to(self.device, non_blocking=True)

    def _dit_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before DiT forward: offload encoders, load DiT."""
        for enc in self.encoders:
            self._to_cpu(enc)
        self._to_gpu(module)
        torch.cuda.synchronize()
        logger.debug("Swapped: encoders -> CPU, DiT -> GPU")

    def _encoder_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before encoder forward: offload DiT, load encoder."""
        for dit_mod in self.dits:
            self._to_cpu(dit_mod)
        self._to_gpu(module)
        torch.cuda.synchronize()
        logger.debug("Swapped: DiT -> CPU, encoder -> GPU")

    def register(self) -> None:
        """Register forward pre-hooks on DiT and encoders."""
        # Hook on each DiT-like module
        for dit_mod in self.dits:
            h = dit_mod.register_forward_pre_hook(self._dit_pre_hook)
            self._handles.append(h)
            logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

        # Hook on each encoder
        for enc in self.encoders:
            h = enc.register_forward_pre_hook(self._encoder_pre_hook)
            self._handles.append(h)
            logger.debug("Registered offload hook for %s", enc.__class__.__name__)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []


class LayerwiseOffloader:
    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        num_gpu_layers: int = 1,
    ):
        assert all(isinstance(m, nn.Module) for m in blocks), "All transformer blocks must be nn.Module"

        self.blocks = blocks
        self.device = device
        self.pin_memory = pin_memory
        self.num_gpu_layers = num_gpu_layers
        self._pre_hook_handles: list = []
        self._post_hook_handles: list = []
        self.n = len(self.blocks)

        self.copy_stream = torch.cuda.Stream()

        # Per-layer synchronization primitive: set after H2D copy completes.
        self._prefetch_done: list[torch.cuda.Event | None] = [None] * self.n

        # Simple state to avoid redundant work.
        self._resident: list[bool] = [False] * self.n
        self._inflight: list[bool] = [False] * self.n

        # layer-id -> {name -> weight}
        self.layer_weights: dict[int, dict[str, torch.Tensor]] = {}
        for idx, block in enumerate(self.blocks):
            self.layer_weights[idx] = self._to_cpu(block)

        self.register_pre_block_hook()
        self.register_post_block_hook()

        self.prefetch_layer(0, non_blocking=False)

    def _to_cpu(self, block: nn.Module) -> dict[str, torch.Tensor]:
        cpu_tensors = {}

        for name, param in block.named_parameters():
            cpu_param = param.detach().to("cpu")
            if self.pin_memory:
                cpu_param = cpu_param.pin_memory()
            cpu_tensors[name] = cpu_param
            param.data = torch.empty((), device=self.device, dtype=param.dtype)
            # param.data = torch.empty_like(param.data, device="meta")

        for name, buf in block.named_buffers():
            cpu_buf = buf.detach().to("cpu")
            if self.pin_memory:
                cpu_buf = cpu_buf.pin_memory()
            cpu_tensors[name] = cpu_buf
            buf.data = torch.empty((), device=self.device, dtype=buf.dtype)
            # buf.data = torch.empty_like(buf.data, device="meta")

        return cpu_tensors

    def register_pre_block_hook(self) -> None:
        @torch.compiler.disable
        def _pre_hook(layer_id: int, module: nn.Module, args: tuple) -> None:
            self._ensure_layer_ready(layer_id)

            # For the last block / layer, prefetch layer 0 (the first layer)
            next_id = (layer_id + 1) % self.n
            self.prefetch_layer(next_id, non_blocking=True)

        for i, layer in enumerate(self.blocks):
            handle = layer.register_forward_pre_hook(partial(_pre_hook, i))
            self._pre_hook_handles.append(handle)

    def register_post_block_hook(self) -> None:
        @torch.compiler.disable
        def _post_hook(layer_id: int, module: nn.Module, args: tuple, output: tuple) -> None:
            # torch.cuda.current_stream().wait_stream(self.copy_stream)
            self.offload_layer(layer_id)
            self._resident[layer_id] = False
            self._prefetch_done[layer_id] = None

        for i, layer in enumerate(self.blocks):
            handle = layer.register_forward_hook(partial(_post_hook, i))
            self._post_hook_handles.append(handle)

    @torch.compiler.disable
    def _ensure_layer_ready(self, layer_id: int) -> None:
        """Called on compute stream right before layer executes."""
        if self._resident[layer_id]:
            evt = self._prefetch_done[layer_id]
            if evt is not None:
                torch.cuda.current_stream(device=self.device).wait_event(evt)
            return

        # If not resident, schedule immediate prefetch and wait for it (JIT fix)
        self.prefetch_layer(layer_id)
        evt = self._prefetch_done[layer_id]
        if evt is not None:
            torch.cuda.current_stream(device=self.device).wait_event(evt)
        self._resident[layer_id] = True

    @torch.compiler.disable
    def prefetch_layer(self, layer_id: int, non_blocking: bool = True) -> None:
        # to gpu
        if layer_id >= len(self.blocks) or layer_id < 0:
            logger.warning(f"Invalid layer id specified: {layer_id}")
            return
        if len(self.blocks) != len(self.layer_weights):
            logger.error("Inconsistent block layers happened")
            return

        self.copy_stream.wait_stream(torch.cuda.current_stream())

        end_layer_id = min(layer_id + self.num_gpu_layers, len(self.blocks))

        with torch.cuda.stream(self.copy_stream):
            for idx in range(layer_id, end_layer_id):
                # logger.info(f" >>> Prefetching layer {idx}")
                if self._resident[idx] or self._inflight[idx]:
                    continue

                self._inflight[idx] = True

                block = self.blocks[idx]
                block_named_parameters: dict[str, nn.Parameter] = dict(block.named_parameters())
                block_named_buffers: dict[str, torch.Tensor] = dict(block.named_buffers())

                for name, cpu_t in self.layer_weights[idx].items():
                    # logger.info(f" >>> Re-materializing {name} on device {self.device}")
                    param_or_buf = (
                        block_named_parameters[name] if name in block_named_parameters else block_named_buffers[name]
                    )

                    gpu_t = torch.empty_like(cpu_t, device=self.device)
                    gpu_t.copy_(cpu_t, non_blocking=non_blocking)
                    # with torch.cuda.stream(self.copy_stream):
                    # self.blocks[idx].to(self.device, non_blocking=non_blocking)
                    # gpu_weight.copy_(cpu_t, non_blocking=non_blocking)

                    param_or_buf.data = gpu_t

                evt = torch.cuda.Event()
                evt.record(self.copy_stream)
                self._prefetch_done[idx] = evt

                self._inflight[idx] = False
                self._resident[idx] = True

    @torch.compiler.disable
    def offload_layer(self, layer_id: int) -> None:
        # to cpu
        if layer_id >= len(self.blocks) or layer_id < 0:
            logger.warning(f"Invalid layer id specified: {layer_id}")
            return

        # logger.info(f" >>> Offloading layer {layer_id}")

        block = self.blocks[layer_id]
        # free GPU residency
        for _, param in block.named_parameters():
            param.data = torch.empty((), device=self.device, dtype=param.dtype)
            # param.data = torch.empty_like(param.data, device="meta")
        for _, buf in block.named_buffers():
            buf.data = torch.empty((), device=self.device, dtype=buf.dtype)
            # buf.data = torch.empty_like(buf.data, device="meta")

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._pre_hook_handles:
            h.remove()
        for h in self._post_hook_handles:
            h.remove()
        self._pre_hook_handles = []
        self._post_hook_handles = []


def apply_offload_hooks(
    model: nn.Module,
    od_config: OmniDiffusionConfig,
    *,
    device: torch.device | None = None,
) -> None:
    """Apply mutual-exclusion offload hooks based on config.

    When enable_cpu_offload is enabled, DiT and encoders swap GPU access:
    - Encoders (text_encoder, text_encoder_2, text_encoder_3, image_encoder)
      run on GPU while DiT is on CPU
    - DiT runs on GPU while encoders are on CPU

    Args:
        model: Diffusion pipeline model
        od_config: OmniDiffusionConfig with offload settings
    """
    enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
    layerwise_offload_dit = getattr(od_config, "layerwise_offload_dit", False)

    if not enable_cpu_offload and not layerwise_offload_dit:
        return

    # Find DiT/transformer modules
    dit_modules: list[nn.Module] = []
    dit_names: list[str] = []
    candidate_attrs = ["transformer", "transformer_2", "dit"]
    for attr in candidate_attrs:
        if not hasattr(model, attr):
            continue
        module_obj = getattr(model, attr)
        if module_obj is None:
            continue

        assert isinstance(module_obj, nn.Module), f"Expected {attr} to be nn.Module, got {type(module_obj)!r}"

        if module_obj in dit_modules:
            continue

        dit_modules.append(module_obj)
        dit_names.append(attr)

    if not dit_modules:
        logger.warning("enable_cpu_offload enabled but no transformer/dit/unet found")
        return

    if device is None:
        try:
            device = next(dit_modules[0].parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin = getattr(od_config, "pin_cpu_memory", True)

    if enable_cpu_offload:
        # Collect all encoders
        encoders: list[nn.Module] = []
        encoder_names: list[str] = []
        for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                encoders.append(getattr(model, attr))
                encoder_names.append(attr)

        if not encoders:
            logger.warning("enable_cpu_offload enabled but no encoders found")
            return

        # Initial state: keep DiT modules on CPU (encoders typically run first)
        for dit_mod in dit_modules:
            dit_mod.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pin and torch.cuda.is_available():
            for dit_mod in dit_modules:
                for p in dit_mod.parameters():
                    if p.data.device.type == "cpu" and not p.data.is_pinned():
                        p.data = p.data.pin_memory()

        # Register sequential offload hooks
        SequentialOffloader(dit_modules, encoders, device, pin).register()
        logger.info(
            "CPU offload enabled: %s <-> %s (mutual exclusion)",
            ", ".join(dit_names),
            ", ".join(encoder_names),
        )

    elif layerwise_offload_dit:
        blocks = []
        for dit_module in dit_modules:
            # HACK: hardcoded for testing
            _blocks = getattr(dit_module, "transformer_blocks", None)
            blocks.extend(list(_blocks))

        LayerwiseOffloader(blocks, device, pin)
        logger.info("Layerwise offload enabled")
