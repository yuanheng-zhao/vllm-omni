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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class OffloadStrategy(Enum):
    """Offload strategy types."""

    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYER_WISE = "layer_wise"  # Block-level offloading with sliding window


@dataclass
class OffloadConfig:
    """Validated offload configuration."""

    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    layerwise_num_gpu_layers: int = 1

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> OffloadConfig:
        """Extract and validate offload settings from OmniDiffusionConfig.

        Enforces mutual exclusion between model-level and layer-wise offloading.
        Layer-wise takes priority if both are enabled.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)
        layerwise_num_gpu_layers = getattr(od_config, "layerwise_num_gpu_layers", 1)

        # Determine strategy (mutual exclusion, layer-wise takes priority)
        if enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
            if enable_cpu_offload:
                logger.info(
                    "Both model-level and layer-wise offloading enabled. "
                    "Layer-wise takes priority, disabling model-level offloading."
                )
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            layerwise_num_gpu_layers=layerwise_num_gpu_layers,
        )


@dataclass
class PipelineModules:
    """Discovered pipeline modules for offloading."""

    dits: list[nn.Module]
    dit_names: list[str]
    encoders: list[nn.Module]
    encoder_names: list[str]
    vae: nn.Module | None = None


class ModuleDiscovery:
    """Discovers pipeline components for offloading."""

    DIT_ATTRS = ["transformer", "transformer_2", "dit"]
    ENCODER_ATTRS = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    VAE_ATTRS = ["vae"]

    @staticmethod
    def discover(pipeline: nn.Module) -> PipelineModules:
        """Discover DiT, encoder, and VAE modules from pipeline.

        Args:
            pipeline: Diffusion pipeline model

        Returns:
            PipelineModules with lists of discovered modules and names
        """
        # Find DiT/transformer modules
        dit_modules: list[nn.Module] = []
        dit_names: list[str] = []
        for attr in ModuleDiscovery.DIT_ATTRS:
            if not hasattr(pipeline, attr):
                continue
            module_obj = getattr(pipeline, attr)
            if module_obj is None:
                continue

            if not isinstance(module_obj, nn.Module):
                logger.warning(f"Expected {attr} to be nn.Module, got {type(module_obj)!r}")
                continue

            # Avoid duplicates
            if module_obj in dit_modules:
                continue

            dit_modules.append(module_obj)
            dit_names.append(attr)

        # Collect all encoders
        encoders: list[nn.Module] = []
        encoder_names: list[str] = []
        for attr in ModuleDiscovery.ENCODER_ATTRS:
            if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
                encoders.append(getattr(pipeline, attr))
                encoder_names.append(attr)

        # Collect VAE
        vae = None
        for attr in ModuleDiscovery.VAE_ATTRS:
            module = getattr(pipeline, attr, None)
            if module is not None:
                vae = module
                break

        return PipelineModules(
            dits=dit_modules,
            dit_names=dit_names,
            encoders=encoders,
            encoder_names=encoder_names,
            vae=vae,
        )


class OffloadBackend(ABC):
    """Base class for CPU offload backends.

    Follows the same pattern as CacheBackend for consistency across
    optimization features in vLLM-Omni.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        """Initialize backend with configuration and target device.

        Args:
            config: OffloadConfig with strategy settings
            device: Target GPU device for online modules
        """
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError("Subclasses must implement enable()")

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError("Subclasses must implement disable()")

    def is_enabled(self) -> bool:
        """Check if offloading is currently active."""
        return self.enabled


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
        if previous_device.type != "cpu":
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

        current_omni_platform.synchronize()

        logger.debug("Swapped: encoders -> CPU, DiT -> GPU")

    def _encoder_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before encoder forward: offload DiT, load encoder."""
        for dit_mod in self.dits:
            self._to_cpu(dit_mod)
        self._to_gpu(module)

        current_omni_platform.synchronize()

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
    """Layer-wise CPU offloading for transformer blocks.

    Keeps only a sliding window of layers (blocks), by default a single layer, on GPU,
    prefetching the next block while the current block computes to approach compute - memcpy overlap.
    Unused blocks are freed on GPU.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        num_gpu_layers: int = 1,
    ):
        assert all(isinstance(m, nn.Module) for m in blocks), "All transformer blocks must be torch.nn.Module"
        assert current_omni_platform.is_cuda(), "Layerwise offloading is only supported on cuda devices for now"

        self.blocks = blocks
        self.device = device
        self.pin_memory = pin_memory
        self.num_gpu_layers = num_gpu_layers
        self.num_blocks = len(self.blocks)
        if self.num_blocks == 0:
            raise ValueError("LayerwiseOffloader requires at least one block, but found 0.")
        if not (1 <= self.num_gpu_layers <= self.num_blocks):
            raise ValueError(f"Invalid num_gpu_layers {self.num_gpu_layers} with {self.num_blocks} blocks")

        self._pre_hook_handles: list = []
        self._post_hook_handles: list = []

        self._copy_stream = torch.cuda.Stream()

        # Per-layer synchronization primitive: set after H2D copy completes.
        self._prefetch_done: list[torch.cuda.Event | None] = [None] * self.num_blocks

        # Simple state to avoid redundant work.
        self._resident: list[bool] = [False] * self.num_blocks

        # Pre-allocate gpu tensors
        # layer-id -> {dtype -> flattened aggregated cpu tensor}
        self.layer_cpu_weights: list[dict[torch.dtype, torch.Tensor]] = []
        self.layer_metadata: list[dict[torch.dtype, list[dict[str, Any]]]] = []

        self.block_parameters: dict[int, dict[str, nn.Parameter]] = {}
        self.block_buffers: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, block in enumerate(self.blocks):
            self.block_parameters[layer_idx] = dict(block.named_parameters())
            self.block_buffers[layer_idx] = dict(block.named_buffers())

            dtype_cpu_flattened_weights, dtype_metadata = self._to_cpu(
                self.block_parameters[layer_idx], self.block_buffers[layer_idx]
            )
            self.layer_cpu_weights.append(dtype_cpu_flattened_weights)
            self.layer_metadata.append(dtype_metadata)

        if self.num_blocks != len(self.layer_cpu_weights):
            logger.error(
                f"Inconsistent block layers happened: # of blocks: {self.num_blocks}; "
                f"# of layer cpu weights: {len(self.layer_cpu_weights)}"
            )

        # Register pre and post forward hooks on each of the blocks
        self.register_block_hooks()

        # Pre-fetch the first layer
        # For subsequent requests, the first layer/block will be pre-fetched
        # during the last layer compute of the previous request.
        self.prefetch_layer(0, non_blocking=False)

    def _to_cpu(
        self, params: dict[str, nn.Parameter], bufs: dict[str, torch.Tensor]
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            total_numel = sum(t.numel() for _, t in name2weights.items())
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

            current_offset = 0
            for name, param_or_buf in name2weights.items():
                numel = param_or_buf.numel()
                cpu_tensor[current_offset : current_offset + numel].copy_(param_or_buf.flatten())
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": param_or_buf.shape,
                    }
                )

                param_or_buf.data = torch.empty((), device=self.device, dtype=dtype)
                current_offset += numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        return dtype_cpu_flattened_weights, dtype_metadata

    def register_block_hooks(self) -> None:
        """Register forward hooks on blocks for prefetching and offloading."""

        def _pre_hook(module: nn.Module, args: tuple, *, layer_idx: int) -> None:
            # For the last block / layer, prefetch layer 0 (the first layer)
            next_id = (layer_idx + 1) % self.num_blocks
            self.prefetch_layer(next_id, non_blocking=True)

        def _post_hook(module: nn.Module, args: tuple, output: tuple, *, layer_idx: int) -> None:
            self.offload_layer(layer_idx)
            self._resident[layer_idx] = False
            self._prefetch_done[layer_idx] = None

        for i, layer in enumerate(self.blocks):
            pre_hook_fn = partial(_pre_hook, layer_idx=i)
            handle = layer.register_forward_pre_hook(pre_hook_fn)
            self._pre_hook_handles.append(handle)

            post_hook_fn = partial(_post_hook, layer_idx=i)
            handle = layer.register_forward_hook(post_hook_fn)
            self._post_hook_handles.append(handle)

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target layer in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        if layer_idx >= self.num_blocks or layer_idx < 0:
            logger.warning(f"Invalid layer id specified: {layer_idx}")
            return

        self._copy_stream.wait_stream(torch.cuda.current_stream())

        layers_to_fetch = [(layer_idx + i) % self.num_blocks for i in range(self.num_gpu_layers)]

        for idx in layers_to_fetch:
            if self._resident[idx]:
                continue

            layer_params = self.block_parameters[idx]
            layer_bufs = self.block_buffers[idx]

            evt = torch.cuda.Event()
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}

            with torch.cuda.stream(self._copy_stream):
                for dtype, cpu_weight in self.layer_cpu_weights[idx].items():
                    gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                    gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                    gpu_weights[dtype] = gpu_weight

                evt.record(self._copy_stream)

            for dtype in self.layer_metadata[idx]:
                ordered_metadata: list[dict[str, Any]] = self.layer_metadata[idx][dtype]

                gpu_weight = gpu_weights[dtype]

                for metadata in ordered_metadata:
                    target_name = metadata["name"]
                    target_param_or_buf = (
                        layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                    )

                    target_param_or_buf.data = gpu_weight[
                        metadata["offset"] : metadata["offset"] + metadata["numel"]
                    ].view(metadata["shape"])

            self._prefetch_done[idx] = evt
            self._resident[idx] = True

    @torch.compiler.disable
    def offload_layer(self, layer_idx: int) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders."""
        if layer_idx >= self.num_blocks or layer_idx < 0:
            logger.warning(f"Invalid layer id specified: {layer_idx}")
            return
        if not self._resident[layer_idx]:
            logger.warning(f"{layer_idx} is not residing on GPU")
            return

        evt = self._prefetch_done[layer_idx]
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)

        # free GPU residency
        for _, param in self.block_parameters[layer_idx].items():
            param.data = torch.empty((), device=self.device, dtype=param.dtype)
        for _, buf in self.block_buffers[layer_idx].items():
            buf.data = torch.empty((), device=self.device, dtype=buf.dtype)

    def remove_all_hooks(self) -> None:
        """Remove all hooks."""
        for h in self._pre_hook_handles:
            h.remove()
        for h in self._post_hook_handles:
            h.remove()
        self._pre_hook_handles.clear()
        self._post_hook_handles.clear()

    @staticmethod
    def get_blocks_attr_name(model: nn.Module) -> str | None:
        """Retrieve blocks attribute name from provided DiT model"""
        return getattr(model.__class__, "_layerwise_offload_blocks_attr", None)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> list[nn.Module]:
        """
        Retrieve a list of blocks from provided DiT model. Blocks attribute name
        are found by `_layerwise_offload_blocks_attr` set to DiT models. For example,

        ```
        class WanTransformer3DModel(nn.Module):
            _layerwise_offload_blocks_attr = "blocks"
        ```
        """
        blocks_attr_name = LayerwiseOffloader.get_blocks_attr_name(model)
        if blocks_attr_name is None:
            logger.warning(
                f"No _layerwise_offload_blocks_attr defined for {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        _blocks = getattr(model, blocks_attr_name, None)
        if _blocks is None:
            logger.warning(
                f"Blocks (layers) '{blocks_attr_name}' not found on {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        return list(_blocks)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Implements mutual-exclusion offloading between DiT transformers and encoders.
    When encoders run, DiT is on CPU. When DiT runs, encoders are on CPU.
    This allows running large models that don't fit entirely on GPU.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        """Initialize model-level offload backend.

        Args:
            config: OffloadConfig with MODEL_LEVEL strategy
            device: Target GPU device
        """
        super().__init__(config, device)
        self._sequential_offloader: SequentialOffloader | None = None

    def enable(self, pipeline: nn.Module) -> None:
        """Enable model-level offloading on pipeline.

        Args:
            pipeline: Diffusion pipeline model
        """
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        # Discover modules
        modules = ModuleDiscovery.discover(pipeline)

        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return

        if not modules.encoders:
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        # Move encoders to GPU
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE to GPU if available
        if modules.vae is not None:
            try:
                modules.vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Initial state: keep DiT modules on CPU (encoders typically run first)
        for dit_mod in modules.dits:
            dit_mod.to("cpu")

        torch.cuda.empty_cache()

        # Pin CPU memory if requested
        if self.config.pin_cpu_memory:
            for dit_mod in modules.dits:
                for p in dit_mod.parameters():
                    if p.data.device.type == "cpu" and not p.data.is_pinned():
                        p.data = p.data.pin_memory()

        # Register sequential offload hooks
        self._sequential_offloader = SequentialOffloader(
            modules.dits, modules.encoders, self.device, self.config.pin_cpu_memory
        )
        self._sequential_offloader.register()

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
        )

    def disable(self) -> None:
        """Disable model-level offloading and cleanup hooks."""
        if not self.enabled:
            return

        if self._sequential_offloader is not None:
            self._sequential_offloader.remove()
            self._sequential_offloader = None

        self.enabled = False
        logger.info("Model-level offloading disabled")


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        """Initialize layer-wise offload backend.

        Args:
            config: OffloadConfig with LAYER_WISE strategy
            device: Target GPU device
        """
        super().__init__(config, device)
        self._layerwise_offloaders: list[LayerwiseOffloader] = []

    def enable(self, pipeline: nn.Module) -> None:
        """Enable layer-wise offloading on pipeline.

        Args:
            pipeline: Diffusion pipeline model
        """
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return

        # Discover modules
        modules = ModuleDiscovery.discover(pipeline)

        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping layer-wise offloading")
            return

        # Move encoders to GPU (they stay resident)
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE to GPU if available
        if modules.vae is not None:
            try:
                modules.vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Setup layer-wise offloading for each DiT
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_name = LayerwiseOffloader.get_blocks_attr_name(dit_module)
            blocks = LayerwiseOffloader.get_blocks_from_dit(dit_module)

            if not blocks_attr_name or not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name == blocks_attr_name:
                    logger.debug(f"Skipped blocks module {name}")
                    continue
                m.to(self.device)
                logger.debug(f"Moved {name} to device {self.device}")

            # Create and register offloader
            offloader = LayerwiseOffloader(
                blocks, self.device, self.config.pin_cpu_memory, self.config.layerwise_num_gpu_layers
            )
            self._layerwise_offloaders.append(offloader)

            # Store reference on DiT module for compatibility
            setattr(dit_module, "_layerwise_offloader", offloader)

            logger.info(
                f"Layer-wise offloading enabled on {len(blocks)} layers (blocks), "
                f"with {self.config.layerwise_num_gpu_layers} kept on device"
            )

        if self._layerwise_offloaders:
            self.enabled = True
        else:
            logger.warning("No layer-wise offloaders created, offloading not enabled")

    def disable(self) -> None:
        """Disable layer-wise offloading and cleanup hooks."""
        if not self.enabled:
            return

        for offloader in self._layerwise_offloaders:
            offloader.remove_all_hooks()

        self._layerwise_offloaders.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")


def get_offload_backend(
    od_config: OmniDiffusionConfig,
    device: torch.device | None = None,
) -> OffloadBackend | None:
    """Create appropriate offload backend based on configuration.

    Args:
        od_config: OmniDiffusionConfig with offload settings
        device: Target device (auto-detected if None)

    Returns:
        OffloadBackend instance or None if offloading disabled

    Example:
        >>> backend = get_offload_backend(od_config, device=torch.device("cuda:0"))
        >>> if backend:
        ...     backend.enable(pipeline)
    """
    # Extract and validate configuration
    config = OffloadConfig.from_od_config(od_config)

    # Return None if no offloading requested
    if config.strategy == OffloadStrategy.NONE:
        return None

    # Validate platform (CUDA required for now)
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
        logger.warning("CPU offloading requires CUDA devices. Skipping offloading.")
        return None

    # Detect device if not provided
    if device is None:
        try:
            device = current_omni_platform.get_torch_device()
        except (NotImplementedError, AttributeError) as exc:
            logger.error("Failed to detect device: %s. Skipping offloading.", exc)
            return None

    # Create appropriate backend
    if config.strategy == OffloadStrategy.MODEL_LEVEL:
        return ModelLevelOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.LAYER_WISE:
        return LayerWiseOffloadBackend(config, device)
    else:
        logger.error("Unknown offload strategy: %s", config.strategy)
        return None


# Legacy
# def apply_offload_hooks(
#     model: nn.Module,
#     od_config: OmniDiffusionConfig,
#     *,
#     device: torch.device | None = None,
# ) -> None:
#     """Apply mutual-exclusion offload hooks based on config.

#     When enable_cpu_offload is enabled, DiT and encoders swap GPU access:
#     - Encoders (text_encoder, text_encoder_2, text_encoder_3, image_encoder)
#       run on GPU while DiT is on CPU
#     - DiT runs on GPU while encoders are on CPU

#     Args:
#         model: Diffusion pipeline model
#         od_config: OmniDiffusionConfig with offload settings
#     """
#     enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
#     enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
#     pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

#     if not enable_cpu_offload and not enable_layerwise_offload:
#         return
#     if enable_cpu_offload and enable_layerwise_offload:
#         # NOTE: Model-wise and layerwise cpu offloading are not supported together at this moment,
#         # consider layerwise offloading has higher priority than model-wise offloading
#         enable_cpu_offload = False
#         logger.info(
#             "Model-wise and layer-wise CPU offloading are not supported together at this moment. "
#             "Automatically disabled model-wise offloading."
#         )
#     # For now, model-wise and layer-wise (block-wise) offloading
#     # are functioning as expected when cuda device is available
#     if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
#         logger.info("CPU Offloading requires cuda devices available. Skipping for now...")
#         return

#     # Find DiT/transformer modules
#     dit_modules: list[nn.Module] = []
#     dit_names: list[str] = []
#     candidate_attrs = ["transformer", "transformer_2", "dit"]
#     for attr in candidate_attrs:
#         if not hasattr(model, attr):
#             continue
#         module_obj = getattr(model, attr)
#         if module_obj is None:
#             continue

#         assert isinstance(module_obj, nn.Module), f"Expected {attr} to be nn.Module, got {type(module_obj)!r}"

#         if module_obj in dit_modules:
#             continue

#         dit_modules.append(module_obj)
#         dit_names.append(attr)

#     if not dit_modules:
#         logger.warning("enable_cpu_offload enabled but no transformer/dit/unet found")
#         return
#     if device is None:
#         try:
#             device = next(dit_modules[0].parameters()).device
#         except StopIteration:
#             try:
#                 device = current_omni_platform.get_torch_device()
#             except (NotImplementedError, AttributeError):
#                 logger.error("Fail to get device of pipeline. Skipping applying offloading hooks")
#                 return

#     # Collect all encoders
#     encoders: list[nn.Module] = []
#     encoder_names: list[str] = []
#     for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
#         if hasattr(model, attr) and getattr(model, attr) is not None:
#             encoders.append(getattr(model, attr))
#             encoder_names.append(attr)
#     if not encoders and enable_cpu_offload:
#         logger.warning("enable_cpu_offload enabled but no encoders found")
#         return
#     for enc in encoders:
#         enc.to(device)

#     # Collect VAE
#     for name in ["vae"]:
#         module = getattr(model, name, None)
#         if module is None:
#             continue
#         try:
#             module.to(device, non_blocking=True)
#         except Exception as exc:
#             logger.debug("Failed to move %s to GPU: %s", name, exc)

#     if enable_cpu_offload:
#         # Initial state: keep DiT modules on CPU (encoders typically run first)
#         for dit_mod in dit_modules:
#             dit_mod.to("cpu")

#         torch.cuda.empty_cache()

#         if pin_cpu_memory:
#             for dit_mod in dit_modules:
#                 for p in dit_mod.parameters():
#                     if p.data.device.type == "cpu" and not p.data.is_pinned():
#                         p.data = p.data.pin_memory()

#         # Register sequential offload hooks
#         SequentialOffloader(dit_modules, encoders, device, pin_cpu_memory).register()
#         logger.info(
#             "CPU offload enabled: %s <-> %s (mutual exclusion)",
#             ", ".join(dit_names),
#             ", ".join(encoder_names),
#         )
#     elif enable_layerwise_offload:
#         logger.info(f"Applying offloading hooks on {dit_names}")

#         for i, dit_module in enumerate(dit_modules):
#             logger.info(f"Applying hook on {dit_names[i]} ({dit_module.__class__.__name__})")
#             blocks_attr_name = LayerwiseOffloader.get_blocks_attr_name(dit_module)
#             blocks = LayerwiseOffloader.get_blocks_from_dit(dit_module)

#             if not blocks_attr_name or not blocks:
#                 logger.warning(
#                     "Target layers (blocks) are not found. "
#                     f"Skipping offloading on {dit_names[i]} ({dit_module.__class__.__name__})"
#                 )
#                 continue

#             # move modules other than blocks to gpu and keep them on gpu
#             for name, m in dit_module.named_children():
#                 # Skip the blocks module (layers to be offloaded)
#                 if name == blocks_attr_name:
#                     logger.debug(f"Skipped module {name}")
#                     continue

#                 m.to(device)
#                 logger.debug(f"Moved {name} to device {device}")

#             # set to the module (transformer)
#             offloader = LayerwiseOffloader(blocks, device, pin_cpu_memory, od_config.layerwise_num_gpu_layers)
#             setattr(dit_module, "_layerwise_offloader", offloader)

#             logger.info(
#                 f"Layerwise offloading enabled on {len(blocks)} layers (blocks), "
#                 f"with {od_config.layerwise_num_gpu_layers} kept on device"
#             )
