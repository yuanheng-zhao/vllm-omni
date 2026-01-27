# Layerwise (Blockwise) Offloading for Diffusion Models

## Overview
Layerwise offload operates at transformer block granularity, keeping a single transformer block, or a specified number of blocks, on GPU while others stay in CPU memory.

Unlike full model-wise CPU offload which swaps entire components like DiT and encoders, layerwise offloading applies a sliding window way of loading and offloading weights between gpu and cpu: while block `i` computes, block `i+1` get prefetched asynchronously via pinned memory. In this way, only partial blocks(s) reside on GPU at any moment during inference, so that greatly decrease the memory occupancy.

## Steps
When `layerwise_offload_dit` is enabled,

1. During model initialization, all components are loaded to CPU first. Then components other than DiT model(s) in the pipeline, such as VAE and encoders, are moved to GPU. The weights of target transformer blocks are collected as contiguous tensors per layer on CPU with pinned memory; and non-block modules (embeddings, norms, etc) in the DiT model are moved to and stay on GPU.
2. The first block(s) are transferred to GPU during initialization of `LayerwiseOffloader`, before the first denoising step of the very first request.
3. As each block executes, the next block prefetches on a separate CUDA stream for compute - memory copy overlap. After execution, the current block is immediately freed from GPU memory.
4. When the last block completes, the first block prefetches for the next denoising step.


Example of hooks of a DiT model with n layers, by default keep a single layer on GPU:
| Layer (block) idx | forward pre-hook               | forward          | forward post-hook         |
|-------------------|--------------------------------|------------------|---------------------------|
| layer-0           | prefetch layer 1 (copy stream) | compute layer 0  | free layer-0 gpu weights  |
| layer-1           | prefetch layer 2 (copy stream) | compute layer 1  | free layer-1 gpu weights  |
| layer-2           | prefetch layer 3 (copy stream) | compute layer 2  | free layer-2 gpu weights  |
| ...               | ...                            | ...              | ...                       |
| layer-(n-1)       | **prefetch layer 0 (copy stream)** | compute layer (n-1) | free layer (n-1) gpu weights  |


## Configuration

- **Python API**: set `layerwise_offload_dit=True` and optionally `layerwise_num_gpu_layers`.

```python
from vllm_omni import Omni

if __name__ == "__main__":
    m = Omni(
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        layerwise_offload_dit=True,
        ...
    )
```

- **CLI**: pass `--enable-layerwise-offload` and `--num-gpu-layers` to the diffusion service entrypoint.

## Supported Models
Models must define `_layerwise_offload_blocks_attr` class attribute so that the layerwise offloader finds the target transformer blocks.
- **Qwen-Image** (`transformer_blocks`)
- **Wan2.2** (`blocks`)


| Architecture | Models | Example HF Models | DiT Model Cls | Blocks Attr Name |
|--------------|--------|-------------------|----------|----------|
| `QwenImagePipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | "transformer_blocks" |
| `Wan22Pipeline` | Wan2.2 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | "blocks" |


## Known Limitations
- Cold start latency increases because of
    1) components are loaded to CPU first at the very first during initialization,  
    2) weight consolidation and pinning
- Performance depends on CPU <-> GPU interconnection (e.g., PCIe bandwidth).
- Support single GPU only for now
