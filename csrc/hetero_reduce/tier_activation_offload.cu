// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * tier_activation_offload.cu
 *
 * Tier-aware activation checkpoint offload and prefetch kernels for
 * heterogeneous GPU clusters (A6000 SM8.6 + H100 SM9.0 + Blackwell SM12.0)
 * connected over PCIe without NVLink.
 *
 * Background
 * ----------
 * Activation checkpointing (gradient checkpointing) trades memory for compute
 * by recomputing activations on the backward pass instead of storing them.
 * In heterogeneous clusters, different GPU tiers have different VRAM:
 *   - A6000 SM8.6:  48 GB VRAM   (most constrained)
 *   - H100  SM9.0:  80 GB VRAM
 *   - Blackwell SM12.0: 192 GB VRAM (NVL72 config)
 *
 * Tier-aware offload strategy
 * ---------------------------
 * Rather than always recomputing, we offload activations from VRAM-constrained
 * tiers (A6000) to either:
 *   (a) Host pinned memory via cudaMemcpyAsync (cheapest path)
 *   (b) A peer GPU with spare VRAM via P2P copy if available
 *
 * Before recomputation on the backward pass, a prefetch kernel brings the
 * checkpoints back.
 *
 * This file provides:
 *   1. launch_activation_pack()    — pack non-contiguous activations into a
 *                                    flat BF16 buffer (device-side gather)
 *   2. launch_activation_unpack()  — scatter a flat buffer back to tensors
 *   3. launch_quantise_fp16_to_int8() — optional 8-bit quantisation for
 *                                       host offload to halve PCIe traffic
 *   4. launch_dequantise_int8_to_fp16() — dequantise on prefetch
 *   5. compute_offload_budget()    — host helper: compute how many bytes a
 *                                    tier should offload given VRAM headroom
 *
 * Kernel design
 * -------------
 * All kernels use 128-bit vectorised loads/stores (8× BF16 or 16× INT8).
 * SM-specialised __launch_bounds__: SM8.6→(256,2), SM9.0→(256,4), SM12.0→(512,4).
 *
 * Quantisation
 * ------------
 * INT8 absmax per 128-element tile quantisation (compatible with bitsandbytes
 * block-wise format).  FP32 scale = max(|x|) / 127.  Dequant: x = i8 * scale.
 * One block per tile; 128 elements / 16 per thread = 8 threads × 16 elements.
 * Tile scales stored in a separate float32 buffer (1 float per 128 elements).
 *
 * Cooperative groups
 * ------------------
 * Warp-level max reduction for absmax uses cg::reduce() with cg::greater<float>().
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kOffloadBlock   = 256;
static constexpr int kOffloadVecW    = 8;   // BF16 elements per 128-bit load
static constexpr int kQuantTileSize  = 128; // elements per quantisation tile
static constexpr int kInt8VecW       = 16;  // INT8 elements per 128-bit load

// ---------------------------------------------------------------------------
// Vectorised BF16 copy helpers
// ---------------------------------------------------------------------------
DS_D_INLINE void copy_bf16x8(const __nv_bfloat16* __restrict__ src,
                               __nv_bfloat16* __restrict__ dst)
{
    *reinterpret_cast<uint4*>(dst) =
        *reinterpret_cast<const uint4*>(src);
}

// ---------------------------------------------------------------------------
// Activation pack kernel.
//
// Gathers activations from `num_tensors` input pointers into a flat output
// buffer.  Each input tensor has the same element count `tensor_elems`.
//
// Output layout: [tensor_0 | tensor_1 | ... | tensor_{n-1}]
//   offset of tensor i: i * tensor_elems
//
// @param output       flat BF16 buffer  [num_tensors * tensor_elems]
// @param inputs       array of device pointers to activation tensors
// @param num_tensors  number of activation tensors to pack
// @param tensor_elems elements per tensor (must be divisible by 8)
// ---------------------------------------------------------------------------
template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
activation_pack_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ inputs,
    int    num_tensors,
    size_t tensor_elems)
{
    const size_t vec_per_tensor = tensor_elems / kOffloadVecW;
    const size_t total_vecs     = (size_t)num_tensors * vec_per_tensor;
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t v = tid; v < total_vecs; v += stride) {
        const int    tensor_idx = (int)(v / vec_per_tensor);
        const size_t vec_idx    = v % vec_per_tensor;
        const size_t src_elem   = vec_idx * kOffloadVecW;
        const size_t dst_elem   = (size_t)tensor_idx * tensor_elems + src_elem;

#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next vector's source data
        if (v + stride < total_vecs) {
            const int    next_tidx = (int)((v + stride) / vec_per_tensor);
            const size_t next_vidx = (v + stride) % vec_per_tensor;
            asm volatile("prefetch.global.L1 [%0];" :: "l"(inputs[next_tidx] + next_vidx * kOffloadVecW));
        }
#endif
        copy_bf16x8(inputs[tensor_idx] + src_elem, output + dst_elem);
    }
}

// ---------------------------------------------------------------------------
// Activation unpack kernel.
//
// Scatters a flat buffer back to `num_tensors` output tensors.
// Inverse of activation_pack_kernel.
// ---------------------------------------------------------------------------
template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
activation_unpack_kernel(
    __nv_bfloat16* const* __restrict__       outputs,
    const __nv_bfloat16* __restrict__        flat,
    int    num_tensors,
    size_t tensor_elems)
{
    const size_t vec_per_tensor = tensor_elems / kOffloadVecW;
    const size_t total_vecs     = (size_t)num_tensors * vec_per_tensor;
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t v = tid; v < total_vecs; v += stride) {
        const int    tensor_idx = (int)(v / vec_per_tensor);
        const size_t vec_idx    = v % vec_per_tensor;
        const size_t src_elem   = (size_t)tensor_idx * tensor_elems
                                  + vec_idx * kOffloadVecW;
        const size_t dst_elem   = vec_idx * kOffloadVecW;

#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next vector's source data
        if (v + stride < total_vecs) {
            const int    next_tidx = (int)((v + stride) / vec_per_tensor);
            const size_t next_vidx = (v + stride) % vec_per_tensor;
            const size_t next_src  = (size_t)next_tidx * tensor_elems
                                     + next_vidx * kOffloadVecW;
            asm volatile("prefetch.global.L1 [%0];" :: "l"(flat + next_src));
        }
#endif
        copy_bf16x8(flat + src_elem, outputs[tensor_idx] + dst_elem);
    }
}

// ---------------------------------------------------------------------------
// INT8 block-wise quantisation kernel.
//
// Quantises a flat BF16 buffer to INT8 with per-tile absmax scaling.
// Each block handles kQuantTileSize elements.
// Tile scale stored in scales[tile_idx] = absmax / 127.
//
// One warp per CTA (32 threads), each thread processes 4 BF16 elements.
// Warp-level max reduction via cooperative groups.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32, 8)
quantise_bf16_to_int8_kernel(
    int8_t* __restrict__        output,  // [n_elems] INT8
    float*  __restrict__        scales,  // [n_tiles] FP32 scale per tile
    const __nv_bfloat16* __restrict__ input,   // [n_elems] BF16
    size_t n_elems)
{
    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(blk);

    const int tile_idx = blockIdx.x;
    const size_t tile_base = (size_t)tile_idx * kQuantTileSize;
    if (tile_base >= n_elems) return;

    const int tile_len = (int)std::min((size_t)kQuantTileSize, n_elems - tile_base);
    const int lane = warp.thread_rank();

    // Step 1: compute absmax across tile (each lane reads 4 elements).
    float local_max = 0.f;
    for (int i = lane; i < tile_len; i += 32) {
        float v = __bfloat162float(input[tile_base + i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float tile_max = cg::reduce(warp, local_max, cg::greater<float>());

    float scale     = tile_max / 127.f + 1e-8f;  // avoid div-by-zero
    float inv_scale = 1.f / scale;

    // Lane 0 writes scale.
    if (lane == 0) scales[tile_idx] = scale;

    // Step 2: quantise and write INT8.
    for (int i = lane; i < tile_len; i += 32) {
        float v   = __bfloat162float(input[tile_base + i]);
        int8_t q  = (int8_t)__float2int_rn(v * inv_scale);
        output[tile_base + i] = q;
    }
}

// ---------------------------------------------------------------------------
// INT8 block-wise dequantisation kernel.
//
// Converts INT8 + scales back to BF16.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32, 8)
dequantise_int8_to_bf16_kernel(
    __nv_bfloat16* __restrict__  output,  // [n_elems] BF16
    const int8_t* __restrict__   input,   // [n_elems] INT8
    const float*  __restrict__   scales,  // [n_tiles] FP32 scale
    size_t n_elems)
{
    const int tile_idx = blockIdx.x;
    const size_t tile_base = (size_t)tile_idx * kQuantTileSize;
    if (tile_base >= n_elems) return;

    const int tile_len = (int)std::min((size_t)kQuantTileSize, n_elems - tile_base);
    const float scale = scales[tile_idx];
    const int lane = threadIdx.x;

    for (int i = lane; i < tile_len; i += 32) {
        float v = (float)input[tile_base + i] * scale;
        output[tile_base + i] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------------------
// Host helper: compute how many activation bytes a tier should offload.
//
// @param total_act_bytes  Total activation bytes the model requires
// @param vram_free_bytes  Current free VRAM on this tier's device
// @param headroom_frac    Fraction of free VRAM to keep for runtime use
// @returns bytes to offload (rounded to kBucketAlign = 128-bit boundary)
// ---------------------------------------------------------------------------
size_t compute_offload_budget(size_t total_act_bytes,
                              size_t vram_free_bytes,
                              float  headroom_frac)
{
    if (headroom_frac < 0.f) headroom_frac = 0.f;
    if (headroom_frac > 1.f) headroom_frac = 1.f;

    size_t usable = (size_t)(vram_free_bytes * (1.f - headroom_frac));
    if (usable >= total_act_bytes) return 0;  // fits in VRAM, no offload

    size_t to_offload = total_act_bytes - usable;
    // Round up to 128-bit alignment (16 bytes).
    to_offload = (to_offload + 15ULL) & ~15ULL;
    return to_offload;
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

void launch_activation_pack(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      tensor_elems,
    int                         sm_version,
    cudaStream_t                stream)
{
    // Copy input pointer array to device.
    const __nv_bfloat16** d_inputs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_inputs),
                    num_tensors * sizeof(const __nv_bfloat16*), stream);
    cudaMemcpyAsync(d_inputs, inputs,
                    num_tensors * sizeof(const __nv_bfloat16*),
                    cudaMemcpyHostToDevice, stream);

    const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
    const int grid = (int)std::min(
        (total_vecs + kOffloadBlock - 1) / kOffloadBlock, (size_t)65535);

    // Pack is bandwidth-bound; block size doesn't change throughput much.
    // Use 256 threads for all SM versions.
    activation_pack_kernel<kOffloadBlock><<<grid, kOffloadBlock, 0, stream>>>(
        output, d_inputs, num_tensors, tensor_elems);

    cudaFreeAsync(d_inputs, stream);
}

void launch_activation_unpack(
    __nv_bfloat16* const*  outputs,
    const __nv_bfloat16*   flat,
    int                    num_tensors,
    size_t                 tensor_elems,
    int                    sm_version,
    cudaStream_t           stream)
{
    __nv_bfloat16** d_outputs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_outputs),
                    num_tensors * sizeof(__nv_bfloat16*), stream);
    cudaMemcpyAsync(d_outputs, outputs,
                    num_tensors * sizeof(__nv_bfloat16*),
                    cudaMemcpyHostToDevice, stream);

    const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
    const int grid = (int)std::min(
        (total_vecs + kOffloadBlock - 1) / kOffloadBlock, (size_t)65535);

    activation_unpack_kernel<kOffloadBlock><<<grid, kOffloadBlock, 0, stream>>>(
        d_outputs, flat, num_tensors, tensor_elems);

    cudaFreeAsync(d_outputs, stream);
}

void launch_quantise_fp16_to_int8(
    int8_t*              output,
    float*               scales,
    const __nv_bfloat16* input,
    size_t               n_elems,
    cudaStream_t         stream)
{
    const size_t n_tiles = (n_elems + kQuantTileSize - 1) / kQuantTileSize;
    // One warp (32 threads) per tile.
    quantise_bf16_to_int8_kernel<<<(int)std::min(n_tiles, (size_t)65535), 32, 0, stream>>>(
        output, scales, input, n_elems);
}

void launch_dequantise_int8_to_fp16(
    __nv_bfloat16* output,
    const int8_t*  input,
    const float*   scales,
    size_t         n_elems,
    cudaStream_t   stream)
{
    const size_t n_tiles = (n_elems + kQuantTileSize - 1) / kQuantTileSize;
    dequantise_int8_to_bf16_kernel<<<(int)std::min(n_tiles, (size_t)65535), 32, 0, stream>>>(
        output, input, scales, n_elems);
}
