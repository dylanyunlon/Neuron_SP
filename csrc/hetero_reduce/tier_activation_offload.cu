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
#include <type_traits>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

// BUG-FIX (#143): unified CUDA kernel error check macro.
// Checks cudaGetLastError() after each kernel launch.
// In debug builds or when HETERO_REDUCE_STRICT_ERRORS is defined this aborts;
// in production it writes to stderr and is a no-op (caller stream stays valid).
#ifndef DS_LAUNCH_CHECK
#  ifdef NDEBUG
#    define DS_LAUNCH_CHECK(stream)                                              \\
       do {                                                                      \\
           cudaError_t _e = cudaGetLastError();                                  \\
           if (_e != cudaSuccess)                                                \\
               fprintf(stderr, "[hetero_reduce] kernel launch error: %s (%s:%d)\\n",\\
                       cudaGetErrorString(_e), __FILE__, __LINE__);              \\
       } while (0)
#  else
#    define DS_LAUNCH_CHECK(stream)                                              \\
       do {                                                                      \\
           cudaError_t _e = cudaGetLastError();                                  \\
           if (_e != cudaSuccess) {                                              \\
               fprintf(stderr, "[hetero_reduce] kernel launch error: %s (%s:%d)\\n",\\
                       cudaGetErrorString(_e), __FILE__, __LINE__);              \\
               abort();                                                          \\
           }                                                                     \\
       } while (0)
#  endif
#endif  // DS_LAUNCH_CHECK


namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// SM8.6/SM9.0: 256 threads, 2 CTAs/SM.  SM12.0 Blackwell: 512 threads, 4 CTAs/SM.
// Larger blocks on Blackwell improve occupancy on the wider 128-wide SMs.
static constexpr int kOffloadBlockSm86   = 256;
static constexpr int kOffloadBlockSm120  = 512;
static constexpr int kOffloadVecW        = 8;    // BF16 elements per 128-bit load
static constexpr int kQuantTileSize      = 128;  // elements per quantisation tile
static constexpr int kInt8VecW           = 16;   // INT8 elements per 128-bit load

// Compile-time policy: select block size and min-CTAs-per-SM by SM version.
template <int SmVer> struct OffloadPolicy {
    static constexpr int kBlockSize      = (SmVer >= 120) ? kOffloadBlockSm120 : kOffloadBlockSm86;
    static constexpr int kMinBlocksPerSM = (SmVer >= 90)  ? 4 : 2;
};

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
template <int SmVer>
__global__ void
__launch_bounds__(OffloadPolicy<SmVer>::kBlockSize, OffloadPolicy<SmVer>::kMinBlocksPerSM)
activation_pack_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ inputs,
    int    num_tensors,
    size_t tensor_elems)
{
    constexpr int kBS = OffloadPolicy<SmVer>::kBlockSize;
    const size_t vec_per_tensor = tensor_elems / kOffloadVecW;
    const size_t total_vecs     = (size_t)num_tensors * vec_per_tensor;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;

    for (size_t v = tid; v < total_vecs; v += stride) {
        const int    tensor_idx = (int)(v / vec_per_tensor);
        const size_t vec_idx    = v % vec_per_tensor;
        const size_t src_elem   = vec_idx * kOffloadVecW;
        const size_t dst_elem   = (size_t)tensor_idx * tensor_elems + src_elem;

        copy_bf16x8(inputs[tensor_idx] + src_elem, output + dst_elem);
    }
}

// ---------------------------------------------------------------------------
// Activation unpack kernel.
//
// Scatters a flat buffer back to `num_tensors` output tensors.
// Inverse of activation_pack_kernel.
// ---------------------------------------------------------------------------
template <int SmVer>
__global__ void
__launch_bounds__(OffloadPolicy<SmVer>::kBlockSize, OffloadPolicy<SmVer>::kMinBlocksPerSM)
activation_unpack_kernel(
    __nv_bfloat16* const* __restrict__       outputs,
    const __nv_bfloat16* __restrict__        flat,
    int    num_tensors,
    size_t tensor_elems)
{
    constexpr int kBS = OffloadPolicy<SmVer>::kBlockSize;
    const size_t vec_per_tensor = tensor_elems / kOffloadVecW;
    const size_t total_vecs     = (size_t)num_tensors * vec_per_tensor;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;

    for (size_t v = tid; v < total_vecs; v += stride) {
        const int    tensor_idx = (int)(v / vec_per_tensor);
        const size_t vec_idx    = v % vec_per_tensor;
        const size_t src_elem   = (size_t)tensor_idx * tensor_elems
                                  + vec_idx * kOffloadVecW;
        const size_t dst_elem   = vec_idx * kOffloadVecW;

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
    if (num_tensors <= 0 || tensor_elems == 0) return;  // BUG-FIX: guard zero-grid launch
    // Copy input pointer array to device.
    const __nv_bfloat16** d_inputs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_inputs),
                    num_tensors * sizeof(const __nv_bfloat16*), stream);
    cudaMemcpyAsync(d_inputs, inputs,
                    num_tensors * sizeof(const __nv_bfloat16*),
                    cudaMemcpyHostToDevice, stream);

    // SM12.0 (Blackwell): 512-thread blocks improve occupancy on wider SMs.
    // SM8.6 / SM9.0: 256-thread blocks.
    // Three-way SM dispatch: SM12.0 (512 threads, 4 CTAs/SM),
    //                          SM9.0  (256 threads, 4 CTAs/SM),
    //                          SM8.6  (256 threads, 2 CTAs/SM).
    // Using block-size as the sole discriminator conflated SM9.0 and SM8.6,
    // giving H100 the wrong kMinBlocksPerSM=2 __launch_bounds__ hint.
    // Fix: dispatch on sm_version first, then block size.
    if (sm_version >= 120) {
        constexpr int kBS = kOffloadBlockSm120;
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_pack_kernel<120><<<grid, kBS, 0, stream>>>(
            output, d_inputs, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        constexpr int kBS = kOffloadBlockSm86;  // same block size, different min-CTAs
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_pack_kernel<90><<<grid, kBS, 0, stream>>>(
            output, d_inputs, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    } else {
        constexpr int kBS = kOffloadBlockSm86;
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_pack_kernel<86><<<grid, kBS, 0, stream>>>(
            output, d_inputs, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    }

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
    if (num_tensors <= 0 || tensor_elems == 0) return;  // BUG-FIX: guard zero-grid launch
    __nv_bfloat16** d_outputs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_outputs),
                    num_tensors * sizeof(__nv_bfloat16*), stream);
    cudaMemcpyAsync(d_outputs, outputs,
                    num_tensors * sizeof(__nv_bfloat16*),
                    cudaMemcpyHostToDevice, stream);

    // Three-way SM dispatch — mirrors launch_activation_pack fix above.
    if (sm_version >= 120) {
        constexpr int kBS = kOffloadBlockSm120;
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_unpack_kernel<120><<<grid, kBS, 0, stream>>>(
            d_outputs, flat, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        constexpr int kBS = kOffloadBlockSm86;
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_unpack_kernel<90><<<grid, kBS, 0, stream>>>(
            d_outputs, flat, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    } else {
        constexpr int kBS = kOffloadBlockSm86;
        const size_t total_vecs = (size_t)num_tensors * (tensor_elems / kOffloadVecW);
        const int grid = (int)std::min((total_vecs + kBS - 1) / kBS, (size_t)65535);
        activation_unpack_kernel<86><<<grid, kBS, 0, stream>>>(
            d_outputs, flat, num_tensors, tensor_elems);
    DS_LAUNCH_CHECK(stream);
    }

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
    DS_LAUNCH_CHECK(stream);
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
    DS_LAUNCH_CHECK(stream);
}
