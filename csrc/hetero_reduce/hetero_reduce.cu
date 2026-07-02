// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_reduce.cu
 *
 * Fused BF16 → FP32 reduce-scatter + FP32 → BF16 writeback kernel for
 * heterogeneous GPU clusters (2×A6000 SM8.6 + 2×Blackwell SM12.0 + 1×H100
 * SM9.0) connected via PCIe without NVLink.
 *
 * Kernel strategy
 * ---------------
 * Each thread processes 8 BF16 elements per step (128-bit vectorised load/store).
 * FP32 accumulation avoids precision loss when summing many gradients.
 * Separate kernel variants for SM 8.6/9.0 vs SM 12.0 differ only in
 * __launch_bounds__ to guide occupancy on each architecture.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"  // DS_D_INLINE, hw_warp_size

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kBlockSize   = 256;
static constexpr int kVecWidth    = 8;     // BF16 elements per thread per step
static constexpr int kMaxTensors  = 32;

// ---------------------------------------------------------------------------
// Vectorised 128-bit load of 8 × BF16.
// ---------------------------------------------------------------------------
DS_D_INLINE void load_bf16x8(const __nv_bfloat16* __restrict__ ptr,
                               float2& acc0, float2& acc1,
                               float2& acc2, float2& acc3)
{
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    acc0.x += __bfloat162float(p[0].x);
    acc0.y += __bfloat162float(p[0].y);
    acc1.x += __bfloat162float(p[1].x);
    acc1.y += __bfloat162float(p[1].y);
    acc2.x += __bfloat162float(p[2].x);
    acc2.y += __bfloat162float(p[2].y);
    acc3.x += __bfloat162float(p[3].x);
    acc3.y += __bfloat162float(p[3].y);
}

// ---------------------------------------------------------------------------
// Vectorised 128-bit store: 8 floats → 8 BF16.
// ---------------------------------------------------------------------------
DS_D_INLINE void store_fp32x8_as_bf16(__nv_bfloat16* __restrict__ ptr,
                                        float2 a, float2 b, float2 c, float2 d)
{
    __nv_bfloat162 ba = {__float2bfloat16(a.x), __float2bfloat16(a.y)};
    __nv_bfloat162 bb = {__float2bfloat16(b.x), __float2bfloat16(b.y)};
    __nv_bfloat162 bc = {__float2bfloat16(c.x), __float2bfloat16(c.y)};
    __nv_bfloat162 bd = {__float2bfloat16(d.x), __float2bfloat16(d.y)};
    uint4 raw;
    raw.x = *reinterpret_cast<const uint32_t*>(&ba);
    raw.y = *reinterpret_cast<const uint32_t*>(&bb);
    raw.z = *reinterpret_cast<const uint32_t*>(&bc);
    raw.w = *reinterpret_cast<const uint32_t*>(&bd);
    *reinterpret_cast<uint4*>(ptr) = raw;
}

// ---------------------------------------------------------------------------
// Reduce kernel — SM 8.6 / 9.0 (conservative occupancy).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlockSize, 2)
fused_bf16_reduce_kernel_sm86(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ inputs,
    int  num_tensors,
    size_t n_elems)
{
    const size_t tid    = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * kBlockSize;

    for (size_t t = tid; t < n_elems / kVecWidth; t += stride) {
        const size_t base = t * kVecWidth;
        float2 acc0 = {0.f, 0.f}, acc1 = {0.f, 0.f};
        float2 acc2 = {0.f, 0.f}, acc3 = {0.f, 0.f};

        for (int i = 0; i < num_tensors; i++) {
            load_bf16x8(inputs[i] + base, acc0, acc1, acc2, acc3);
        }
        store_fp32x8_as_bf16(output + base, acc0, acc1, acc2, acc3);
    }
}

// ---------------------------------------------------------------------------
// Reduce kernel — SM 12.0 (Blackwell: higher occupancy hint).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlockSize, 4)
fused_bf16_reduce_kernel_sm120(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ inputs,
    int  num_tensors,
    size_t n_elems)
{
    const size_t tid    = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * kBlockSize;

    for (size_t t = tid; t < n_elems / kVecWidth; t += stride) {
        const size_t base = t * kVecWidth;
        float2 acc0 = {0.f, 0.f}, acc1 = {0.f, 0.f};
        float2 acc2 = {0.f, 0.f}, acc3 = {0.f, 0.f};

        for (int i = 0; i < num_tensors; i++) {
            load_bf16x8(inputs[i] + base, acc0, acc1, acc2, acc3);
        }
        store_fp32x8_as_bf16(output + base, acc0, acc1, acc2, acc3);
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrapper
// ---------------------------------------------------------------------------
void launch_fused_bf16_reduce(__nv_bfloat16*              output,
                               const __nv_bfloat16* const* inputs,
                               int                         num_tensors,
                               size_t                      n_elems,
                               int                         sm_version,
                               cudaStream_t                stream)
{
    // Allocate a small device buffer for the pointer array.
    const __nv_bfloat16** d_inputs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_inputs),
                    num_tensors * sizeof(const __nv_bfloat16*), stream);
    cudaMemcpyAsync(d_inputs, inputs,
                    num_tensors * sizeof(const __nv_bfloat16*),
                    cudaMemcpyHostToDevice, stream);

    const size_t vec_elems = n_elems / kVecWidth;
    const int grid = static_cast<int>(
        std::min((vec_elems + kBlockSize - 1) / kBlockSize, (size_t)65535));

    if (sm_version >= 120) {
        fused_bf16_reduce_kernel_sm120<<<grid, kBlockSize, 0, stream>>>(
            output, d_inputs, num_tensors, n_elems);
    } else {
        fused_bf16_reduce_kernel_sm86<<<grid, kBlockSize, 0, stream>>>(
            output, d_inputs, num_tensors, n_elems);
    }

    cudaFreeAsync(d_inputs, stream);
}
