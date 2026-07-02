// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_reduce.h
 *
 * Fused BF16->FP32 reduce-scatter + FP32->BF16 cast kernel for heterogeneous
 * GPU clusters (SM 8.6 / 9.0 / 12.0) running over PCIe without NVLink.
 *
 * Design goals:
 *   - Accept a list of gradient tensors in BF16, reduce them in FP32
 *     accumulation, and write the result back as BF16 in one fused pass.
 *   - Support per-tier bucket_size so low-bandwidth tiers can use larger
 *     buckets without stalling faster tiers.
 *   - Support non-uniform shard assignment across heterogeneous GPU tiers:
 *     H100 (SM 9.0) receives the largest shard, Blackwell (SM 12.0) next,
 *     A6000 (SM 8.6) the smallest.
 *   - Zero copies: input and output pointers may alias (in-place).
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Tier descriptor: one entry per physical GPU tier in the heterogeneous pool.
// ---------------------------------------------------------------------------
struct HeteroTierDesc {
    int device_id;       // CUDA device ordinal
    int sm_version;      // e.g. 86, 90, 120
    size_t bucket_size;  // gradient bytes per reduce-scatter bucket
};

// ---------------------------------------------------------------------------
// Launch wrappers — callable from C++ / pybind11 binding code.
// ---------------------------------------------------------------------------

/**
 * fused_bf16_reduce
 *
 * Reduces `num_tensors` BF16 input pointers, each of `n_elems` elements, into
 * a single BF16 output buffer using FP32 accumulation.
 *
 * All pointers must live on the same CUDA device; the caller is responsible
 * for cross-device copies when orchestrating the full reduce-scatter.
 *
 * For small tensors (≤ 32 K elements) with few inputs, a warp-cooperative
 * path is used: lanes within each warp accumulate different input tensors
 * and then perform a cg::reduce() to fold partial results.
 *
 * @param output      [out] BF16 output buffer (device), length n_elems
 * @param inputs      [in]  Array of device pointers to BF16 input tensors
 * @param num_tensors Number of input tensors to reduce
 * @param n_elems     Number of BF16 elements per tensor
 * @param sm_version  SM version of the active device (86, 90, 120, …)
 * @param stream      CUDA stream to launch on
 */
void launch_fused_bf16_reduce(__nv_bfloat16* output,
                               const __nv_bfloat16* const* inputs,
                               int num_tensors,
                               size_t n_elems,
                               int sm_version,
                               cudaStream_t stream);

/**
 * compute_hetero_shard_ranges
 *
 * Given N GPU tiers, compute the element range each tier should reduce and
 * store during a heterogeneous reduce-scatter.  Higher-SM-version devices
 * receive proportionally larger shards.
 *
 * Weight assignment:  SM 12.0 → 4, SM 9.0 → 3, SM 8.6 → 1.
 * All shard boundaries are aligned to 8-element (kVecWidth) boundaries.
 *
 * @param tiers       [in]  Array of tier descriptors
 * @param num_tiers   Number of tiers
 * @param total_elems Total BF16 elements in the gradient tensor
 * @param out_offsets [out] Per-tier start offset (caller-allocated, length num_tiers)
 * @param out_counts  [out] Per-tier element count (caller-allocated, length num_tiers)
 */
void compute_hetero_shard_ranges(const HeteroTierDesc* tiers,
                                  int num_tiers,
                                  size_t total_elems,
                                  size_t* out_offsets,
                                  size_t* out_counts);

/**
 * launch_hetero_reduce_scatter
 *
 * Launches the fused reduce-scatter kernel on the *current* device.
 * All `num_tensors` inputs are read in full, but only the local shard
 * [shard_offset, shard_offset + shard_count) is reduced and written to
 * `output`.  The output buffer is shard_count elements (not full tensor).
 *
 * @param output       [out] BF16 output buffer, length shard_count
 * @param inputs       [in]  Array of device pointers to BF16 input tensors
 * @param num_tensors  Number of input tensors
 * @param shard_offset Starting element index in the full gradient tensor
 * @param shard_count  Number of elements this device writes
 * @param sm_version   SM version of the current device
 * @param stream       CUDA stream
 */
void launch_hetero_reduce_scatter(__nv_bfloat16* output,
                                   const __nv_bfloat16* const* inputs,
                                   int num_tensors,
                                   size_t shard_offset,
                                   size_t shard_count,
                                   int sm_version,
                                   cudaStream_t stream);

/**
 * fused_swiglu_ln
 *
 * Fused SwiGLU activation followed by RMS LayerNorm.
 * Specialised at compile time for SM 8.6, 9.0, and 12.0 via template dispatch.
 *
 * gate_proj and up_proj are the two halves produced by a gated-MLP linear
 * layer (each shape [batch, hidden_size]).  The output is the SwiGLU result
 * after in-place RMS-LN normalisation.
 *
 * @param output     [out] BF16 output [batch, hidden_size]
 * @param gate_proj  [in]  BF16 gate projection  [batch, hidden_size]
 * @param up_proj    [in]  BF16 up   projection  [batch, hidden_size]
 * @param ln_weight  [in]  FP32 LN weight [hidden_size]
 * @param batch      Batch size (rows)
 * @param hidden     Hidden size (cols, must be divisible by 8)
 * @param eps        LayerNorm epsilon
 * @param sm_version SM version of the active device
 * @param stream     CUDA stream
 */
void launch_fused_swiglu_ln(__nv_bfloat16* output,
                             const __nv_bfloat16* gate_proj,
                             const __nv_bfloat16* up_proj,
                             const float* ln_weight,
                             int batch,
                             int hidden,
                             float eps,
                             int sm_version,
                             cudaStream_t stream);
