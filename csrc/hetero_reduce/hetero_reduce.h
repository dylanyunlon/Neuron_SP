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
