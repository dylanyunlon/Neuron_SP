// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #23: fused RoPE for heterogeneous head counts

/*
 * fused_rope.h — Public API for fused Rotary Position Embedding kernels.
 *
 * This header wraps the lower-level hetero_reduce RoPE kernels and adds
 * a fused Q+K simultaneous apply path for GQA models where Q and K have
 * different head counts (e.g. Llama-2: 32 Q heads, 8 KV heads).
 *
 * The fused Q+K kernel processes both tensors in a single launch, sharing
 * the cos/sin cache reads across Q and K heads for better L2 utilisation.
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

/**
 * launch_fused_rope_qk
 *
 * Simultaneously applies RoPE to both Q and K tensors in a single kernel
 * launch.  This is more efficient than two separate launches because:
 *   1. cos/sin cache is read once and shared across Q and K.
 *   2. Only one kernel launch overhead (matters on PCIe-bound systems).
 *   3. Better SM utilisation when Q and K head counts differ (GQA).
 *
 * @param q_output    [out] BF16 rotated Q [B, S, num_heads_q, D]
 * @param k_output    [out] BF16 rotated K [B, S, num_heads_kv, D]
 * @param q_input     [in]  BF16 Q input   [B, S, num_heads_q, D]
 * @param k_input     [in]  BF16 K input   [B, S, num_heads_kv, D]
 * @param cos_cache   [in]  FP32 cos cache [S, D/2] (nullptr for cacheless)
 * @param sin_cache   [in]  FP32 sin cache [S, D/2] (nullptr for cacheless)
 * @param batch       Batch size
 * @param seq_len     Sequence length
 * @param num_heads_q Number of Q attention heads
 * @param num_heads_kv Number of K/V attention heads (GQA)
 * @param head_dim    Head dimension (must be even)
 * @param neox_style  true → NeoX/Llama style, false → GPT-J interleaved
 * @param base        RoPE base frequency (cacheless mode)
 * @param pos_offset  Position offset (cacheless mode)
 * @param sm_version  SM version (86, 90, 120)
 * @param stream      CUDA stream
 */
void launch_fused_rope_qk(__nv_bfloat16*       q_output,
                           __nv_bfloat16*       k_output,
                           const __nv_bfloat16* q_input,
                           const __nv_bfloat16* k_input,
                           const float*         cos_cache,
                           const float*         sin_cache,
                           int                  batch,
                           int                  seq_len,
                           int                  num_heads_q,
                           int                  num_heads_kv,
                           int                  head_dim,
                           bool                 neox_style,
                           float                base,
                           int                  pos_offset,
                           int                  sm_version,
                           cudaStream_t         stream);
