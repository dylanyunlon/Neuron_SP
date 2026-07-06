// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_attention.h  —  NeurIPS 2026  DES-LOC + AutoSP  (addresses #135)
 *
 * Fused scaled dot-product attention kernel for heterogeneous GPU clusters:
 *   2× A6000 (SM8.6) + 1× H100 (SM9.0) + 2× Blackwell (SM12.0)
 *
 * Design overview:
 *   - Flash-Attention-style online softmax tiled over K dimension.
 *     O(N) HBM reads vs. O(N²) for the naive materialise-then-softmax path.
 *   - SM-conditional tile sizes via KernelPolicy<SmVer>:
 *       SM8.6: Br=64,  Bc=64  (A6000, 48 GB GDDR6X)
 *       SM9.0: Br=128, Bc=128 (H100,  80 GB HBM3)
 *       SM12.0: Br=128, Bc=64 (Blackwell)
 *   - Multi-head / GQA: kv_head = q_head / gqa_ratio (runtime remapping).
 *   - Causal masking: applied inside the K-tile loop, no extra memory.
 *   - SWA support: pass window_left / window_right for sliding-window attention.
 *   - LSE (log-sum-exp) saved during forward for backward recompute.
 *
 * See fused_attention.cu for implementation details.
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ===========================================================================
// Forward pass
// ===========================================================================

/**
 * launch_fused_attention
 *
 * Fused BF16 multi-head scaled dot-product attention.  Computes:
 *
 *   O[b, h, i, :] = softmax(Q[b,h,i,:] · Kᵀ[b,hkv,:,:] * scale) · V[b,hkv,:,:]
 *
 * The log-sum-exp (LSE) is saved for the backward recompute path.
 *
 * Memory layout for all tensors: [batch, heads, seq, head_dim] (packed, row-major).
 *
 * For GQA:  num_kv_heads < num_q_heads; kv_head = q_head / gqa_ratio.
 * For MHA:  num_kv_heads == num_q_heads.
 * For MQA:  num_kv_heads == 1.
 *
 * SWA (sliding-window attention):
 *   Pass window_left  >= 0 to limit attention to the last `window_left` past tokens.
 *   Pass window_right >= 0 to limit attention to the next `window_right` future tokens.
 *   Pass -1 for either to allow full (unbounded) attention in that direction.
 *   causal=true is equivalent to window_left=-1, window_right=0.
 *
 * @param output         [out] BF16  [batch, num_q_heads, seq_q, head_dim]
 * @param lse            [out] FP32  [batch, num_q_heads, seq_q]  or nullptr to skip
 * @param query          [in]  BF16  [batch, num_q_heads, seq_q, head_dim]
 * @param key            [in]  BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param value          [in]  BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param batch_size     Batch dimension B
 * @param num_q_heads    Number of query heads Hq
 * @param num_kv_heads   Number of key/value heads Hkv (1 for MQA, Hq for MHA)
 * @param seq_q          Query sequence length
 * @param seq_k          Key/value sequence length
 * @param head_dim       Per-head dimension D  (must be divisible by 2)
 * @param softmax_scale  Attention scale, typically 1/√D
 * @param causal         True for causal (decoder) attention
 * @param window_left    SWA left window (-1 = unbounded)
 * @param window_right   SWA right window (-1 = unbounded)
 * @param dropout_p      Attention dropout probability (0.0 to disable)
 * @param philox_seed    Philox RNG seed (must match Python CUDARNGTracker)
 * @param philox_offset  Philox RNG offset (per-batch / per-layer offset)
 * @param sm_version     SM version of the active device (86, 90, 120, …)
 * @param stream         CUDA stream
 */
void launch_fused_attention(
    __nv_bfloat16*       output,
    float*               lse,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int    batch_size,
    int    num_q_heads,
    int    num_kv_heads,
    int    seq_q,
    int    seq_k,
    int    head_dim,
    float  softmax_scale,
    bool   causal,
    int    window_left,
    int    window_right,
    float  dropout_p,
    uint64_t philox_seed,
    uint64_t philox_offset,
    int      sm_version,
    cudaStream_t stream);

// ===========================================================================
// Backward pass
// ===========================================================================

/**
 * launch_fused_attention_bwd
 *
 * Backward pass for fused attention.  Recomputes softmax probabilities from
 * saved Q, K, V, output O, and log-sum-exp (LSE), then computes:
 *
 *   dV  = Pᵀ  dO
 *   dS  = P ⊙ (dO·Vᵀ − Di)   where Di = rowsum(dO ⊙ O)
 *   dQ  = dS  K * scale
 *   dK  = dSᵀ Q * scale
 *
 * Gradients are accumulated (+=) into pre-allocated output tensors.
 * The caller must zero-initialise dq_out, dk_out, dv_out beforehand.
 *
 * NOTE: For the initial #135 integration, the Python-level recompute checkpoint
 * path in dot_product_attention.py (torch.autograd.checkpoint) is the default
 * backward.  This kernel provides an explicit fused alternative when called
 * from a custom torch.autograd.Function.  See issue #138 for full wiring.
 *
 * @param dq_out         [in/out] BF16  [batch, num_q_heads, seq_q, head_dim]
 * @param dk_out         [in/out] BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param dv_out         [in/out] BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param query          [in]  BF16  [batch, num_q_heads, seq_q, head_dim]
 * @param key            [in]  BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param value          [in]  BF16  [batch, num_kv_heads, seq_k, head_dim]
 * @param output         [in]  BF16  [batch, num_q_heads, seq_q, head_dim]  (fwd output)
 * @param d_output       [in]  BF16  [batch, num_q_heads, seq_q, head_dim]  (upstream dO)
 * @param lse            [in]  FP32  [batch, num_q_heads, seq_q]             (from fwd)
 * @param batch_size     Batch dimension B
 * @param num_q_heads    Number of query heads
 * @param num_kv_heads   Number of key/value heads
 * @param seq_q          Query sequence length
 * @param seq_k          Key/value sequence length
 * @param head_dim       Per-head dimension D
 * @param softmax_scale  Attention scale, typically 1/√D
 * @param causal         True for causal attention
 * @param sm_version     SM version of the active device (86, 90, 120, …)
 * @param stream         CUDA stream
 */
void launch_fused_attention_bwd(
    __nv_bfloat16*       dq_out,
    __nv_bfloat16*       dk_out,
    __nv_bfloat16*       dv_out,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const __nv_bfloat16* output,
    const __nv_bfloat16* d_output,
    const float*         lse,
    int    batch_size,
    int    num_q_heads,
    int    num_kv_heads,
    int    seq_q,
    int    seq_k,
    int    head_dim,
    float  softmax_scale,
    bool   causal,
    int    sm_version,
    cudaStream_t stream);
