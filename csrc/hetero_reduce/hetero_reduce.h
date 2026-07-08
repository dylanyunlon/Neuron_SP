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

/**
 * launch_fused_swiglu_ln_fwd_save
 *
 * Forward pass variant that additionally writes rms_inv[row] into a
 * caller-allocated FP32 device buffer (length = batch).  Pass this buffer
 * to launch_fused_swiglu_ln_backward to avoid recomputing the RMS
 * denominator on the backward pass.
 *
 * @param output       [out] BF16 output        [batch, hidden]
 * @param rms_inv_out  [out] FP32 rms_inv       [batch]
 * @param gate_proj    [in]  BF16 gate          [batch, hidden]
 * @param up_proj      [in]  BF16 up            [batch, hidden]
 * @param ln_weight    [in]  FP32 LN weight     [hidden]
 * @param batch        Batch size
 * @param hidden       Hidden size (divisible by 8)
 * @param eps          LayerNorm epsilon
 * @param sm_version   SM version (86 / 90 / 120)
 * @param stream       CUDA stream
 */
void launch_fused_swiglu_ln_fwd_save(__nv_bfloat16*       output,
                                      float*               rms_inv_out,
                                      const __nv_bfloat16* gate_proj,
                                      const __nv_bfloat16* up_proj,
                                      const float*         ln_weight,
                                      int batch,
                                      int hidden,
                                      float eps,
                                      int sm_version,
                                      cudaStream_t stream);

/**
 * launch_fused_swiglu_ln_backward
 *
 * Backward pass for fused SwiGLU + RMSNorm.  Computes gradients for
 * gate_proj, up_proj, and ln_weight given the upstream gradient d_output.
 *
 * The caller must:
 *   1. Have run launch_fused_swiglu_ln_fwd_save to obtain rms_inv_buf.
 *   2. Zero d_ln_weight before the first backward call (gradients accumulate
 *      via atomicAdd across the batch).
 *
 * Math (per row i):
 *   dot_i     = Σ_j  d_out[i,j] · w_j · swiglu(gate[i,j], up[i,j])
 *   d_s_j     = rms_inv_i · w_j · d_out[i,j]
 *               - rms_inv_i³ · swiglu_j · dot_i / hidden
 *   d_gate_j  = d_s_j · up_j · σ(gate_j) · (1 + gate_j · (1 − σ(gate_j)))
 *   d_up_j    = d_s_j · gate_j · σ(gate_j)
 *   d_w_j    += d_out[i,j] · swiglu[i,j] · rms_inv_i
 *
 * @param d_gate       [out] BF16 gradient for gate_proj [batch, hidden]
 * @param d_up         [out] BF16 gradient for up_proj   [batch, hidden]
 * @param d_ln_weight  [out] FP32 gradient for ln_weight [hidden] — ACCUMULATES
 * @param d_output     [in]  BF16 upstream gradient      [batch, hidden]
 * @param gate_proj    [in]  BF16 forward gate input     [batch, hidden]
 * @param up_proj      [in]  BF16 forward up input       [batch, hidden]
 * @param ln_weight    [in]  FP32 LN weight              [hidden]
 * @param rms_inv_buf  [in]  FP32 rms_inv from fwd_save  [batch]
 * @param batch        Batch size
 * @param hidden       Hidden size (divisible by 8)
 * @param eps          LayerNorm epsilon (must match forward call)
 * @param sm_version   SM version (86 / 90 / 120)
 * @param stream       CUDA stream
 */
void launch_fused_swiglu_ln_backward(__nv_bfloat16*       d_gate,
                                      __nv_bfloat16*       d_up,
                                      float*               d_ln_weight,
                                      const __nv_bfloat16* d_output,
                                      const __nv_bfloat16* gate_proj,
                                      const __nv_bfloat16* up_proj,
                                      const float*         ln_weight,
                                      const float*         rms_inv_buf,
                                      int batch,
                                      int hidden,
                                      float eps,
                                      int sm_version,
                                      cudaStream_t stream);

// ===========================================================================
// fused_rope_hetero — Heterogeneous-head-count RoPE kernel
// ===========================================================================

/**
 * launch_rope_cache
 *
 * Precomputes the cosine / sine lookup tables for RoPE on device.
 * cos_cache[s, k] = cos(theta_k * (s + pos_offset))
 * sin_cache[s, k] = sin(theta_k * (s + pos_offset))
 * where theta_k = base^(-2k / head_dim).
 *
 * @param cos_cache   [out] FP32 device buffer [seq_len, head_dim/2]
 * @param sin_cache   [out] FP32 device buffer [seq_len, head_dim/2]
 * @param seq_len     Number of sequence positions to fill
 * @param head_dim    Full head dimension (half_dim = head_dim/2 pairs)
 * @param base        RoPE base frequency (default 10000.f)
 * @param pos_offset  Global position offset for packed sequences
 * @param stream      CUDA stream
 */
void launch_rope_cache(float*       cos_cache,
                       float*       sin_cache,
                       int          seq_len,
                       int          head_dim,
                       float        base,
                       int          pos_offset,
                       cudaStream_t stream);

/**
 * launch_fused_rope_hetero
 *
 * Applies Rotary Position Embedding to a BF16 query/key tensor.
 * Supports heterogeneous head counts (32/64/128 heads) parametrised at
 * runtime.  SM-dispatched for SM 8.6, 9.0, and 12.0.
 *
 * @param output      [out] BF16 tensor [B, S, H, D] (may alias input)
 * @param input       [in]  BF16 tensor [B, S, H, D]
 * @param cos_cache   [in]  FP32 device buffer [S, D/2]
 * @param sin_cache   [in]  FP32 device buffer [S, D/2]
 * @param batch       Batch size
 * @param seq_len     Sequence length
 * @param num_heads   Number of attention heads
 * @param head_dim    Head dimension (must be even)
 * @param neox_style  true → Llama/NeoX style; false → GPT-J interleaved
 * @param sm_version  SM version of active device (86, 90, 120)
 * @param stream      CUDA stream
 */
void launch_fused_rope_hetero(__nv_bfloat16*       output,
                               const __nv_bfloat16* input,
                               const float*         cos_cache,
                               const float*         sin_cache,
                               int                  batch,
                               int                  seq_len,
                               int                  num_heads,
                               int                  head_dim,
                               bool                 neox_style,
                               int                  sm_version,
                               cudaStream_t         stream);

// ===========================================================================
// pcie_adaptive_allreduce — PCIe-aware gradient bucketing and allreduce
// ===========================================================================

/**
 * PcieGradChunk
 *
 * Describes one contiguous shard of gradient data for packing.
 * Declared here for use in both pcie_adaptive_allreduce.cu and binding.cpp.
 */
struct PcieGradChunk {
    const __nv_bfloat16* src;    // device pointer to gradient tensor
    size_t               offset; // starting element index within src
    size_t               length; // number of BF16 elements in this chunk
};

/**
 * compute_pcie_bucket_size
 *
 * Returns the recommended bucket size in bytes for gradient packing,
 * targeting ~1 ms of PCIe transfer latency at the given bandwidth.
 *
 * @param pcie_bw_gbps  Measured or estimated PCIe bandwidth in GB/s
 * @returns             Bucket size in bytes (multiple of 16, clamped)
 */
size_t compute_pcie_bucket_size(float pcie_bw_gbps);

/**
 * launch_pcie_gradient_pack
 *
 * Gathers non-contiguous gradient shards from multiple tensors into
 * a flat BF16 bucket buffer (device-side gather).
 *
 * @param bucket        [out] Flat BF16 device buffer [bucket_elems]
 * @param chunks        [in]  Array of chunk descriptors (host pointer)
 * @param num_chunks    Number of gradient shards to pack
 * @param bucket_elems  Total elements in output bucket (sum of chunk lengths)
 * @param sm_version    SM version of active device
 * @param stream        CUDA stream
 */
void launch_pcie_gradient_pack(__nv_bfloat16*       bucket,
                                const PcieGradChunk* chunks,
                                int                  num_chunks,
                                size_t               bucket_elems,
                                int                  sm_version,
                                cudaStream_t         stream);

/**
 * launch_pcie_ring_reduce
 *
 * Ring-allreduce reduce phase: accumulates peer bucket `src` into
 * local accumulator `dst` using BF16→FP32→BF16 precision.
 *
 * @param dst         [in/out] BF16 local accumulator [n_elems]
 * @param src         [in]     BF16 incoming peer bucket [n_elems]
 * @param n_elems     Number of elements (must be divisible by 8)
 * @param sm_version  SM version
 * @param stream      CUDA stream
 */
void launch_pcie_ring_reduce(__nv_bfloat16*       dst,
                              const __nv_bfloat16* src,
                              size_t               n_elems,
                              int                  sm_version,
                              cudaStream_t         stream);

/**
 * launch_pcie_allreduce_finalise
 *
 * Divides the allreduce sum by world_size and optionally writes to a
 * separate output buffer.
 *
 * @param out         [out] BF16 output buffer [n_elems]
 * @param src         [in]  BF16 sum buffer [n_elems]
 * @param n_elems     Number of elements (divisible by 8)
 * @param world_size  Number of participating GPUs
 * @param sm_version  SM version
 * @param stream      CUDA stream
 */
void launch_pcie_allreduce_finalise(__nv_bfloat16*       out,
                                     const __nv_bfloat16* src,
                                     size_t               n_elems,
                                     int                  world_size,
                                     int                  sm_version,
                                     cudaStream_t         stream);

// ===========================================================================
// tier_activation_offload — Tier-aware activation checkpoint offload
// ===========================================================================

/**
 * compute_offload_budget
 *
 * Computes how many activation bytes a tier must offload to host/peer
 * given its current free VRAM and a safety headroom fraction.
 *
 * @param total_act_bytes  Total activation storage required by the model
 * @param vram_free_bytes  Current free VRAM on this tier's device
 * @param headroom_frac    Fraction of free VRAM to keep unused (e.g. 0.10)
 * @returns bytes to offload, rounded to 16-byte boundary (0 if fits in VRAM)
 */
size_t compute_offload_budget(size_t total_act_bytes,
                               size_t vram_free_bytes,
                               float  headroom_frac);

/**
 * launch_activation_pack
 *
 * Gathers activation tensors into a flat BF16 offload buffer.
 * Output layout: [tensor_0 | tensor_1 | ... | tensor_{n-1}]
 *
 * @param output        [out] Flat BF16 buffer [num_tensors * tensor_elems]
 * @param inputs        [in]  Array of device pointers to activation tensors
 * @param num_tensors   Number of tensors to pack
 * @param tensor_elems  Elements per tensor (divisible by 8)
 * @param sm_version    SM version
 * @param stream        CUDA stream
 */
void launch_activation_pack(__nv_bfloat16*              output,
                             const __nv_bfloat16* const* inputs,
                             int                         num_tensors,
                             size_t                      tensor_elems,
                             int                         sm_version,
                             cudaStream_t                stream);

/**
 * launch_activation_unpack
 *
 * Scatters a flat BF16 buffer back to individual activation tensors.
 * Inverse of launch_activation_pack.
 *
 * @param outputs       [out] Array of device pointers to destination tensors
 * @param flat          [in]  Flat BF16 buffer [num_tensors * tensor_elems]
 * @param num_tensors   Number of tensors to unpack
 * @param tensor_elems  Elements per tensor (divisible by 8)
 * @param sm_version    SM version
 * @param stream        CUDA stream
 */
void launch_activation_unpack(__nv_bfloat16* const*  outputs,
                               const __nv_bfloat16*   flat,
                               int                    num_tensors,
                               size_t                 tensor_elems,
                               int                    sm_version,
                               cudaStream_t           stream);

/**
 * launch_quantise_fp16_to_int8
 *
 * Block-wise INT8 quantisation of a BF16 activation buffer.
 * Tile size = 128 elements.  Scale = absmax / 127 per tile.
 * Halves PCIe traffic when offloading to host pinned memory.
 *
 * @param output   [out] INT8 quantised buffer [n_elems]
 * @param scales   [out] FP32 per-tile scales  [ceil(n_elems / 128)]
 * @param input    [in]  BF16 input buffer [n_elems]
 * @param n_elems  Number of elements
 * @param stream   CUDA stream
 */
void launch_quantise_fp16_to_int8(int8_t*              output,
                                   float*               scales,
                                   const __nv_bfloat16* input,
                                   size_t               n_elems,
                                   cudaStream_t         stream);

/**
 * launch_dequantise_int8_to_fp16
 *
 * Inverse of launch_quantise_fp16_to_int8.  Called during activation
 * prefetch before recomputation on the backward pass.
 *
 * @param output   [out] BF16 dequantised buffer [n_elems]
 * @param input    [in]  INT8 quantised buffer [n_elems]
 * @param scales   [in]  FP32 per-tile scales  [ceil(n_elems / 128)]
 * @param n_elems  Number of elements
 * @param stream   CUDA stream
 */
void launch_dequantise_int8_to_fp16(__nv_bfloat16* output,
                                     const int8_t*  input,
                                     const float*   scales,
                                     size_t         n_elems,
                                     cudaStream_t   stream);

// ===========================================================================
// Additional API — Worker-12 (Opus) additions
// ===========================================================================

/**
 * hetero_bucket_size_elems
 *
 * Returns the policy-recommended gradient bucket size in BF16 elements for
 * a given SM version.  Derived from KernelPolicy<SmVer>::kBucketElems.
 *
 * H100  (SM 9.0):     4M elements (32 MB) — large L2, maximise reuse
 * A6000 (SM 8.6):   512K elements  (4 MB) — small L2, avoid thrashing
 * Blackwell (SM12.0): 2M elements (16 MB) — moderate L2
 *
 * @param sm_version  SM version (86, 90, 120, …)
 * @returns           Recommended bucket size in BF16 elements
 */
size_t hetero_bucket_size_elems(int sm_version);

/**
 * compute_adaptive_chunk_size
 *
 * Computes adaptive ring-allreduce chunk size targeting kTargetOverlapMs ms
 * of PCIe transfer time per chunk, based on measured or estimated bandwidth.
 *
 * @param pcie_bw_gbps  PCIe bandwidth in GB/s (from probe or estimate)
 * @returns             Chunk size in bytes, aligned to 16 bytes
 */
size_t compute_adaptive_chunk_size(float pcie_bw_gbps);

/**
 * probe_pcie_bandwidth
 *
 * Sends a kProbeSizeBytes (4 MB) test buffer from src_device to dst_device
 * via cudaMemcpyPeerAsync, times it with CUDA events, and returns the
 * measured bandwidth in GB/s.  Caches results in a static table.
 *
 * Requires peer access to be enabled (cudaDeviceEnablePeerAccess).
 * Falls back to 8.0 GB/s on allocation failure.
 *
 * @param src_device  CUDA device ordinal of sender
 * @param dst_device  CUDA device ordinal of receiver
 * @returns           Measured PCIe bandwidth in GB/s
 */
float probe_pcie_bandwidth(int src_device, int dst_device);

/**
 * launch_pcie_ring_reduce_step
 *
 * Single ring-reduce step: accum_buf[i] += recv_buf[i] (in FP32, stored BF16).
 * Used by the double-buffered pipeline orchestration.
 *
 * @param accum_buf      [in/out] BF16 accumulator [chunk_elems]
 * @param recv_buf       [in]     BF16 received chunk [chunk_elems]
 * @param chunk_elems    Number of BF16 elements
 * @param sm_version     SM version for kernel dispatch
 * @param compute_stream CUDA stream for the reduce kernel
 */
void launch_pcie_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream);

// ===========================================================================
// fused_layernorm_residual — addresses #110
// ===========================================================================

/**
 * launch_fused_layernorm_residual
 *
 * Fused residual addition + RMS LayerNorm kernel.
 *
 * For each row i of [batch × hidden]:
 *   residual_i[j] += input_i[j]
 *   output_i[j]   = residual_i[j] * ln_weight[j]
 *                   * rsqrt(mean(residual_i²) + ε)
 *
 * The residual stream is updated in-place; the normalized result is written
 * to a separate output buffer (which may alias residual for single-buffer mode).
 *
 * @param output      [out] BF16 LN output [batch, hidden]
 * @param residual    [in/out] BF16 residual stream [batch, hidden] (updated in-place)
 * @param input       [in]  BF16 new contribution [batch, hidden]
 * @param ln_weight   [in]  FP32 RMSNorm scale [hidden]
 * @param batch       Batch size
 * @param hidden      Hidden size (must be divisible by 8)
 * @param eps         RMSNorm epsilon
 * @param sm_version  SM version of the active device (86, 90, 120)
 * @param stream      CUDA stream
 */
void launch_fused_layernorm_residual(__nv_bfloat16*       output,
                                      __nv_bfloat16*       residual,
                                      const __nv_bfloat16* input,
                                      const float*         ln_weight,
                                      int                  batch,
                                      int                  hidden,
                                      float                eps,
                                      int                  sm_version,
                                      cudaStream_t         stream);

// ===========================================================================
// cross_entropy_tp — Tensor-parallel cross-entropy loss  (addresses #110)
// ===========================================================================

/**
 * launch_cross_entropy_tp_forward
 *
 * Phase-1 forward pass: compute local (max, sum_exp, label_logit) scalars
 * for each sample from this TP rank's vocab shard.
 *
 * The caller reduces these across TP ranks:
 *   global_max     = AllReduce_max(local_max)
 *   global_sum_exp = AllReduce_sum(local_sum_exp * exp(local_max - global_max))
 *   global_logit   = AllReduce_sum(local_logit)
 *
 * @param local_max      [out] FP32 [batch] — max logit in this shard
 * @param local_sum_exp  [out] FP32 [batch] — Σ exp(logit - local_max)
 * @param local_logit    [out] FP32 [batch] — logit at label position (0 if not in shard)
 * @param logits         [in]  BF16 [batch, v_local] — this rank's logit shard
 * @param labels         [in]  int32 [batch] — global vocab label per sample
 * @param batch          Batch size
 * @param v_local        Local vocabulary size (= V / tp_size)
 * @param shard_offset   Global vocab index of logits[:,0]
 * @param sm_version     SM version of the active device (86, 90, 120)
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_forward(float*               local_max,
                                      float*               local_sum_exp,
                                      float*               local_logit,
                                      const __nv_bfloat16* logits,
                                      const int*           labels,
                                      int                  batch,
                                      int                  v_local,
                                      int                  shard_offset,
                                      int                  sm_version,
                                      cudaStream_t         stream);

/**
 * launch_cross_entropy_tp_loss
 *
 * Phase-2: compute per-sample cross-entropy loss from globally-reduced scalars.
 *
 *   loss[i] = log(global_sum_exp[i]) + global_max[i] - global_logit[i]
 *
 * @param loss           [out] FP32 [batch] — per-sample CE loss
 * @param global_max     [in]  FP32 [batch] — globally reduced max
 * @param global_sum_exp [in]  FP32 [batch] — globally reduced sum of exp
 * @param global_logit   [in]  FP32 [batch] — globally reduced label logit
 * @param batch          Batch size
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_loss(float*       loss,
                                   const float* global_max,
                                   const float* global_sum_exp,
                                   const float* global_logit,
                                   int          batch,
                                   cudaStream_t stream);

/**
 * launch_cross_entropy_tp_backward
 *
 * Backward pass: compute softmax gradient w.r.t. this rank's logit shard.
 *
 *   d_logits[row, j] = (softmax(logit)[row,j] -
 *                       1{shard_offset + j == label[row]}) / batch_size
 *
 * Written in-place into d_logits (may alias logits after forward is complete).
 *
 * @param d_logits       [out] BF16 [batch, v_local] — gradient output
 * @param logits         [in]  BF16 [batch, v_local] — forward logit shard
 * @param labels         [in]  int32 [batch] — global label indices
 * @param global_max     [in]  FP32 [batch] — globally reduced max per sample
 * @param log_sum_exp    [in]  FP32 [batch] — log(global_sum_exp) per sample
 * @param batch          Batch size
 * @param v_local        Local vocabulary size
 * @param shard_offset   Global vocab index of logits[:,0]
 * @param inv_batch      1.f / batch_size (pre-computed by caller)
 * @param sm_version     SM version of the active device (86, 90, 120)
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_backward(__nv_bfloat16*       d_logits,
                                       const __nv_bfloat16* logits,
                                       const int*           labels,
                                       const float*         global_max,
                                       const float*         log_sum_exp,
                                       int                  batch,
                                       int                  v_local,
                                       int                  shard_offset,
                                       float                inv_batch,
                                       int                  sm_version,
                                       cudaStream_t         stream);

// ===========================================================================
// VocabPartition — heterogeneous (non-uniform) TP vocab split  (#141)
//
// In homogeneous TP all ranks receive V/tp_size tokens.  In a heterogeneous
// cluster (mixed SM8.6/SM9.0/SM12.0) it is advantageous to assign more vocab
// tokens to faster GPUs.  VocabPartition captures the arbitrary shard
// boundaries so the kernel can work correctly regardless of shard size.
//
// Design: two-pass approach
//   Pass 1 (local)  — each rank runs cross_entropy_tp_forward_hetero_kernel
//                     over its own shard of size v_local[rank].  Produces
//                     (local_max, local_sum_exp, local_logit) — identical
//                     semantics to the uniform path but v_local is now
//                     per-rank rather than V/tp_size.
//   Pass 2 (global) — caller performs:
//                       global_max     = AllReduce_max(local_max)
//                       global_sum_exp = AllReduce_sum(
//                                          local_sum_exp
//                                          * exp(local_max - global_max))
//                       global_logit   = AllReduce_sum(local_logit)
//                     and then calls launch_cross_entropy_tp_loss as before.
//
// The two new launch wrappers below are drop-in replacements for the uniform
// versions; the only difference is that v_local (the shard width) is now
// supplied at call-time rather than being an implicit V/tp_size constant.
// ===========================================================================

/**
 * VocabPartition
 *
 * Describes this rank's slice of the full vocabulary in a non-uniform TP
 * split.  Populated by compute_hetero_vocab_partition() below, or filled
 * directly by the caller for custom splits.
 *
 * Fields:
 *   v_local       — number of vocab tokens assigned to this rank
 *   shard_offset  — global vocab index of logits[:,0] on this rank
 *   tp_size       — total number of TP ranks (informational)
 *   rank          — index of this rank within the TP group
 */
struct VocabPartition {
    int v_local;       // vocab tokens on this rank
    int shard_offset;  // first global vocab index owned by this rank
    int tp_size;       // total TP ranks
    int rank;          // this rank's index in [0, tp_size)
};

/**
 * compute_hetero_vocab_partition
 *
 * Compute per-rank VocabPartition entries for a non-uniform vocabulary split.
 * Tokens are distributed proportionally to each rank's SM-version weight:
 *   SM 12.0 (Blackwell) → weight 4
 *   SM  9.0 (H100)      → weight 3
 *   SM  8.6 (A6000)     → weight 1
 * All shard boundaries are aligned to 8-element (kVecBF16) boundaries.
 * The last rank absorbs any residual from alignment.
 *
 * @param out_parts   [out] Array of VocabPartition, length tp_size (caller-alloc)
 * @param sm_versions [in]  SM version of each rank, length tp_size
 * @param tp_size     Number of TP ranks
 * @param vocab_size  Total vocabulary size V (must be divisible by 8 for best perf)
 */
void compute_hetero_vocab_partition(VocabPartition* out_parts,
                                     const int*      sm_versions,
                                     int             tp_size,
                                     int             vocab_size);

/**
 * launch_cross_entropy_tp_forward_hetero
 *
 * Phase-1 forward pass for a non-uniform vocab shard described by `vp`.
 * Identical semantics to launch_cross_entropy_tp_forward; the shard width
 * and offset are taken from `vp` rather than separate parameters.
 *
 * Two-pass local-max / local-sum strategy:
 *   Pass A (this kernel): compute local (max, sum_exp, label_logit) over
 *           logits[0..batch-1][0..vp.v_local-1].
 *   Pass B (caller, cross-rank): AllReduce to obtain global scalars, then
 *           call launch_cross_entropy_tp_loss as usual.
 *
 * @param local_max      [out] FP32 [batch]
 * @param local_sum_exp  [out] FP32 [batch]
 * @param local_logit    [out] FP32 [batch]  (0.0 for samples whose label is not in vp)
 * @param logits         [in]  BF16 [batch, vp.v_local]
 * @param labels         [in]  int32 [batch] global vocab label per sample
 * @param batch          Batch size
 * @param vp             VocabPartition for this rank
 * @param sm_version     SM version of the active device (86, 90, 120)
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_forward_hetero(
    float*               local_max,
    float*               local_sum_exp,
    float*               local_logit,
    const __nv_bfloat16* logits,
    const int*           labels,
    int                  batch,
    VocabPartition       vp,
    int                  sm_version,
    cudaStream_t         stream);

/**
 * launch_cross_entropy_tp_backward_hetero
 *
 * Backward pass for a non-uniform vocab shard described by `vp`.
 * Identical semantics to launch_cross_entropy_tp_backward.
 *
 * @param d_logits       [out] BF16 [batch, vp.v_local]
 * @param logits         [in]  BF16 [batch, vp.v_local]
 * @param labels         [in]  int32 [batch]
 * @param global_max     [in]  FP32 [batch]
 * @param log_sum_exp    [in]  FP32 [batch]
 * @param batch          Batch size
 * @param vp             VocabPartition for this rank
 * @param inv_batch      1.f / batch_size
 * @param sm_version     SM version (86, 90, 120)
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_backward_hetero(
    __nv_bfloat16*       d_logits,
    const __nv_bfloat16* logits,
    const int*           labels,
    const float*         global_max,
    const float*         log_sum_exp,
    int                  batch,
    VocabPartition       vp,
    float                inv_batch,
    int                  sm_version,
    cudaStream_t         stream);

// ===========================================================================
// fused_gradient_allreduce — INT8 mixed-precision compressed all-reduce
//   for heterogeneous PCIe topology  (fused_gradient_allreduce.cu)
// ===========================================================================

/**
 * launch_gradient_compress
 *
 * Phase 1: compress a BF16 gradient tensor to INT8 with per-block FP32 scales.
 *
 * Each kBlockElems-element block of the gradient is independently quantised:
 *   scale[b]    = max|x[b]| / 127
 *   out_int8[i] = round(x[i] / scale[block(i)])  clamped to [-128, 127]
 *
 * Uses cub::BlockReduce to compute per-block ℓ∞ norms efficiently.
 *
 * @param out_int8    [out] INT8 device buffer, n_elems bytes
 * @param out_scale   [out] FP32 device buffer, ceil(n_elems/256) floats
 * @param input       [in]  BF16 device buffer, n_elems elements
 * @param n_elems     Number of BF16 gradient elements
 * @param sm_version  SM version (86, 90, 120)
 * @param stream      CUDA stream
 */
void launch_gradient_compress(int8_t*               out_int8,
                               float*                out_scale,
                               const __nv_bfloat16*  input,
                               size_t                n_elems,
                               int                   sm_version,
                               cudaStream_t          stream);

/**
 * launch_int8_ring_reduce_step
 *
 * Fused INT8 ring-allreduce accumulation: dequantise dst and src (with their
 * respective per-block scales), sum them, re-quantise the result and update
 * dst_int8 and dst_scale in-place.  Keeps the entire ring reduce in INT8,
 * halving PCIe traffic vs. BF16 ring reduce.
 *
 * @param dst_int8    [in/out] INT8 accumulator chunk
 * @param dst_scale   [in/out] FP32 per-block scales for dst_int8
 * @param src_int8    [in]     INT8 received chunk from ring peer
 * @param src_scale   [in]     FP32 per-block scales for src_int8
 * @param n_elems     Number of elements in this chunk
 * @param sm_version  SM version (86, 90, 120)
 * @param stream      CUDA compute stream
 */
void launch_int8_ring_reduce_step(int8_t*       dst_int8,
                                   float*        dst_scale,
                                   const int8_t* src_int8,
                                   const float*  src_scale,
                                   size_t        n_elems,
                                   int           sm_version,
                                   cudaStream_t  stream);

/**
 * launch_gradient_decompress
 *
 * Phase 3: reconstruct BF16 gradient from INT8 data + per-block FP32 scales.
 *   output[i] = int8_data[i] × (scale[block(i)] / 127)
 *
 * @param output     [out] BF16 device buffer, n_elems elements
 * @param int8_data  [in]  INT8 device buffer, n_elems bytes
 * @param scale_buf  [in]  FP32 per-block scales, ceil(n_elems/256) floats
 * @param n_elems    Number of elements
 * @param sm_version SM version (86, 90, 120)
 * @param stream     CUDA stream
 */
void launch_gradient_decompress(__nv_bfloat16* output,
                                 const int8_t*  int8_data,
                                 const float*   scale_buf,
                                 size_t         n_elems,
                                 int            sm_version,
                                 cudaStream_t   stream);

/**
 * launch_gradient_allreduce_finalise
 *
 * Applies the averaging step (÷ world_size) to all per-block scale factors
 * after the ring reduce-scatter completes.  No INT8 data is touched;
 * only the scale_buf is updated, so the final decompress produces
 * the mean gradient directly.
 *
 * @param scale_buf   [in/out] FP32 per-block scales
 * @param n_elems     Total gradient elements (used to compute n_scale_blocks)
 * @param world_size  Number of participating ranks
 * @param stream      CUDA stream
 */
void launch_gradient_allreduce_finalise(float*       scale_buf,
                                         size_t       n_elems,
                                         int          world_size,
                                         cudaStream_t stream);

/**
 * launch_fused_gradient_allreduce
 *
 * High-level 3-phase pipeline: compress → ring-allreduce → decompress.
 * Executes the full gradient all-reduce in INT8 (2× PCIe bandwidth reduction
 * vs. BF16) using a double-buffered ring over heterogeneous PCIe topology.
 *
 * The caller must pre-allocate all staging buffers and provide per-peer
 * device pointers (from cudaIpcGetMemHandle / cudaIpcOpenMemHandle or
 * cudaMallocAsync on peer-accessible devices).
 *
 * @param grad            [in/out] BF16 gradient [n_elems], updated in-place
 * @param int8_staging    working INT8 buffer    [n_elems bytes]
 * @param scale_staging   working scale buffer   [ceil(n_elems/256) floats]
 * @param ping_int8       receive ping buffer    [n_elems bytes]
 * @param pong_int8       receive pong buffer    [n_elems bytes]
 * @param ping_scale      receive ping scale     [ceil(n_elems/256) floats]
 * @param pong_scale      receive pong scale     [ceil(n_elems/256) floats]
 * @param peer_int8       device pointers to peers' int8_staging, length world_size
 * @param peer_scale      device pointers to peers' scale_staging, length world_size
 * @param rank            this rank in [0, world_size)
 * @param world_size      number of participating GPUs
 * @param n_elems         BF16 gradient elements
 * @param sm_version      SM version of the current device (86, 90, 120)
 * @param transfer_stream CUDA stream for cudaMemcpyPeerAsync
 * @param compute_stream  CUDA stream for compress/reduce/decompress kernels
 * @param xfer_events     two pre-created CUDA events for double-buffer sync
 */
void launch_fused_gradient_allreduce(__nv_bfloat16*  grad,
                                      int8_t*         int8_staging,
                                      float*          scale_staging,
                                      int8_t*         ping_int8,
                                      int8_t*         pong_int8,
                                      float*          ping_scale,
                                      float*          pong_scale,
                                      int8_t* const*  peer_int8,
                                      float*  const*  peer_scale,
                                      int             rank,
                                      int             world_size,
                                      size_t          n_elems,
                                      int             sm_version,
                                      cudaStream_t    transfer_stream,
                                      cudaStream_t    compute_stream,
                                      cudaEvent_t     xfer_events[2]);

/**
 * gradient_compress_bytes
 *
 * Returns the INT8 staging buffer size (bytes) required for n_elems gradient
 * elements.  Always equals n_elems * sizeof(int8_t).
 *
 * @param n_elems  Number of BF16 gradient elements
 * @returns        Required INT8 buffer size in bytes
 */
size_t gradient_compress_bytes(size_t n_elems);

/**
 * gradient_scale_bytes
 *
 * Returns the per-block scale buffer size (bytes) required alongside an
 * n_elems-element compressed gradient.
 *
 * @param n_elems  Number of BF16 gradient elements
 * @returns        Required scale buffer size in bytes
 */
size_t gradient_scale_bytes(size_t n_elems);

// ===========================================================================
// fused_adam_heterogeneous — Per-tier LR-scaled Adam optimizer kernel
//   (fused_adam_heterogeneous.cu)
// ===========================================================================

/**
 * hetero_adam_lr_scale
 *
 * Returns the default per-tier learning-rate scale factor proportional to
 * each GPU tier's relative throughput:
 *   SM 12.0 (Blackwell) → 4.0
 *   SM  9.0 (H100)      → 3.0
 *   SM  8.6 (A6000)     → 1.0
 *
 * Python-level schedulers may override this per-step.
 *
 * @param sm_version  SM version of the current device (86, 90, 120, …)
 * @returns           Float scale factor ≥ 1.0
 */
float hetero_adam_lr_scale(int sm_version);

/**
 * launch_fused_adam_heterogeneous
 *
 * Fused Adam optimizer update with per-tier learning-rate scaling, designed
 * for heterogeneous A6000/H100/Blackwell mixed-GPU clusters.
 *
 * Applies the standard Adam update (Kingma & Ba, 2015) with decoupled weight
 * decay (AdamW) and an effective learning rate of lr_base × lr_scale:
 *
 *   m_t   = β₁·m_{t-1} + (1−β₁)·g_t
 *   v_t   = β₂·v_{t-1} + (1−β₂)·g_t²
 *   m̂_t   = m_t · bc1            (bc1 = 1/(1−β₁ᵗ), pre-computed by caller)
 *   v̂_t   = v_t · bc2            (bc2 = 1/(1−β₂ᵗ))
 *   θ_t   = (1−lr_eff·wd)·θ_{t-1} − lr_eff · m̂_t / (√v̂_t + ε)
 *
 * where lr_eff = lr_base × lr_scale.
 *
 * BF16 params and gradients; FP32 moments; optional FP32 master-weight copy.
 * Uses 128-bit vectorised loads (8 BF16 / 4 FP32 per instruction).
 * SM-specialised via AdamPolicy<SmVer>: SM8.6→(256,2), SM9.0→(256,4),
 * SM12.0→(512,4).
 *
 * @param params        [in/out] BF16 working parameters [n_elems]
 * @param master_params [in/out] FP32 master copy [n_elems], or nullptr
 * @param exp_avg       [in/out] FP32 first-moment  (m) [n_elems]
 * @param exp_avg_sq    [in/out] FP32 second-moment (v) [n_elems]
 * @param grads         [in]     BF16 gradients [n_elems]
 * @param n_elems       Number of parameter elements
 * @param lr_base       Base learning rate (before tier scaling)
 * @param lr_scale      Per-tier LR scale (use hetero_adam_lr_scale() for defaults)
 * @param beta1         Adam β₁ (typically 0.9)
 * @param beta2         Adam β₂ (typically 0.999)
 * @param bc1           Bias correction 1 = 1/(1−β₁^step)
 * @param bc2           Bias correction 2 = 1/(1−β₂^step)
 * @param eps           Adam ε (typically 1e-8)
 * @param weight_decay  Decoupled weight-decay coefficient (0.0 to disable)
 * @param sm_version    SM version of the current device (86, 90, 120, …)
 * @param stream        CUDA stream
 */
void launch_fused_adam_heterogeneous(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    const __nv_bfloat16* grads,
    size_t               n_elems,
    float                lr_base,
    float                lr_scale,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    int                  sm_version,
    cudaStream_t         stream);

/**
 * launch_fused_adamw_amsgrad_heterogeneous
 *
 * Extended heterogeneous Adam launch supporting AMSGrad, per-tier gradient
 * clipping, and FP8-E4M3 gradient inputs.
 *
 * AMSGrad (Reddi et al., 2018) maintains a running maximum of the
 * bias-corrected second moment, ensuring a non-increasing effective step size:
 *   v̂_max_t = max(v̂_max_{t-1}, v̂_t)
 *   θ_t     = θ_{t-1} − lr_eff · m̂_t / (√v̂_max_t + ε)
 *
 * Gradient clipping:
 *   clip_scale = min(1, clip_norm / global_grad_norm).
 *   Use launch_grad_norm_sq to accumulate the per-shard squared norm, then
 *   reduce across tiers on the host before computing clip_scale.
 *
 * FP8-E4M3 gradient path:
 *   Set grad_dtype = 1 (kGradFP8_E4M3) and pass a non-null uint8_t* grads.
 *   fp8_grad_scale absorbs the per-tensor quantisation scale so the kernel
 *   sees properly scaled FP32 gradients.  Requires CUDA >= 12.1 on SM >= 8.9
 *   for hardware decode; older toolchains use a software fallback.
 *
 * @param params            [in/out] BF16 working parameters [n_elems]
 * @param master_params     [in/out] FP32 master copy [n_elems], or nullptr
 * @param exp_avg           [in/out] FP32 first-moment  (m) [n_elems]
 * @param exp_avg_sq        [in/out] FP32 second-moment (v) [n_elems]
 * @param exp_avg_sq_max    [in/out] FP32 v_max [n_elems], or nullptr (→ classic Adam)
 * @param grads             [in]     Gradient buffer (BF16 or FP8-E4M3) [n_elems]
 * @param n_elems           Number of parameter elements
 * @param lr_base           Base learning rate (before tier scaling)
 * @param lr_scale          Per-tier LR scale (use hetero_adam_lr_scale())
 * @param beta1             Adam β₁ (typically 0.9)
 * @param beta2             Adam β₂ (typically 0.999)
 * @param bc1               Bias correction 1 = 1/(1−β₁^step)
 * @param bc2               Bias correction 2 = 1/(1−β₂^step)
 * @param eps               Adam ε (typically 1e-8)
 * @param weight_decay      Decoupled weight-decay coefficient (0.0 to disable)
 * @param clip_scale        Gradient clip multiplier = min(1, clip_norm/global_norm).
 *                          Pass 1.0f to disable clipping.
 * @param fp8_grad_scale    Per-tensor FP8 quantisation scale (ignored for BF16)
 * @param grad_dtype        0 = BF16, 1 = FP8-E4M3
 * @param sm_version        SM version of the current device (86, 90, 120, …)
 * @param stream            CUDA stream
 */
void launch_fused_adamw_amsgrad_heterogeneous(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    float*               exp_avg_sq_max,
    const void*          grads,
    size_t               n_elems,
    float                lr_base,
    float                lr_scale,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    float                clip_scale,
    float                fp8_grad_scale,
    int                  grad_dtype,
    int                  sm_version,
    cudaStream_t         stream);

/**
 * launch_grad_norm_sq
 *
 * Accumulates ‖g‖² from a BF16 gradient shard into a device-resident FP32
 * scalar accumulator via atomicAdd, for use in global gradient norm clipping.
 *
 * The caller must zero *norm_sq_accum (e.g. via cudaMemsetAsync) before the
 * first call in a training step.  After all shards are processed, copy the
 * scalar to host, take the square root to obtain the global gradient norm,
 * then compute:
 *   clip_scale = (clip_norm > 0 && global_norm > clip_norm)
 *                  ? clip_norm / global_norm : 1.0f;
 * and pass it to launch_fused_adamw_amsgrad_heterogeneous.
 *
 * @param grads         [in]  BF16 gradient shard (device), length n_elems
 * @param n_elems       Number of gradient elements
 * @param norm_sq_accum [in/out] Device scalar accumulator (pre-zeroed by caller)
 * @param sm_version    SM version of the current device (86, 90, 120, …)
 * @param stream        CUDA stream
 */
void launch_grad_norm_sq(
    const __nv_bfloat16* grads,
    size_t               n_elems,
    float*               norm_sq_accum,
    int                  sm_version,
    cudaStream_t         stream);

// ===========================================================================
// fused_layernorm_residual extended — Welford full LN + optional bias + FP32 out
// ====================================================================

/**
 * launch_fused_layernorm_residual_ex
 *
 * Extended version supporting:
 *   - Full LayerNorm (Welford variance, mean subtraction) OR RMSNorm
 *   - Optional bias: residual += input + bias (one fused pass)
 *   - Optional FP32 output: writes normalised activations as FP32 for
 *     tensor-parallel column layers
 *
 * @param output       [out] BF16 LN output [batch, hidden]
 * @param residual     [in/out] BF16 residual stream [batch, hidden]
 * @param input        [in]  BF16 new contribution [batch, hidden]
 * @param bias         [in]  BF16 optional bias [hidden] (nullptr to skip)
 * @param ln_weight    [in]  FP32 RMSNorm/LN scale [hidden]
 * @param output_fp32  [out] FP32 LN output [batch, hidden] (nullptr to skip)
 * @param batch        Batch size
 * @param hidden       Hidden size (must be divisible by 8)
 * @param eps          LayerNorm epsilon
 * @param full_ln      true = full LayerNorm (Welford), false = RMSNorm
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA stream
 */
void launch_fused_layernorm_residual_ex(
    __nv_bfloat16*       output,
    __nv_bfloat16*       residual,
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    const float*         ln_weight,
    float*               output_fp32,
    int                  batch,
    int                  hidden,
    float                eps,
    bool                 full_ln,
    int                  sm_version,
    cudaStream_t         stream);

// ===========================================================================
// fused_rope_hetero cacheless mode
// ===========================================================================

/**
 * launch_fused_rope_cacheless
 *
 * Applies RoPE by computing sin/cos on-the-fly (no precomputed cache).
 * Useful for very long sequences where the [S, D/2] cache exceeds L2 capacity.
 * Slightly higher arithmetic cost than the cached path; use only when cache
 * memory is unavailable.
 *
 * @param output      [out] BF16 output [B, S, H, D]
 * @param input       [in]  BF16 input  [B, S, H, D]
 * @param batch       Batch size
 * @param seq_len     Sequence length
 * @param num_heads   Number of attention heads
 * @param head_dim    Head dimension (must be even)
 * @param base        RoPE base frequency (default 10000.f)
 * @param pos_offset  Global position offset (for packed sequences)
 * @param neox_style  true → Llama/NeoX, false → GPT-J interleaved
 * @param sm_version  SM version (86, 90, 120)
 * @param stream      CUDA stream
 */
void launch_fused_rope_cacheless(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    int                  batch,
    int                  seq_len,
    int                  num_heads,
    int                  head_dim,
    float                base,
    int                  pos_offset,
    bool                 neox_style,
    int                  sm_version,
    cudaStream_t         stream);

// ===========================================================================
// grad_norm_sq — improved version with Kahan compensation + FP8 support
// ===========================================================================

/**
 * launch_grad_norm_sq_fp8
 *
 * Accumulates ‖g‖² for FP8-E4M3 gradients.  Companion to launch_grad_norm_sq
 * for BF16 gradients.
 *
 * @param grads        [in]  FP8-E4M3 gradient buffer [n_elems bytes]
 * @param n_elems      Number of gradient elements
 * @param norm_sq_accum [in/out] FP32 device scalar (caller zeroes before first call)
 * @param fp8_scale    Per-tensor FP8 scale factor
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA stream
 */
void launch_grad_norm_sq_fp8(
    const uint8_t* grads,
    size_t         n_elems,
    float*         norm_sq_accum,
    float          fp8_scale,
    int            sm_version,
    cudaStream_t   stream);


// ===========================================================================
// pcie_adaptive_allreduce — adaptive bucketing  (issue #24)
// ===========================================================================

/**
 * AdaptiveBucketIndex — device-side scatter descriptor for one gradient region.
 *
 * Used by launch_pcie_adaptive_unpack to scatter a received flat bucket buffer
 * back to destination gradient tensor pointers.
 *
 * Fields:
 *   dst      — device pointer to the target gradient tensor region.
 *   src_off  — element offset within the received flat bucket.
 *   n_elems  — BF16 elements to copy from bucket[src_off] → dst[0..n_elems).
 */
struct AdaptiveBucketIndex {
    __nv_bfloat16* dst;
    size_t         src_off;
    size_t         n_elems;
};

/**
 * BucketPlan — host-side result of pcie_adaptive_bucket_plan().
 *
 * Per-rank adaptive bucket sizes are derived from probed PCIe BW so that
 * each ring edge transfers a chunk sized to fill ~kTargetOverlapMs ms of
 * PCIe bandwidth, preventing fast intra-NUMA links from stalling on the same
 * chunk size as slow cross-NUMA links.
 */
struct BucketPlan {
    static constexpr int kMaxRanks  = 64;
    static constexpr int kMaxChunks = 1024;

    int    world_size;
    size_t bucket_sizes  [kMaxRanks];   // bytes per rank's send bucket
    size_t bucket_offsets[kMaxRanks];   // prefix-sum of bucket_sizes (flat buf)
    int    chunk_bucket  [kMaxChunks];  // chunk c → assigned rank index
    int    num_chunks;
};

/**
 * pcie_adaptive_bucket_plan
 *
 * Probes PCIe BW for each ring edge and assigns gradient chunks to per-rank
 * send buckets proportional to measured bandwidth (greedy bin-packing).
 *
 * @param plan        [out] Populated BucketPlan
 * @param chunks      [in]  Gradient chunk descriptors, length num_chunks
 * @param num_chunks  Number of gradient chunks (≤ BucketPlan::kMaxChunks)
 * @param device_ids  CUDA device ordinals per rank, length world_size
 * @param world_size  Number of participating ranks (≤ BucketPlan::kMaxRanks)
 */
void pcie_adaptive_bucket_plan(
    BucketPlan*          plan,
    const PcieGradChunk* chunks,
    int                  num_chunks,
    const int*           device_ids,
    int                  world_size);

/**
 * launch_pcie_adaptive_unpack
 *
 * Scatters a received flat bucket buffer back to gradient tensor regions.
 * Inverse of launch_pcie_gradient_pack; dispatches
 * pcie_adaptive_unpack_kernel with binary-search scatter indexing.
 *
 * @param bucket       [in]  Received flat BF16 bucket [total_elems]
 * @param index        [in]  Host scatter index array (num_entries entries)
 * @param num_entries  Number of AdaptiveBucketIndex entries
 * @param total_elems  Total BF16 elements (= Σ index[i].n_elems)
 * @param sm_version   SM version for block-size selection
 * @param stream       CUDA stream
 */
void launch_pcie_adaptive_unpack(
    const __nv_bfloat16*       bucket,
    const AdaptiveBucketIndex* index,
    int                        num_entries,
    size_t                     total_elems,
    int                        sm_version,
    cudaStream_t               stream);

/**
 * tma_block_load_smem_bytes
 *
 * Returns the shared-memory bytes consumed per CTA by the TMA tile buffer
 * in tma_ring_reduce_sm90_kernel.  Use for occupancy analysis / smem checks.
 */
size_t tma_block_load_smem_bytes();

/**
 * launch_tma_hetero_reduce_step
 *
 * TMA-accelerated ring-reduce step for SM9.0+ (issue #72).
 * Loads src_buf into shared memory via cp.async.bulk.tensor (TMA), then
 * accumulates dst += src in FP32 with BF16 I/O.
 *
 * Falls back to launch_pcie_ring_reduce_step on SM < 9.0 or CUDA < 12.
 *
 * @param accum_buf      [in/out] BF16 local accumulator [chunk_elems]
 * @param src_buf        [in]     BF16 peer bucket [chunk_elems]
 * @param chunk_elems    Number of BF16 elements (divisible by 8)
 * @param sm_version     SM version (86, 90, 120)
 * @param compute_stream CUDA stream
 */
void launch_tma_hetero_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ src_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream);

// pcie_adaptive_allreduce — NUMA-aware topology-aware dispatch  (issue #138)
// ===========================================================================

/**
 * TopoInfo — NUMA node and PCIe switch affinity for each rank.
 *
 * Populated by query_topo_info() and consumed by select_allreduce_algo()
 * to choose between direct, ring, and tree allreduce algorithms.
 *
 * Fields:
 *   world_size   — total participating ranks.
 *   numa_node    — NUMA node index per rank (0-indexed).
 *   pcie_switch  — PCIe switch domain index per rank (0-indexed).
 *   device_id    — CUDA device ordinal per rank.
 *   num_numa     — count of distinct NUMA nodes present.
 *   num_switches — count of distinct PCIe switch domains present.
 */
struct TopoInfo;   // full definition in pcie_adaptive_allreduce.cu

/**
 * AllreduceAlgo — algorithm selected by select_allreduce_algo().
 *
 *   kDirect  payload < 256 KB   root pulls all shards point-to-point
 *   kRing    256 KB ≤ payload < 64 MB   ring allreduce
 *   kTree    payload ≥ 64 MB, pow2 world, multi-switch   recursive-halving tree
 */
enum class AllreduceAlgo : int {
    kDirect = 0,
    kRing   = 1,
    kTree   = 2,
};

/**
 * query_topo_info
 *
 * Probes CUDA runtime attributes to fill TopoInfo for world_size ranks.
 * Falls back to NEURON_PCIE_SWITCH_WIDTH env var (default 4 GPUs/switch).
 *
 * @param topo        [out] Topology descriptor to populate
 * @param device_ids  [in]  CUDA device ordinals, one per rank
 * @param world_size  Number of participating ranks (≤ TopoInfo::kMaxRanks)
 */
void query_topo_info(TopoInfo* topo, const int* device_ids, int world_size);

/**
 * select_allreduce_algo
 *
 * Returns the optimal algorithm for the given payload and topology.
 *
 * Thresholds:
 *   payload < 256 KB                              → kDirect
 *   256 KB ≤ payload < 64 MB                     → kRing
 *   payload ≥ 64 MB, pow2 world, multi-switch     → kTree
 *   payload ≥ 64 MB, non-pow2 or single-switch   → kRing (fallback)
 *
 * @param topo          Populated topology descriptor
 * @param payload_bytes Allreduce payload size in bytes
 * @returns             AllreduceAlgo enum value
 */
AllreduceAlgo select_allreduce_algo(const TopoInfo& topo, size_t payload_bytes);

/**
 * build_numa_ring_order
 *
 * Produces a NUMA-locality-optimised ring permutation.
 * Ranks sharing a NUMA node are placed adjacent in the ring to minimise
 * cross-NUMA PCIe traffic per ring step.
 *
 * @param topo        Topology descriptor
 * @param ring_order  [out] Permutation array, caller-allocated length world_size
 */
void build_numa_ring_order(const TopoInfo& topo, int* ring_order);

/**
 * launch_pcie_tree_reduce_step
 *
 * Single recursive-halving tree-reduce step:
 *   accum_buf[i] += recv_buf[i]  (BF16 → FP32 accumulation → BF16)
 *
 * Mirrors launch_pcie_ring_reduce_step; used by the tree-allreduce
 * orchestration for each reduce phase step.
 *
 * @param accum_buf      [in/out] BF16 accumulator [chunk_elems]
 * @param recv_buf       [in]     BF16 received shard [chunk_elems]
 * @param chunk_elems    Number of BF16 elements in this step's shard
 * @param sm_version     SM version for kernel dispatch (86, 90, 120)
 * @param compute_stream CUDA stream for the reduce kernel
 */
void launch_pcie_tree_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream);

// ===========================================================================
// hetero_ring_allreduce — Heterogeneous ring allreduce for PCIe-only topology
//   (hetero_ring_allreduce.cu)
//
// Ring allreduce across 5 GPUs on 2 NUMA nodes connected by PCIe without
// NVLink.  Bandwidth-aware chunking: 4 MB intra-NUMA, 2 MB cross-NUMA.
// Double-buffered with independent transfer and compute streams.
// ===========================================================================

/**
 * NumaAwareRingDesc (forward declaration for callers).
 *
 * Initialise with hetero_ring_init(); pass to launch_hetero_ring_allreduce().
 * Full definition in hetero_ring_allreduce.cu.
 */
struct NumaAwareRingDesc;

/**
 * hetero_ring_init
 *
 * Initialises a NumaAwareRingDesc for a world_size-GPU PCIe ring.
 *
 * Default NUMA assignment (if numa_nodes == nullptr):
 *   For world_size = 5: positions {0,1,2} → NUMA-0; {3,4} → NUMA-1.
 *
 * @param desc        [out] Topology descriptor (caller-allocated)
 * @param device_ids  [in]  CUDA device ordinals for ring positions 0..P-1
 * @param numa_nodes  [in]  NUMA node per ring position, or nullptr for default
 * @param sm_versions [in]  SM version (86, 90, 120) per ring position
 * @param world_size  Number of participating GPUs (≤ 8)
 * @param this_rank   Ring position of the calling process (0-indexed)
 */
void hetero_ring_init(
    NumaAwareRingDesc* desc,
    const int*         device_ids,
    const int*         numa_nodes,
    const int*         sm_versions,
    int                world_size,
    int                this_rank);

/**
 * launch_hetero_ring_reduce_step
 *
 * Single reduce-scatter step: accum[i] += recv[i]  (BF16 → FP32 → BF16).
 * Dispatches SM-specialised kernel (SM8.6 / SM9.0 / SM12.0 cp.async).
 *
 * @param accum        [in/out] BF16 accumulator [chunk_elems]
 * @param recv         [in]     BF16 received chunk [chunk_elems]
 * @param chunk_elems  Number of BF16 elements
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA compute stream
 */
void launch_hetero_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      stream);

/**
 * launch_hetero_ring_gather_step
 *
 * Single all-gather step: output[i] = recv[i]  (128-bit vectorised copy).
 * No accumulation — the incoming chunk is already fully reduced.
 *
 * @param output       [out] BF16 destination [chunk_elems]
 * @param recv         [in]  BF16 received fully-reduced chunk [chunk_elems]
 * @param chunk_elems  Number of BF16 elements
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA compute stream
 */
void launch_hetero_ring_gather_step(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      stream);

/**
 * launch_hetero_ring_allreduce
 *
 * Full reduce-scatter + all-gather ring allreduce for a heterogeneous
 * PCIe cluster.  Both phases execute (P-1) steps each with double-buffered
 * DMA/compute overlap.
 *
 * Caller must pre-allocate on this rank's device:
 *   data         — full BF16 gradient tensor [total_elems]  (in-place)
 *   ping_buf     — receive ping buffer [hetero_ring_max_chunk_bytes() / 2]
 *   pong_buf     — receive pong buffer [hetero_ring_max_chunk_bytes() / 2]
 *   peer_data[r] — peer-mapped device pointer to rank r's data buffer
 *   xfer_done[2] — two pre-created CUDA events for double-buffer sync
 *
 * @param data          [in/out] BF16 gradient [total_elems], modified in-place
 * @param ping_buf      Receive ping buffer (device, max chunk size)
 * @param pong_buf      Receive pong buffer (device, max chunk size)
 * @param peer_data     Device pointers to all ranks' data buffers (length P)
 * @param desc          Topology descriptor from hetero_ring_init()
 * @param total_elems   Number of BF16 gradient elements
 * @param sm_version    SM version of this rank's device
 * @param stream_xfer   CUDA stream for cudaMemcpyPeerAsync (DMA)
 * @param stream_comp   CUDA stream for reduce/gather kernels (compute)
 * @param xfer_done     Two pre-created CUDA events for double-buffer handshake
 */
void launch_hetero_ring_allreduce(
    __nv_bfloat16*          data,
    __nv_bfloat16*          ping_buf,
    __nv_bfloat16*          pong_buf,
    __nv_bfloat16* const*   peer_data,
    const NumaAwareRingDesc& desc,
    size_t                  total_elems,
    int                     sm_version,
    cudaStream_t            stream_xfer,
    cudaStream_t            stream_comp,
    cudaEvent_t             xfer_done[2]);

/**
 * hetero_ring_intra_numa_chunk_bytes / hetero_ring_cross_numa_chunk_bytes
 *
 * Returns the bandwidth-aware chunk size constants for buffer sizing.
 *   Intra-NUMA: 4 MB  (targets ~0.5 ms at 32 GB/s PCIe 4.0 x16)
 *   Cross-NUMA: 2 MB  (targets ~0.5 ms at 16 GB/s cross-switch PCIe)
 *   Max:        4 MB  (use for ping/pong buffer allocation)
 */
size_t hetero_ring_intra_numa_chunk_bytes();
size_t hetero_ring_cross_numa_chunk_bytes();
size_t hetero_ring_max_chunk_bytes();

/**
 * hetero_ring_sm_block_size
 *
 * Returns the thread-block size used by the reduce/gather kernels for a
 * given SM version.  Useful for occupancy analysis and buffer alignment.
 *
 * SM8.6 (A6000): 128  (fewer SMs -> smaller blocks)
 * SM9.0  (H100): 256
 * SM12.0 (Blackwell): 512
 *
 * @param sm_version  SM version (86, 90, 120)
 * @returns           Thread-block size
 */
int hetero_ring_sm_block_size(int sm_version);

// ===========================================================================
// fused_rope_qk — Simultaneous Q+K RoPE for GQA  (issue #23)
// ===========================================================================

/**
 * launch_fused_rope_qk
 *
 * Applies RoPE to both Q and K in a single kernel launch.
 * Handles GQA where Q and K have different head counts.
 *
 * @param q_output     [out] BF16 rotated Q [B, S, Hq, D]
 * @param k_output     [out] BF16 rotated K [B, S, Hkv, D]
 * @param q_input      [in]  BF16 Q input   [B, S, Hq, D]
 * @param k_input      [in]  BF16 K input   [B, S, Hkv, D]
 * @param cos_cache    [in]  FP32 [S, D/2] (nullptr for cacheless)
 * @param sin_cache    [in]  FP32 [S, D/2] (nullptr for cacheless)
 * @param batch        Batch size
 * @param seq_len      Sequence length
 * @param num_heads_q  Q attention heads
 * @param num_heads_kv K/V attention heads (GQA)
 * @param head_dim     Head dimension (must be even)
 * @param neox_style   true → NeoX/Llama, false → GPT-J
 * @param base         RoPE base frequency (cacheless mode)
 * @param pos_offset   Position offset (cacheless mode)
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA stream
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

// ===========================================================================
// fused_mlp kernels — SwiGLU + pre-LN + residual  (issue #25)
// ===========================================================================

/**
 * launch_fused_swiglu
 *
 * Single kernel for gate × σ(gate) × up (replaces 3 kernel launches).
 */
void launch_fused_swiglu(__nv_bfloat16*       output,
                          const __nv_bfloat16* gate_proj,
                          const __nv_bfloat16* up_proj,
                          int                  batch,
                          int                  hidden,
                          int                  sm_version,
                          cudaStream_t         stream);

/**
 * launch_fused_pre_ln_attn
 *
 * Fused pre-LayerNorm for attention input:
 *   output = RMSNorm(residual) × ln_weight
 */
void launch_fused_pre_ln_attn(__nv_bfloat16*       output,
                               const __nv_bfloat16* residual,
                               const float*         ln_weight,
                               int                  batch,
                               int                  hidden,
                               float                eps,
                               int                  sm_version,
                               cudaStream_t         stream);

/**
 * launch_fused_residual_rmsnorm
 *
 * Fused residual add + RMSNorm:
 *   residual += input; output = RMSNorm(residual, ln_weight, eps)
 */
void launch_fused_residual_rmsnorm(__nv_bfloat16*       output,
                                    __nv_bfloat16*       residual,
                                    const __nv_bfloat16* input,
                                    const float*         ln_weight,
                                    int                  batch,
                                    int                  hidden,
                                    float                eps,
                                    int                  sm_version,
                                    cudaStream_t         stream);

// ---------------------------------------------------------------------------
// Issue #124: TP-aware fused cross-entropy completions
// ---------------------------------------------------------------------------

/**
 * launch_cross_entropy_tp_log_finalise
 *
 * Computes log(global_sum_exp[i]) in-place for i in [0, batch).
 * Call this AFTER dist.all_reduce(local_sum_exp, op=SUM) to convert the
 * global sum_exp into the log_sum_exp required by the backward kernel.
 *
 * @param global_sum_exp  [in/out] FP32 device buffer [batch], modified in-place
 * @param batch           Number of samples
 * @param stream          CUDA stream
 */
void launch_cross_entropy_tp_log_finalise(
    float*       global_sum_exp,
    int          batch,
    cudaStream_t stream);

/**
 * launch_cross_entropy_tp_forward_with_log
 *
 * Forward kernel that writes four output buffers per row:
 *   local_max      — for AllReduce_max across TP ranks
 *   local_sum_exp  — for AllReduce_sum across TP ranks
 *   local_log_sum  — log(local_sum_exp); valid directly for tp_size=1,
 *                    must be recomputed after AllReduce for tp_size > 1.
 *   local_logit    — label logit; for AllReduce_sum across TP ranks.
 *
 * @param local_max      [out] FP32 [batch]
 * @param local_sum_exp  [out] FP32 [batch]
 * @param local_log_sum  [out] FP32 [batch] — log of local partial sum
 * @param local_logit    [out] FP32 [batch]
 * @param logits         [in]  BF16 [batch × v_local]
 * @param labels         [in]  Int32 [batch] — global vocab indices
 * @param batch          Batch size (= grid dim)
 * @param v_local        Vocab shard width on this rank
 * @param shard_offset   First global vocab index on this rank
 * @param sm_version     86, 90, or 120
 * @param stream         CUDA stream
 */
void launch_cross_entropy_tp_forward_with_log(
    float*               local_max,
    float*               local_sum_exp,
    float*               local_log_sum,
    float*               local_logit,
    const __nv_bfloat16* logits,
    const int*           labels,
    int                  batch,
    int                  v_local,
    int                  shard_offset,
    int                  sm_version,
    cudaStream_t         stream);
