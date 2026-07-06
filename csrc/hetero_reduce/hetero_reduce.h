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
