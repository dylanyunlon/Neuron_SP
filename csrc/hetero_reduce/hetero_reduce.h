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
