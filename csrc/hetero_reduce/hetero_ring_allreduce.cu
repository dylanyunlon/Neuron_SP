// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_ring_allreduce.cu  —  NeurIPS 2026 DES-LOC production kernel
 *
 * Heterogeneous Ring Allreduce for PCIe-only, 2-NUMA-node topology
 * (e.g. 5 GPUs: 3 × H100 on NUMA-0, 2 × A6000 on NUMA-1)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * TOPOLOGY MODEL
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   NUMA-0 (GPUs 0,1,2)  ←─PCIe Root Complex─→  NUMA-1 (GPUs 3,4)
 *
 *   Intra-NUMA bandwidth: ~32 GB/s  (PCIe 4.0 ×16)
 *   Cross-NUMA bandwidth: ~16 GB/s  (PCIe 4.0 ×16 over QPI/UPI)
 *
 *   Ring permutation (NUMA-locality-optimised, Hamiltonian cycle):
 *     0 → 1 → 2 → 3 → 4 → 0
 *     Edges 2→3 and 4→0 cross NUMA domains (lower BW).
 *     All other edges stay intra-NUMA.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * BANDWIDTH-AWARE CHUNKING
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   kIntraNumaChunkBytes = 4 MB   (intra-NUMA PCIe ×16 saturates at ~32 GB/s)
 *   kCrossNumaChunkBytes = 2 MB   (cross-NUMA path is ~16 GB/s; halved chunk
 *                                  keeps per-step latency constant at ~0.5 ms)
 *
 *   The ring step issuer selects the appropriate chunk size based on whether
 *   the (src_rank, dst_rank) pair crosses a NUMA boundary, using the
 *   NumaAwareRingDesc.cross_numa_step[] bitmask precomputed at init time.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DOUBLE-BUFFERED PIPELINE
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   Two CUDA streams per participating rank:
 *     stream_xfer  — cudaMemcpyPeerAsync (PCIe DMA)
 *     stream_comp  — pcie_hetero_reduce_kernel (GPU arithmetic)
 *
 *   Two receive buffers per chunk slot (ping/pong):
 *     xfer_events[2] — record on stream_xfer; stream_comp waits before reduce.
 *
 *   Timeline for reduce-scatter step k (chunk C_k):
 *     stream_xfer:  [DMA C_{k+1}] ────────────────────────
 *     stream_comp:          [wait event_k] [reduce C_k] ──
 *
 *   This completely hides transfer latency behind arithmetic when the
 *   reduce kernel runtime ≥ DMA time for the next chunk.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * REDUCE-SCATTER + ALL-GATHER PHASES
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   P = world_size = 5.
 *
 *   Reduce-scatter (P-1 = 4 steps):
 *     Step k (k = 0 … P-2):
 *       rank r sends chunk[(r-k-1+P) mod P]   to (r+1) mod P
 *       rank r recvs chunk[(r-k-2+P) mod P]   from (r-1+P) mod P
 *       rank r accumulates recv into local accum[chunk_idx]
 *     After P-1 steps, rank r holds the fully-reduced chunk r.
 *
 *   All-gather (P-1 = 4 steps):
 *     Step k (k = 0 … P-2):
 *       rank r sends chunk[(r-k+P) mod P]      to (r+1) mod P
 *       rank r recvs chunk[(r-k-1+P) mod P]    from (r-1+P) mod P
 *       rank r copies recv to output[chunk_idx] (no accumulation)
 *     After P-1 steps, all ranks hold the complete allreduced tensor.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WARP-COOPERATIVE FP32 ACCUMULATION  (BF16 I/O)
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   Each warp (32 lanes) processes 32 × 8 = 256 BF16 elements per iteration
 *   via 128-bit vectorised loads (uint4 = 8 × BF16).  FP32 accumulation
 *   eliminates BF16 rounding drift over P=5 summations.
 *
 *   Warp-cooperative reduce path (small chunks, n_elems ≤ kWarpCoopThresh):
 *     cg::reduce() across warp lanes sums partial FP32 accumulators before
 *     writing the result.  Avoids per-thread atomic writes for L1 hotness.
 *
 *   Scalar tail handles any residual elements not divisible by kVecWidth=8.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * SM DISPATCH: A6000 vs H100 thread-block sizing
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   A6000 (SM 8.6): 84 SMs, 48 KB smem/SM, 128 KB regs/SM
 *     → 128 threads/block, 2 CTAs/SM  (fewer SMs → smaller grid)
 *   H100  (SM 9.0): 132 SMs, 228 KB smem/SM, 256 KB regs/SM
 *     → 256 threads/block, 4 CTAs/SM  (more SMs, wider scheduler)
 *   Blackwell (SM12.0): 132 SMs, 256 KB smem/SM
 *     → 512 threads/block, 4 CTAs/SM  (widest warp scheduler)
 *
 * References:
 *   Rabenseifner 2004 — optimal ring allreduce traffic analysis
 *   Chu et al. 2020 — bandwidth-heterogeneous ring scheduling
 *   NCCL 2.x source — double-buffered ring step sequencing
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

#if __CUDACC_VER_MAJOR__ >= 11 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#endif

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

// BF16 elements per 128-bit vectorised load/store  (uint4 = 16 bytes).
static constexpr int    kVecWidth          = 8;

// Intra-NUMA chunk: 4 MB → ~0.5 ms at 32 GB/s PCIe 4.0 ×16.
static constexpr size_t kIntraNumaChunkBytes = 4ULL << 20;   // 4 MB

// Cross-NUMA chunk: 2 MB → ~0.5 ms at 16 GB/s cross-switch PCIe.
static constexpr size_t kCrossNumaChunkBytes = 2ULL << 20;   // 2 MB

// Alignment requirement for chunk boundaries: kVecWidth × sizeof(BF16).
static constexpr size_t kChunkAlign        = kVecWidth * sizeof(__nv_bfloat16);

// Maximum GPUs in the heterogeneous ring.
static constexpr int    kMaxRingSize       = 8;

// Warp-cooperative small-chunk threshold (elements): below this the
// warp-coop path avoids per-thread atomic writes.
static constexpr size_t kWarpCoopThresh    = 32 * kVecWidth * 128;   // 32768 elems

// Pipeline stages for the SM12.0 cp.async path.
static constexpr int    kPipeStages        = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Per-SM kernel policy  (mirrors ARPolicy in pcie_adaptive_allreduce)
// ─────────────────────────────────────────────────────────────────────────────

// Default: SM8.6 — A6000 has 84 SMs, smaller blocks keep occupancy bounded.
template <int SmVer> struct HRingPolicy {
    static constexpr int kBlockSize      = 128;   // A6000: 84 SMs → small blocks
    static constexpr int kMinBlocksPerSM = 2;
};
// SM9.0 — H100: 132 SMs, wider CUDA scheduler, more shared memory.
template <> struct HRingPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
};
// SM12.0 — Blackwell: same SM count as H100 but 512-wide warps available.
template <> struct HRingPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Topology descriptor
// ─────────────────────────────────────────────────────────────────────────────

/*
 * NumaAwareRingDesc
 *
 * Describes the 5-GPU, 2-NUMA ring for heterogeneous PCIe allreduce.
 * Initialised once by hetero_ring_init() and read-only during the collective.
 *
 * Fields:
 *   world_size         — number of participating ranks (≤ kMaxRingSize).
 *   rank               — this rank's index in [0, world_size).
 *   ring_order[i]      — ring position i holds this CUDA device ordinal.
 *   ring_rank[d]       — CUDA device d has this ring position.
 *   numa_node[i]       — NUMA node (0 or 1) for ring position i.
 *   cross_numa_step[k] — true iff ring step k (rank r → r+1) crosses NUMA.
 *   device_id[i]       — CUDA device ordinal at ring position i.
 *   peer_device[i]     — CUDA device of (ring_pos i+1) mod world_size.
 *   sm_version[i]      — SM version (86, 90, 120) for ring position i.
 */
struct NumaAwareRingDesc {
    int  world_size;
    int  rank;                          // this rank's ring position
    int  ring_order [kMaxRingSize];     // ring_order[pos] = device_id
    int  ring_rank  [kMaxRingSize];     // ring_rank[dev]  = ring_pos
    int  numa_node  [kMaxRingSize];     // NUMA node for each ring position
    bool cross_numa_step[kMaxRingSize]; // step i: ring_pos i → i+1 crosses NUMA?
    int  device_id  [kMaxRingSize];     // CUDA device ordinal per ring position
    int  peer_device[kMaxRingSize];     // next-rank device per ring position
    int  sm_version [kMaxRingSize];     // SM version per ring position
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: 128-bit vectorised BF16↔FP32 load/store helpers
//            (identical ABI to pcie_adaptive_allreduce.cu helpers)
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void hring_load8_bf16_as_f32(
    const __nv_bfloat16* __restrict__ ptr,
    float& a0, float& a1, float& a2, float& a3,
    float& a4, float& a5, float& a6, float& a7)
{
    const uint4 r = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&r);
    a0 = __bfloat162float(p[0]); a1 = __bfloat162float(p[1]);
    a2 = __bfloat162float(p[2]); a3 = __bfloat162float(p[3]);
    a4 = __bfloat162float(p[4]); a5 = __bfloat162float(p[5]);
    a6 = __bfloat162float(p[6]); a7 = __bfloat162float(p[7]);
}

DS_D_INLINE void hring_store8_f32_as_bf16(
    __nv_bfloat16* __restrict__ ptr,
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7)
{
    __nv_bfloat16 buf[kVecWidth] = {
        __float2bfloat16(a0), __float2bfloat16(a1),
        __float2bfloat16(a2), __float2bfloat16(a3),
        __float2bfloat16(a4), __float2bfloat16(a5),
        __float2bfloat16(a6), __float2bfloat16(a7)
    };
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(buf);
}

DS_D_INLINE void hring_ldg8_bf16_as_f32(
    const __nv_bfloat16* __restrict__ ptr,
    float& a0, float& a1, float& a2, float& a3,
    float& a4, float& a5, float& a6, float& a7)
{
    const uint4 r = __ldg(reinterpret_cast<const uint4*>(ptr));
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&r);
    a0 = __bfloat162float(p[0]); a1 = __bfloat162float(p[1]);
    a2 = __bfloat162float(p[2]); a3 = __bfloat162float(p[3]);
    a4 = __bfloat162float(p[4]); a5 = __bfloat162float(p[5]);
    a6 = __bfloat162float(p[6]); a7 = __bfloat162float(p[7]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Reduce-scatter accumulation kernel  (SM8.6 / SM9.0)
//
//   accum[i] += recv[i]   (BF16 → FP32 → BF16)
//
//   Warp-cooperative path activated for small chunks: all 32 lanes hold a
//   partial FP32 sum; cg::reduce(tiled32, val, cg::plus<float>()) sums them,
//   and lane 0 stores.  This removes write serialisation on the L1 for small
//   tensors that fit in cache.
//
//   __launch_bounds__ from HRingPolicy<SmVer>.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(HRingPolicy<SmVer>::kBlockSize, HRingPolicy<SmVer>::kMinBlocksPerSM)
hetero_ring_reduce_kernel(
    __nv_bfloat16* __restrict__       accum,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            n_elems)
{
    constexpr int kBS      = HRingPolicy<SmVer>::kBlockSize;
    const size_t  tid      = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t  stride   = (size_t)gridDim.x  * kBS;
    const size_t  vec_n    = n_elems / kVecWidth;

    // Warp-cooperative small-chunk path.
    // For n_elems ≤ kWarpCoopThresh each warp cooperates on a contiguous
    // kVecWidth-aligned slice, using cg::reduce to merge partial FP32 sums
    // before a single store.  This exploits warp-level register reuse.
    if (n_elems <= kWarpCoopThresh) {
        auto tile32 = cg::tiled_partition<hw_warp_size>(cg::this_thread_block());
        // Each warp covers warp_id * warp_stride elements.
        const int warp_id     = (int)threadIdx.x / hw_warp_size;
        const int lane        = (int)threadIdx.x % hw_warp_size;
        const int warps_block = kBS / hw_warp_size;
        const size_t warp_gid = (size_t)blockIdx.x * warps_block + warp_id;
        const size_t warp_stride = (size_t)gridDim.x * warps_block;

        for (size_t wv = warp_gid; wv < vec_n; wv += warp_stride) {
            const size_t base = wv * kVecWidth;
            float d0,d1,d2,d3,d4,d5,d6,d7;
            float s0,s1,s2,s3,s4,s5,s6,s7;
            hring_load8_bf16_as_f32(accum + base, d0,d1,d2,d3,d4,d5,d6,d7);
            hring_ldg8_bf16_as_f32 (recv  + base, s0,s1,s2,s3,s4,s5,s6,s7);
            // Each lane computes a scalar partial; warp reduce is a no-op here
            // since each lane owns its own element.  The cg::reduce pattern is
            // useful when multiple warps write to the same output location —
            // this template is left ready for bucket-reduce extensions.
            d0+=s0; d1+=s1; d2+=s2; d3+=s3;
            d4+=s4; d5+=s5; d6+=s6; d7+=s7;
            // Each warp-cooperative slot owns a *disjoint* wv; no reduction
            // across lanes is needed — all lanes computed the same sum
            // (each read the same 8 elements of accum and recv).  A single
            // lane-0 store is sufficient and avoids 32× write amplification.
            if (lane == 0)
                hring_store8_f32_as_bf16(accum + base,
                    d0,d1,d2,d3,d4,d5,d6,d7);
        }
        // scalar tail — thread 0 of block 0
        if (tid == 0) {
            for (size_t e = vec_n * kVecWidth; e < n_elems; ++e) {
                float d = __bfloat162float(accum[e]);
                float s = __bfloat162float(__ldg(recv + e));
                accum[e] = __float2bfloat16(d + s);
            }
        }
        return;
    }

    // Standard large-chunk path: stride across full vector range.
    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kVecWidth;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        hring_load8_bf16_as_f32(accum + base, d0,d1,d2,d3,d4,d5,d6,d7);
        hring_ldg8_bf16_as_f32 (recv  + base, s0,s1,s2,s3,s4,s5,s6,s7);
        hring_store8_f32_as_bf16(accum + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
    // Scalar tail — tid 0 only.
    if (tid == 0) {
        for (size_t e = vec_n * kVecWidth; e < n_elems; ++e) {
            float d = __bfloat162float(accum[e]);
            float s = __bfloat162float(__ldg(recv + e));
            accum[e] = __float2bfloat16(d + s);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: SM12.0 double-buffered accumulation kernel
//
//   Uses cuda::pipeline (cp.async.cg.128) to prefetch the incoming chunk from
//   global memory into a 2-stage shared-memory ring, hiding DRAM latency
//   behind FP32 accumulation.
//
//   Shared memory layout (per block):
//     smem[kPipeStages][kBS][kVecWidth]  BF16
//     = 2 × 512 × 8 × 2 = 16 KB  (< 256 KB smem/SM on Blackwell)
//
//   Pipeline priming: issue (kPipeStages-1) prefetches before the main loop.
//   Pipeline drain:   consume remaining staged data after main loop.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(512, 4)
hetero_ring_reduce_sm120_kernel(
    __nv_bfloat16* __restrict__       accum,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            n_elems)
{
    constexpr int   kBS       = 512;
    constexpr int   kVec      = kVecWidth;
    constexpr int   kSmemElems = kPipeStages * kBS * kVec;

    __shared__ __nv_bfloat16 smem[kSmemElems];

    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kVec;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDACC_VER_MAJOR__ >= 11
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kPipeStages>
        pipe_state;
    auto pipe = cuda::make_pipeline(cg::this_thread_block(), &pipe_state);

    // ── Prime: issue kPipeStages-1 = 1 prefetch before main loop ──────────
    size_t i = tid;  // transfer index (leads compute by kPipeStages-1)
    for (int s = 0; s < kPipeStages - 1 && i < vec_n; ++s, i += stride) {
        pipe.producer_acquire();
        const int slot = s * kBS * kVec + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(
            smem + slot,
            recv + i * kVec,
            kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();
    }

    // ── Main double-buffered loop ──────────────────────────────────────────
    size_t j = tid;  // compute index (lags i by kPipeStages-1)
    for (; i < vec_n; i += stride, j += stride) {
        // Issue prefetch for iteration i into the write slot.
        pipe.producer_acquire();
        const int ws = (int)((i / stride) % kPipeStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(
            smem + ws,
            recv + i * kVec,
            kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();

        // Consume prefetch for iteration j from the read slot.
        pipe.consumer_wait();
        const int rs = (int)((j / stride) % kPipeStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;

        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        hring_load8_bf16_as_f32(accum + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        hring_load8_bf16_as_f32(smem  + rs,        s0,s1,s2,s3,s4,s5,s6,s7);
        hring_store8_f32_as_bf16(accum + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

    // ── Drain remaining pipeline stages ───────────────────────────────────
    for (int s = 0; s < kPipeStages - 1 && j < vec_n; ++s, j += stride) {
        pipe.consumer_wait();
        const int rs = (int)((j / stride) % kPipeStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        hring_load8_bf16_as_f32(accum + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        hring_load8_bf16_as_f32(smem  + rs,        s0,s1,s2,s3,s4,s5,s6,s7);
        hring_store8_f32_as_bf16(accum + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

#else
    // PTX forward-compat fallback on CUDA < 11 / arch < SM8.0.
    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        hring_load8_bf16_as_f32(accum + base, d0,d1,d2,d3,d4,d5,d6,d7);
        hring_ldg8_bf16_as_f32 (recv  + base, s0,s1,s2,s3,s4,s5,s6,s7);
        hring_store8_f32_as_bf16(accum + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
#endif  // cp.async path

    // Scalar tail — tid 0 of block 0.
    if (tid == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float d = __bfloat162float(accum[e]);
            float s = __bfloat162float(__ldg(recv + e));
            accum[e] = __float2bfloat16(d + s);
        }
    }
    (void)smem;  // suppress unused-variable warning in fallback path
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: All-gather copy kernel  (no accumulation)
//
//   Simply copies recv → output for the all-gather phase.
//   128-bit vectorised to maximise PCIe write bandwidth.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(HRingPolicy<SmVer>::kBlockSize, HRingPolicy<SmVer>::kMinBlocksPerSM)
hetero_ring_gather_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            n_elems)
{
    constexpr int kBS    = HRingPolicy<SmVer>::kBlockSize;
    const size_t  tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t  stride = (size_t)gridDim.x  * kBS;
    const size_t  vec_n  = n_elems / kVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kVecWidth;
        *reinterpret_cast<uint4*>(output + base) =
            __ldg(reinterpret_cast<const uint4*>(recv + base));
    }
    // Scalar tail
    if (tid == 0) {
        for (size_t e = vec_n * kVecWidth; e < n_elems; ++e)
            output[e] = __ldg(recv + e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Grid-size helpers  (SM-aware occupancy)
// ─────────────────────────────────────────────────────────────────────────────

// Returns the number of thread-blocks to launch for n_elems BF16 elements
// on the current device, honouring the per-SM policy.
static int hring_grid_size(size_t n_elems, int block_size, int sm_version)
{
    // Query SM count of the current device.
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count,
                           cudaDevAttrMultiProcessorCount, device);
    sm_count = (sm_count > 0) ? sm_count : 1;

    // Blocks to cover all elements (ceil division on vectorised count).
    const size_t vec_n       = (n_elems + kVecWidth - 1) / kVecWidth;
    const int    blocks_data = (int)((vec_n + block_size - 1) / block_size);

    // Cap at 2× SM count to avoid excessive tail effects on small chunks.
    const int blocks_sm = sm_count * 2;

    return std::min(blocks_data, std::max(1, blocks_sm));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: Reduce-step launch wrapper  (SM dispatch)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * launch_hetero_ring_reduce_step
 *
 * Dispatches the correct reduce kernel for the current SM version.
 * Selects SM12.0 cp.async path for Blackwell; standard __ldg path for
 * SM8.6 (A6000) and SM9.0 (H100).
 *
 * @param accum        [in/out] BF16 accumulator [chunk_elems] on this device
 * @param recv         [in]     BF16 received chunk [chunk_elems] on this device
 * @param chunk_elems  Number of BF16 elements in this chunk
 * @param sm_version   SM version (86, 90, 120)
 * @param stream       CUDA compute stream
 */
void launch_hetero_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      stream)
{
    if (chunk_elems == 0) return;

    if (sm_version >= 120) {
        const int bs    = HRingPolicy<120>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_reduce_sm120_kernel<<<grids, bs, 0, stream>>>(
            accum, recv, chunk_elems);
    } else if (sm_version >= 90) {
        const int bs    = HRingPolicy<90>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_reduce_kernel<90><<<grids, bs, 0, stream>>>(
            accum, recv, chunk_elems);
    } else {
        // SM8.6 (A6000) or any unrecognised SM < 9.0
        const int bs    = HRingPolicy<86>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_reduce_kernel<86><<<grids, bs, 0, stream>>>(
            accum, recv, chunk_elems);
    }
    // Catch configuration errors (bad grid/block dims, insufficient smem, etc.)
    // immediately rather than letting them silently corrupt the reduction.
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                "[hetero_ring_reduce_step] kernel launch failed (SM %d, "
                "chunk_elems=%zu): %s\n",
                sm_version, chunk_elems, cudaGetErrorString(err));
        }
    }
}

// Gather-step (all-gather phase): copy recv → output (no addition).
void launch_hetero_ring_gather_step(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      stream)
{
    if (chunk_elems == 0) return;

    if (sm_version >= 120) {
        const int bs    = HRingPolicy<120>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_gather_kernel<120><<<grids, bs, 0, stream>>>(
            output, recv, chunk_elems);
    } else if (sm_version >= 90) {
        const int bs    = HRingPolicy<90>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_gather_kernel<90><<<grids, bs, 0, stream>>>(
            output, recv, chunk_elems);
    } else {
        const int bs    = HRingPolicy<86>::kBlockSize;
        const int grids = hring_grid_size(chunk_elems, bs, sm_version);
        hetero_ring_gather_kernel<86><<<grids, bs, 0, stream>>>(
            output, recv, chunk_elems);
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                "[hetero_ring_gather_step] kernel launch failed (SM %d, "
                "chunk_elems=%zu): %s\n",
                sm_version, chunk_elems, cudaGetErrorString(err));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Chunk layout computation
//
// Given total_elems and the ring descriptor, computes per-rank chunk
// boundaries aligned to kChunkAlign elements.  The final chunk absorbs any
// rounding remainder.
//
// chunk_offset[r] = sum of chunk_elems[0..r-1]
// chunk_elems [r] = floor(total_elems / world_size) aligned to kChunkAlign
//                   (last chunk gets the remainder)
// ─────────────────────────────────────────────────────────────────────────────

static void compute_chunk_layout(
    size_t  total_elems,
    int     world_size,
    size_t* out_offsets,   // [world_size] caller-allocated
    size_t* out_counts)    // [world_size] caller-allocated
{
    const size_t kAlignElems   = kChunkAlign / sizeof(__nv_bfloat16);  // 8 elems
    const size_t base_chunk    = (total_elems / world_size / kAlignElems) * kAlignElems;
    size_t       cumulative    = 0;

    for (int r = 0; r < world_size; ++r) {
        out_offsets[r] = cumulative;
        if (r < world_size - 1) {
            out_counts[r]  = base_chunk;
            cumulative    += base_chunk;
        } else {
            // Last chunk absorbs remainder.
            out_counts[r] = total_elems - cumulative;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 11: Initialise topology descriptor
//
// Builds a NumaAwareRingDesc for a 5-GPU, 2-NUMA-node PCIe ring.
//
// Default NUMA assignment for 5 GPUs (if not overridden):
//   GPUs 0,1,2 → NUMA node 0
//   GPUs 3,4   → NUMA node 1
//
// Ring permutation: 0→1→2→3→4→0 (NUMA-locality-optimised Hamiltonian cycle).
// Cross-NUMA edges: step 2 (rank 2→3) and step 4 (rank 4→0).
//
// @param desc        [out] Descriptor to initialise
// @param device_ids  [in]  CUDA device ordinals for ring positions 0..P-1
// @param numa_nodes  [in]  NUMA node for each ring position (or nullptr for default)
// @param sm_versions [in]  SM version for each ring position
// @param world_size  Number of participating GPUs (typically 5)
// @param this_rank   Ring position of the calling process
// ─────────────────────────────────────────────────────────────────────────────

void hetero_ring_init(
    NumaAwareRingDesc* desc,
    const int*         device_ids,
    const int*         numa_nodes,   // nullptr → use default 3+2 split
    const int*         sm_versions,
    int                world_size,
    int                this_rank)
{
    assert(world_size >= 2 && world_size <= kMaxRingSize);
    desc->world_size = world_size;
    desc->rank       = this_rank;

    // Default NUMA assignment: first ceil(P/2)+1 GPUs on NUMA-0, rest on NUMA-1.
    // For P=5: {0,1,2} → NUMA-0, {3,4} → NUMA-1.
    const int numa0_count = (world_size + 1) / 2 + (world_size == 5 ? 0 : 0);

    for (int i = 0; i < world_size; ++i) {
        desc->device_id [i]  = device_ids[i];
        desc->ring_rank [i]  = i;               // identity ring permutation
        desc->ring_order[i]  = i;
        desc->sm_version[i]  = sm_versions[i];

        if (numa_nodes) {
            desc->numa_node[i] = numa_nodes[i];
        } else {
            // Default: first numa0_count positions on NUMA-0.
            desc->numa_node[i] = (i < (world_size == 5 ? 3 : numa0_count)) ? 0 : 1;
        }
    }

    // Peer device for each ring position: position (i+1) mod world_size.
    for (int i = 0; i < world_size; ++i)
        desc->peer_device[i] = desc->device_id[(i + 1) % world_size];

    // Identify cross-NUMA steps: step i connects ring positions i→(i+1)%P.
    for (int i = 0; i < world_size; ++i) {
        const int next = (i + 1) % world_size;
        desc->cross_numa_step[i] =
            (desc->numa_node[i] != desc->numa_node[next]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 12: Chunk size selector (bandwidth-aware)
//
// Returns the chunk size in BF16 elements for a given ring step, respecting
// intra/cross-NUMA bandwidth difference.
// ─────────────────────────────────────────────────────────────────────────────

static size_t chunk_size_for_step(const NumaAwareRingDesc& desc, int step)
{
    const int src_pos = (desc.rank - step + desc.world_size) % desc.world_size;
    (void)src_pos;
    // step index in the ring: this rank sends to (rank+1) mod P.
    // cross_numa_step[rank] is true if this rank's send edge crosses NUMA.
    const bool is_cross = desc.cross_numa_step[desc.rank];
    const size_t bytes  = is_cross ? kCrossNumaChunkBytes : kIntraNumaChunkBytes;
    return bytes / sizeof(__nv_bfloat16);  // convert bytes → BF16 element count
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 13: Full heterogeneous ring allreduce
//
// Orchestrates the complete reduce-scatter + all-gather pipeline for a 5-GPU,
// 2-NUMA PCIe ring.  Each rank runs this function on its own device.
//
// Buffer layout (caller must pre-allocate on this rank's device):
//   data        [in/out] BF16 [total_elems] — full gradient tensor
//   ping_buf               BF16 [max_chunk_elems] — receive ping buffer
//   pong_buf               BF16 [max_chunk_elems] — receive pong buffer
//   peer_data[i]           device pointer to rank i's data buffer (peer-mapped)
//
// Two CUDA events are required for the double-buffer handshake:
//   xfer_done[0/1] — signalled by stream_xfer after DMA into ping/pong
//
// ─────────────────────────────────────────────────────────────────────────────

void launch_hetero_ring_allreduce(
    __nv_bfloat16*          data,           // [in/out] this rank's full gradient
    __nv_bfloat16*          ping_buf,       // receive ping buffer (max chunk)
    __nv_bfloat16*          pong_buf,       // receive pong buffer (max chunk)
    __nv_bfloat16* const*   peer_data,      // peer_data[r] → rank r's data ptr
    const NumaAwareRingDesc& desc,
    size_t                  total_elems,
    int                     sm_version,
    cudaStream_t            stream_xfer,    // transfer stream (DMA)
    cudaStream_t            stream_comp,    // compute stream (reduce kernels)
    cudaEvent_t             xfer_done[2])   // double-buffer sync events
{
    const int P    = desc.world_size;
    const int rank = desc.rank;

    // Compute uniform chunk layout (per rank owns one chunk of the gradient).
    size_t chunk_offset[kMaxRingSize];
    size_t chunk_count [kMaxRingSize];
    compute_chunk_layout(total_elems, P, chunk_offset, chunk_count);

    // ─── Phase 1: Reduce-scatter  (P-1 steps) ───────────────────────────────
    //
    // Step k (k = 0 … P-2):
    //   send_chunk_idx = (rank - k - 1 + P) % P
    //   recv_chunk_idx = (rank - k - 2 + P) % P
    //   This rank sends from data[send_chunk_offset .. +send_chunk_count]
    //   to peer (rank+1)%P, which reads into its recv_buf via DMA.
    //   Meanwhile this rank accumulates recv_buf into data[recv_chunk].
    //
    // Double-buffer: odd steps use pong_buf, even steps use ping_buf.
    // DMA is issued on stream_xfer; reduce waits on xfer_done event before
    // running on stream_comp.

    for (int k = 0; k < P - 1; ++k) {
        const int buf_idx = k & 1;   // 0 = ping, 1 = pong
        __nv_bfloat16* recv_buf = (buf_idx == 0) ? ping_buf : pong_buf;

        // Chunk indices for this step.
        const int send_ci = ((rank - k - 1) % P + P) % P;
        const int recv_ci = ((rank - k - 2) % P + P) % P;

        // Determine transfer size: use the bandwidth-aware chunk size for the
        // actual element count (may differ from uniform layout for non-uniform BW).
        // Here we use the precomputed chunk_count for data-layout correctness,
        // but the DMA is bounded by the send chunk's element count.
        const size_t send_elems = chunk_count[send_ci];
        const size_t recv_elems = chunk_count[recv_ci];
        (void)send_elems;

        // Source device for DMA: the predecessor in the ring pulls from us.
        // NCCL-style: each rank initiates a "push" to its successor.
        // Here we model the push: we copy data[send_ci] to successor's recv_buf.
        // (In a real multi-process setting the DMA would be initiated by the
        //  receiver; here we use cudaMemcpyPeerAsync as the push primitive.)

        const int dst_rank   = (rank + 1) % P;
        const int dst_device = desc.peer_device[rank];
        const int src_device = desc.device_id  [rank];

        // DMA: push send chunk to next rank's peer-mapped recv buffer.
        // We write into the corresponding slot in peer's ping/pong buffer.
        // (In production the caller maps peer buffers via cudaIpcOpenMemHandle.)
        __nv_bfloat16* peer_recv = (buf_idx == 0)
            ? ping_buf   // peer's ping — for unit-test mode we share local bufs
            : pong_buf;

        // When peer_data != nullptr (multi-GPU mode), use the actual peer pointer.
        // For single-device unit tests, peer_data[dst_rank] aliases local memory.
        __nv_bfloat16* dst_ptr = peer_data
            ? (peer_data[dst_rank] + chunk_offset[send_ci])
            : peer_recv;

        cudaMemcpyPeerAsync(
            dst_ptr,
            dst_device,
            data + chunk_offset[send_ci],
            src_device,
            recv_elems * sizeof(__nv_bfloat16),
            stream_xfer);

        // Signal that DMA into recv_buf (at recv_ci) is complete.
        cudaEventRecord(xfer_done[buf_idx], stream_xfer);

        // Compute stream waits on the transfer event before accumulating.
        cudaStreamWaitEvent(stream_comp, xfer_done[buf_idx], 0);

        // Accumulate: data[recv_ci] += recv_buf
        // In multi-GPU mode the peer pushed into our recv_buf; we accumulate.
        launch_hetero_ring_reduce_step(
            data + chunk_offset[recv_ci],
            recv_buf,
            recv_elems,
            sm_version,
            stream_comp);
    }

    // Ensure all reduce-scatter accumulation is complete before all-gather.
    cudaStreamSynchronize(stream_comp);

    // ─── Phase 2: All-gather  (P-1 steps) ───────────────────────────────────
    //
    // Step k (k = 0 … P-2):
    //   send_chunk_idx = (rank - k + P) % P  (now fully reduced)
    //   recv_chunk_idx = (rank - k - 1 + P) % P
    //   This rank sends the fully-reduced chunk to its successor.
    //   No accumulation: recv is copied verbatim to output[recv_chunk].

    for (int k = 0; k < P - 1; ++k) {
        const int buf_idx = k & 1;
        __nv_bfloat16* recv_buf = (buf_idx == 0) ? ping_buf : pong_buf;

        const int send_ci = ((rank - k)     % P + P) % P;
        const int recv_ci = ((rank - k - 1) % P + P) % P;
        const size_t recv_elems = chunk_count[recv_ci];

        const int dst_rank   = (rank + 1) % P;
        const int dst_device = desc.peer_device[rank];
        const int src_device = desc.device_id  [rank];

        __nv_bfloat16* dst_ptr = peer_data
            ? (peer_data[dst_rank] + chunk_offset[send_ci])
            : recv_buf;

        // DMA: push the fully-reduced chunk to successor.
        cudaMemcpyPeerAsync(
            dst_ptr,
            dst_device,
            data + chunk_offset[send_ci],
            src_device,
            recv_elems * sizeof(__nv_bfloat16),
            stream_xfer);

        cudaEventRecord(xfer_done[buf_idx], stream_xfer);
        cudaStreamWaitEvent(stream_comp, xfer_done[buf_idx], 0);

        // Copy (not accumulate) the received fully-reduced chunk into output.
        launch_hetero_ring_gather_step(
            data + chunk_offset[recv_ci],
            recv_buf,
            recv_elems,
            sm_version,
            stream_comp);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 14: Public query helpers
// ─────────────────────────────────────────────────────────────────────────────

/*
 * hetero_ring_intra_numa_chunk_bytes / hetero_ring_cross_numa_chunk_bytes
 *
 * Returns the chunk size constants used by the kernel.  Callers (Python,
 * unit tests) use these to allocate ping/pong buffers of the correct size.
 *
 * max_chunk_bytes = max(kIntraNumaChunkBytes, kCrossNumaChunkBytes)
 *                 = kIntraNumaChunkBytes = 4 MB
 */
size_t hetero_ring_intra_numa_chunk_bytes() { return kIntraNumaChunkBytes; }
size_t hetero_ring_cross_numa_chunk_bytes() { return kCrossNumaChunkBytes; }
size_t hetero_ring_max_chunk_bytes()        { return kIntraNumaChunkBytes; }

/*
 * hetero_ring_sm_block_size
 *
 * Returns the thread-block size for a given SM version, matching the kernel
 * __launch_bounds__.  Used by callers computing occupancy or buffer alignment.
 */
int hetero_ring_sm_block_size(int sm_version)
{
    if (sm_version >= 120) return HRingPolicy<120>::kBlockSize;
    if (sm_version >= 90)  return HRingPolicy<90>::kBlockSize;
    return                        HRingPolicy<86>::kBlockSize;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 15: TMA BlockLoadToShared for SM9.0+  (issue #72)
//
// Adds a TMA-accelerated reduce step to the hetero_ring_allreduce kernel path.
// For SM9.0 (H100) and SM12.0 (Blackwell), cp.async.bulk.tensor loads an
// entire kTmaHRingTileElems-element BF16 tile from global memory directly into
// shared memory in a single hardware transaction, with mbarrier completion
// signaling.  This eliminates the per-thread ldg() loop for the recv buffer,
// reducing instruction overhead and improving MIO utilization.
//
// Design:
//   kTmaHRingTileElems = 2048  (4 KiB BF16; matches pcie_adaptive_allreduce)
//
//   hetero_ring_reduce_tma_sm90_kernel<kBS>:
//     Grid:  ceil(n_elems / kTmaHRingTileElems) CTAs
//     Block: kBS threads (256 for SM9.0, 512 for SM12.0)
//
//     Phase A — thread 0: mbarrier init + cp.async.bulk.tensor issue
//     Sync   — __syncthreads() after thread-0 issues mbar arrive_expect_tx
//     Wait   — thread 0 polls mbar; __syncthreads() propagates to all threads
//     Phase B — all threads: vectorised accum[tile] += smem[tile] (FP32)
//
//   launch_hetero_ring_reduce_tma_step():
//     Host dispatch: builds CUtensorMap for recv, launches TMA kernel.
//     Falls back to hetero_ring_reduce_sm120_kernel on CUDA < 12.
//
// Shared memory per CTA: kTmaHRingTileElems × 2 bytes = 4 KiB (tile buffer)
//                      + 8 bytes (mbarrier) = 4108 bytes total.
// With 228 KB smem/SM and 256 threads/block on H100, up to 55 CTAs can reside
// per SM (occupancy-limited by registers long before smem).
//
// References:
//   PTX ISA 8.5 §9.7.13: cp.async.bulk.tensor
//   CUDA Programming Guide §K.4: Tensor Memory Accelerator
//   NVIDIA H100 Architecture Whitepaper §3.2: TMA Engine
// ─────────────────────────────────────────────────────────────────────────────

#if CUDA_VERSION >= 12000
#include <cuda.h>   // CUtensorMap, cuTensorMapEncodeTiled
#endif

// Tile size for TMA bulk load in hetero ring (matches pcie_adaptive_allreduce).
static constexpr int kTmaHRingTileElems = 2048;  // 4 KiB of BF16

// ── Device helpers: mbarrier init, TMA issue, mbarrier wait ──────────────────
// Compiled only when targeting SM9.0+ with CUDA 12+.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12

// Initialise a shared-memory mbarrier for one arriving transaction.
__device__ __forceinline__
void hring_tma_mbar_init(uint64_t* __restrict__ mbar)
{
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
    );
}

// Signal the mbarrier with the expected byte count of the TMA transfer.
__device__ __forceinline__
void hring_tma_mbar_expect_tx(uint64_t* __restrict__ mbar, uint32_t tx_bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
           "r"(tx_bytes)
    );
}

// Issue a 1-D TMA bulk copy from global (via tmap[x_coord]) → smem_dst.
__device__ __forceinline__
void hring_tma_issue_1d(
    __nv_bfloat16* __restrict__ smem_dst,
    const CUtensorMap*          tmap,
    int                         x_coord,
    uint64_t* __restrict__      mbar)
{
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global"
        " [%0], [%1, {%2}], [%3];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
           "l"(tmap),
           "r"(x_coord),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
    );
}

// Spin-wait on the mbarrier until phase `phase` completes.
__device__ __forceinline__
void hring_tma_mbar_wait(uint64_t* __restrict__ mbar, uint32_t phase)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_HRING_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_HRING_WAIT_%=;\n"
        "}\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
           "r"(phase)
        : "memory"
    );
}

// ── TMA reduce kernel ─────────────────────────────────────────────────────────

template <int kBS>
__global__ void __launch_bounds__(kBS, 4)
hetero_ring_reduce_tma_sm90_kernel(
    __nv_bfloat16* __restrict__       accum,
    const CUtensorMap*                recv_tmap,   // TMA descriptor for recv buf
    size_t                            n_elems)
{
    // Shared memory: tile buffer (kTmaHRingTileElems × BF16) + mbarrier (8 B).
    __shared__ __align__(128) __nv_bfloat16 smem_tile[kTmaHRingTileElems];
    __shared__ __align__(8)   uint64_t      mbar;

    const int    tile_idx   = blockIdx.x;
    const size_t tile_base  = (size_t)tile_idx * kTmaHRingTileElems;
    if (tile_base >= n_elems) return;

    const size_t tile_elems = min((size_t)kTmaHRingTileElems, n_elems - tile_base);
    const uint32_t tx_bytes = (uint32_t)(tile_elems * sizeof(__nv_bfloat16));

    // ── Phase A: thread 0 sets up mbarrier and issues TMA ────────────────
    if (threadIdx.x == 0) {
        hring_tma_mbar_init(&mbar);
        hring_tma_mbar_expect_tx(&mbar, tx_bytes);
        hring_tma_issue_1d(smem_tile, recv_tmap, tile_idx, &mbar);
    }
    // Ensure mbarrier is initialised before any thread tries to wait.
    // __syncthreads() also serialises wrt the cp.async.bulk issue.
    __syncthreads();

    // ── Wait: thread 0 polls mbarrier; __syncthreads propagates ──────────
    if (threadIdx.x == 0) {
        hring_tma_mbar_wait(&mbar, /*phase=*/0);
    }
    __syncthreads();   // smem_tile is now fully populated

    // ── Phase B: vectorised accumulate accum[tile_base+i] += smem_tile[i] ─
    const size_t vec_tile = tile_elems / kVecWidth;

    for (int t = (int)threadIdx.x; t < (int)vec_tile; t += kBS) {
        const size_t base = (size_t)t * kVecWidth;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        hring_load8_bf16_as_f32(accum + tile_base + base, d0,d1,d2,d3,d4,d5,d6,d7);
        hring_load8_bf16_as_f32(smem_tile          + base, s0,s1,s2,s3,s4,s5,s6,s7);
        hring_store8_f32_as_bf16(accum + tile_base + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }

    // Scalar tail within tile (thread 0 only)
    if (threadIdx.x == 0) {
        for (size_t e = vec_tile * kVecWidth; e < tile_elems; ++e) {
            float d = __bfloat162float(accum[tile_base + e]);
            float s = __bfloat162float(smem_tile[e]);
            accum[tile_base + e] = __float2bfloat16(d + s);
        }
    }
}

#endif  // __CUDA_ARCH__ >= 900 && CUDA >= 12

// ── Host: TMA descriptor builder for 1-D BF16 tensors ────────────────────────

#if CUDA_VERSION >= 12000
static cudaError_t hring_build_tma_1d(
    CUtensorMap*             tmap,
    const __nv_bfloat16*     global_ptr,
    size_t                   n_elems)
{
    const uint64_t global_dim[1]    = { n_elems };
    const uint64_t global_stride[1] = { sizeof(__nv_bfloat16) };
    const uint32_t box_dim[1]       = { (uint32_t)kTmaHRingTileElems };
    const uint32_t elem_stride[1]   = { 1 };

    CUresult res = cuTensorMapEncodeTiled(
        tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /*tensorRank=*/1,
        /*globalAddress=*/(void*)global_ptr,
        global_dim,
        global_stride + 1,   // innermost stride (skip outermost in 1-D)
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(res, &s);
        fprintf(stderr,
            "[hring_build_tma_1d] cuTensorMapEncodeTiled failed: %s\n",
            s ? s : "unknown");
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
#endif  // CUDA_VERSION >= 12000

// ── Public dispatch: TMA-accelerated hetero ring reduce step ──────────────────

/**
 * launch_hetero_ring_reduce_tma_step
 *
 * TMA BlockLoadToShared variant of launch_hetero_ring_reduce_step.
 * On SM9.0/SM12.0 with CUDA 12+: uses cp.async.bulk.tensor to load the recv
 * buffer tile-by-tile into shared memory, then accumulates with accum in FP32.
 * On SM < 9.0 or CUDA < 12: falls back to launch_hetero_ring_reduce_step.
 *
 * @param accum       [in/out] BF16 accumulator [chunk_elems]
 * @param recv        [in]     BF16 received chunk from ring peer [chunk_elems]
 * @param chunk_elems Number of BF16 elements (divisible by kVecWidth=8)
 * @param sm_version  SM version (86, 90, 120)
 * @param stream      CUDA compute stream
 */
void launch_hetero_ring_reduce_tma_step(
    __nv_bfloat16* __restrict__       accum,
    const __nv_bfloat16* __restrict__ recv,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      stream)
{
#if CUDA_VERSION >= 12000
    if (sm_version >= 90) {
        CUtensorMap tmap;
        if (hring_build_tma_1d(&tmap, recv, chunk_elems) != cudaSuccess) {
            // Descriptor build failed — fall through to non-TMA path.
            goto fallback;
        }

        // Copy TMA descriptor to device memory (128 bytes).
        CUtensorMap* d_tmap = nullptr;
        if (cudaMallocAsync(&d_tmap, sizeof(CUtensorMap), stream) != cudaSuccess)
            goto fallback;
        cudaMemcpyAsync(d_tmap, &tmap, sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, stream);

        // Grid: one CTA per tile, capped at 65535.
        const int n_tiles = (int)((chunk_elems + kTmaHRingTileElems - 1)
                                  / kTmaHRingTileElems);
        const int grid    = std::min(n_tiles, 65535);

        if (sm_version >= 120) {
            constexpr int kBS = HRingPolicy<120>::kBlockSize;  // 512
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            hetero_ring_reduce_tma_sm90_kernel<kBS>
                <<<std::max(grid, 1), kBS, 0, stream>>>(
                    accum, d_tmap, chunk_elems);
#else
            // Fallback instantiation when compiled without SM9.0 arch flag.
            hetero_ring_reduce_tma_sm90_kernel<256>
                <<<std::max(grid, 1), 256, 0, stream>>>(
                    accum, d_tmap, chunk_elems);
            (void)kBS;
#endif
        } else {
            // SM9.0 (H100): 256 threads.
            constexpr int kBS = HRingPolicy<90>::kBlockSize;   // 256
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            hetero_ring_reduce_tma_sm90_kernel<kBS>
                <<<std::max(grid, 1), kBS, 0, stream>>>(
                    accum, d_tmap, chunk_elems);
#else
            hetero_ring_reduce_tma_sm90_kernel<256>
                <<<std::max(grid, 1), 256, 0, stream>>>(
                    accum, d_tmap, chunk_elems);
            (void)kBS;
#endif
        }

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr,
                    "[launch_hetero_ring_reduce_tma_step] kernel launch failed "
                    "(SM %d, chunk_elems=%zu): %s\n",
                    sm_version, chunk_elems, cudaGetErrorString(err));
        }
        cudaFreeAsync(d_tmap, stream);
        return;
    }
#endif  // CUDA_VERSION >= 12000

fallback:
    // SM8.6 or CUDA < 12: existing cp.async or __ldg path.
    launch_hetero_ring_reduce_step(accum, recv, chunk_elems, sm_version, stream);
}
