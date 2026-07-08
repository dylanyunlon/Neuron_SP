// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * pcie_adaptive_allreduce.cu  —  NeurIPS 2026 DES-LOC production kernel
 *
 * PCIe-aware adaptive allreduce with topology-aware algorithm selection:
 *   1. SM12.0 async bulk copy paths (cp.async.bulk / TMA where available)
 *   2. Ring-based PCIe allreduce with true double-buffering
 *   3. Compute-communication overlap via separate CUDA streams + events
 *   4. Runtime bandwidth probe driving adaptive chunk sizing
 *   5. NUMA-aware TopoInfo struct driving ring/tree/direct dispatch
 *
 * ═══════════════════════════════════════════════════════════════════════
 * TOPOLOGY-AWARE ALGORITHM SELECTION  (issue #138)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * TopoInfo captures NUMA node membership and PCIe switch affinity for each
 * participating rank so the dispatcher can pick the optimal collective:
 *
 *   payload < kDirectThresholdBytes  → direct reduce (rank 0 pulls all)
 *     Lowest latency for tiny tensors; PCIe setup cost dominates otherwise.
 *
 *   kDirectThresholdBytes ≤ payload < kRingThresholdBytes  → ring allreduce
 *     Classic Rabenseifner ring: bandwidth-optimal when all links are
 *     symmetric; O(2(N-1)/N × payload) traffic.
 *
 *   payload ≥ kRingThresholdBytes  → recursive-halving tree allreduce
 *     Logarithmic depth (log2 N steps) hides latency on large payloads
 *     when the PCIe tree has ≥ 2 NUMA nodes or ≥ 2 PCIe switches.
 *     Falls back to ring if world_size is not a power of two.
 *
 * NUMA placement rules:
 *   - Ranks sharing a NUMA node (numa_node[] equal) use direct cudaMemcpy;
 *     no PCIe traffic crosses the CPU interconnect.
 *   - Cross-NUMA ring step reordering places ranks within the same NUMA
 *     node adjacent in the ring to maximise intra-node bandwidth.
 *   - Tree reduction assigns parent/child edges to prefer intra-switch links
 *     (pcie_switch[] equal) before crossing to another switch domain.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * SM12.0 ASYNC BULK COPY (Blackwell / GB200)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Blackwell introduces TMA (Tensor Memory Accelerator) with cp.async.bulk
 * for bulk DRAM→SMEM transfers.  For the ring-reduce kernel:
 *   • On SM12.0: we use __pipeline_memcpy_async() (cp.async.cg.128) to
 *     stage input from global memory into a shared-memory prefetch buffer
 *     while the prior iteration's data is being processed.  This hides
 *     global memory latency behind FP32 accumulation.
 *   • On SM8.6/SM9.0: we use __ldg() (L2-cached non-temporal loads)
 *     which still hides latency on HBM3 with sufficient ILP.
 *
 * Implementation uses cuda::pipeline (CUDA 11.4+) for SM8.0+ cp.async,
 * and the SM12.0 specialisation bumps to 512-thread blocks for the
 * wider scheduler.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DOUBLE-BUFFERED RING ALLREDUCE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Classic ring allreduce (Rabenseifner 2004) adapted for PCIe:
 *
 *   Phase 1: Reduce-scatter  (world_size-1 steps)
 *     step k: rank r receives chunk[(r-k-1) mod ws] from (r-1) mod ws
 *             and accumulates into local accum[chunk_idx].
 *
 *   Phase 2: All-gather      (world_size-1 steps)
 *     step k: rank r receives fully-reduced chunk[(r-k) mod ws] from
 *             (r-1) mod ws and copies to local output[chunk_idx].
 *
 * Double-buffer: two recv buffers (ping/pong) alternate per step.
 * Transfer stream handles cudaMemcpyPeerAsync; compute stream handles
 * pcie_ring_reduce_kernel.  A CUDA event per buffer signals transfer
 * completion, allowing the compute stream to start reduction immediately
 * without blocking the transfer stream.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * RECURSIVE-HALVING TREE ALLREDUCE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * For large payloads (≥ kRingThresholdBytes) on power-of-two world sizes:
 *
 *   Reduce phase (log2 N steps, bottom-up):
 *     step s (s = 0 … log2(N)-1):
 *       pair distance d = 2^s
 *       ranks with (rank & (2d-1)) == d receive from rank ^ d and accumulate.
 *
 *   Broadcast phase (log2 N steps, top-down):
 *     step s (s = log2(N)-1 … 0):
 *       ranks with (rank & (2d-1)) == 0 send to rank ^ d.
 *
 * NUMA affinity: within each halving step, transfer direction is chosen so
 * same-switch rank pairs (topo.pcie_switch[a] == topo.pcie_switch[b]) are
 * matched in earlier (lower-distance) steps before crossing PCIe switches.
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// CUDA pipeline for cp.async (SM8.0+)
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#endif

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

// BF16 elements per 128-bit vectorised load/store.
static constexpr int    kARVecWidth      = 8;

// Double-buffer pipeline stages for cp.async prefetch.
static constexpr int    kPipelineStages  = 2;

// Minimum/maximum chunk sizes (bytes). Chunk is the unit transferred per
// ring step; it covers kARVecWidth BF16 elements per thread iteration.
static constexpr size_t kMinChunkBytes   = (1ULL << 20);        //   1 MB
static constexpr size_t kMaxChunkBytes   = (256ULL << 20);      // 256 MB
static constexpr size_t kChunkAlign      = kARVecWidth * sizeof(__nv_bfloat16);

// Bandwidth probe: 4 MB saturates PCIe in < 0.5 ms at 8 GB/s.
static constexpr size_t kProbeSizeBytes  = 4UL << 20;

// Target overlap: ~5 ms of PCIe transfer hidden per ring step.
static constexpr float  kTargetOverlapMs = 5.0f;

// ─────────────────────────────────────────────────────────────────────────────
// Topology-aware algorithm selection thresholds  (issue #138)
//
//   payload < kDirectThresholdBytes   → direct reduce  (root pulls all shards)
//   kDirectThresholdBytes ≤ payload
//                  < kRingThresholdBytes  → ring allreduce
//   payload ≥ kRingThresholdBytes        → tree allreduce (if pow2 world)
//
// Rationale:
//   Direct: PCIe per-transfer setup cost ~10 µs; at 16 GB/s that breaks even
//   with ring at ~160 KB.  We round to 256 KB for safety margin.
//
//   Ring→Tree crossover: ring traffic = 2(N-1)/N × P ≈ 2P for large N.
//   Tree traffic = 2 × P × (N-1)/N per step × log2(N) steps, but hides
//   long-haul latency across NUMA domains by reducing hop count.
//   At 64 MB (32 M BF16 elements) the latency gain dominates bandwidth cost
//   for N ≥ 4 with ≥ 2 NUMA nodes.
// ─────────────────────────────────────────────────────────────────────────────

// 256 KB payload threshold: below this use direct reduce.
static constexpr size_t kDirectThresholdBytes = 256ULL * 1024;

// 64 MB payload threshold: at or above this use tree allreduce (if eligible).
static constexpr size_t kRingThresholdBytes   = 64ULL * 1024 * 1024;

// Per-SM block sizes and min-CTAs-per-SM for ring-reduce kernels.
// SM12.0 (Blackwell): 512 threads, 4 CTAs/SM
// SM9.0  (H100):      256 threads, 4 CTAs/SM
// SM8.6  (A6000):     256 threads, 2 CTAs/SM
template <int SmVer> struct ARPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
};
template <> struct ARPolicy<86> {
    // A6000: 84 SMs, 48 GB GDDR6X, 6 MB L2.
    // 256 threads, 2 CTAs/SM — conservative for register headroom.
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
};
template <> struct ARPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
};
template <> struct ARPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Topology descriptor  (issue #138)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * TopoInfo — NUMA and PCIe switch affinity for each rank in the collective.
 *
 * Fields:
 *   world_size   — total number of participating ranks / GPUs.
 *   numa_node    — NUMA node index for each rank (0-indexed).
 *                  Ranks sharing a numa_node[] value communicate without
 *                  crossing the CPU memory bus.
 *   pcie_switch  — PCIe switch domain for each rank (0-indexed).
 *                  Ranks sharing a pcie_switch[] value are connected behind
 *                  the same PCIe switch and incur lower cross-switch latency.
 *   device_id    — CUDA device ordinal for each rank.
 *   num_numa     — number of distinct NUMA nodes present (≥ 1).
 *   num_switches — number of distinct PCIe switch domains present (≥ 1).
 *
 * Construction:
 *   Populate via query_topo_info() which calls cudaDeviceGetAttribute() for
 *   CUDA-visible topology hints and falls back to sane defaults when the
 *   runtime doesn't expose NUMA information.
 */
struct TopoInfo {
    static constexpr int kMaxRanks = 64;

    int world_size;
    int numa_node   [kMaxRanks];  // NUMA node for each rank
    int pcie_switch [kMaxRanks];  // PCIe switch domain for each rank
    int device_id   [kMaxRanks];  // CUDA device ordinal for each rank
    int num_numa;                 // distinct NUMA nodes
    int num_switches;             // distinct PCIe switch domains
};

/**
 * AllreduceAlgo — algorithm chosen by the topology-aware dispatcher.
 */
enum class AllreduceAlgo : int {
    kDirect = 0,  // root collects all shards directly (small payloads)
    kRing   = 1,  // ring allreduce (medium payloads or non-pow2 world)
    kTree   = 2,  // recursive-halving tree (large payloads, pow2 world)
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Bandwidth cache
// ─────────────────────────────────────────────────────────────────────────────

struct BandwidthCache {
    float bw_gbps;    // 0.0 = not yet probed
    int   step_count;
};

static constexpr int kMaxDevices = 16;
static BandwidthCache g_bw_cache[kMaxDevices][kMaxDevices] = {};

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: 128-bit vectorised load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void ar_load8_f32(
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

// Read-only L2 cache variant: issues LD.GLOBAL.NC.128 (non-coherent) via __ldg.
// Mirrors hring_ldg8_bf16_as_f32 in hetero_ring_allreduce.cu.
// Use instead of ar_load8_f32 for read-only (src / recv) buffers.
DS_D_INLINE void ar_ldg8_f32(
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

DS_D_INLINE void ar_store8_bf16(
    __nv_bfloat16* __restrict__ ptr,
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7)
{
    __nv_bfloat16 buf[8] = {
        __float2bfloat16(a0), __float2bfloat16(a1),
        __float2bfloat16(a2), __float2bfloat16(a3),
        __float2bfloat16(a4), __float2bfloat16(a5),
        __float2bfloat16(a6), __float2bfloat16(a7)
    };
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(buf);
}

// __ldg() wrapper for read-only global-memory loads.
DS_D_INLINE uint4 ldg128(const __nv_bfloat16* __restrict__ ptr)
{
    return __ldg(reinterpret_cast<const uint4*>(ptr));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Ring-reduce accumulation kernel
//
//   dst[i] += src[i]  (BF16 in → FP32 accumulation → BF16 out)
//
//   SM8.6 / SM9.0 path: __ldg() for src (read-only L2 hint), direct
//     load for dst (may be in L1 from prior step).
//
//   SM12.0 path: two-stage cp.async.cg pipeline — one iteration prefetches
//     src into shared-memory while the prior iteration's data is processed.
//     This completely hides src global-memory latency behind arithmetic.
//
//   __launch_bounds__ from ARPolicy<SmVer>.
// ─────────────────────────────────────────────────────────────────────────────

// ── SM8.6 / SM9.0 version (standard __ldg path) ──────────────────────────────
template <int SmVer>
__global__ void
__launch_bounds__(ARPolicy<SmVer>::kBlockSize, ARPolicy<SmVer>::kMinBlocksPerSM)
pcie_ring_reduce_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    constexpr int kBS   = ARPolicy<SmVer>::kBlockSize;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;

        // src: read-only L2 cache hint via LD.GLOBAL.NC (ar_ldg8_f32)
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_ldg8_f32(src + base, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
    // Scalar tail
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
}

// ── SM12.0 Blackwell: cp.async double-buffered prefetch path ─────────────────
// Uses cuda::pipeline to issue cp.async.cg.128 instructions that move data
// from global memory to shared memory asynchronously, hiding DRAM latency.
// Two smem stages (ping/pong) alternate: while stage A is being processed,
// stage B is being loaded from global memory.
//
// Shared memory per block: 2 × kBlockSize × kARVecWidth × sizeof(BF16)
//   = 2 × 512 × 8 × 2 = 16 KB  (fits comfortably in 256 KB smem/SM)
__global__ void __launch_bounds__(512, 4)
pcie_ring_reduce_sm120_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    constexpr int kBS      = 512;
    constexpr int kVec     = kARVecWidth;
    // Shared memory: 2 pipeline stages × kBS threads × 8 BF16 each
    constexpr int kSmemElems = kPipelineStages * kBS * kVec;
    __shared__ __nv_bfloat16 smem[kSmemElems];

    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kVec;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDACC_VER_MAJOR__ >= 11
    // cp.async pipeline — two stages.
    namespace pipe_ns = cuda;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kPipelineStages>
        pipe_state;
    auto pipe = cuda::make_pipeline(cg::this_thread_block(), &pipe_state);

    // Prime the pipeline: issue kPipelineStages-1 prefetches before the loop.
    size_t i = tid;
    for (int s = 0; s < kPipelineStages - 1 && i < vec_n; ++s, i += stride) {
        pipe.producer_acquire();
        const size_t base       = i * kVec;
        const int    smem_slot  = s * kBS * kVec + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(
            smem + smem_slot,
            src  + base,
            kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();
    }

    // Main double-buffered loop.
    size_t j = tid;  // j: the "compute" index (lags i by kPipelineStages-1)
    for (; i < vec_n; i += stride, j += stride) {
        // Prefetch next iteration into the write stage.
        pipe.producer_acquire();
        const int ws = ((i / stride) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(smem + ws, src + i * kVec, kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();

        // Wait for the read stage (which holds src[j..j+kVec]).
        pipe.consumer_wait();
        const int rs = (((j / stride)) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;

        float d0,d1,d2,d3,d4,d5,d6,d7;
        ar_load8_f32(dst + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(smem + rs, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

    // Drain remaining pipeline stages.
    for (int s = 0; s < kPipelineStages - 1 && j < vec_n; ++s, j += stride) {
        pipe.consumer_wait();
        const int rs = (((j / stride)) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        ar_load8_f32(dst + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(smem + rs, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

#else
    // Fallback for toolchains that don't support cp.async: plain __ldg.
    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_ldg8_f32(src + base, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
#endif

    // Scalar tail (thread 0 only)
    if (tid == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
    (void)smem;  // silence unused-variable warnings in non-SM12 fallback
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Tree-reduce accumulation kernel  (issue #138)
//
// Used in the recursive-halving tree allreduce for large payloads.
// Functionally identical to the ring-reduce kernel but compiled with a
// separate entry point so profiling can distinguish tree vs ring steps.
//
// dst[i] += src[i]  (BF16 in → FP32 accumulation → BF16 out)
//
// The tree reduce phase calls this on the receiver rank after receiving a
// shard from its child rank in the recursive-halving schedule.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(ARPolicy<SmVer>::kBlockSize, ARPolicy<SmVer>::kMinBlocksPerSM)
pcie_tree_reduce_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    constexpr int kBS   = ARPolicy<SmVer>::kBlockSize;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_ldg8_f32(src + base, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Finalisation kernel — divide by world_size
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(ARPolicy<SmVer>::kBlockSize, ARPolicy<SmVer>::kMinBlocksPerSM)
pcie_allreduce_finalise_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems,
    float  inv_world_size)
{
    constexpr int kBS   = ARPolicy<SmVer>::kBlockSize;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;
        float a0,a1,a2,a3,a4,a5,a6,a7;
        ar_ldg8_f32(src + base, a0,a1,a2,a3,a4,a5,a6,a7);
        ar_store8_bf16(out + base,
            a0*inv_world_size, a1*inv_world_size,
            a2*inv_world_size, a3*inv_world_size,
            a4*inv_world_size, a5*inv_world_size,
            a6*inv_world_size, a7*inv_world_size);
    }
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e)
            out[e] = __float2bfloat16(__bfloat162float(__ldg(src + e)) * inv_world_size);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Gradient packing kernel — binary-search chunk lookup
//
// Gathers non-contiguous gradient shards into a flat BF16 bucket.
// Uses a precomputed prefix-sum array for O(log C) chunk lookup per vector,
// replacing the prior O(C) linear scan.
// ─────────────────────────────────────────────────────────────────────────────

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
pcie_gradient_pack_kernel(
    __nv_bfloat16* __restrict__         bucket,
    const PcieGradChunk* __restrict__   chunks,
    const size_t* __restrict__          prefix_ends,  // [num_chunks]
    int    num_chunks,
    size_t bucket_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < bucket_elems / kARVecWidth; i += stride) {
        const size_t elem_base = i * kARVecWidth;

        // Binary search: smallest c such that prefix_ends[c] > elem_base.
        int lo = 0, hi = num_chunks - 1, ci = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (prefix_ends[mid] <= elem_base)
                lo = mid + 1;
            else { ci = mid; hi = mid - 1; }
        }

        const size_t chunk_start = (ci == 0) ? 0 : prefix_ends[ci - 1];
        const size_t within      = elem_base - chunk_start;
        const __nv_bfloat16* src = chunks[ci].src + chunks[ci].offset + within;

        // 128-bit BF16→BF16 copy (no conversion needed).
        *reinterpret_cast<uint4*>(bucket + elem_base) =
            *reinterpret_cast<const uint4*>(src);
    }

    // Scalar tail (at most kARVecWidth-1 elements).
    if (tid == 0) {
        const size_t vec_start = (bucket_elems / kARVecWidth) * kARVecWidth;
        size_t cum = 0;
        for (int c = 0; c < num_chunks && cum < bucket_elems; ++c) {
            const size_t chunk_end = cum + chunks[c].length;
            for (size_t e = vec_start; e < bucket_elems; ++e) {
                if (e >= cum && e < chunk_end)
                    bucket[e] = chunks[c].src[chunks[c].offset + (e - cum)];
            }
            cum = chunk_end;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: Runtime bandwidth probe
// ─────────────────────────────────────────────────────────────────────────────

float probe_pcie_bandwidth(int src_device, int dst_device)
{
    BandwidthCache& entry = g_bw_cache[src_device][dst_device];

    // Return cached result on subsequent calls; bw_gbps == 0.0 means uncached.
    // Re-check after potential concurrent write: if another thread raced us and
    // already filled the cache by the time we enter, use that value.
    // (CUDA serialises host-side CUDA API calls within a process, so this is safe
    // without a mutex under the assumption that probes are issued from a single
    // orchestrator thread before any training streams are launched.)
    if (entry.bw_gbps > 0.f) return entry.bw_gbps;

    void* src_buf = nullptr;
    void* dst_buf = nullptr;
    cudaEvent_t   ev_start = nullptr, ev_stop = nullptr;
    cudaStream_t  probe_stream = nullptr;
    float         bw_gbps = -1.f;  // -1 = probe failed

    // ── Allocate resources (all errors are fatal for the probe) ──
    cudaSetDevice(src_device);
    if (cudaMalloc(&src_buf, kProbeSizeBytes) != cudaSuccess) goto cleanup;
    if (cudaMemset(src_buf, 0, kProbeSizeBytes) != cudaSuccess) goto cleanup;

    cudaSetDevice(dst_device);
    if (cudaMalloc(&dst_buf, kProbeSizeBytes) != cudaSuccess) goto cleanup;

    cudaSetDevice(src_device);
    if (cudaEventCreate(&ev_start) != cudaSuccess) goto cleanup;
    if (cudaEventCreate(&ev_stop)  != cudaSuccess) goto cleanup;
    if (cudaStreamCreate(&probe_stream) != cudaSuccess) goto cleanup;

    {
        // Warm-up: prime PCIe TLB entries and fill dst L2 cache.
        if (cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                                kProbeSizeBytes, probe_stream) != cudaSuccess) goto cleanup;
        if (cudaStreamSynchronize(probe_stream) != cudaSuccess) goto cleanup;

        // Timed transfer: record start/stop events around a single peer copy.
        if (cudaEventRecord(ev_start, probe_stream) != cudaSuccess) goto cleanup;
        if (cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                                kProbeSizeBytes, probe_stream) != cudaSuccess) goto cleanup;
        if (cudaEventRecord(ev_stop, probe_stream) != cudaSuccess) goto cleanup;
        if (cudaStreamSynchronize(probe_stream) != cudaSuccess) goto cleanup;

        float elapsed_ms = 0.f;
        if (cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop) != cudaSuccess
            || elapsed_ms <= 0.f) goto cleanup;

        // Unidirectional bandwidth (bytes/s → GB/s).
        bw_gbps = (float)kProbeSizeBytes / (elapsed_ms * 1e-3f) / 1e9f;
    }

cleanup:
    // Destroy resources in reverse-allocation order; ignore errors during cleanup.
    if (probe_stream) { cudaStreamDestroy(probe_stream); probe_stream = nullptr; }
    if (ev_stop)      { cudaEventDestroy(ev_stop);  ev_stop  = nullptr; }
    if (ev_start)     { cudaEventDestroy(ev_start); ev_start = nullptr; }
    if (src_buf) { cudaSetDevice(src_device); cudaFree(src_buf); src_buf = nullptr; }
    if (dst_buf) { cudaSetDevice(dst_device); cudaFree(dst_buf); dst_buf = nullptr; }
    cudaSetDevice(src_device);

    if (bw_gbps > 0.f) {
        // Double-check cache race: write only if still uncached (concurrent call may have won).
        if (entry.bw_gbps <= 0.f) {
            entry.bw_gbps    = bw_gbps;
            entry.step_count = 0;
        }
    } else {
        // Probe failed — emit a warning; do NOT cache so the next call can retry.
        fprintf(stderr,
            "[probe_pcie_bandwidth] WARNING: probe from device %d to %d failed; "
            "falling back to 8 GB/s default (likely PCIe peer access not enabled).\n",
            src_device, dst_device);
        return 8.f;
    }

    return entry.bw_gbps;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Adaptive chunk sizing
// ─────────────────────────────────────────────────────────────────────────────

size_t compute_adaptive_chunk_size(float pcie_bw_gbps)
{
    // Chunk size = bandwidth × target_overlap_time.
    // This ensures ~kTargetOverlapMs ms of PCIe transfer is hidden per ring step.
    float chunk_f = pcie_bw_gbps * 1e9f * kTargetOverlapMs * 1e-3f;
    size_t raw = (size_t)chunk_f;
    raw = (raw / kChunkAlign) * kChunkAlign;

    // PCIe Gen5 floor: when measured bandwidth ≥ 28 GB/s (≈ unidirectional Gen5
    // x16 theoretical 32 GB/s minus overhead), raise the minimum chunk to 8 MB.
    // Gen4 x16 peaks at ~16 GB/s unidirectional so 28 GB/s is a safe threshold.
    // Override via NEURON_PCIE_MIN_CHUNK_MB (bytes, rounded to kChunkAlign).
    static const size_t kEffMinChunkBytes = []() -> size_t {
        const char* env = std::getenv("NEURON_PCIE_MIN_CHUNK_MB");
        if (env && *env) {
            size_t mb = (size_t)std::atol(env);
            if (mb > 0) return ((mb << 20) / kChunkAlign) * kChunkAlign;
        }
        return 0;  // 0 = not set; select dynamically below
    }();

    size_t min_chunk = kMinChunkBytes;  // default: 1 MB (Gen4-appropriate)
    if (pcie_bw_gbps >= 28.f)
        min_chunk = 8ULL << 20;         // 8 MB floor for Gen5 x16
    if (kEffMinChunkBytes > 0)
        min_chunk = kEffMinChunkBytes;  // env-var takes precedence

    raw = std::max(raw, min_chunk);
    raw = std::min(raw, kMaxChunkBytes);
    return raw;
}

size_t compute_pcie_bucket_size(float pcie_bw_gbps)
{
    return compute_adaptive_chunk_size(pcie_bw_gbps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 11: Topology query and algorithm selection  (issue #138)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * query_topo_info
 *
 * Probes CUDA runtime attributes to fill a TopoInfo struct.
 * Uses cudaDeviceGetAttribute(cudaDevAttrHostNativeAtomicSupported) as a
 * proxy for NUMA-local access; falls back to assuming all ranks share one
 * NUMA node when the attribute is unavailable.
 *
 * PCIe switch domain is approximated by the integer division
 * device_id / kDevicesPerSwitch (default 4: typical 4-GPU PLX switch).
 * Override with the NEURON_PCIE_SWITCH_WIDTH env var if your topology differs.
 *
 * @param topo          [out] Populated topology descriptor
 * @param device_ids    [in]  CUDA device ordinals for each rank
 * @param world_size    Number of participating ranks
 */
void query_topo_info(TopoInfo* topo, const int* device_ids, int world_size)
{
    // Devices-per-switch: how many GPUs share a PCIe switch port.
    // NEURON_PCIE_SWITCH_WIDTH overrides; default 4 (common PLX / Broadcom).
    const char* env_width = getenv("NEURON_PCIE_SWITCH_WIDTH");
    int devs_per_switch = (env_width && env_width[0]) ? atoi(env_width) : 4;
    if (devs_per_switch < 1) devs_per_switch = 1;

    topo->world_size = world_size;
    topo->num_numa     = 0;
    topo->num_switches = 0;

    int max_numa_seen   = -1;
    int max_switch_seen = -1;

    for (int r = 0; r < world_size; ++r) {
        const int dev = device_ids[r];
        topo->device_id[r] = dev;

        // NUMA heuristic: query cudaDevAttrIsMultiGpuBoard as a topology hint.
        // Devices sharing a board tend to sit in the same NUMA domain.
        // This is a best-effort approximation; production deployments should
        // inject accurate NUMA mappings via the binding layer (pybind11 / C API).
        int numa_val = 0;
#if CUDART_VERSION >= 9000
        // cudaDevAttrHostNativeAtomicSupported != 0 on NUMA-local GPUs only.
        int native_atomic = 0;
        if (cudaDeviceGetAttribute(&native_atomic,
                cudaDevAttrHostNativeAtomicSupported, dev) == cudaSuccess) {
            numa_val = native_atomic ? 0 : (dev / 4);
        } else {
            numa_val = dev / 4;
        }
#else
        numa_val = dev / 4;
#endif
        topo->numa_node[r] = numa_val;
        if (numa_val > max_numa_seen) max_numa_seen = numa_val;

        const int sw = dev / devs_per_switch;
        topo->pcie_switch[r] = sw;
        if (sw > max_switch_seen) max_switch_seen = sw;
    }

    topo->num_numa     = max_numa_seen + 1;
    topo->num_switches = max_switch_seen + 1;
}

/**
 * select_allreduce_algo
 *
 * Topology-aware algorithm dispatcher for issue #138.
 *
 * Decision logic:
 *
 *   1. payload_bytes < kDirectThresholdBytes  → kDirect
 *      Small tensors: PCIe setup overhead dominates; root collects all shards
 *      via point-to-point cudaMemcpyPeerAsync.  Single-NUMA clusters use this
 *      up to kDirectThresholdBytes regardless of world_size.
 *
 *   2. payload_bytes < kRingThresholdBytes  → kRing
 *      Medium tensors: ring allreduce is bandwidth-optimal (traffic = 2P).
 *      Used even for multi-NUMA clusters at this size range because the
 *      NUMA-reordered ring (ranks within the same NUMA node placed adjacent)
 *      keeps hot transfers intra-node.
 *
 *   3. payload_bytes ≥ kRingThresholdBytes AND is_power_of_two(world_size)
 *      AND topo.num_switches > 1  → kTree
 *      Large tensors with a multi-switch PCIe topology: recursive-halving
 *      tree reduces the number of cross-switch hops from O(N) to O(log N),
 *      hiding inter-NUMA latency at the cost of slightly higher bandwidth
 *      usage (traffic = 2P × (N-1)/N × log2 N / N).
 *
 *   4. payload_bytes ≥ kRingThresholdBytes AND
 *      (non-power-of-two OR single-switch)  → kRing
 *      Tree requires power-of-two world size; fall back to ring otherwise.
 *      Single-switch clusters don't benefit from tree's latency reduction.
 *
 * @param topo          Populated topology descriptor
 * @param payload_bytes Total allreduce payload in bytes
 * @returns             AllreduceAlgo enum selecting the algorithm
 */
AllreduceAlgo select_allreduce_algo(const TopoInfo& topo, size_t payload_bytes)
{
    // Threshold 1: direct for tiny payloads.
    if (payload_bytes < kDirectThresholdBytes)
        return AllreduceAlgo::kDirect;

    // Threshold 2: ring for medium payloads.
    if (payload_bytes < kRingThresholdBytes)
        return AllreduceAlgo::kRing;

    // Threshold 3: tree for large payloads on eligible topologies.
    //   Requirements: power-of-two world_size AND multi-switch topology.
    const int ws = topo.world_size;
    const bool is_pow2     = (ws > 0) && ((ws & (ws - 1)) == 0);
    const bool multi_switch = topo.num_switches > 1;

    if (is_pow2 && multi_switch)
        return AllreduceAlgo::kTree;

    // Fallback: ring (handles non-pow2 world and single-switch).
    return AllreduceAlgo::kRing;
}

/**
 * build_numa_ring_order
 *
 * Constructs a NUMA-aware ring permutation for kRing mode.
 * Ranks sharing the same NUMA node are placed adjacent in the ring to
 * maximise intra-node bandwidth and reduce cross-NUMA PCIe hops.
 *
 * Algorithm:
 *   1. Group ranks by numa_node.
 *   2. Within each NUMA group, sort by device_id (arbitrary stable order).
 *   3. Concatenate groups; the resulting order is the ring order.
 *
 * @param topo        Topology descriptor
 * @param ring_order  [out] Caller-allocated array of length world_size;
 *                    ring_order[i] = original rank index for ring position i.
 */
void build_numa_ring_order(const TopoInfo& topo, int* ring_order)
{
    const int ws = topo.world_size;
    bool placed[TopoInfo::kMaxRanks] = {};
    int  pos = 0;

    for (int numa = 0; numa < topo.num_numa && pos < ws; ++numa) {
        for (int r = 0; r < ws; ++r) {
            if (!placed[r] && topo.numa_node[r] == numa) {
                ring_order[pos++] = r;
                placed[r] = true;
            }
        }
    }
    // Append any unplaced ranks (should not happen with well-formed TopoInfo).
    for (int r = 0; r < ws; ++r) {
        if (!placed[r]) ring_order[pos++] = r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 12: Tree-reduce launch helper  (issue #138)
//
// launch_pcie_tree_reduce_step dispatches the pcie_tree_reduce_kernel for a
// single recursive-halving step.  API mirrors launch_pcie_ring_reduce_step
// so the orchestration layer can treat both uniformly.
// ─────────────────────────────────────────────────────────────────────────────

void launch_pcie_tree_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream)
{
    const size_t vec_n = chunk_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = ARPolicy<120>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_tree_reduce_kernel<120>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else if (sm_version >= 90) {
        constexpr int kBS = ARPolicy<90>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_tree_reduce_kernel<90>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else {
        constexpr int kBS = ARPolicy<86>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_tree_reduce_kernel<86>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[pcie_tree_reduce_step] kernel launch failed (SM %d, "
                "chunk_elems=%zu): %s\n",
                sm_version, chunk_elems, cudaGetErrorString(err));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 13: Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_pcie_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream)
{
    const size_t vec_n = chunk_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = ARPolicy<120>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_sm120_kernel
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else if (sm_version >= 90) {
        constexpr int kBS = ARPolicy<90>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<90>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else {
        constexpr int kBS = ARPolicy<86>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<86>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[pcie_ring_reduce_step] kernel launch failed (SM %d, "
                "chunk_elems=%zu): %s\n",
                sm_version, chunk_elems, cudaGetErrorString(err));
    }
}

void launch_pcie_gradient_pack(
    __nv_bfloat16*       bucket,
    const PcieGradChunk* chunks,
    int                  num_chunks,
    size_t               bucket_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    constexpr int kPackBS = 256;

    // Build device-side descriptors and prefix sums.
    PcieGradChunk* d_chunks = nullptr;
    size_t*        d_prefix = nullptr;
    cudaMallocAsync(&d_chunks, num_chunks * sizeof(PcieGradChunk), stream);
    cudaMallocAsync(&d_prefix, num_chunks * sizeof(size_t), stream);

    cudaMemcpyAsync(d_chunks, chunks, num_chunks * sizeof(PcieGradChunk),
                    cudaMemcpyHostToDevice, stream);

    // Prefix-sum on host → copy to device.
    size_t* h_prefix = new size_t[num_chunks];
    size_t cum = 0;
    for (int c = 0; c < num_chunks; ++c) { cum += chunks[c].length; h_prefix[c] = cum; }
    cudaMemcpyAsync(d_prefix, h_prefix, num_chunks * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    delete[] h_prefix;

    const size_t vec_elems = bucket_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kPackBS - 1) / kPackBS, (size_t)65535);

    pcie_gradient_pack_kernel<kPackBS>
        <<<std::max(grid,1), kPackBS, 0, stream>>>(
            bucket, d_chunks, d_prefix, num_chunks, bucket_elems);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[pcie_gradient_pack] kernel launch failed "
                "(num_chunks=%d, bucket_elems=%zu): %s\n",
                num_chunks, bucket_elems, cudaGetErrorString(err));
    }

    cudaFreeAsync(d_chunks, stream);
    cudaFreeAsync(d_prefix, stream);
}

void launch_pcie_ring_reduce(
    __nv_bfloat16*       dst,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    launch_pcie_ring_reduce_step(dst, src, n_elems, sm_version, stream);
}

void launch_pcie_allreduce_finalise(
    __nv_bfloat16*       out,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  world_size,
    int                  sm_version,
    cudaStream_t         stream)
{
    const float inv_ws = 1.f / (float)world_size;
    const size_t vec_n = n_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = ARPolicy<120>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<120>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    } else if (sm_version >= 90) {
        constexpr int kBS = ARPolicy<90>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<90>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    } else {
        constexpr int kBS = ARPolicy<86>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<86>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[pcie_allreduce_finalise] kernel launch failed (SM %d, "
                "n_elems=%zu): %s\n",
                sm_version, n_elems, cudaGetErrorString(err));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 14: PCIe-aware adaptive bucketing  (issue #24)
//
// Problem: a flat ring allreduce packs all gradient chunks into uniform-size
// buckets, ignoring that different PCIe links have different bandwidths.
// Slow cross-NUMA links stall the ring when they must transfer the same bucket
// size as faster intra-NUMA links.
//
// Solution: per-rank adaptive bucket sizing.
//   1. For each (src_rank, dst_rank) ring edge, probe_pcie_bandwidth() gives
//      measured BW.  compute_adaptive_chunk_size() maps BW → chunk bytes.
//   2. pcie_adaptive_bucket_plan() assigns each gradient chunk to a bucket
//      slot, respecting per-edge chunk limits.  It outputs:
//        • BucketPlan::bucket_sizes[rank] — bytes for this rank's send bucket.
//        • BucketPlan::chunk_to_bucket[]  — which bucket each gradient chunk
//          belongs to, for device-side gather/scatter.
//        • BucketPlan::bucket_offsets[]   — prefix-sum of bucket_sizes for
//          flat buffer addressing.
//   3. The device-side adaptive unpack kernel pcie_adaptive_unpack_kernel()
//      scatters a received flat bucket back to gradient tensor offsets using
//      a compact index table (AdaptiveBucketIndex), avoiding a second
//      pcie_gradient_pack_kernel launch on the receive side.
//
// Kernel: pcie_adaptive_unpack_kernel
//   Inverse of pcie_gradient_pack_kernel: for each (dst_ptr, src_offset,
//   length) entry in the index table, copies bucket[src_offset..+length] to
//   dst_ptr[0..length-1] using 128-bit vectorised stores.
//
// Host API: pcie_adaptive_bucket_plan / launch_pcie_adaptive_unpack
// ─────────────────────────────────────────────────────────────────────────────

// Maximum gradient chunks supported per bucket plan.
static constexpr int kMaxGradChunks = 1024;

// Maximum participating ranks for a bucket plan.
static constexpr int kMaxBucketRanks = 64;

/**
 * AdaptiveBucketIndex — device-side scatter descriptor for one bucket entry.
 *
 * After a bucket is received from a ring peer, the unpack kernel reads this
 * table to scatter contiguous bucket regions back to the destination gradient
 * tensor pointers.
 *
 *   dst       — device pointer to the target gradient tensor region.
 *   src_off   — byte offset within the received flat bucket buffer.
 *   n_elems   — BF16 elements to copy from bucket[src_off] → dst[0..].
 */
struct AdaptiveBucketIndex {
    __nv_bfloat16* dst;      // destination gradient tensor pointer (device)
    size_t         src_off;  // element offset within the flat bucket
    size_t         n_elems;  // number of BF16 elements
};

/**
 * BucketPlan — host-side result from pcie_adaptive_bucket_plan().
 *
 * All arrays are indexed by rank [0, world_size).
 * chunk_bucket[c] gives the rank whose send bucket gradient chunk c belongs to.
 */
struct BucketPlan {
    static constexpr int kMaxRanks  = kMaxBucketRanks;
    static constexpr int kMaxChunks = kMaxGradChunks;

    int    world_size;
    size_t bucket_sizes  [kMaxRanks];   // bytes per rank's send bucket
    size_t bucket_offsets[kMaxRanks];   // prefix-sum of bucket_sizes (flat buf)
    int    chunk_bucket  [kMaxChunks];  // chunk c → assigned rank
    int    num_chunks;
};

// ── Device kernel: scatter flat bucket → gradient tensor regions ──────────────
//
// Each thread handles kARVecWidth BF16 elements (128-bit vector store).
// index[i].{dst, src_off, n_elems} covers at most one contiguous gradient
// region; entries are sorted by src_off so accesses to `bucket` are coalesced.
//
// Launch: 1D grid over total bucket elements / kARVecWidth.
// Shared memory: none.  Register pressure: ~10 regs per thread.

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
pcie_adaptive_unpack_kernel(
    const __nv_bfloat16* __restrict__      bucket,
    const AdaptiveBucketIndex* __restrict__ index,
    const size_t* __restrict__              prefix_ends,  // [num_entries]
    int    num_entries,
    size_t total_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < total_elems / kARVecWidth; i += stride) {
        const size_t elem_base = i * kARVecWidth;

        // Binary search: find entry whose range covers elem_base.
        int lo = 0, hi = num_entries - 1, ei = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (prefix_ends[mid] <= elem_base)
                lo = mid + 1;
            else { ei = mid; hi = mid - 1; }
        }

        const size_t entry_start = (ei == 0) ? 0 : prefix_ends[ei - 1];
        const size_t within      = elem_base - entry_start;
        const size_t src_base    = index[ei].src_off + within;

        // 128-bit copy: bucket[src_base..+8] → dst[within..+8]
        *reinterpret_cast<uint4*>(index[ei].dst + within) =
            *reinterpret_cast<const uint4*>(bucket + src_base);
    }

    // Scalar tail (thread 0 handles residual elements not in vec loop)
    if (tid == 0) {
        const size_t vec_start = (total_elems / kARVecWidth) * kARVecWidth;
        size_t cum = 0;
        for (int e_idx = 0; e_idx < num_entries && cum < total_elems; ++e_idx) {
            const size_t entry_end = cum + index[e_idx].n_elems;
            for (size_t e = vec_start; e < total_elems && e < entry_end; ++e)
                index[e_idx].dst[e - cum] = bucket[index[e_idx].src_off + (e - cum)];
            cum = entry_end;
        }
    }
}

// ── Host: adaptive bucket plan ────────────────────────────────────────────────

/**
 * pcie_adaptive_bucket_plan
 *
 * Assigns gradient chunks to per-rank send buckets proportional to the
 * measured PCIe bandwidth on each ring edge.
 *
 * Algorithm:
 *   For each rank r in [0, world_size):
 *     bw[r] = probe_pcie_bandwidth(device_ids[r], device_ids[(r+1) % ws])
 *     bucket_bytes[r] = compute_adaptive_chunk_size(bw[r])
 *
 *   Chunks are assigned round-robin to ranks in decreasing bucket-size order
 *   (greedy bin-packing approximation), capped so each chunk fits within
 *   bucket_bytes[rank] without fragmenting a 128-bit boundary.
 *
 * @param plan        [out] Populated BucketPlan
 * @param chunks      [in]  Gradient chunk descriptors (num_chunks entries)
 * @param num_chunks  Number of gradient chunks
 * @param device_ids  CUDA device ordinals for each rank, length world_size
 * @param world_size  Number of participating ranks
 */
void pcie_adaptive_bucket_plan(
    BucketPlan*          plan,
    const PcieGradChunk* chunks,
    int                  num_chunks,
    const int*           device_ids,
    int                  world_size)
{
    if (!plan || !chunks || !device_ids
        || num_chunks <= 0 || world_size <= 0
        || world_size > BucketPlan::kMaxRanks
        || num_chunks > BucketPlan::kMaxChunks) {
        fprintf(stderr, "[pcie_adaptive_bucket_plan] invalid arguments\n");
        return;
    }

    plan->world_size = world_size;
    plan->num_chunks = num_chunks;

    // ── Step 1: probe per-edge BW and derive target bucket sizes ──────────
    size_t bw_chunk_bytes[BucketPlan::kMaxRanks] = {};
    for (int r = 0; r < world_size; ++r) {
        const int src_dev = device_ids[r];
        const int dst_dev = device_ids[(r + 1) % world_size];
        const float bw = probe_pcie_bandwidth(src_dev, dst_dev);
        bw_chunk_bytes[r] = compute_adaptive_chunk_size(bw);
    }

    // ── Step 2: greedy assignment of chunks to ranks ──────────────────────
    // Track fill level (bytes) per rank's bucket.
    size_t fill[BucketPlan::kMaxRanks] = {};
    for (int r = 0; r < world_size; ++r)
        fill[r] = 0;

    for (int c = 0; c < num_chunks; ++c) {
        const size_t chunk_bytes = chunks[c].length * sizeof(__nv_bfloat16);

        // Find the rank with the most remaining capacity that can hold chunk c.
        int best_rank = -1;
        size_t best_room = 0;

        for (int r = 0; r < world_size; ++r) {
            const size_t cap  = bw_chunk_bytes[r];
            const size_t room = (fill[r] < cap) ? (cap - fill[r]) : 0;
            if (room >= chunk_bytes && room > best_room) {
                best_room = room;
                best_rank = r;
            }
        }

        if (best_rank < 0) {
            // No rank can fit this chunk without overflow — assign to the
            // rank with the largest bucket limit (spillover allowed; the
            // actual transfer will span multiple ring steps).
            best_rank = 0;
            for (int r = 1; r < world_size; ++r)
                if (bw_chunk_bytes[r] > bw_chunk_bytes[best_rank])
                    best_rank = r;
        }

        plan->chunk_bucket[c] = best_rank;
        fill[best_rank] += chunk_bytes;
    }

    // ── Step 3: finalise bucket sizes (actual fill, aligned to kChunkAlign) ─
    for (int r = 0; r < world_size; ++r) {
        size_t sz = fill[r];
        // Round up to kChunkAlign (= 8 × sizeof(BF16) = 16 bytes).
        sz = ((sz + kChunkAlign - 1) / kChunkAlign) * kChunkAlign;
        plan->bucket_sizes[r] = sz;
    }

    // ── Step 4: prefix-sum of bucket sizes for flat buffer layout ─────────
    size_t cum = 0;
    for (int r = 0; r < world_size; ++r) {
        plan->bucket_offsets[r] = cum;
        cum += plan->bucket_sizes[r];
    }
}

/**
 * launch_pcie_adaptive_unpack
 *
 * Scatters a flat received bucket buffer back to gradient tensor regions.
 * Called on the receive side after a ring transfer completes, replacing a
 * separate pcie_gradient_pack_kernel invocation.
 *
 * @param bucket        [in]  Flat BF16 bucket received from ring peer [total_elems]
 * @param index         [in]  Host-side scatter index array (num_entries entries)
 * @param num_entries   Number of AdaptiveBucketIndex entries
 * @param total_elems   Total BF16 elements in bucket (sum of index[].n_elems)
 * @param sm_version    SM version for block-size selection
 * @param stream        CUDA stream
 */
void launch_pcie_adaptive_unpack(
    const __nv_bfloat16*       bucket,
    const AdaptiveBucketIndex* index,
    int                        num_entries,
    size_t                     total_elems,
    int                        sm_version,
    cudaStream_t               stream)
{
    if (!bucket || !index || num_entries <= 0 || total_elems == 0) return;

    constexpr int kUnpackBS = 256;

    // Build device-side index and prefix-end arrays.
    AdaptiveBucketIndex* d_index  = nullptr;
    size_t*              d_prefix = nullptr;
    cudaMallocAsync(&d_index,  num_entries * sizeof(AdaptiveBucketIndex), stream);
    cudaMallocAsync(&d_prefix, num_entries * sizeof(size_t), stream);

    cudaMemcpyAsync(d_index, index, num_entries * sizeof(AdaptiveBucketIndex),
                    cudaMemcpyHostToDevice, stream);

    // Build host prefix-end array: prefix_ends[i] = Σ_{j≤i} index[j].n_elems
    size_t* h_prefix = new size_t[num_entries];
    size_t  cum = 0;
    for (int i = 0; i < num_entries; ++i) {
        cum += index[i].n_elems;
        h_prefix[i] = cum;
    }
    cudaMemcpyAsync(d_prefix, h_prefix, num_entries * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    delete[] h_prefix;

    const size_t vec_elems = total_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kUnpackBS - 1) / kUnpackBS, (size_t)65535);

    pcie_adaptive_unpack_kernel<kUnpackBS>
        <<<std::max(grid, 1), kUnpackBS, 0, stream>>>(
            bucket, d_index, d_prefix, num_entries, total_elems);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "[pcie_adaptive_unpack] kernel launch failed "
                "(num_entries=%d, total_elems=%zu): %s\n",
                num_entries, total_elems, cudaGetErrorString(err));
    }

    cudaFreeAsync(d_index,  stream);
    cudaFreeAsync(d_prefix, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 15: TMA-based BlockLoadToShared for SM9.0+  (issue #72)
//
// SM9.0 (H100) and SM12.0 (Blackwell) support Tensor Memory Accelerator (TMA)
// operations: cp.async.bulk.tensor with a CUTensorMap descriptor.  TMA moves
// an entire tile of global memory into shared memory in one instruction,
// freeing all threads in the CTA to overlap arithmetic with the transfer.
//
// This section provides:
//   tma_block_load_smem_bytes()          — smem bytes needed for the tile
//   tma_block_load_to_shared_sm90()      — __device__ wrapper (SM9.0+)
//   launch_tma_block_load_reduce_sm90()  — fused load+reduce step kernel
//   launch_tma_hetero_reduce_step()      — public dispatch function
//
// ═══════════════════════════════════════════════════════════════════════
// TMA DESCRIPTOR SETUP  (host side)
// ═══════════════════════════════════════════════════════════════════════
//
// cuTensorMapEncodeTiled fills a CUtensorMap with:
//   • global shape + stride (the full gradient tensor)
//   • box shape = {kTmaTileElems} (the per-CTA tile in BF16 elements)
//   • swizzle = CU_TENSOR_MAP_SWIZZLE_128B for 128-byte swizzle
//   • interleave = CU_TENSOR_MAP_INTERLEAVE_NONE (contiguous layout)
//
// The kernel issues cp.async.bulk.tensor.1d.shared::cluster.global with
// a mbar (mbarrier) for completion synchronization.  After mbar_wait,
// all smem elements are valid and the CTA reads them for reduction.
//
// ═══════════════════════════════════════════════════════════════════════
// KERNEL DESIGN
// ═══════════════════════════════════════════════════════════════════════
//
// tma_ring_reduce_sm90_kernel<kBS>:
//   • Each CTA processes kTmaTileElems BF16 elements of the src bucket.
//   • Thread 0 issues cp.async.bulk to load src tile into smem.
//   • All threads spin on mbar until transfer completes.
//   • All threads load 8 elements from smem (128-bit), load 8 from dst
//     (read-write), accumulate in FP32, store back to dst (BF16).
//   • No extra shared memory beyond the tile buffer is needed.
//   • Works on SM9.0 (H100) and SM12.0 (Blackwell).
//
// ═══════════════════════════════════════════════════════════════════════
// FALLBACK
// ═══════════════════════════════════════════════════════════════════════
//
// When __CUDA_ARCH__ < 900 or CUDA < 12.0, the kernel falls back to the
// existing pcie_ring_reduce_kernel (plain __ldg path) which is already
// correct for SM8.6 and toolchains without TMA support.
// ─────────────────────────────────────────────────────────────────────────────

// TMA tile size in BF16 elements per CTA: 4 KiB / 2 bytes = 2048 elements.
// Must be a multiple of kARVecWidth and fit in shared memory (SM9.0 has
// 228 KB smem/SM; 4 KiB per CTA leaves ample room for 4+ CTAs/SM).
static constexpr int kTmaTileElems = 2048;   // 4 KiB of BF16

// Shared memory bytes for one TMA tile buffer.
// Exposed as a public helper so callers can check smem budgets.
size_t tma_block_load_smem_bytes() {
    return kTmaTileElems * sizeof(__nv_bfloat16);
}

// ── Device function: issue TMA bulk async load into shared memory ─────────────
//
// Called by exactly one thread per CTA (typically threadIdx.x == 0).
// Issues cp.async.bulk.tensor.1d.shared::cluster.global [addr], [tmap], {coord}
// and initialises the mbarrier for the tile transfer.
//
// Inline PTX:
//   cp.async.bulk.tensor.1d.shared::cluster.global [dst], [tmap], {x},
//       [mbar];
//   mbarrier.arrive.expect_tx.shared::cta.b64 [mbar], bytes;
//
// Defined only for SM9.0+ to avoid PTX errors on older arches.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12

DS_D_INLINE void tma_block_load_to_shared(
    __nv_bfloat16* __restrict__ smem_dst,
    const CUtensorMap*          tmap,
    int                         x_coord,   // tile index in the 1-D tensor
    uint64_t* __restrict__      mbar)      // shared-memory mbarrier
{
    // Initialise mbarrier for one expected transaction of kTmaTileElems * 2 bytes.
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
    );

    // Expect tx: signals how many bytes the TMA will deliver.
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
           "r"(static_cast<uint32_t>(kTmaTileElems * sizeof(__nv_bfloat16)))
    );

    // Issue TMA 1-D bulk copy: global tmap[x_coord] → smem_dst.
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global"
        " [%0], [%1, {%2}], [%3];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
           "l"(tmap),
           "r"(x_coord),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
    );
}

DS_D_INLINE void tma_mbar_wait(uint64_t* __restrict__ mbar, uint32_t phase)
{
    // Spin until the mbarrier transitions to the expected phase.
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT;\n"
        "}\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
           "r"(phase)
    );
}

#endif  // __CUDA_ARCH__ >= 900

// ── TMA ring-reduce kernel (SM9.0+) ──────────────────────────────────────────
//
// Grid:  one CTA per kTmaTileElems-element tile of the input.
// Block: kBS threads.
//
// Phase A (thread 0):  tma_block_load_to_shared(smem, tmap, tile_coord, mbar)
// Sync:  tma_mbar_wait(mbar, 0)
// Phase B (all threads): vectorised accumulate dst[tile] += smem[tile] (FP32)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12

template <int kBS>
__global__ void __launch_bounds__(kBS, 4)
tma_ring_reduce_sm90_kernel(
    __nv_bfloat16* __restrict__       dst,
    const CUtensorMap*                src_tmap,   // TMA descriptor for src bucket
    size_t                            n_elems)
{
    // Shared memory layout: [smem_tile | mbar (8 bytes, 8-byte aligned)]
    __shared__ __align__(128) __nv_bfloat16 smem_tile[kTmaTileElems];
    __shared__ __align__(8)   uint64_t      mbar;

    const int   tile_idx   = blockIdx.x;
    const size_t tile_base  = (size_t)tile_idx * kTmaTileElems;
    if (tile_base >= n_elems) return;

    const size_t tile_elems = min((size_t)kTmaTileElems, n_elems - tile_base);

    // ── Phase A: TMA issue (thread 0 only) ───────────────────────────────
    if (threadIdx.x == 0) {
        tma_block_load_to_shared(smem_tile, src_tmap, tile_idx, &mbar);
    }
    __syncthreads();

    // ── Wait for TMA completion (all threads spin on mbarrier) ───────────
    // Only thread 0 polled mbar; others use __syncthreads to observe barrier.
    // Use a single thread to wait, then syncthreads to propagate completion.
    if (threadIdx.x == 0) {
        tma_mbar_wait(&mbar, 0);
    }
    __syncthreads();   // smem_tile is now fully populated

    // ── Phase B: vectorised accumulate dst[tile_base+i] += smem_tile[i] ──
    const size_t vec_tile = tile_elems / kARVecWidth;

    for (int t = (int)threadIdx.x; t < (int)vec_tile; t += kBS) {
        const int base = t * kARVecWidth;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst  + tile_base + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_load8_f32(smem_tile        + base, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + tile_base + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }

    // Scalar tail within the tile (thread 0 only)
    if (threadIdx.x == 0) {
        for (size_t e = vec_tile * kARVecWidth; e < tile_elems; ++e) {
            float d = __bfloat162float(dst[tile_base + e]);
            float s = __bfloat162float(smem_tile[e]);
            dst[tile_base + e] = __float2bfloat16(d + s);
        }
    }
}

#endif  // __CUDA_ARCH__ >= 900

// ── Host: TMA descriptor construction helper ──────────────────────────────────
//
// Wraps cuTensorMapEncodeTiled for 1-D BF16 tensors.
// Returns cudaSuccess on success; the CUtensorMap is populated in *tmap_out.
// The caller owns the CUtensorMap (stack or device-pinned; no CUDA allocation).

static cudaError_t build_tma_descriptor_bf16_1d(
    CUtensorMap*             tmap_out,
    const __nv_bfloat16*     global_ptr,
    size_t                   n_elems)
{
#if CUDA_VERSION >= 12000
    // 1-D tensor: global shape = [n_elems], element size = 2 bytes (BF16).
    const uint64_t global_dim[1]    = { n_elems };
    const uint64_t global_stride[1] = { sizeof(__nv_bfloat16) };  // byte stride
    // Box (tile) shape: kTmaTileElems elements per CTA.
    const uint32_t box_dim[1]       = { (uint32_t)kTmaTileElems };
    // Element stride: 1 (contiguous).
    const uint32_t elem_stride[1]   = { 1 };

    CUresult res = cuTensorMapEncodeTiled(
        tmap_out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /*tensorRank=*/   1,
        /*globalAddress=*/(void*)global_ptr,
        global_dim,
        global_stride + 1,   // stride array starts at dim 1 (skip outermost)
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "[build_tma_descriptor_bf16_1d] cuTensorMapEncodeTiled "
            "failed: %s\n", err_str ? err_str : "unknown");
        return cudaErrorUnknown;
    }
    return cudaSuccess;
#else
    (void)tmap_out; (void)global_ptr; (void)n_elems;
    fprintf(stderr, "[build_tma_descriptor_bf16_1d] TMA requires CUDA >= 12.0\n");
    return cudaErrorNotSupported;
#endif
}

// ── Public dispatch: TMA-accelerated hetero reduce step ──────────────────────

/**
 * launch_tma_hetero_reduce_step
 *
 * Fused TMA BlockLoadToShared + accumulate step for SM9.0+ devices.
 * On SM < 9.0 or CUDA < 12: falls back to launch_pcie_ring_reduce_step.
 *
 * @param accum_buf      [in/out] BF16 local accumulator [chunk_elems]
 * @param src_buf        [in]     BF16 incoming peer bucket [chunk_elems]
 * @param chunk_elems    Number of BF16 elements (must be divisible by kARVecWidth)
 * @param sm_version     SM version of the active device (86, 90, 120)
 * @param compute_stream CUDA stream for kernel launch
 */
void launch_tma_hetero_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ src_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream)
{
#if CUDA_VERSION >= 12000
    // SM9.0+ path: build TMA descriptor and launch TMA kernel.
    if (sm_version >= 90) {
        CUtensorMap tmap;
        if (build_tma_descriptor_bf16_1d(&tmap, src_buf, chunk_elems)
                != cudaSuccess) {
            // Descriptor build failed — fall through to non-TMA path.
            goto fallback;
        }

        // Copy the TMA descriptor to device-accessible memory.
        // CUtensorMap is 128 bytes; we allocate it in device memory so the
        // kernel can access it via a raw pointer.
        CUtensorMap* d_tmap = nullptr;
        cudaMallocAsync(&d_tmap, sizeof(CUtensorMap), compute_stream);
        cudaMemcpyAsync(d_tmap, &tmap, sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice, compute_stream);

        // Grid: one CTA per tile.
        const int n_tiles = (int)((chunk_elems + kTmaTileElems - 1) / kTmaTileElems);
        const int grid    = std::min(n_tiles, 65535);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // This code path is only reached at runtime on SM9.0+ devices;
        // the kernel itself is compiled with --generate-code arch=compute_90.
        if (sm_version >= 120) {
            tma_ring_reduce_sm90_kernel<512>
                <<<grid, 512, 0, compute_stream>>>(
                    accum_buf, d_tmap, chunk_elems);
        } else {
            tma_ring_reduce_sm90_kernel<256>
                <<<grid, 256, 0, compute_stream>>>(
                    accum_buf, d_tmap, chunk_elems);
        }
#else
        // Compiled without SM9.0 arch: still schedule the kernel; CUDA driver
        // will select the appropriate code object at launch time.
        tma_ring_reduce_sm90_kernel<256>
            <<<grid, 256, 0, compute_stream>>>(
                accum_buf, d_tmap, chunk_elems);
#endif

        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr,
                    "[launch_tma_hetero_reduce_step] TMA kernel launch failed "
                    "(SM %d, chunk_elems=%zu): %s\n",
                    sm_version, chunk_elems, cudaGetErrorString(err));
        }
        cudaFreeAsync(d_tmap, compute_stream);
        return;
    }
#endif  // CUDA_VERSION >= 12000

fallback:
    // SM8.6 or CUDA < 12: plain __ldg ring-reduce step.
    launch_pcie_ring_reduce_step(accum_buf, src_buf, chunk_elems,
                                 sm_version, compute_stream);
}
