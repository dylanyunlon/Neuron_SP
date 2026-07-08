// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #22: tier-aware activation checkpointing offload

/*
 * pinned_pool.h — Pre-allocated pinned CPU memory pool for async H2D/D2H
 *
 * Design:
 *   A per-tier pool of cudaHostAlloc'd pinned memory, sized at init based on
 *   the tier's offload budget (A6000 needs more offload than H100).
 *   Double-buffered: two slabs (ping/pong) allow concurrent DMA in both
 *   directions without synchronisation between adjacent layers.
 *
 *   Pool layout:
 *     slab[0]  : [0 .. slab_bytes)           — ping buffer
 *     slab[1]  : [slab_bytes .. 2*slab_bytes) — pong buffer
 *
 *   Each slab is large enough to hold the maximum single-layer activation
 *   tensor.  The OffloadManager tracks which slab is in-flight and which
 *   is available for the next offload/prefetch operation.
 *
 *   Alignment: all slab bases are aligned to 4096 bytes (page boundary)
 *   for optimal DMA throughput.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// Alignment for pinned memory allocations (OS page boundary).
static constexpr size_t kPinnedPoolAlign = 4096;

// Round up to alignment boundary.
inline size_t align_up(size_t n, size_t alignment)
{
    return (n + alignment - 1) & ~(alignment - 1);
}

/**
 * PinnedPool — Double-buffered pinned host memory pool.
 *
 * Lifecycle:
 *   1. Construct with PinnedPool(slab_bytes) — allocates 2 × slab_bytes pinned.
 *   2. Call get_slab(phase) to obtain a host pointer for DMA.
 *      phase ∈ {0, 1} selects ping or pong buffer.
 *   3. Destruct or call release() to free pinned memory.
 *
 * Thread safety: NOT thread-safe.  Caller must serialize access or use
 * one pool per CUDA stream / tier.
 */
struct PinnedPool {
    void*  base;         // cudaHostAlloc'd base pointer (both slabs)
    size_t slab_bytes;   // usable bytes per slab (aligned)
    size_t total_bytes;  // = 2 * slab_bytes
    bool   valid;        // false if allocation failed

    PinnedPool() : base(nullptr), slab_bytes(0), total_bytes(0), valid(false) {}

    /**
     * Allocate a double-buffered pinned pool.
     *
     * @param slab_sz  Bytes per slab (will be rounded up to kPinnedPoolAlign).
     *                 Total allocation = 2 × aligned_slab_sz.
     * @return true on success, false on allocation failure.
     */
    bool init(size_t slab_sz)
    {
        slab_bytes  = align_up(slab_sz, kPinnedPoolAlign);
        total_bytes = 2 * slab_bytes;

        cudaError_t err = cudaHostAlloc(&base, total_bytes,
                                        cudaHostAllocDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "[PinnedPool] cudaHostAlloc(%zu bytes) failed: %s\n",
                    total_bytes, cudaGetErrorString(err));
            base  = nullptr;
            valid = false;
            return false;
        }
        valid = true;
        return true;
    }

    /**
     * Get a host pointer to the specified slab.
     *
     * @param phase  0 = ping, 1 = pong.
     * @return Host pointer to the start of the slab, or nullptr if invalid.
     */
    void* get_slab(int phase) const
    {
        if (!valid || !base) return nullptr;
        return static_cast<uint8_t*>(base) + (size_t)(phase & 1) * slab_bytes;
    }

    /**
     * Release all pinned memory.  Safe to call multiple times.
     */
    void release()
    {
        if (base) {
            cudaFreeHost(base);
            base = nullptr;
        }
        valid = false;
    }

    ~PinnedPool()
    {
        release();
    }

    // Non-copyable, movable.
    PinnedPool(const PinnedPool&) = delete;
    PinnedPool& operator=(const PinnedPool&) = delete;

    PinnedPool(PinnedPool&& o) noexcept
        : base(o.base), slab_bytes(o.slab_bytes),
          total_bytes(o.total_bytes), valid(o.valid)
    {
        o.base  = nullptr;
        o.valid = false;
    }

    PinnedPool& operator=(PinnedPool&& o) noexcept
    {
        if (this != &o) {
            release();
            base        = o.base;
            slab_bytes  = o.slab_bytes;
            total_bytes = o.total_bytes;
            valid       = o.valid;
            o.base  = nullptr;
            o.valid = false;
        }
        return *this;
    }
};
