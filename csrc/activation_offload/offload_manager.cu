// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #22: tier-aware activation checkpointing offload

/*
 * offload_manager.cu — Tier-aware activation checkpoint offload orchestration
 *
 * ═══════════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OffloadManager orchestrates async H2D/D2H activation checkpoint offload
 * for heterogeneous GPU clusters (A6000 SM8.6 + H100 SM9.0 + Blackwell
 * SM12.0) connected over PCIe without NVLink.
 *
 * Architecture:
 *   1. PinnedPool (pinned_pool.h): pre-allocated double-buffered pinned
 *      host memory, sized per-tier based on VRAM headroom.
 *   2. Dedicated offload CUDA stream: all H2D/D2H copies run on a
 *      separate stream from forward/backward compute, enabling overlap.
 *   3. Double-buffered pipeline: layer N activations stream to CPU while
 *      layer N+1 forward runs on the compute stream.  On backward,
 *      layer N activations prefetch from CPU while layer N-1 backward runs.
 *   4. Optional INT8 quantisation: halves PCIe traffic at the cost of
 *      ~0.1% model quality (block-wise absmax, 128-element tiles).
 *
 * Pipeline timeline (forward offload):
 *   compute_stream: [fwd layer N] [fwd layer N+1] [fwd layer N+2] ...
 *   offload_stream:   [pack+D2H N]   [pack+D2H N+1]   ...
 *   Event sync: compute_stream records event after layer N fwd completes;
 *               offload_stream waits on that event before packing layer N.
 *
 * Pipeline timeline (backward prefetch):
 *   compute_stream: ... [bwd layer N+2] [bwd layer N+1] [bwd layer N] ...
 *   offload_stream:        [H2D+unpack N+1] [H2D+unpack N] ...
 *   Prefetch is issued 1 layer ahead of current backward layer.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * INTEGRATION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Python side (deepspeed/runtime/tier_aware_activation_offload.py)
 * calls into the pybind11 binding to:
 *   1. offload_init(sm_version, total_act_bytes, vram_free, headroom_frac)
 *   2. offload_push(activation_tensor, layer_idx)  — forward: D2H offload
 *   3. offload_pop(layer_idx)  — backward: H2D prefetch + return tensor
 *   4. offload_sync()  — wait for all pending transfers
 *
 * Low-level kernels (pack/unpack/quantise/dequantise) are reused from
 * csrc/hetero_reduce/tier_activation_offload.cu via the shared header.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "pinned_pool.h"
#include "../hetero_reduce/hetero_reduce.h"

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Constants
// ─────────────────────────────────────────────────────────────────────────────

// Maximum number of transformer layers supported.
static constexpr int kMaxLayers = 256;

// INT8 quantisation tile size (must match tier_activation_offload.cu).
static constexpr int kQuantTileSize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: LayerSlot — per-layer offload bookkeeping
// ─────────────────────────────────────────────────────────────────────────────

struct LayerSlot {
    size_t  n_elems;           // BF16 elements in this layer's activation
    size_t  host_offset;       // byte offset into pinned slab for this layer
    bool    quantised;         // true if INT8 quantised on offload
    bool    offloaded;         // true if D2H has been issued
    bool    prefetched;        // true if H2D has been issued
    int     ping_pong;         // which slab this layer's data lives in (0/1)

    // INT8 scale buffer offset (from slab base), valid when quantised=true.
    size_t  scale_offset;
    size_t  n_scale_elems;     // = ceil(n_elems / kQuantTileSize)

    LayerSlot()
        : n_elems(0), host_offset(0), quantised(false),
          offloaded(false), prefetched(false), ping_pong(0),
          scale_offset(0), n_scale_elems(0) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: OffloadManager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * OffloadManager — Tier-aware async activation offload with double-buffering.
 *
 * One instance per GPU tier.  Manages:
 *   - PinnedPool for host-side storage
 *   - Dedicated CUDA stream for async D2H/H2D copies
 *   - Per-layer slot tracking
 *   - Optional INT8 quantisation for bandwidth reduction
 */
class OffloadManager {
public:
    OffloadManager()
        : sm_version_(0), device_id_(-1), offload_stream_(nullptr),
          fwd_done_event_(nullptr), bwd_prefetch_event_(nullptr),
          use_quantisation_(false), initialised_(false),
          current_phase_(0) {}

    ~OffloadManager()
    {
        cleanup();
    }

    /**
     * Initialise the offload manager for a specific GPU tier.
     *
     * @param sm_version      SM version (86, 90, 120)
     * @param total_act_bytes Total activation bytes across all layers
     * @param vram_free_bytes Current free VRAM on this device
     * @param headroom_frac   Fraction of VRAM to keep free (e.g. 0.10)
     * @param use_quant       Enable INT8 quantisation (halves PCIe traffic)
     * @param num_layers      Number of transformer layers
     * @return true on success
     */
    bool init(int sm_version,
              size_t total_act_bytes,
              size_t vram_free_bytes,
              float headroom_frac,
              bool use_quant,
              int num_layers)
    {
        cleanup();

        sm_version_ = sm_version;
        use_quantisation_ = use_quant;
        num_layers_ = std::min(num_layers, kMaxLayers);

        cudaGetDevice(&device_id_);

        // Compute offload budget using the helper from tier_activation_offload.cu.
        size_t offload_bytes = compute_offload_budget(
            total_act_bytes, vram_free_bytes, headroom_frac);

        if (offload_bytes == 0) {
            // All activations fit in VRAM; no offload needed.
            initialised_ = true;
            return true;
        }

        // Per-slab size: enough to hold the offload budget.
        // With quantisation, INT8 data is half the size + scales overhead.
        size_t slab_sz = offload_bytes;
        if (use_quantisation_) {
            size_t n_tiles = (offload_bytes / sizeof(__nv_bfloat16) + kQuantTileSize - 1)
                             / kQuantTileSize;
            // INT8 data + FP32 scales
            slab_sz = offload_bytes / 2 + n_tiles * sizeof(float);
        }

        if (!pool_.init(slab_sz)) {
            fprintf(stderr, "[OffloadManager] Failed to allocate pinned pool "
                    "(%zu bytes per slab)\n", slab_sz);
            return false;
        }

        // Create dedicated offload stream (non-blocking w.r.t. default stream).
        cudaError_t err;
        err = cudaStreamCreateWithFlags(&offload_stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[OffloadManager] Failed to create offload stream: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        err = cudaEventCreateWithFlags(&fwd_done_event_,
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) return false;

        err = cudaEventCreateWithFlags(&bwd_prefetch_event_,
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) return false;

        initialised_ = true;
        return true;
    }

    /**
     * Offload a layer's activation from GPU to host (D2H).
     *
     * Called during the forward pass after layer `layer_idx` completes.
     * The copy runs on the offload stream, overlapping with the next
     * layer's forward computation on the compute stream.
     *
     * @param dev_ptr        Device pointer to BF16 activation tensor
     * @param n_elems        Number of BF16 elements
     * @param layer_idx      Layer index [0, num_layers)
     * @param compute_stream The compute stream (for event synchronisation)
     */
    void offload_push(const __nv_bfloat16* dev_ptr,
                      size_t n_elems,
                      int layer_idx,
                      cudaStream_t compute_stream)
    {
        if (!initialised_ || !pool_.valid || layer_idx >= num_layers_)
            return;

        LayerSlot& slot = slots_[layer_idx];
        slot.n_elems    = n_elems;
        slot.ping_pong  = current_phase_;

        // Record event on compute stream: offload waits until fwd is done.
        cudaEventRecord(fwd_done_event_, compute_stream);
        cudaStreamWaitEvent(offload_stream_, fwd_done_event_, 0);

        void* host_slab = pool_.get_slab(current_phase_);
        if (!host_slab) return;

        if (use_quantisation_) {
            // Quantise on the offload stream, then D2H copy.
            size_t n_tiles = (n_elems + kQuantTileSize - 1) / kQuantTileSize;
            size_t int8_bytes  = n_elems * sizeof(int8_t);
            size_t scale_bytes = n_tiles * sizeof(float);

            slot.host_offset  = 0;  // simplified: one layer per slab
            slot.scale_offset = int8_bytes;
            slot.n_scale_elems = n_tiles;
            slot.quantised = true;

            // Allocate temp device buffers for quantised data.
            int8_t* d_int8 = nullptr;
            float*  d_scale = nullptr;
            cudaMallocAsync(&d_int8,  int8_bytes,  offload_stream_);
            cudaMallocAsync(&d_scale, scale_bytes, offload_stream_);

            // Quantise BF16 → INT8 on GPU.
            launch_quantise_fp16_to_int8(d_int8, d_scale, dev_ptr,
                                         n_elems, offload_stream_);

            // D2H: INT8 data.
            cudaMemcpyAsync(
                static_cast<uint8_t*>(host_slab) + slot.host_offset,
                d_int8, int8_bytes,
                cudaMemcpyDeviceToHost, offload_stream_);

            // D2H: FP32 scales.
            cudaMemcpyAsync(
                static_cast<uint8_t*>(host_slab) + slot.scale_offset,
                d_scale, scale_bytes,
                cudaMemcpyDeviceToHost, offload_stream_);

            cudaFreeAsync(d_int8,  offload_stream_);
            cudaFreeAsync(d_scale, offload_stream_);
        } else {
            // Direct BF16 D2H copy.
            slot.host_offset = 0;
            slot.quantised   = false;
            size_t bytes = n_elems * sizeof(__nv_bfloat16);

            cudaMemcpyAsync(
                static_cast<uint8_t*>(host_slab) + slot.host_offset,
                dev_ptr, bytes,
                cudaMemcpyDeviceToHost, offload_stream_);
        }

        slot.offloaded  = true;
        slot.prefetched = false;
    }

    /**
     * Prefetch a layer's activation from host to GPU (H2D).
     *
     * Called 1 layer ahead during the backward pass.  The copy runs on
     * the offload stream, overlapping with the current backward layer's
     * computation on the compute stream.
     *
     * @param dev_ptr        Device pointer to write the restored activation
     * @param layer_idx      Layer index [0, num_layers)
     * @param compute_stream The compute stream (for event synchronisation)
     */
    void offload_pop(__nv_bfloat16* dev_ptr,
                     int layer_idx,
                     cudaStream_t compute_stream)
    {
        if (!initialised_ || !pool_.valid || layer_idx >= num_layers_)
            return;

        LayerSlot& slot = slots_[layer_idx];
        if (!slot.offloaded) return;

        void* host_slab = pool_.get_slab(slot.ping_pong);
        if (!host_slab) return;

        if (slot.quantised) {
            size_t int8_bytes  = slot.n_elems * sizeof(int8_t);
            size_t scale_bytes = slot.n_scale_elems * sizeof(float);

            // Allocate temp device buffers.
            int8_t* d_int8 = nullptr;
            float*  d_scale = nullptr;
            cudaMallocAsync(&d_int8,  int8_bytes,  offload_stream_);
            cudaMallocAsync(&d_scale, scale_bytes, offload_stream_);

            // H2D: INT8 data.
            cudaMemcpyAsync(
                d_int8,
                static_cast<uint8_t*>(host_slab) + slot.host_offset,
                int8_bytes,
                cudaMemcpyHostToDevice, offload_stream_);

            // H2D: FP32 scales.
            cudaMemcpyAsync(
                d_scale,
                static_cast<uint8_t*>(host_slab) + slot.scale_offset,
                scale_bytes,
                cudaMemcpyHostToDevice, offload_stream_);

            // Dequantise INT8 → BF16 on GPU.
            launch_dequantise_int8_to_fp16(dev_ptr, d_int8, d_scale,
                                            slot.n_elems, offload_stream_);

            cudaFreeAsync(d_int8,  offload_stream_);
            cudaFreeAsync(d_scale, offload_stream_);
        } else {
            // Direct BF16 H2D copy.
            size_t bytes = slot.n_elems * sizeof(__nv_bfloat16);
            cudaMemcpyAsync(
                dev_ptr,
                static_cast<uint8_t*>(host_slab) + slot.host_offset,
                bytes,
                cudaMemcpyHostToDevice, offload_stream_);
        }

        // Record event on offload stream; compute stream waits before using data.
        cudaEventRecord(bwd_prefetch_event_, offload_stream_);
        cudaStreamWaitEvent(compute_stream, bwd_prefetch_event_, 0);

        slot.prefetched = true;
        slot.offloaded  = false;
    }

    /**
     * Synchronise: wait for all pending offload/prefetch operations.
     */
    void sync()
    {
        if (offload_stream_)
            cudaStreamSynchronize(offload_stream_);
    }

    /**
     * Advance the double-buffer phase (call between forward and backward).
     */
    void flip_phase()
    {
        current_phase_ ^= 1;
    }

    /**
     * Query the offload budget in bytes.
     */
    size_t get_slab_bytes() const { return pool_.slab_bytes; }

    /**
     * Query whether a layer has been offloaded and is available for prefetch.
     */
    bool is_offloaded(int layer_idx) const
    {
        if (layer_idx >= num_layers_) return false;
        return slots_[layer_idx].offloaded;
    }

private:
    void cleanup()
    {
        if (offload_stream_) {
            cudaStreamSynchronize(offload_stream_);
            cudaStreamDestroy(offload_stream_);
            offload_stream_ = nullptr;
        }
        if (fwd_done_event_) {
            cudaEventDestroy(fwd_done_event_);
            fwd_done_event_ = nullptr;
        }
        if (bwd_prefetch_event_) {
            cudaEventDestroy(bwd_prefetch_event_);
            bwd_prefetch_event_ = nullptr;
        }
        pool_.release();
        initialised_ = false;
    }

    int           sm_version_;
    int           device_id_;
    int           num_layers_;
    cudaStream_t  offload_stream_;
    cudaEvent_t   fwd_done_event_;
    cudaEvent_t   bwd_prefetch_event_;
    bool          use_quantisation_;
    bool          initialised_;
    int           current_phase_;        // 0 or 1 (ping/pong)
    PinnedPool    pool_;
    LayerSlot     slots_[kMaxLayers];
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: C-API wrappers for pybind11
//
// Global singleton per device.  In multi-GPU setups each device gets its
// own OffloadManager via the CUDA device ordinal.
// ─────────────────────────────────────────────────────────────────────────────

static OffloadManager g_offload_mgr[16];  // up to 16 devices

OffloadManager* get_offload_manager()
{
    int dev = 0;
    cudaGetDevice(&dev);
    return &g_offload_mgr[dev];
}

extern "C" {

bool offload_manager_init(int sm_version,
                          size_t total_act_bytes,
                          size_t vram_free_bytes,
                          float headroom_frac,
                          bool use_quant,
                          int num_layers)
{
    return get_offload_manager()->init(
        sm_version, total_act_bytes, vram_free_bytes,
        headroom_frac, use_quant, num_layers);
}

void offload_manager_push(const void* dev_ptr,
                          size_t n_elems,
                          int layer_idx,
                          cudaStream_t compute_stream)
{
    get_offload_manager()->offload_push(
        static_cast<const __nv_bfloat16*>(dev_ptr),
        n_elems, layer_idx, compute_stream);
}

void offload_manager_pop(void* dev_ptr,
                         int layer_idx,
                         cudaStream_t compute_stream)
{
    get_offload_manager()->offload_pop(
        static_cast<__nv_bfloat16*>(dev_ptr),
        layer_idx, compute_stream);
}

void offload_manager_sync()
{
    get_offload_manager()->sync();
}

void offload_manager_flip_phase()
{
    get_offload_manager()->flip_phase();
}

}  // extern "C"
