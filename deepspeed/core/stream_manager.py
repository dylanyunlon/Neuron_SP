"""
Insight I3: centralized stream management (Megatron M3724)

Framework-level CUDA stream allocator for DES-LOC / Neuron_SP.

Background
----------
Megatron M3724 identified that ad-hoc stream creation across different
modules (DDP, ZeRO-3, shard sync, comm overlap) leads to:
  1. Unbounded stream proliferation — every module allocates its own streams,
     NCCL has a per-stream overhead, and on PCIe-only topologies (2×A6000 +
     1×H100 NVL + 2×Blackwell PCIe) this degrades scheduling efficiency.
  2. No visibility into total stream count per GPU tier — A6000 has less
     concurrency than H100 NVL; allocating the same number of streams on
     both is wasteful on A6000 and underutilised on H100.
  3. Missed overlap opportunities — compute and comm streams created by
     different modules cannot be coordinated without a central registry.

Design
------
StreamManager is a process-global singleton.  Each GPU type can have at
most ``max_streams`` compute streams and ``max_streams`` comm streams.
Callers request a stream by (gpu_type, role) and get a round-robin
assignment from the pool.  The pool is created lazily on first access.

GPU type strings
----------------
  "h100_nvl"   — H100 NVL (highest concurrency, NVLink fabric)
  "a6000"      — A6000 (PCIe, moderate concurrency)
  "blackwell"  — Blackwell PCIe (PCIe, similar profile to A6000)
  "default"    — fallback for unknown GPU types

Usage
-----
    from deepspeed.core.stream_manager import StreamManager

    compute_stream = StreamManager.get_compute_stream("h100_nvl")
    comm_stream    = StreamManager.get_comm_stream("a6000")

    with torch.cuda.stream(compute_stream):
        output = model(input)

    with torch.cuda.stream(comm_stream):
        dist.all_reduce(grad_buffer, async_op=True)
"""

# Insight I3: centralized stream management (Megatron M3724)

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Per-GPU-type stream budget
# ---------------------------------------------------------------------------

# Maximum number of compute / comm streams to allocate per GPU type.
# These defaults are tuned for the Neuron_SP heterogeneous cluster topology:
#   H100 NVL   — more SMs, NVLink → can sustain more concurrent streams
#   A6000      — PCIe fabric, fewer SMs → fewer streams avoids scheduling waste
#   Blackwell  — PCIe, treated similarly to A6000 until profiling says otherwise
_DEFAULT_MAX_STREAMS: Dict[str, int] = {
    "h100_nvl": 4,
    "a6000": 2,
    "blackwell": 2,
    "default": 2,
}


# ---------------------------------------------------------------------------
# StreamPool — internal pool for one (gpu_type, role) pair
# ---------------------------------------------------------------------------

class _StreamPool:
    """Round-robin pool of CUDA streams for a single (gpu_type, role) pair."""

    def __init__(self, max_streams: int) -> None:
        self._max = max_streams
        self._streams: List[torch.cuda.Stream] = []
        self._cursor: int = 0
        self._lock = threading.Lock()

    def get(self) -> torch.cuda.Stream:
        with self._lock:
            if len(self._streams) < self._max:
                # Lazily create streams up to the budget
                s = torch.cuda.Stream()
                self._streams.append(s)
                return s
            # Round-robin among existing streams
            s = self._streams[self._cursor % len(self._streams)]
            self._cursor += 1
            return s

    def all_streams(self) -> List[torch.cuda.Stream]:
        with self._lock:
            return list(self._streams)


# ---------------------------------------------------------------------------
# StreamManager — public API
# ---------------------------------------------------------------------------

class StreamManager:
    """Insight I3: centralized stream management (Megatron M3724).

    Process-global singleton that manages CUDA stream pools per GPU type and
    role (compute vs comm).  All modules in DES-LOC / Neuron_SP should obtain
    streams through this class instead of calling ``torch.cuda.Stream()``
    directly.
    """

    _lock: threading.Lock = threading.Lock()
    _compute_pools: Dict[str, _StreamPool] = {}
    _comm_pools: Dict[str, _StreamPool] = {}

    # Override max_streams per gpu_type at startup if needed:
    #   StreamManager.max_streams["h100_nvl"] = 8
    max_streams: Dict[str, int] = dict(_DEFAULT_MAX_STREAMS)

    @classmethod
    def _get_pool(
        cls,
        role_pools: Dict[str, _StreamPool],
        gpu_type: str,
    ) -> _StreamPool:
        """Return (creating if necessary) the pool for *gpu_type* in *role_pools*."""
        key = gpu_type if gpu_type in cls.max_streams else "default"
        if key not in role_pools:
            with cls._lock:
                # Double-checked locking
                if key not in role_pools:
                    role_pools[key] = _StreamPool(cls.max_streams[key])
        return role_pools[key]

    @classmethod
    def get_compute_stream(cls, gpu_type: str = "default") -> torch.cuda.Stream:
        """Insight I3: centralized stream management (Megatron M3724).

        Return a CUDA stream suitable for compute kernels on *gpu_type*.
        Streams are reused round-robin up to max_streams[gpu_type].

        Args:
            gpu_type: One of "h100_nvl", "a6000", "blackwell", or "default".
                      Unknown types fall back to "default".

        Returns:
            A ``torch.cuda.Stream`` from the compute pool for this GPU type.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("StreamManager.get_compute_stream() requires CUDA")
        pool = cls._get_pool(cls._compute_pools, gpu_type)
        return pool.get()

    @classmethod
    def get_comm_stream(cls, gpu_type: str = "default") -> torch.cuda.Stream:
        """Insight I3: centralized stream management (Megatron M3724).

        Return a CUDA stream suitable for communication collectives on *gpu_type*.
        Communication streams are kept separate from compute streams so that
        NCCL and compute kernels can overlap without head-of-line blocking.

        On PCIe topologies (A6000, Blackwell) the comm budget is intentionally
        smaller than on H100 NVL because PCIe fabric concurrency is limited.

        Args:
            gpu_type: One of "h100_nvl", "a6000", "blackwell", or "default".

        Returns:
            A ``torch.cuda.Stream`` from the comm pool for this GPU type.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("StreamManager.get_comm_stream() requires CUDA")
        pool = cls._get_pool(cls._comm_pools, gpu_type)
        return pool.get()

    @classmethod
    def get_shard_sync_stream(cls, gpu_type: str = "default") -> torch.cuda.Stream:
        """Convenience alias: return a comm stream for FP32→BF16 shard sync.

        Equivalent to ``get_comm_stream(gpu_type)`` but named for clarity at
        call sites in desloc_engine.py that previously created an ad-hoc stream.

        Insight I3: centralized stream management (Megatron M3724)
        """
        return cls.get_comm_stream(gpu_type)

    @classmethod
    def configure(cls, gpu_type: str, max_streams: int) -> None:
        """Override the stream budget for *gpu_type*.

        Must be called before the first ``get_compute_stream`` /
        ``get_comm_stream`` call for this *gpu_type*.  Raises RuntimeError if
        a pool has already been created.

        Args:
            gpu_type: GPU type key (e.g. "h100_nvl").
            max_streams: New maximum stream count (≥ 1).
        """
        if max_streams < 1:
            raise ValueError(f"max_streams must be >= 1, got {max_streams}")
        with cls._lock:
            if gpu_type in cls._compute_pools or gpu_type in cls._comm_pools:
                raise RuntimeError(
                    f"Cannot reconfigure StreamManager for '{gpu_type}' after "
                    "streams have already been allocated. Call configure() before "
                    "the first get_compute_stream() / get_comm_stream() call."
                )
            cls.max_streams[gpu_type] = max_streams

    @classmethod
    def reset(cls) -> None:
        """Destroy all stream pools (useful for tests / re-initialisation)."""
        with cls._lock:
            cls._compute_pools.clear()
            cls._comm_pools.clear()

    @classmethod
    def debug_info(cls) -> Dict[str, object]:
        """Return a dict summarising current stream pool state for debugging."""
        info: Dict[str, object] = {}
        for key, pool in cls._compute_pools.items():
            info[f"compute:{key}"] = {
                "count": len(pool.all_streams()),
                "max": pool._max,
                "cursor": pool._cursor,
            }
        for key, pool in cls._comm_pools.items():
            info[f"comm:{key}"] = {
                "count": len(pool.all_streams()),
                "max": pool._max,
                "cursor": pool._cursor,
            }
        return info
