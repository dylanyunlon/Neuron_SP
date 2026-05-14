# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# M343-C34: BloomBee DHT bridge for heterogeneous AutoSP.
#
# This module adapts BloomBee's hivemind-based decentralized peer discovery
# to work with Neuron_SP's compile-time sequence parallelism. It enables:
#
# 1. DHT-based GPU capability announcement (peers declare their specs)
# 2. Dynamic SP group formation from discovered peers
# 3. Remote A2A proxy for WAN-connected sequence-parallel peers
# 4. Adaptive sync gating based on peer reachability

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# BloomBee imports are optional — this module works standalone for NCCL clusters
# and adds DHT functionality when hivemind is available.
_HIVEMIND_AVAILABLE = False
try:
    import hivemind
    from hivemind.dht import DHT, DHTValue
    from hivemind.utils import get_dht_time
    _HIVEMIND_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# DHT key schema for GPU capability announcements
# Pattern: BloomBee's dht_utils.py uses declare_active_modules to announce
# which transformer blocks a peer serves. We announce GPU capabilities.
# ---------------------------------------------------------------------------

DHT_NEURONSP_PREFIX = "neuronsp_gpu_caps"
DHT_CAPABILITY_EXPIRY_SECS = 120  # re-announce every 2 minutes


@dataclass
class PeerCapability:
    """Serializable GPU capability record for DHT announcement."""
    peer_id: str
    rank: int
    device_name: str
    compute_capability: Tuple[int, int]
    memory_total_gb: float
    memory_bandwidth_gbps: float
    tier: int
    nvlink_available: bool
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['compute_capability'] = list(d['compute_capability'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'PeerCapability':
        d['compute_capability'] = tuple(d['compute_capability'])
        return cls(**d)


def announce_capability_via_dht(
    dht: 'DHT',
    capability: 'PeerCapability',
) -> bool:
    """Announce this peer's GPU capability to the DHT.

    Pattern: BloomBee cli/run_server.py calls declare_active_modules
    at startup. We do the same for GPU capabilities so the mesh planner
    can discover heterogeneous peers before forming SP groups.
    """
    if not _HIVEMIND_AVAILABLE:
        logger.warning("[DHTBridge] hivemind not available, skipping DHT announcement")
        return False

    capability.timestamp = time.time()
    key = f"{DHT_NEURONSP_PREFIX}:{capability.peer_id}"

    try:
        dht.store(
            key=key,
            value=json.dumps(capability.to_dict()),
            expiration_time=get_dht_time() + DHT_CAPABILITY_EXPIRY_SECS,
        )
        logger.info(f"[DHTBridge] Announced capability: {capability.device_name} "
                     f"tier={capability.tier} rank={capability.rank}")
        return True
    except Exception as e:
        logger.warning(f"[DHTBridge] Failed to announce: {e}")
        return False


def discover_peers_via_dht(
    dht: 'DHT',
    timeout_secs: float = 10.0,
) -> List[PeerCapability]:
    """Discover GPU-capable peers from the DHT.

    Pattern: BloomBee's SequenceManager.make_sequence queries DHT
    to find available transformer block servers. We query for GPU
    capability announcements to form heterogeneous SP groups.
    """
    if not _HIVEMIND_AVAILABLE:
        return []

    try:
        # Get all keys with our prefix
        # hivemind's get returns (value, expiration_time) or None
        result = dht.get(f"{DHT_NEURONSP_PREFIX}:*", latest=True)
        if result is None:
            return []

        peers = []
        if isinstance(result, dict):
            for key, (value, _expiry) in result.items():
                try:
                    cap = PeerCapability.from_dict(json.loads(value))
                    # Only include peers announced within the expiry window
                    if time.time() - cap.timestamp < DHT_CAPABILITY_EXPIRY_SECS:
                        peers.append(cap)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return peers
    except Exception as e:
        logger.warning(f"[DHTBridge] Discovery failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Remote A2A proxy for WAN-connected peers
# Pattern: BloomBee's remote_forward_backward.py sends activations
# via hivemind P2P (gRPC). For AutoSP's All-to-All across WAN peers,
# we serialize the scatter/gather into P2P send/recv pairs.
#
# CRITICAL DESIGN NOTE: This is a FALLBACK for peers without NCCL
# connectivity. For NCCL-connected peers, use the standard all_to_all.py.
# ---------------------------------------------------------------------------

@dataclass
class RemoteA2AConfig:
    """Configuration for remote All-to-All over P2P."""
    use_compression: bool = True
    compression_codec: str = "fp16"  # fp16, int8, none
    max_chunk_bytes: int = 64 * 1024 * 1024  # 64MB per P2P message
    timeout_secs: float = 30.0
    retry_count: int = 3


class RemoteA2AProxy:
    """Proxy for All-to-All across WAN-connected peers.

    This is NOT a replacement for NCCL All-to-All. It's a bridge for
    the case where some peers in an SP group are connected via WAN
    (e.g., BloomBee's P2P network) rather than NVLink/PCIe/IB.

    The proxy decomposes All-to-All into point-to-point transfers:
      For sp_size=P, each rank sends (P-1) chunks and receives (P-1) chunks.
      Standard A2A: O(1) collective with NCCL ring/tree algorithms.
      P2P fallback: O(P) sequential sends with optional compression.

    This is inherently slower but enables heterogeneous clusters where
    some nodes aren't in the same NCCL communicator.

    Pattern: TE's cp_p2p_fwd_fused_attn (context_parallel.py:857)
    implements ring-style P2P for context parallelism. We adapt this
    for the Ulysses A2A pattern.
    """

    def __init__(self, config: Optional[RemoteA2AConfig] = None):
        self.config = config or RemoteA2AConfig()
        self._compression_enabled = self.config.use_compression
        self._stats = {"sends": 0, "recvs": 0, "bytes_sent": 0, "bytes_recv": 0}

    def compress_tensor(self, tensor: 'torch.Tensor') -> Tuple[bytes, dict]:
        """Compress tensor for P2P transfer.

        Pattern: BloomBee's lossless_transport.py handles tensor
        serialization for remote forward/backward. We add lossy
        compression (fp16 quantization) for bandwidth efficiency.
        """
        import torch
        meta = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "codec": self.config.compression_codec,
        }

        if self.config.compression_codec == "fp16" and tensor.dtype == torch.float32:
            compressed = tensor.half().cpu().numpy().tobytes()
        elif self.config.compression_codec == "int8":
            # Bug Risk 4 fix: Per-CHANNEL absmax quantization instead of per-tensor.
            # Attention activations have heavy-tailed distributions where outlier
            # values in a few channels would destroy precision for the rest.
            # Per-channel quantization preserves dynamic range per feature.
            # Pattern: TE's fp8_utils.py uses per-tensor amax but with separate
            # scale per gemm output; we use per-last-dim scale for activations.
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])  # [*, hidden_dim]
            absmax = flat.abs().amax(dim=0)  # [hidden_dim]
            scales = absmax / 127.0
            scales = scales.clamp(min=1e-8)  # prevent div-by-zero
            quantized = (flat / scales.unsqueeze(0)).clamp(-127, 127).to(torch.int8)
            compressed = quantized.cpu().numpy().tobytes()
            meta["scales"] = scales.cpu().tolist()
            meta["orig_shape"] = list(orig_shape)
        else:
            compressed = tensor.cpu().numpy().tobytes()
            meta["codec"] = "none"

        return compressed, meta

    def decompress_tensor(
        self, data: bytes, meta: dict, device: 'torch.device'
    ) -> 'torch.Tensor':
        """Decompress tensor received via P2P."""
        import torch
        import numpy as np

        shape = meta["shape"]
        original_dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
        codec = meta["codec"]

        if codec == "fp16":
            arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
            tensor = torch.from_numpy(arr.copy()).to(dtype=original_dtype, device=device)
        elif codec == "int8":
            # Per-channel dequantization (matches per-channel quantization)
            orig_shape = meta.get("orig_shape", shape)
            hidden_dim = orig_shape[-1]
            flat_shape = (-1, hidden_dim)
            arr = np.frombuffer(data, dtype=np.int8).reshape(flat_shape)
            tensor = torch.from_numpy(arr.copy()).to(dtype=torch.float32, device=device)
            scales = torch.tensor(meta["scales"], dtype=torch.float32, device=device)
            tensor = tensor * scales.unsqueeze(0)
            tensor = tensor.reshape(orig_shape).to(original_dtype)
        else:
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            tensor = torch.from_numpy(arr.copy()).to(dtype=original_dtype, device=device)

        return tensor

    def get_stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Adaptive sync gating for decentralized heterogeneous training
# Pattern: DES-LOC's Kx-period gating (engine.py:2735) decides when to
# AllReduce gradients. For heterogeneous clusters, we adapt Kx based on:
# 1. Peer availability (BloomBee reachability.py check_reachability)
# 2. Bandwidth tier of the slowest peer in the DP group
# 3. Gradient staleness across heterogeneous compute speeds
# ---------------------------------------------------------------------------

@dataclass
class HeteroSyncConfig:
    """Configuration for heterogeneous-aware sync gating."""
    base_Kx: int = 1  # Base sync period (DES-LOC Algorithm 1)
    min_Kx: int = 1
    max_Kx: int = 32
    # If the slowest peer in DP group is >2x slower than fastest,
    # increase Kx to avoid straggler-induced blocking.
    straggler_threshold: float = 2.0
    straggler_Kx_multiplier: float = 2.0
    # If peer becomes unreachable, temporarily skip sync
    unreachable_grace_steps: int = 5
    # Enable adaptive Kx based on gradient variance
    adaptive_enabled: bool = False


class HeteroSyncGate:
    """Adaptive sync gate for heterogeneous clusters.

    Extends DES-LOC's fixed Kx with dynamic adjustment based on:
    - Per-tier compute speed (fast GPUs don't wait for slow ones)
    - Peer reachability (unreachable peers get grace period)
    - Gradient divergence (increase Kx when gradients are similar)

    Pattern: Megatron's DistributedDataParallel.finish_grad_sync()
    waits for all buckets before proceeding. We gate the wait based
    on heterogeneous conditions.
    """

    def __init__(self, config: Optional[HeteroSyncConfig] = None):
        self.config = config or HeteroSyncConfig()
        self._effective_Kx = self.config.base_Kx
        self._step = 0
        self._peer_last_seen: Dict[int, float] = {}
        self._grad_norms: List[float] = []

    def should_sync(self, step: int) -> bool:
        """Decide whether to perform gradient AllReduce this step.

        Returns True if this is a sync step, False to skip.
        """
        self._step = step

        if self._effective_Kx <= 1:
            return True  # Kx=1 = standard DDP, always sync

        return (step % self._effective_Kx) == 0

    def update_peer_status(self, rank: int, reachable: bool):
        """Update peer reachability status.

        Pattern: BloomBee server/reachability.py checks peer health
        via gRPC ping. We use this signal to adjust sync behavior.
        """
        if reachable:
            self._peer_last_seen[rank] = time.time()

    def update_grad_norm(self, grad_norm: float):
        """Track gradient norms for adaptive Kx."""
        self._grad_norms.append(grad_norm)
        if len(self._grad_norms) > 100:
            self._grad_norms = self._grad_norms[-100:]

    def adapt_Kx(
        self,
        tier_infos: Optional[Dict[int, Any]] = None,
    ):
        """Dynamically adjust Kx based on cluster conditions.

        Called periodically (e.g., every 100 steps) to re-evaluate
        the sync period.

        BIDIRECTIONAL adaptation (Bug Risk 3 fix):
        - INCREASE Kx when cluster is heterogeneous or gradients are stable
        - DECREASE Kx when gradient variance spikes (divergence detection)

        Pattern: Megatron's gradient clipping adaptively limits grad norm.
        We adaptively limit sync period based on gradient divergence.
        """
        if not self.config.adaptive_enabled:
            return

        new_Kx = self.config.base_Kx

        # Factor 1: Straggler detection from tier info
        if tier_infos and len(tier_infos) > 1:
            scores = [t.compute_score() if hasattr(t, 'compute_score')
                      else 0.0 for t in tier_infos.values()]
            if min(scores) > 0 and max(scores) / min(scores) > self.config.straggler_threshold:
                new_Kx = int(new_Kx * self.config.straggler_Kx_multiplier)

        # Factor 2: Gradient variance — BIDIRECTIONAL
        if len(self._grad_norms) >= 20:
            recent = self._grad_norms[-20:]
            mean_norm = sum(recent) / len(recent)
            variance = sum((x - mean_norm) ** 2 for x in recent) / len(recent)
            cv = (variance ** 0.5) / (mean_norm + 1e-8)  # coefficient of variation

            if cv < 0.1:
                # Very stable gradients → increase Kx (sync less often)
                new_Kx = min(new_Kx * 2, self.config.max_Kx)
            elif cv > 0.5:
                # High variance → DECREASE Kx (sync MORE often to prevent divergence)
                # This is the critical Bug Risk 3 fix: without this branch,
                # the adaptation only ever increases Kx, which can cause
                # catastrophic divergence on small models.
                new_Kx = max(new_Kx // 2, self.config.min_Kx)

        # Factor 3: Detect divergence from recent loss trend
        # If gradient norms are trending upward (explosion), force frequent sync
        if len(self._grad_norms) >= 10:
            first_half = sum(self._grad_norms[-10:-5]) / 5
            second_half = sum(self._grad_norms[-5:]) / 5
            if second_half > first_half * 2.0:  # grad norm doubled
                new_Kx = self.config.min_Kx
                logger.warning(
                    f"[HeteroSync] Gradient explosion detected "
                    f"({first_half:.4f} → {second_half:.4f}). "
                    f"Forcing Kx={new_Kx} for recovery.")

        # Clamp
        self._effective_Kx = max(self.config.min_Kx,
                                  min(new_Kx, self.config.max_Kx))

    @property
    def effective_Kx(self) -> int:
        return self._effective_Kx


class RemoteA2ADoubleBuffer:

    def __init__(self, config: Optional[RemoteA2AConfig] = None):
        self._config = config or RemoteA2AConfig()
        self._proxy = RemoteA2AProxy(self._config)
        self._selector = 0
        self._send_buffers = [None, None]
        self._recv_buffers = [None, None]

    def allocate(self, shape, dtype):
        import torch as _t
        for i in range(2):
            self._send_buffers[i] = _t.empty(shape, dtype=dtype, device='cpu')
            self._recv_buffers[i] = _t.empty(shape, dtype=dtype, device='cpu')

    def current_send(self):
        return self._send_buffers[self._selector]

    def current_recv(self):
        return self._recv_buffers[self._selector]

    def swap(self):
        self._selector ^= 1

    def compress_and_stage(self, tensor):
        compressed, meta = self._proxy.compress_tensor(tensor)
        return compressed, meta

    def decompress_from_stage(self, data, meta, device):
        return self._proxy.decompress_tensor(data, meta, device)

    def free(self):
        self._send_buffers = [None, None]
        self._recv_buffers = [None, None]
        self._selector = 0
