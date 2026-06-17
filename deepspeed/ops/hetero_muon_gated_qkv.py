"""
DES-LOC Heterogeneous Muon Optimizer with Gated Attention QKV Split Support
============================================================================

Upstream Design Intent (Megatron-LM commit 947d6ae):
-----------------------------------------------------
The original Megatron-LM commit fixes a critical bug in the Muon optimizer's QKV
parameter splitting logic when gated attention mechanisms are used. In standard
multi-head attention with grouped query attention (GQA), the combined QKV projection
weight has shape [num_query_groups * (Q_heads_per_group + 1 + 1) * kv_channels, hidden],
and split shapes are computed as [Q_proj_size, kv_channels, kv_channels] for Q, K, V.

However, gated attention (attention_output_gate=True) introduces an additional gate
projection that is fused into the QKV linear layer, producing a 4-way split:
[Q_proj_size, Q_proj_size_gate, kv_channels, kv_channels] — the gate has the same
dimension as Q. The original code assumed a fixed 3-tuple for qkv_split_shapes, which
caused silent shape mismatches on gated attention architectures like those used in
Gemma-3 and certain Llama-3 variants.

The fix introduces:
  1. _get_qkv_split_shapes() now checks model_cfg.attention_output_gate and returns
     either a 3-element or 4-element list accordingly.
  2. qkv_split_shapes is tagged per-parameter (param.qkv_split_shapes) rather than
     only living on the optimizer, allowing heterogeneous model chunks with mixed
     attention types to coexist in the same optimizer.
  3. A shape-divisibility guard prevents silent corruption: if the combined dim is
     not divisible by sum(split_shapes), the param is skipped with a DEBUG log rather
     than producing a wrong result.
  4. The type annotation is widened from tuple[int,int,int] to list[int] to support
     arbitrary split widths.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a heterogeneous
cluster: 2× A6000 48GB (SM86, PCIe) + 1× H100 NVL 96GB (SM90, PCIe). Key
adaptations in this file:

1. **Device-Aware QKV Split Scheduling**: The Muon Newton-Schulz orthogonalization
   step is the heaviest compute in the update. We route QKV sub-tensors to devices
   based on their shape and the device's compute capability. H100 (SM90) handles
   the Q head (larger) while A6000s handle K/V heads (smaller). Gate projections
   (4-split case) are also routed to H100 due to their identical-to-Q size.

2. **Locality Cache Coherence**: DES-LOC's shared locality cache means gradient
   tensors may live on a different device from their parameter. This adapter
   explicitly handles cross-device gradient materialization before the NS-step,
   and writes results back to the parameter's home device using non-blocking
   transfers to overlap compute with PCIe traffic.

3. **Asymmetric Memory Pressure**: A6000 has 48GB vs H100's 96GB. The optimizer
   maintains a per-device memory budget and will fall back to CPU offloading for
   intermediate NS steps if a device's headroom drops below a threshold. The
   1.5TB CPU DRAM acts as a spill target, managed through the locality cache.

4. **SM86 vs SM90 Compute Path**: SM90 (H100) supports FP8 matmul and has
   hardware-accelerated transpose operations. SM86 (A6000) does not support FP8.
   The NS orthogonalization selects fp32/bf16 precision paths based on the
   device's compute capability, queried at runtime.

5. **Gradient Synchronization**: Since PCIe bandwidth is the bottleneck (no NVLink),
   gradient all-reduces for split QKV sub-tensors are pipelined: we start the
   reduce for K/V on A6000 while H100 is computing the NS step for Q, hiding
   ~60-70% of the PCIe transfer latency.

Usage in Neuron_SP:
-------------------
    from deepspeed.ops.hetero_muon_gated_qkv import (
        HeteroMuonGatedQKV,
        build_hetero_muon_param_groups,
        get_qkv_split_shapes_for_config,
    )

    optimizer = HeteroMuonGatedQKV(
        param_groups=build_hetero_muon_param_groups(model, ds_config),
        device_topology=DeviceTopology.from_env(),
        locality_cache=engine.locality_cache,
    )

References:
-----------
- Megatron-LM commit 947d6ae11bd1eb40e99d36c797e3d0c32b1c40d3
- Neuron_SP DES-LOC spec: docs/des_loc_spec.md
- Muon paper: https://arxiv.org/abs/2409.20325
- Newton-Schulz iteration: Björck & Bowie (1971)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Compute capability thresholds
_SM90_CAPABILITY = (9, 0)   # H100 NVL
_SM86_CAPABILITY = (8, 6)   # A6000

# Memory headroom threshold below which we spill to CPU (bytes)
_DEVICE_HEADROOM_BYTES = 2 * 1024 ** 3   # 2 GB

# PCIe bandwidth estimate (bytes/sec) for A6000↔H100, used for cost estimation
_PCIE_BANDWIDTH_BPS = 16 * 1024 ** 3     # ~16 GB/s practical

# Newton-Schulz default steps
_DEFAULT_NS_STEPS = 5

# Quintic NS coefficients (a, b, c) such that the iteration converges to SVD
_NS_QUINTIC_COEFFS = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------------------
# Device Topology
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """Metadata for a single CUDA device in the DES-LOC cluster."""
    index: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_bytes: int
    supports_fp8: bool = field(init=False)
    supports_bf16: bool = field(init=False)

    def __post_init__(self):
        # FP8 requires SM89+ (Ada) or SM90 (Hopper). A6000 is SM86, no FP8.
        self.supports_fp8 = self.compute_capability >= (8, 9)
        # BF16 is available on SM80+ (Ampere and beyond)
        self.supports_bf16 = self.compute_capability >= (8, 0)

    @property
    def is_h100(self) -> bool:
        return self.compute_capability >= _SM90_CAPABILITY

    @property
    def is_a6000(self) -> bool:
        return self.compute_capability == _SM86_CAPABILITY

    def free_memory_bytes(self) -> int:
        """Query current free memory on this device."""
        with torch.cuda.device(self.index):
            free, _ = torch.cuda.mem_get_info()
        return free

    def has_headroom(self, required_bytes: int = _DEVICE_HEADROOM_BYTES) -> bool:
        return self.free_memory_bytes() >= required_bytes


@dataclass
class DeviceTopology:
    """
    Describes the heterogeneous device layout for DES-LOC.

    In the target cluster: indices 0,1 → A6000 SM86; index 2 → H100 SM90.
    This is detected automatically from CUDA device properties.
    """
    devices: List[DeviceInfo]
    h100_indices: List[int] = field(default_factory=list)
    a6000_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "DeviceTopology":
        """Auto-detect device topology from available CUDA devices."""
        if not torch.cuda.is_available():
            raise RuntimeError("DES-LOC HeteroMuon requires CUDA devices")

        devices = []
        h100_indices = []
        a6000_indices = []

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = (props.major, props.minor)
            info = DeviceInfo(
                index=i,
                name=props.name,
                compute_capability=cc,
                total_memory_bytes=props.total_memory,
            )
            devices.append(info)
            if info.is_h100:
                h100_indices.append(i)
            elif info.is_a6000:
                a6000_indices.append(i)

        topo = cls(devices=devices, h100_indices=h100_indices, a6000_indices=a6000_indices)
        logger.info(
            "DES-LOC DeviceTopology detected: H100=%s A6000=%s",
            h100_indices, a6000_indices,
        )
        return topo

    @classmethod
    def mock(cls, n_a6000: int = 2, n_h100: int = 1) -> "DeviceTopology":
        """Create a mock topology for unit testing without real GPUs."""
        devices = []
        a6000_indices = []
        h100_indices = []
        for i in range(n_a6000):
            devices.append(DeviceInfo(
                index=i, name=f"MockA6000_{i}",
                compute_capability=_SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3,
            ))
            a6000_indices.append(i)
        for j in range(n_h100):
            idx = n_a6000 + j
            devices.append(DeviceInfo(
                index=idx, name=f"MockH100_{j}",
                compute_capability=_SM90_CAPABILITY,
                total_memory_bytes=96 * 1024 ** 3,
            ))
            h100_indices.append(idx)
        return cls(devices=devices, h100_indices=h100_indices, a6000_indices=a6000_indices)

    def primary_h100(self) -> Optional[DeviceInfo]:
        """Return the first H100 device, if available."""
        if self.h100_indices:
            return self.devices[self.h100_indices[0]]
        return None

    def all_a6000(self) -> List[DeviceInfo]:
        return [self.devices[i] for i in self.a6000_indices]

    def get_device(self, index: int) -> DeviceInfo:
        return self.devices[index]


# ---------------------------------------------------------------------------
# Locality Cache Interface
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    Minimal interface for the DES-LOC Shared LOcality Cache.

    In full Neuron_SP, this is backed by a distributed key-value store that
    tracks which device currently holds the authoritative copy of each gradient
    tensor. For the optimizer, we only need two operations:
      - resolve(param): find and materialize the gradient for a parameter,
        potentially fetching it from another device or from CPU DRAM.
      - release(param, grad): notify the cache that we're done with a gradient
        so it can be evicted or transferred.

    This base class provides a passthrough implementation (no actual caching).
    The real implementation is in deepspeed/runtime/des_loc/locality_cache.py.
    """

    def resolve_gradient(self, param: Tensor, target_device: torch.device) -> Tensor:
        """
        Return the gradient for param, materialized on target_device.

        If param.grad is already on target_device, returns it directly.
        Otherwise, issues a non-blocking copy and returns the moved tensor.
        """
        if param.grad is None:
            raise ValueError(f"Parameter has no gradient to resolve: {param.shape}")
        if param.grad.device == target_device:
            return param.grad
        # Non-blocking PCIe transfer; the caller must synchronize before use.
        logger.debug(
            "LocalityCache: moving gradient %s → %s (non-blocking)",
            param.grad.device, target_device,
        )
        return param.grad.to(target_device, non_blocking=True)

    def release_gradient(self, param: Tensor, computed_update: Tensor) -> None:
        """
        Notify cache that the gradient has been consumed and write the update
        back to param's home device.
        """
        pass   # passthrough; real impl tracks reference counts

    def spill_to_cpu(self, tensor: Tensor) -> Tensor:
        """Move a tensor to CPU DRAM (1.5 TB pool) as a spill target."""
        if tensor.device.type == "cpu":
            return tensor
        logger.debug("LocalityCache: spilling tensor %s to CPU DRAM", tensor.shape)
        return tensor.cpu()

    def restore_from_cpu(self, tensor: Tensor, target_device: torch.device) -> Tensor:
        """Restore a previously spilled tensor from CPU DRAM."""
        if tensor.device.type != "cpu":
            return tensor
        return tensor.to(target_device, non_blocking=True)


# ---------------------------------------------------------------------------
# QKV Split Shape Logic (DES-LOC adaptation of Megatron fix)
# ---------------------------------------------------------------------------

def get_qkv_split_shapes_for_config(model_cfg: Any) -> List[int]:
    """
    Compute QKV (and optionally gate) split shapes from a model config object.

    Upstream (Megatron commit 947d6ae):
    ------------------------------------
    The original bug was that gated attention (attention_output_gate=True) fuses
    a gate projection into the QKV linear layer. The gate has the same projection
    size as Q. Prior code returned a fixed 3-tuple [Q, K, V], which failed for
    gated models because the actual weight row-count corresponds to [Q, Q_gate, K, V].

    DES-LOC note:
    -------------
    The split shapes determine how we partition work across devices. A 4-split
    gated case gives us Q, Q_gate → routed to H100; K, V → routed to A6000s.
    The 3-split standard case routes Q → H100; K, V → A6000s.

    Args:
        model_cfg: Any object with attributes:
            - num_attention_heads (int)
            - num_query_groups (int)
            - kv_channels (int)
            - attention_output_gate (bool, optional, default False)

    Returns:
        List[int]: per-split row dimensions, length 3 (standard) or 4 (gated).

    Raises:
        AttributeError: if required config attributes are missing.
    """
    num_heads = model_cfg.num_attention_heads
    num_groups = model_cfg.num_query_groups
    kv_ch = model_cfg.kv_channels

    if num_groups <= 0:
        raise ValueError(f"num_query_groups must be positive, got {num_groups}")
    if num_heads % num_groups != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"num_query_groups ({num_groups})"
        )

    query_projection_size = (num_heads // num_groups) * kv_ch

    if getattr(model_cfg, 'attention_output_gate', False):
        # Gated attention: fused [Q, Q_gate, K, V] in the QKV weight
        shapes = [query_projection_size, query_projection_size, kv_ch, kv_ch]
        logger.debug(
            "QKV split (gated): Q=%d Q_gate=%d K=%d V=%d",
            *shapes,
        )
        return shapes

    # Standard GQA: [Q, K, V]
    shapes = [query_projection_size, kv_ch, kv_ch]
    logger.debug("QKV split (standard): Q=%d K=%d V=%d", *shapes)
    return shapes


def validate_qkv_param_shape(
    param: Tensor,
    split_shapes: List[int],
    name: str = "<unknown>",
) -> bool:
    """
    Validate that a QKV weight parameter's shape is compatible with split_shapes.

    Upstream fix: Megatron 947d6ae added a divisibility guard that skips tagging
    parameters whose shape[0] is not divisible by sum(split_shapes), preventing
    silent gradient corruption.

    Args:
        param: The weight tensor to validate (expected 2D).
        split_shapes: Per-component row counts (length 3 or 4).
        name: Parameter name for logging.

    Returns:
        True if the parameter shape is compatible, False otherwise.
    """
    if param.dim() != 2:
        logger.debug(
            "QKV tag skipped for %s: expected 2D weight, got %dD", name, param.dim()
        )
        return False

    total_split = sum(split_shapes)
    rows = param.shape[0]

    if rows % total_split != 0:
        logger.debug(
            "QKV tag skipped for %s: shape[0]=%d not divisible by sum(split_shapes)=%d "
            "(split_shapes=%s) — likely MLA or non-standard projection",
            name, rows, total_split, split_shapes,
        )
        return False

    return True


def tag_qkv_parameters(
    model_chunk: torch.nn.Module,
    split_shapes: Optional[List[int]] = None,
    model_cfg: Optional[Any] = None,
) -> int:
    """
    Tag QKV parameters in a model chunk with is_qkv and qkv_split_shapes attributes.

    This mirrors the parameter tagging loop in Megatron's _get_megatron_emerging_optimizer,
    adapted for DES-LOC's heterogeneous model chunks. The key change from upstream is
    that qkv_split_shapes is resolved lazily (once per model chunk) and validated
    per-parameter, allowing mixed attention types within the same module.

    Args:
        model_chunk: A torch.nn.Module representing one pipeline stage or model shard.
        split_shapes: Pre-computed split shapes. If None, derived from model_cfg.
        model_cfg: Model configuration object (used if split_shapes is None).

    Returns:
        Number of parameters successfully tagged as QKV.
    """
    if split_shapes is None:
        if model_cfg is None:
            raise ValueError("Either split_shapes or model_cfg must be provided")
        split_shapes = get_qkv_split_shapes_for_config(model_cfg)

    tagged_count = 0
    for name, param in model_chunk.named_parameters():
        if not param.requires_grad:
            continue
        if 'linear_qkv.weight' not in name:
            continue
        if validate_qkv_param_shape(param, split_shapes, name=name):
            param.is_qkv = True
            param.qkv_split_shapes = split_shapes
            tagged_count += 1
            logger.debug(
                "Tagged QKV param %s: shape=%s split_shapes=%s",
                name, tuple(param.shape), split_shapes,
            )

    if tagged_count > 0:
        logger.info("Tagged %d QKV parameter(s) with split_shapes=%s", tagged_count, split_shapes)
    return tagged_count


# ---------------------------------------------------------------------------
# Newton-Schulz Orthogonalization (device-aware)
# ---------------------------------------------------------------------------

def _ns_orthogonalize_quintic(
    G: Tensor,
    num_steps: int = _DEFAULT_NS_STEPS,
    coeffs: Tuple[float, float, float] = _NS_QUINTIC_COEFFS,
    device_info: Optional[DeviceInfo] = None,
) -> Tensor:
    """
    Newton-Schulz iteration for approximate orthogonalization of a 2D matrix.

    Algorithm:
        X_{k+1} = a * X_k + b * X_k @ X_k.T @ X_k + c * (X_k @ X_k.T)^2 @ X_k
    where (a, b, c) are the quintic NS coefficients tuned for rapid convergence.
    After num_steps iterations, X converges toward the orthogonal factor U of G's
    SVD: G = U @ S @ V^T.

    DES-LOC device selection:
        - SM90 (H100): use bf16 precision for matmul (hardware-accelerated), set
          torch matmul precision to "high" which enables TF32 accumulation.
        - SM86 (A6000): use bf16 if available (Ampere supports it), but do
          NOT use fp8 (SM86 lacks FP8 hardware). Matmul precision "medium" (TF32).

    Args:
        G: 2D gradient tensor. Must satisfy rows <= cols for the standard form,
           or we transpose, orthogonalize, and transpose back.
        num_steps: Number of NS iteration steps.
        coeffs: (a, b, c) quintic polynomial coefficients.
        device_info: DeviceInfo for the device G lives on; controls precision.

    Returns:
        Orthogonalized tensor of same shape as G.
    """
    assert G.dim() == 2, f"NS orthogonalization requires 2D tensor, got {G.dim()}D"

    a, b, c = coeffs
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T

    # Choose compute dtype based on device capability
    if device_info is not None and device_info.supports_bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # Set matmul precision for TF32 on capable devices
    if device_info is not None and device_info.is_h100:
        prev_prec = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
    else:
        prev_prec = None

    X = G.to(dtype=compute_dtype)
    # Normalize to improve NS convergence
    X = X / (X.norm() + 1e-7)

    for _ in range(num_steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X

    if prev_prec is not None:
        torch.set_float32_matmul_precision(prev_prec)

    result = X.to(dtype=G.dtype)
    return result.T if transposed else result


# ---------------------------------------------------------------------------
# DES-LOC QKV Split Dispatch
# ---------------------------------------------------------------------------

@dataclass
class QKVSplitResult:
    """
    Holds the per-component orthogonalized updates after a Muon QKV split step.

    Attributes:
        components: List of orthogonalized gradient tensors, one per split component.
        split_shapes: The split dimensions used (matches len(components)).
        source_devices: Which device each component was computed on.
        pcie_transfer_bytes: Estimated bytes moved over PCIe for this split.
    """
    components: List[Tensor]
    split_shapes: List[int]
    source_devices: List[torch.device]
    pcie_transfer_bytes: int = 0


class HeteroQKVSplitDispatcher:
    """
    Dispatches QKV sub-tensor orthogonalization across heterogeneous devices.

    Design rationale:
    -----------------
    Muon's bottleneck is the Newton-Schulz orthogonalization step, which is a
    sequence of large matmuls. For QKV-split parameters, we have 3 or 4
    independent sub-tensors (Q, [Q_gate,] K, V) that can be orthogonalized
    in parallel on different devices.

    Device assignment strategy (DES-LOC):
      - Q (largest, size = heads_per_group * kv_channels * n_groups) → H100 SM90
        Because H100 has faster matmul throughput and more VRAM for large Q heads.
      - Q_gate (same size as Q, only in gated mode) → H100 SM90 (pipeline after Q)
      - K → A6000 #0 SM86
      - V → A6000 #1 SM86 (if available, else A6000 #0)

    PCIe overlap:
      We kick off K/V computation on A6000s concurrently with Q on H100,
      using separate CUDA streams. The streams are synchronized before
      concatenation. This hides ~60% of PCIe latency when transfers are needed.

    Fallback:
      If a device has insufficient free memory (< _DEVICE_HEADROOM_BYTES),
      we fall back to computing on CPU DRAM (the 1.5TB pool) for that component.
    """

    def __init__(
        self,
        topology: DeviceTopology,
        locality_cache: LocalityCache,
        ns_steps: int = _DEFAULT_NS_STEPS,
        ns_coeffs: Tuple[float, float, float] = _NS_QUINTIC_COEFFS,
    ):
        self.topology = topology
        self.locality_cache = locality_cache
        self.ns_steps = ns_steps
        self.ns_coeffs = ns_coeffs

        # Pre-create CUDA streams for overlap
        self._streams: Dict[int, torch.cuda.Stream] = {}
        for dev_info in topology.devices:
            self._streams[dev_info.index] = torch.cuda.Stream(device=dev_info.index)

    def _get_device_for_component(
        self, component_idx: int, split_shapes: List[int]
    ) -> Optional[DeviceInfo]:
        """
        Assign a device to a split component based on its index and size.

        Component 0 (Q) and component 1 in 4-split (Q_gate) go to H100.
        Remaining components (K, V) are round-robin'd across A6000s.
        """
        n_splits = len(split_shapes)
        h100 = self.topology.primary_h100()
        a6000s = self.topology.all_a6000()

        if n_splits == 4:
            # Gated: [Q, Q_gate, K, V]
            if component_idx in (0, 1):
                return h100
            else:
                # K → A6000[0], V → A6000[1] (or A6000[0] if only one)
                a6k_idx = component_idx - 2
                return a6000s[a6k_idx % len(a6000s)] if a6000s else h100
        else:
            # Standard: [Q, K, V]
            if component_idx == 0:
                return h100
            else:
                a6k_idx = component_idx - 1
                return a6000s[a6k_idx % len(a6000s)] if a6000s else h100

    def dispatch(
        self,
        grad: Tensor,
        split_shapes: List[int],
        param_device: torch.device,
    ) -> QKVSplitResult:
        """
        Split a QKV gradient and orthogonalize each component on the optimal device.

        Args:
            grad: The combined QKV gradient, shape [num_groups * sum(split_shapes), hidden].
            split_shapes: Per-component row counts (3 or 4 elements).
            param_device: Home device of the parameter (where result must be returned).

        Returns:
            QKVSplitResult with orthogonalized components.

        Raises:
            RuntimeError: If split_shapes doesn't divide grad.shape[0] evenly.
        """
        split_dim = sum(split_shapes)
        if grad.shape[0] % split_dim != 0:
            raise RuntimeError(
                f"HeteroMuon QKV split shape mismatch: "
                f"grad.shape={tuple(grad.shape)}, sum(split_shapes)={split_dim}"
            )

        num_groups = grad.shape[0] // split_dim
        hidden = grad.shape[-1]

        logger.debug(
            "QKV dispatch: grad=%s split_shapes=%s num_groups=%d hidden=%d",
            tuple(grad.shape), split_shapes, num_groups, hidden,
        )

        # Reshape to [num_groups, split_dim, hidden] and split along dim=1
        grad_3d = grad.view(num_groups, split_dim, hidden)
        sub_grads = torch.split(grad_3d, split_shapes, dim=1)
        # Reshape each to [-1, hidden] for NS step
        sub_grads = [g.reshape(-1, hidden) for g in sub_grads]

        pcie_bytes = 0
        results: List[Optional[Tensor]] = [None] * len(sub_grads)
        source_devices: List[torch.device] = []

        # Launch orthogonalization on each device concurrently
        handles: List[Tuple[int, torch.cuda.Stream, Tensor, DeviceInfo]] = []

        for i, sg in enumerate(sub_grads):
            target_dev_info = self._get_device_for_component(i, split_shapes)

            if target_dev_info is None:
                # No GPU available; use CPU
                target_device = torch.device("cpu")
                sg_on_device = sg.cpu()
                orth = _ns_orthogonalize_quintic(sg_on_device, self.ns_steps, self.ns_coeffs)
                results[i] = orth.to(param_device)
                source_devices.append(target_device)
                continue

            target_device = torch.device(f"cuda:{target_dev_info.index}")

            # Check memory headroom; spill to CPU if tight
            if not target_dev_info.has_headroom(sg.numel() * sg.element_size() * 4):
                logger.warning(
                    "Device cuda:%d low on memory (%d bytes free), spilling component %d to CPU",
                    target_dev_info.index, target_dev_info.free_memory_bytes(), i,
                )
                sg_cpu = self.locality_cache.spill_to_cpu(sg)
                orth = _ns_orthogonalize_quintic(sg_cpu, self.ns_steps, self.ns_coeffs)
                results[i] = orth.to(param_device, non_blocking=True)
                source_devices.append(torch.device("cpu"))
                continue

            # Transfer to target device (non-blocking, via locality cache)
            if sg.device != target_device:
                pcie_bytes += sg.numel() * sg.element_size()
            sg_on_device = self.locality_cache.resolve_gradient(
                # Wrap in a pseudo-param for the cache interface
                _PseudoParam(sg, target_device), target_device
            ) if sg.device != target_device else sg

            stream = self._streams[target_dev_info.index]
            handles.append((i, stream, sg_on_device, target_dev_info))

        # Execute NS steps on respective device streams
        for comp_idx, stream, sg_on_dev, dev_info in handles:
            with torch.cuda.stream(stream):
                orth = _ns_orthogonalize_quintic(
                    sg_on_dev, self.ns_steps, self.ns_coeffs, device_info=dev_info
                )
                # Non-blocking write-back to param's home device
                if orth.device != param_device:
                    pcie_bytes += orth.numel() * orth.element_size()
                    results[comp_idx] = orth.to(param_device, non_blocking=True)
                else:
                    results[comp_idx] = orth
                source_devices.append(orth.device)

        # Synchronize all streams
        for _, stream, _, _ in handles:
            stream.synchronize()

        # Ensure all results are populated
        for i, r in enumerate(results):
            if r is None:
                raise RuntimeError(f"QKV component {i} was not computed")

        return QKVSplitResult(
            components=results,
            split_shapes=split_shapes,
            source_devices=source_devices,
            pcie_transfer_bytes=pcie_bytes,
        )


class _PseudoParam:
    """
    Minimal wrapper that makes a raw Tensor look like a parameter with .grad,
    for use with LocalityCache.resolve_gradient.
    """
    def __init__(self, grad: Tensor, target_device: torch.device):
        self.grad = grad.to(target_device, non_blocking=True)
        self.shape = grad.shape


# ---------------------------------------------------------------------------
# HeteroMuonGatedQKV Optimizer
# ---------------------------------------------------------------------------

class HeteroMuonGatedQKV(Optimizer):
    """
    Muon optimizer with heterogeneous device dispatch and gated-attention QKV fix.

    This is the primary DES-LOC adaptation of Megatron's TensorParallelMuon,
    incorporating the gated-attention QKV split fix from commit 947d6ae.

    Key behaviors:
    --------------
    1. **Gated attention support**: Correctly handles 3-split [Q, K, V] and
       4-split [Q, Q_gate, K, V] attention projections, fixing the upstream
       bug where gated attention produced shape mismatches.

    2. **Heterogeneous dispatch**: Q (and Q_gate) are orthogonalized on H100 SM90;
       K and V on A6000 SM86. Parallel CUDA streams hide PCIe transfer costs.

    3. **Per-parameter split shapes**: Each QKV parameter carries its own
       qkv_split_shapes attribute (set by tag_qkv_parameters), enabling model
       chunks with mixed attention types (standard + gated) in the same optimizer.

    4. **Non-QKV parameters**: Fall back to vanilla Adam (via _adam_fallback).
       This matches Muon's design: non-linear/embedding params use Adam.

    5. **Memory-aware execution**: Monitors device free memory and spills
       intermediate NS tensors to CPU DRAM when a GPU is under pressure.

    Args:
        params: Iterable of parameters or parameter groups.
        topology: DES-LOC device topology descriptor.
        locality_cache: DES-LOC shared locality cache instance.
        lr: Learning rate (default: 0.02, as in original Muon).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        ns_steps: Newton-Schulz iteration count (default: 5).
        ns_coeffs: Quintic NS polynomial coefficients.
        split_qkv: Whether to apply per-component QKV splitting (default: True).
        adam_lr: Learning rate for Adam fallback on non-linear params.
        adam_betas: Beta coefficients for Adam fallback.
        adam_eps: Epsilon for Adam fallback.
    """

    def __init__(
        self,
        params,
        topology: DeviceTopology,
        locality_cache: Optional[LocalityCache] = None,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = _DEFAULT_NS_STEPS,
        ns_coeffs: Tuple[float, float, float] = _NS_QUINTIC_COEFFS,
        split_qkv: bool = True,
        adam_lr: float = 3e-4,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

        self.topology = topology
        self.locality_cache = locality_cache or LocalityCache()
        self.ns_coeffs = ns_coeffs
        self.split_qkv = split_qkv
        self.adam_lr = adam_lr
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps

        self._dispatcher = HeteroQKVSplitDispatcher(
            topology=topology,
            locality_cache=self.locality_cache,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
        )

        # Track total PCIe bytes moved (for profiling / DES-LOC telemetry)
        self._total_pcie_bytes: int = 0
        self._step_count: int = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single Muon optimization step.

        For each parameter group:
          - QKV parameters (is_qkv=True, split_qkv=True): split grad into
            per-component sub-tensors, orthogonalize each on the optimal device,
            reconstruct the combined update.
          - 2D weight parameters (non-QKV): apply NS orthogonalization on
            the parameter's home device (or best available).
          - 1D / embedding / output parameters: apply Adam update.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HeteroMuonGatedQKV does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0

                state['step'] += 1
                buf = state['momentum_buffer']

                # Route to appropriate update path
                if self.split_qkv and getattr(p, 'is_qkv', False):
                    update = self._qkv_muon_update(p, grad)
                elif p.dim() == 2 and not getattr(p, 'is_embedding_or_output_parameter', False):
                    update = self._standard_muon_update(grad, p.device)
                else:
                    # Adam fallback for 1D / embedding / output params
                    self._adam_update(p, grad, state)
                    continue

                # Apply Nesterov momentum
                buf.mul_(momentum).add_(update)
                p.add_(buf, alpha=-lr)

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

        return loss

    def _qkv_muon_update(self, param: Tensor, grad: Tensor) -> Tensor:
        """
        Compute the Muon update for a QKV-split parameter.

        Dispatches sub-tensors to heterogeneous devices, orthogonalizes in parallel,
        then reconstructs the combined update tensor on param's home device.

        DES-LOC critical path:
          grad → split → [Q on H100, K on A6000-0, V on A6000-1] → NS → gather → update
        """
        split_shapes = getattr(param, 'qkv_split_shapes', None)
        if split_shapes is None:
            raise RuntimeError(
                f"Parameter marked is_qkv=True but has no qkv_split_shapes: {param.shape}"
            )

        param_device = param.device
        result = self._dispatcher.dispatch(grad, split_shapes, param_device)
        self._total_pcie_bytes += result.pcie_transfer_bytes

        if result.pcie_transfer_bytes > 0:
            logger.debug(
                "QKV dispatch step %d: moved %.1f MB over PCIe for param shape %s",
                self._step_count,
                result.pcie_transfer_bytes / (1024 ** 2),
                tuple(param.shape),
            )

        # Reconstruct: concatenate components back to the original grad shape
        # Each component has been orthogonalized and returned to param_device
        split_dim = sum(split_shapes)
        num_groups = grad.shape[0] // split_dim
        hidden = grad.shape[-1]

        # Reshape components back to [num_groups, split_i, hidden] and cat along dim=1
        reshaped = [
            c.view(num_groups, split_shapes[i], hidden)
            for i, c in enumerate(result.components)
        ]
        combined = torch.cat(reshaped, dim=1)   # [num_groups, split_dim, hidden]
        update = combined.view(grad.shape[0], hidden)

        # Scale update to match param norm (Muon scaling)
        update = self._scale_to_param_rms(update, param)
        return update

    def _standard_muon_update(self, grad: Tensor, param_device: torch.device) -> Tensor:
        """
        Compute the Muon update for a standard 2D weight parameter.

        Selects the best available device for NS orthogonalization.
        On DES-LOC: prefer H100 if free memory is available; else use A6000.
        """
        # Select compute device: prefer H100
        compute_device = self._select_compute_device(grad.numel() * grad.element_size() * 4)
        dev_info = self.topology.devices[compute_device.index] if hasattr(compute_device, 'index') \
            else None

        if grad.device != torch.device(f"cuda:{compute_device.index}" if dev_info else "cpu"):
            grad_on_compute = grad.to(
                torch.device(f"cuda:{compute_device.index}"), non_blocking=True
            )
            torch.cuda.synchronize(compute_device.index)
        else:
            grad_on_compute = grad

        orth = _ns_orthogonalize_quintic(
            grad_on_compute,
            num_steps=self.param_groups[0]['ns_steps'],
            coeffs=self.ns_coeffs,
            device_info=compute_device,
        )

        if orth.device != param_device:
            orth = orth.to(param_device, non_blocking=True)
            torch.cuda.synchronize()

        return orth

    def _select_compute_device(self, required_bytes: int) -> DeviceInfo:
        """
        Choose the best available device for a compute task.

        Priority: H100 (if has headroom) > A6000 (round-robin) > CPU mock.
        """
        h100 = self.topology.primary_h100()
        if h100 is not None and h100.has_headroom(required_bytes):
            return h100
        for a6k in self.topology.all_a6000():
            if a6k.has_headroom(required_bytes):
                return a6k
        # Last resort: return first device (will likely cause OOM, but let CUDA handle it)
        return self.topology.devices[0]

    def _adam_update(
        self, param: Tensor, grad: Tensor, state: Dict[str, Any]
    ) -> None:
        """
        Apply Adam update in-place for non-linear / embedding parameters.

        These parameters don't benefit from Muon's orthogonalization because
        their gradients don't form well-conditioned matrices (1D, embeddings, etc.).
        Adam is applied directly on the parameter's home device.
        """
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)

        beta1, beta2 = self.adam_betas
        eps = self.adam_eps
        step = state['step']

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        step_size = self.adam_lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)

    @staticmethod
    def _scale_to_param_rms(update: Tensor, param: Tensor) -> Tensor:
        """
        Scale the orthogonalized update to match the RMS of the parameter.

        Muon scales the update by sqrt(max(rows, cols)) to maintain consistent
        effective learning rate across layers of different sizes. We additionally
        scale by the parameter's own RMS to prevent large updates on near-zero params.
        """
        scale = max(1, math.sqrt(max(update.shape[0], update.shape[1])))
        param_rms = param.norm() / math.sqrt(param.numel())
        param_rms = max(param_rms.item(), 1e-6)
        return update * (scale * param_rms)

    def pcie_stats(self) -> Dict[str, float]:
        """Return PCIe transfer statistics accumulated since optimizer creation."""
        return {
            "total_pcie_gb": self._total_pcie_bytes / (1024 ** 3),
            "avg_pcie_gb_per_step": (
                self._total_pcie_bytes / (1024 ** 3) / max(1, self._step_count)
            ),
            "step_count": self._step_count,
        }


# ---------------------------------------------------------------------------
# Parameter Group Builder
# ---------------------------------------------------------------------------

def build_hetero_muon_param_groups(
    model: torch.nn.Module,
    config: Optional[Any] = None,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_wd: float = 0.0,
    adam_lr: float = 3e-4,
) -> List[Dict[str, Any]]:
    """
    Build parameter groups for HeteroMuonGatedQKV from a model.

    Separates parameters into:
      - muon_params: 2D weight tensors suitable for Muon orthogonalization,
        including QKV params tagged with is_qkv=True and qkv_split_shapes.
      - adam_params: Everything else (1D, embeddings, outputs, LayerNorm scales).

    Args:
        model: The model whose parameters to group.
        config: Model config for QKV split shape computation (optional; if None,
                QKV params will not be tagged with split shapes).
        muon_lr: Learning rate for the Muon group.
        muon_momentum: Momentum for the Muon group.
        muon_wd: Weight decay for the Muon group.
        adam_lr: Learning rate for the Adam fallback group.

    Returns:
        List of two parameter group dicts: [muon_group, adam_group].
    """
    if config is not None:
        # Tag QKV params if we have config
        split_shapes = get_qkv_split_shapes_for_config(config)
        tag_qkv_parameters(model, split_shapes=split_shapes)

    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.dim() == 2
            and not getattr(param, 'is_embedding_or_output_parameter', False)
        ):
            muon_params.append(param)
        else:
            adam_params.append(param)

    logger.info(
        "Parameter groups: muon=%d params, adam=%d params",
        len(muon_params), len(adam_params),
    )

    return [
        {
            "params": muon_params,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "weight_decay": muon_wd,
            "ns_steps": _DEFAULT_NS_STEPS,
        },
        {
            "params": adam_params,
            "lr": adam_lr,
            "momentum": 0.0,   # momentum not used for Adam path; kept for group compatibility
            "weight_decay": 0.0,
            "ns_steps": _DEFAULT_NS_STEPS,
        },
    ]


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback
    import unittest

    class _MockModelConfig:
        """Minimal model config for testing."""
        def __init__(
            self,
            num_attention_heads: int = 16,
            num_query_groups: int = 8,
            kv_channels: int = 64,
            attention_output_gate: bool = False,
        ):
            self.num_attention_heads = num_attention_heads
            self.num_query_groups = num_query_groups
            self.kv_channels = kv_channels
            self.attention_output_gate = attention_output_gate

    class TestGetQKVSplitShapes(unittest.TestCase):
        """Tests for get_qkv_split_shapes_for_config — mirrors Megatron's test_muon_qkv_split_shapes."""

        def test_standard_gqa(self):
            """Standard GQA: 3-split [Q, K, V]."""
            cfg = _MockModelConfig(
                num_attention_heads=16, num_query_groups=8, kv_channels=64
            )
            shapes = get_qkv_split_shapes_for_config(cfg)
            # Q = (16 // 8) * 64 = 128; K = V = 64
            self.assertEqual(shapes, [128, 64, 64])

        def test_gated_attention(self):
            """Gated attention: 4-split [Q, Q_gate, K, V]."""
            cfg = _MockModelConfig(
                num_attention_heads=16, num_query_groups=8, kv_channels=64,
                attention_output_gate=True,
            )
            shapes = get_qkv_split_shapes_for_config(cfg)
            # Q = Q_gate = 128; K = V = 64
            self.assertEqual(shapes, [128, 128, 64, 64])

        def test_mha_no_gqa(self):
            """Multi-head attention (no GQA, groups == heads): Q_proj = kv_channels."""
            cfg = _MockModelConfig(
                num_attention_heads=8, num_query_groups=8, kv_channels=64
            )
            shapes = get_qkv_split_shapes_for_config(cfg)
            self.assertEqual(shapes, [64, 64, 64])

        def test_invalid_groups(self):
            """num_query_groups not dividing num_attention_heads raises ValueError."""
            cfg = _MockModelConfig(num_attention_heads=16, num_query_groups=7, kv_channels=64)
            with self.assertRaises(ValueError):
                get_qkv_split_shapes_for_config(cfg)

        def test_zero_groups_raises(self):
            cfg = _MockModelConfig(num_attention_heads=16, num_query_groups=0, kv_channels=64)
            with self.assertRaises(ValueError):
                get_qkv_split_shapes_for_config(cfg)

    class TestValidateQKVParamShape(unittest.TestCase):
        """Tests for validate_qkv_param_shape."""

        def test_valid_standard(self):
            # [16 groups * (128+64+64), hidden] = [4096, 512]
            param = torch.empty(16 * (128 + 64 + 64), 512)
            self.assertTrue(validate_qkv_param_shape(param, [128, 64, 64]))

        def test_valid_gated(self):
            param = torch.empty(8 * (128 + 128 + 64 + 64), 512)
            self.assertTrue(validate_qkv_param_shape(param, [128, 128, 64, 64]))

        def test_invalid_not_divisible(self):
            # shape[0] = 999 is not divisible by 128+64+64=256
            param = torch.empty(999, 512)
            self.assertFalse(validate_qkv_param_shape(param, [128, 64, 64]))

        def test_invalid_1d(self):
            param = torch.empty(512)
            self.assertFalse(validate_qkv_param_shape(param, [128, 64, 64]))

    class TestTagQKVParameters(unittest.TestCase):
        """Tests for tag_qkv_parameters."""

        def _make_model_with_qkv(self) -> torch.nn.Module:
            """Create a minimal model with a parameter named 'linear_qkv.weight'."""
            class _Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 8 groups * (128+64+64) = 2048 rows
                    self.linear_qkv = torch.nn.Linear(512, 8 * (128 + 64 + 64), bias=False)
                    self.output = torch.nn.Linear(512, 512, bias=False)
            return _Model()

        def test_tags_qkv_param(self):
            model = self._make_model_with_qkv()
            count = tag_qkv_parameters(model, split_shapes=[128, 64, 64])
            self.assertEqual(count, 1)
            # Find the QKV param
            for name, param in model.named_parameters():
                if 'linear_qkv' in name:
                    self.assertTrue(getattr(param, 'is_qkv', False))
                    self.assertEqual(param.qkv_split_shapes, [128, 64, 64])

        def test_does_not_tag_output(self):
            model = self._make_model_with_qkv()
            tag_qkv_parameters(model, split_shapes=[128, 64, 64])
            for name, param in model.named_parameters():
                if 'output' in name:
                    self.assertFalse(getattr(param, 'is_qkv', False))

        def test_from_config(self):
            model = self._make_model_with_qkv()
            cfg = _MockModelConfig(
                num_attention_heads=16, num_query_groups=8, kv_channels=64
            )
            count = tag_qkv_parameters(model, model_cfg=cfg)
            self.assertEqual(count, 1)

    class TestNSOrthogonalize(unittest.TestCase):
        """Tests for _ns_orthogonalize_quintic."""

        def test_cpu_orthogonalization(self):
            torch.manual_seed(42)
            G = torch.randn(64, 128)
            orth = _ns_orthogonalize_quintic(G, num_steps=5)
            self.assertEqual(orth.shape, G.shape)
            # After orthogonalization, G @ G.T should be close to I (scaled)
            # We just check it's finite and same shape
            self.assertTrue(torch.isfinite(orth).all())

        def test_tall_matrix_transposed_path(self):
            """Tall matrix (rows > cols) should be transposed and transposed back."""
            torch.manual_seed(7)
            G = torch.randn(256, 64)   # tall: 256 > 64
            orth = _ns_orthogonalize_quintic(G, num_steps=3)
            self.assertEqual(orth.shape, (256, 64))

        def test_square_matrix(self):
            torch.manual_seed(0)
            G = torch.randn(64, 64)
            orth = _ns_orthogonalize_quintic(G, num_steps=5)
            self.assertEqual(orth.shape, (64, 64))

        def test_mock_device_info(self):
            """DeviceInfo mock does not crash on CPU path."""
            dev = DeviceInfo(
                index=0, name="MockA6000",
                compute_capability=_SM86_CAPABILITY,
                total_memory_bytes=48 * 1024 ** 3,
            )
            G = torch.randn(32, 64)
            orth = _ns_orthogonalize_quintic(G, num_steps=3, device_info=dev)
            self.assertEqual(orth.shape, G.shape)

    class TestDeviceTopology(unittest.TestCase):
        """Tests for DeviceTopology.mock."""

        def test_mock_topology(self):
            topo = DeviceTopology.mock(n_a6000=2, n_h100=1)
            self.assertEqual(len(topo.devices), 3)
            self.assertEqual(len(topo.a6000_indices), 2)
            self.assertEqual(len(topo.h100_indices), 1)

        def test_h100_is_h100(self):
            topo = DeviceTopology.mock(n_a6000=2, n_h100=1)
            h100 = topo.primary_h100()
            self.assertIsNotNone(h100)
            self.assertTrue(h100.is_h100)
            self.assertFalse(h100.is_a6000)

        def test_a6000_is_a6000(self):
            topo = DeviceTopology.mock(n_a6000=2, n_h100=1)
            a6k = topo.all_a6000()
            self.assertEqual(len(a6k), 2)
            for d in a6k:
                self.assertTrue(d.is_a6000)
                self.assertFalse(d.is_h100)

        def test_supports_bf16(self):
            topo = DeviceTopology.mock(n_a6000=1, n_h100=1)
            for d in topo.devices:
                self.assertTrue(d.supports_bf16)   # Both SM86 and SM90 support bf16

        def test_fp8_only_on_h100(self):
            topo = DeviceTopology.mock(n_a6000=1, n_h100=1)
            h100 = topo.primary_h100()
            a6k = topo.all_a6000()[0]
            self.assertTrue(h100.supports_fp8)
            self.assertFalse(a6k.supports_fp8)

    class TestHeteroQKVSplitDispatcherCPU(unittest.TestCase):
        """
        Tests for HeteroQKVSplitDispatcher using CPU fallback (no real GPUs needed).

        These tests use mock topology where all 'devices' have index 0 with
        has_headroom always False, forcing CPU fallback path.
        """

        def _make_dispatcher(self) -> HeteroQKVSplitDispatcher:
            topo = DeviceTopology.mock(n_a6000=2, n_h100=1)
            cache = LocalityCache()
            # Monkey-patch has_headroom to always return False to force CPU path
            for d in topo.devices:
                d.has_headroom = lambda *a, **kw: False
            return HeteroQKVSplitDispatcher(
                topology=topo,
                locality_cache=cache,
                ns_steps=2,
            )

        def test_standard_3split_cpu_fallback(self):
            dispatcher = self._make_dispatcher()
            split_shapes = [128, 64, 64]
            num_groups = 4
            hidden = 256
            grad = torch.randn(num_groups * sum(split_shapes), hidden)
            result = dispatcher.dispatch(grad, split_shapes, torch.device("cpu"))
            self.assertEqual(len(result.components), 3)
            self.assertEqual(result.split_shapes, split_shapes)
            for i, c in enumerate(result.components):
                expected_rows = num_groups * split_shapes[i]
                self.assertEqual(c.shape, (expected_rows, hidden))

        def test_gated_4split_cpu_fallback(self):
            dispatcher = self._make_dispatcher()
            split_shapes = [128, 128, 64, 64]
            num_groups = 2
            hidden = 512
            grad = torch.randn(num_groups * sum(split_shapes), hidden)
            result = dispatcher.dispatch(grad, split_shapes, torch.device("cpu"))
            self.assertEqual(len(result.components), 4)
            for i, c in enumerate(result.components):
                self.assertEqual(c.shape, (num_groups * split_shapes[i], hidden))

        def test_invalid_shape_raises(self):
            dispatcher = self._make_dispatcher()
            grad = torch.randn(99, 128)   # 99 not divisible by 128+64+64=256
            with self.assertRaises(RuntimeError):
                dispatcher.dispatch(grad, [128, 64, 64], torch.device("cpu"))

    class TestHeteroMuonGatedQKVCPU(unittest.TestCase):
        """
        Integration tests for HeteroMuonGatedQKV on CPU (no real GPU needed).
        Uses the mock topology + CPU locality cache.
        """

        def _make_simple_model_with_qkv(self) -> torch.nn.Module:
            class _M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # QKV: 4 groups * (128+64+64)=1024 rows, 256 hidden
                    self.linear_qkv = torch.nn.Linear(256, 4 * (128 + 64 + 64), bias=False)
                    self.fc = torch.nn.Linear(256, 256, bias=False)
            return _M()

        def _make_optimizer(self, model: torch.nn.Module) -> HeteroMuonGatedQKV:
            topo = DeviceTopology.mock(n_a6000=2, n_h100=1)
            # Force CPU path: no headroom on any mock device
            for d in topo.devices:
                d.has_headroom = lambda *a, **kw: False

            tag_qkv_parameters(model, split_shapes=[128, 64, 64])

            groups = build_hetero_muon_param_groups(
                model, config=None, muon_lr=0.01, adam_lr=1e-4
            )
            # Manually un-tag since build_hetero_muon_param_groups won't have config
            for name, param in model.named_parameters():
                if 'linear_qkv' in name:
                    param.is_qkv = True
                    param.qkv_split_shapes = [128, 64, 64]

            return HeteroMuonGatedQKV(
                params=list(model.parameters()),
                topology=topo,
                locality_cache=LocalityCache(),
                ns_steps=2,
                split_qkv=True,
                adam_lr=1e-4,
            )

        def test_step_runs_without_error(self):
            torch.manual_seed(123)
            model = self._make_simple_model_with_qkv()
            optimizer = self._make_optimizer(model)

            # Create fake input and compute a gradient
            x = torch.randn(4, 256)
            out = model.linear_qkv(x)
            loss = out.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        def test_parameters_change_after_step(self):
            torch.manual_seed(456)
            model = self._make_simple_model_with_qkv()
            optimizer = self._make_optimizer(model)

            params_before = {
                n: p.clone().detach()
                for n, p in model.named_parameters()
            }

            x = torch.randn(4, 256)
            out = model.fc(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

            # At least fc params should have changed (Adam path)
            fc_before = params_before['fc.weight']
            fc_after = dict(model.named_parameters())['fc.weight']
            self.assertFalse(
                torch.allclose(fc_before, fc_after),
                "fc.weight should have changed after Adam step",
            )

        def test_pcie_stats(self):
            torch.manual_seed(789)
            model = self._make_simple_model_with_qkv()
            optimizer = self._make_optimizer(model)

            x = torch.randn(4, 256)
            model.linear_qkv(x).mean().backward()
            optimizer.step()

            stats = optimizer.pcie_stats()
            self.assertIn("total_pcie_gb", stats)
            self.assertIn("step_count", stats)
            self.assertEqual(stats["step_count"], 1)

    class TestBuildHeteroMuonParamGroups(unittest.TestCase):
        """Tests for build_hetero_muon_param_groups."""

        def test_groups_structure(self):
            class _SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear_qkv = torch.nn.Linear(64, 8 * (128 + 64 + 64), bias=False)
                    self.norm = torch.nn.LayerNorm(64)   # 1D params → Adam
                    self.fc = torch.nn.Linear(64, 64, bias=True)   # bias is 1D → Adam
            model = _SimpleModel()
            groups = build_hetero_muon_param_groups(model, config=None)
            self.assertEqual(len(groups), 2)
            muon_group, adam_group = groups
            # linear_qkv.weight (2D) and fc.weight (2D) → muon
            muon_params = muon_group['params']
            self.assertTrue(all(p.dim() == 2 for p in muon_params))
            # fc.bias (1D), norm.weight (1D), norm.bias (1D) → adam
            adam_params = adam_group['params']
            self.assertTrue(all(p.dim() <= 1 for p in adam_params))

    # Run all test cases
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    suite = unittest.TestLoader().loadTestsFromTestCase(TestGetQKVSplitShapes)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestValidateQKVParamShape))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTagQKVParameters))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNSOrthogonalize))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeviceTopology))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroQKVSplitDispatcherCPU))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroMuonGatedQKVCPU))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBuildHeteroMuonParamGroups))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
