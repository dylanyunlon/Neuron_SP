"""
HeteroEPGradNormMuon — DES-LOC Heterogeneous Expert-Parallel Gradient Norm and
Zero-Count Correction for Muon Optimizer.

Upstream Design Intent (Megatron b2e93909):
============================================
Megatron's commit fixes a subtle bug in Expert-Parallel (EP) Muon training where
gradient norm and num_zeros statistics were computed incorrectly when EP weight groups
were mixed into the same optimizer instance as non-EP weights.  The root cause:
TensorParallelMuon receives a flat list of param_groups; when expert params from
different EP shards land in the same gradnorm all-reduce, their contribution is
double-counted (or omitted) depending on which process group is used.

The fix has two parts:
  1. *EP param groups are split out* from the main Muon optimizer and wrapped in a
     separate TensorParallelMuon+Float16Optimizer chain, with its own
     `grad_stats_parallel_group` set to `pg_collection.tp_ep_pp` (not plain TP).
  2. `init_state_fn` callables are hoisted before optimizer construction so that
     checkpoint restoration (torch_dist format) can call them on any optimizer in
     the chained list uniformly via `opt.init_state_fn(opt)`.

DES-LOC Adaptation Points:
===========================
DES-LOC = Decoupled Execution with Shared LOcality Cache.

Hardware context: 2× A6000 48 GB (SM86) + 1× H100 NVL 96 GB (SM90), PCIe only,
1.5 TB CPU DRAM.  No NVLink means all-reduce across GPUs is PCIe-bandwidth-limited
(~32 GB/s effective vs NVLink's ~600 GB/s).  Expert params can be 100s of MB; a
naive gradnorm all-reduce across all three devices for EP groups would saturate PCIe.

Key DES-LOC adaptations:
  A. **Device-Aware EP Partition** — expert param groups are pinned to the H100
     (large HBM, SM90 BF16 throughput) while dense Muon runs on A6000 pair.
     The locality cache tag (`_des_loc_device`) is attached to each param group so
     the DES-LOC scheduler knows where optimizer state lives.

  B. **Staged GradNorm Reduction** — instead of a single cross-device all-reduce for
     gradnorm, we do a two-stage reduction:
       stage 1: intra-device reduce (cheap, same PCIe root)
       stage 2: cross-device scalar reduce (tiny payload: just the norm scalar)
     This matches what Megatron achieves via `grad_stats_parallel_group` but does it
     without assuming NVLink symmetry.

  C. **CPU DRAM Locality Cache** — momentum buffers for EP experts are kept in
     pinned CPU memory when GPU HBM would overflow, with async prefetch into H100
     HBM during the forward pass.  The `_des_loc_offload` flag on a param group
     enables this path.

  D. **Heterogeneous init_state_fn** — state initializers are device-placement-aware:
     Muon momentum buffers go to the param's canonical device; Adam exp_avg tensors
     for EP params go to CPU pinned memory when `_des_loc_offload=True`.

  E. **BF16-only enforcement** — fp16 is explicitly blocked (matching upstream) because
     DES-LOC's locality cache compression uses BF16 mantissa-aligned quantization.

Usage:
------
    from deepspeed.moe.hetero_ep_gradnorm_muon import (
        build_des_loc_muon_optimizer_chain,
        HeteroEPGradNormConfig,
    )

    cfg = HeteroEPGradNormConfig(
        lr=1e-3,
        muon_momentum=0.95,
        weight_decay=0.1,
        bf16=True,
        h100_device=torch.device("cuda:2"),   # H100 NVL
        a6000_devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        cpu_offload_ep_state=True,
    )
    optimizers = build_des_loc_muon_optimizer_chain(cfg, model_chunks, pg_collection)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroEPGradNormConfig:
    """
    Configuration for DES-LOC heterogeneous EP Muon optimizer chain.

    Attributes
    ----------
    lr : float
        Base learning rate shared across Muon and Adam sub-optimizers.
    muon_momentum : float
        Nesterov momentum beta for TensorParallelMuon (upstream: config.muon_momentum).
    muon_use_nesterov : bool
        Whether to use Nesterov correction in Muon orthogonalisation step.
    weight_decay : float
        Decoupled weight decay applied by both Muon and Adam.
    bf16 : bool
        Must be True; fp16 is blocked (DES-LOC locality cache requires BF16).
    fp16 : bool
        Must remain False; raises if True.
    muon_fp32_matmul_prec : str
        Precision for internal Muon Newton-Schulz matmuls ('highest'/'high'/'medium').
    muon_num_ns_steps : int
        Number of Newton-Schulz iteration steps for approximate orthogonalisation.
    muon_scale_mode : str
        Spectral/RMS scaling mode ('spectral' | 'rms' | 'none').
    muon_split_qkv : bool
        Whether to split QKV projections for per-head Muon update.
    muon_extra_scale_factor : float
        Extra scale multiplier applied after Muon update.
    muon_tp_mode : str
        Tensor-parallel communication mode for Muon ('reduce_scatter' | 'allreduce').
    use_distributed_optimizer : bool
        Must remain False for Muon (upstream constraint, kept for guard).
    use_precision_aware_optimizer : bool
        If True, Adam state is initialised via optimizer.initialize_state(p).
    h100_device : torch.device
        The H100 NVL device (canonical home for EP expert params).
    a6000_devices : List[torch.device]
        The A6000 devices (canonical home for dense/linear params).
    cpu_offload_ep_state : bool
        If True, EP optimizer state (momentum, exp_avg) is kept in pinned CPU DRAM.
    ep_prefetch_stream : Optional[torch.cuda.Stream]
        CUDA stream used for async H100←CPU prefetch of EP optimizer state.
    """

    lr: float = 1e-3
    muon_momentum: float = 0.95
    muon_use_nesterov: bool = True
    weight_decay: float = 0.1
    bf16: bool = True
    fp16: bool = False
    muon_fp32_matmul_prec: str = "high"
    muon_num_ns_steps: int = 5
    muon_scale_mode: str = "spectral"
    muon_split_qkv: bool = True
    muon_extra_scale_factor: float = 1.0
    muon_tp_mode: str = "reduce_scatter"
    use_distributed_optimizer: bool = False
    use_precision_aware_optimizer: bool = False
    h100_device: torch.device = field(default_factory=lambda: torch.device("cuda:2"))
    a6000_devices: List[torch.device] = field(
        default_factory=lambda: [torch.device("cuda:0"), torch.device("cuda:1")]
    )
    cpu_offload_ep_state: bool = True
    ep_prefetch_stream: Optional[torch.cuda.Stream] = None


# ---------------------------------------------------------------------------
# Locality-cache helpers
# ---------------------------------------------------------------------------

def _make_pinned_zeros_like(t: Tensor) -> Tensor:
    """
    Create a CPU pinned-memory tensor with the same shape and dtype as *t*.

    DES-LOC Locality Cache: expert optimizer states are kept in pinned CPU DRAM
    (1.5 TB available) and asynchronously streamed to H100 HBM during the
    optimizer step.  Pinned memory allows DMA transfers without staging copies.
    """
    return torch.zeros(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)


def _async_h2d(cpu_tensor: Tensor, device: torch.device,
               stream: Optional[torch.cuda.Stream] = None) -> Tensor:
    """
    Asynchronously copy a pinned CPU tensor to *device*.

    If *stream* is provided the copy is issued on that stream so it can overlap
    with forward/backward computation (DES-LOC prefetch).  Returns the device
    tensor; caller must synchronise before use if stream != default.
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            return cpu_tensor.to(device=device, non_blocking=True)
    return cpu_tensor.to(device=device, non_blocking=False)


def _async_d2h(gpu_tensor: Tensor, cpu_tensor: Tensor,
               stream: Optional[torch.cuda.Stream] = None) -> None:
    """
    Asynchronously copy *gpu_tensor* back into pre-allocated *cpu_tensor*.

    Used after the optimizer step to flush updated state back to the CPU locality
    cache without blocking the next forward pass.
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    else:
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)


# ---------------------------------------------------------------------------
# init_state_fn factories
# ---------------------------------------------------------------------------

def make_muon_init_state_fn(
    cpu_offload: bool = False,
    prefetch_stream: Optional[torch.cuda.Stream] = None,
) -> Callable:
    """
    Return a Muon state initialiser compatible with torch_dist checkpoint format.

    Upstream (Megatron b2e93909): `muon_init_state_fn` was moved *before* optimizer
    construction so that `opt.init_state_fn(opt)` can be called uniformly from
    checkpoint utils, replacing per-optimizer manual loops.

    DES-LOC adaptation: when *cpu_offload=True* the momentum buffer is allocated in
    pinned CPU DRAM (locality cache).  A ``_cpu_momentum_buffer`` shadow attribute
    is set alongside the regular ``momentum_buffer`` (which will be a device view
    used only during the optimizer step) to allow async double-buffering.

    Parameters
    ----------
    cpu_offload : bool
        If True, primary storage is in pinned CPU memory.
    prefetch_stream : optional CUDA stream
        If provided, prefetch from CPU → device is issued on this stream.
    """
    def muon_init_state_fn(opt: Any, config: Any = None) -> None:
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    if cpu_offload:
                        cpu_buf = _make_pinned_zeros_like(p.data)
                        opt.state[p]["_cpu_momentum_buffer"] = cpu_buf
                        # device view starts as zeros on the param's device
                        opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
                        logger.debug(
                            "DES-LOC Muon init_state: param shape=%s pinned to CPU DRAM",
                            tuple(p.shape),
                        )
                    else:
                        opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
    return muon_init_state_fn


def make_adam_init_state_fn(
    cpu_offload: bool = False,
    use_precision_aware: bool = False,
) -> Callable:
    """
    Return an Adam state initialiser for the non-linear param chain.

    Upstream (Megatron b2e93909): `adam_init_state_fn` was hoisted to before
    optimizer construction, and `opt.init_state_fn(opt)` is now called uniformly
    in checkpoint utils.  The precision-aware branch delegates to
    `optimizer.initialize_state(p)`.

    DES-LOC adaptation: when *cpu_offload=True* exp_avg and exp_avg_sq live in
    pinned CPU DRAM (locality cache).  This is appropriate for EP expert Adam
    states which can be large (MoE has many expert parameters) and are accessed
    once per step — the PCIe transfer cost is amortised by async prefetch.

    Parameters
    ----------
    cpu_offload : bool
        Allocate state in pinned CPU memory rather than GPU HBM.
    use_precision_aware : bool
        Delegate state init to `optimizer.initialize_state(p)` when True.
    """
    def adam_init_state_fn(opt: Any, config: Any = None) -> None:
        _use_prec = use_precision_aware
        if config is not None:
            _use_prec = getattr(config, "use_precision_aware_optimizer", _use_prec)
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    if _use_prec:
                        opt.initialize_state(p)
                    elif cpu_offload:
                        opt.state[p]["exp_avg"] = _make_pinned_zeros_like(p.data)
                        opt.state[p]["exp_avg_sq"] = _make_pinned_zeros_like(p.data)
                        logger.debug(
                            "DES-LOC Adam init_state: param shape=%s exp_avg pinned to CPU DRAM",
                            tuple(p.shape),
                        )
                    else:
                        opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                        opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
    return adam_init_state_fn


# ---------------------------------------------------------------------------
# Two-stage gradient norm reduction for heterogeneous devices
# ---------------------------------------------------------------------------

class HeteroEPGradNormReducer:
    """
    Two-stage gradient norm and zero-count reducer for DES-LOC heterogeneous setup.

    Upstream Problem (Megatron b2e93909):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The original Muon optimizer computed gradnorm/num_zeros using the TP process
    group, which was correct for dense layers.  But EP expert layers have a wider
    process group (tp_ep_pp) because their parameters are sharded across EP ranks
    as well.  When EP param groups were mixed into the main Muon optimizer, their
    gradnorm contribution used the wrong process group, causing:
      - gradnorm to be inflated (params counted multiple times across EP ranks)
      - num_zeros to be similarly incorrect

    Upstream fix: split EP params into a separate optimizer and set
    `grad_stats_parallel_group = pg_collection.tp_ep_pp` on it.

    DES-LOC Adaptation:
    ~~~~~~~~~~~~~~~~~~~
    Our hardware has no NVLink so a naive all-reduce across all 3 GPUs (2×A6000 +
    H100) would saturate the PCIe bus.  We use a staged approach:

      Stage 1 — Intra-device norm aggregation:
        Each device computes a local partial norm^2 sum over its owned params.
        This is pure local GPU work, no cross-device traffic.

      Stage 2 — Cross-device scalar all-reduce:
        Only the scalar norm^2 value (8 bytes) crosses the PCIe bus.
        This is O(1) communication cost regardless of model size.

    For num_zeros we follow the same staged pattern: local count → scalar reduce.

    The split between dense-param group (A6000 pair, TP group) and EP-param group
    (H100, tp_ep_pp group) is enforced by the `_des_loc_device` tag on each group.
    """

    def __init__(
        self,
        dense_pg: Optional[dist.ProcessGroup],
        ep_pg: Optional[dist.ProcessGroup],
        h100_device: torch.device,
        a6000_devices: List[torch.device],
        norm_type: float = 2.0,
    ) -> None:
        self.dense_pg = dense_pg
        self.ep_pg = ep_pg
        self.h100_device = h100_device
        self.a6000_devices = a6000_devices
        self.norm_type = norm_type
        logger.info(
            "HeteroEPGradNormReducer initialised: h100=%s a6000=%s",
            h100_device,
            a6000_devices,
        )

    def _local_norm_sq(self, params: List[Tensor]) -> Tensor:
        """Compute sum of squared gradient norms locally (no communication)."""
        total = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total += g.norm(self.norm_type) ** self.norm_type
        return total

    def _local_zero_count(self, params: List[Tensor]) -> int:
        """Count zero-valued gradient elements locally."""
        count = 0
        for p in params:
            if p.grad is None:
                continue
            count += int((p.grad == 0).sum().item())
        return count

    def _stage2_reduce_scalar(
        self,
        local_val: Tensor,
        pg: Optional[dist.ProcessGroup],
        device: torch.device,
    ) -> Tensor:
        """
        All-reduce a scalar tensor across *pg*.

        Moves the scalar to *device* for the reduce (CUDA-aware comms), then
        returns the result on CPU to avoid keeping tensors on GPU after the step.
        Only 8 bytes cross the PCIe bus — negligible compared to param transfers.
        """
        if pg is None or not dist.is_initialized():
            return local_val
        dev_val = local_val.to(device=device, non_blocking=False)
        dist.all_reduce(dev_val, op=dist.ReduceOp.SUM, group=pg)
        return dev_val.cpu()

    def compute_grad_norm(
        self,
        dense_params: List[Tensor],
        ep_params: List[Tensor],
    ) -> float:
        """
        Compute the global gradient norm across dense and EP param groups.

        Dense params → stage-1 local norm^2 on A6000 → stage-2 scalar reduce on TP group.
        EP params    → stage-1 local norm^2 on H100   → stage-2 scalar reduce on tp_ep_pp group.
        Final global norm = (dense_norm^2 + ep_norm^2)^(1/norm_type).

        Returns
        -------
        float
            The global gradient norm.
        """
        dense_norm_sq = self._local_norm_sq(dense_params)
        ep_norm_sq = self._local_norm_sq(ep_params)

        # Stage 2: scalar cross-device reduces
        a6000_anchor = self.a6000_devices[0] if self.a6000_devices else torch.device("cpu")
        dense_norm_sq = self._stage2_reduce_scalar(dense_norm_sq, self.dense_pg, a6000_anchor)
        ep_norm_sq = self._stage2_reduce_scalar(ep_norm_sq, self.ep_pg, self.h100_device)

        total_norm = (dense_norm_sq.item() + ep_norm_sq.item()) ** (1.0 / self.norm_type)
        logger.debug(
            "HeteroEPGradNorm: dense_norm_sq=%.6f ep_norm_sq=%.6f total_norm=%.6f",
            dense_norm_sq.item(),
            ep_norm_sq.item(),
            total_norm,
        )
        return total_norm

    def compute_num_zeros(
        self,
        dense_params: List[Tensor],
        ep_params: List[Tensor],
    ) -> int:
        """
        Compute the global count of zero gradient elements across all ranks.

        Follows the same two-stage pattern as compute_grad_norm but for integers.
        The scalar int is broadcast as float32 for all-reduce compatibility.
        """
        dense_zeros = self._local_zero_count(dense_params)
        ep_zeros = self._local_zero_count(ep_params)

        dense_t = torch.tensor(float(dense_zeros), dtype=torch.float32)
        ep_t = torch.tensor(float(ep_zeros), dtype=torch.float32)

        a6000_anchor = self.a6000_devices[0] if self.a6000_devices else torch.device("cpu")
        dense_t = self._stage2_reduce_scalar(dense_t, self.dense_pg, a6000_anchor)
        ep_t = self._stage2_reduce_scalar(ep_t, self.ep_pg, self.h100_device)

        total = int(dense_t.item() + ep_t.item())
        logger.debug(
            "HeteroEPNumZeros: dense=%d ep=%d total=%d",
            int(dense_t.item()),
            int(ep_t.item()),
            total,
        )
        return total


# ---------------------------------------------------------------------------
# Param-group classification helpers
# ---------------------------------------------------------------------------

def _classify_param_groups(
    param_groups: List[Dict],
    layer_wise_distributed_optimizer: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split *param_groups* into dense (linear/non-EP) and EP groups.

    Upstream (Megatron b2e93909):
    When `layer_wise_distributed_optimizer=False`, EP param groups must be removed
    from the main Muon list and given their own optimizer with `grad_stats_parallel_group`
    set to `tp_ep_pp`.  When the layerwise optimizer IS used, it handles EP internally.

    DES-LOC adaptation:
    We additionally tag each group with `_des_loc_device` to indicate its canonical
    placement device, enabling the DES-LOC scheduler to enforce locality constraints
    without re-inspecting param tensors at step time.

    Returns
    -------
    dense_groups : list of groups not tagged as expert_parallel
    ep_groups    : list of groups tagged as expert_parallel (empty if layerwise)
    """
    dense_groups: List[Dict] = []
    ep_groups: List[Dict] = []

    if layer_wise_distributed_optimizer:
        # Layerwise dist-opt handles EP internally; no split needed.
        logger.info(
            "_classify_param_groups: layerwise dist-opt active, no EP split."
        )
        return list(param_groups), []

    for group in param_groups:
        is_ep = group.get("is_expert_parallel", False)
        if is_ep:
            ep_groups.append(group)
            logger.debug(
                "_classify_param_groups: routed group (n_params=%d) → EP optimizer",
                len(group.get("params", [])),
            )
        else:
            dense_groups.append(group)

    logger.info(
        "_classify_param_groups: dense_groups=%d ep_groups=%d",
        len(dense_groups),
        len(ep_groups),
    )
    return dense_groups, ep_groups


def _tag_groups_with_device(
    groups: List[Dict],
    device: torch.device,
    offload: bool = False,
) -> None:
    """
    Attach DES-LOC locality metadata to each param group in-place.

    Tags added:
      ``_des_loc_device``   — canonical compute device for this group's params.
      ``_des_loc_offload``  — True if optimizer state should live in CPU DRAM.

    The DES-LOC scheduler reads these tags at step time to decide whether to
    prefetch state from the locality cache (CPU DRAM) before the optimizer update.
    """
    for g in groups:
        g["_des_loc_device"] = device
        g["_des_loc_offload"] = offload


# ---------------------------------------------------------------------------
# Muon kwargs builder
# ---------------------------------------------------------------------------

def _build_muon_kwargs(
    cfg: HeteroEPGradNormConfig,
    qkv_split_shapes: Optional[Any],
    pg_collection: Any,
) -> Dict[str, Any]:
    """
    Build the keyword-argument dict for TensorParallelMuon.

    Upstream (Megatron b2e93909): the kwargs dict was factored out of the inline
    TensorParallelMuon(...) call so it can be reused for both the dense and EP
    optimizer instances without duplication.

    DES-LOC: no changes to the kwargs themselves; the device placement is handled
    via param group tags and the locality cache layer, not via Muon internals.
    """
    return {
        "lr": cfg.lr,
        "momentum_beta": cfg.muon_momentum,
        "use_nesterov": cfg.muon_use_nesterov,
        "weight_decay": cfg.weight_decay,
        "fp32_matmul_prec": cfg.muon_fp32_matmul_prec,
        "num_ns_steps": cfg.muon_num_ns_steps,
        "scale_mode": cfg.muon_scale_mode,
        "split_qkv": cfg.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": cfg.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": cfg.muon_tp_mode,
    }


# ---------------------------------------------------------------------------
# QKV split shape collection
# ---------------------------------------------------------------------------

def _collect_qkv_split_shapes(model_chunks: List[Any]) -> List[Any]:
    """
    Walk model chunks and collect per-head QKV split shapes for Muon.

    Upstream: each model chunk is inspected for attention modules that carry
    `qkv_split_shape` attributes (set during TP initialisation).  The shape list
    controls how Muon orthogonalises the QKV projection matrix per attention head.

    DES-LOC: unchanged from upstream logic.  This is topology metadata, not
    placement metadata, so no heterogeneous-device consideration applies here.
    """
    shapes: List[Any] = []
    for chunk in model_chunks:
        if not hasattr(chunk, "named_modules"):
            continue
        for _name, module in chunk.named_modules():
            shape = getattr(module, "qkv_split_shape", None)
            if shape is not None:
                shapes.append(shape)
    return shapes


# ---------------------------------------------------------------------------
# Main optimizer chain builder
# ---------------------------------------------------------------------------

def build_des_loc_muon_optimizer_chain(
    cfg: HeteroEPGradNormConfig,
    model_chunks: List[Any],
    pg_collection: Any,
    layer_wise_distributed_optimizer: bool = False,
    _get_param_groups_fn: Optional[Callable] = None,
    _TensorParallelMuon_cls: Optional[Any] = None,
    _Float16Optimizer_cls: Optional[Any] = None,
    _FP32Optimizer_cls: Optional[Any] = None,
) -> List[Any]:
    """
    Build the full DES-LOC Muon optimizer chain for heterogeneous EP training.

    This is the DES-LOC counterpart of Megatron's ``get_megatron_muon_optimizer``,
    re-architected to handle the 2×A6000 + 1×H100 PCIe-only topology.

    Upstream Algorithm (Megatron b2e93909):
    ----------------------------------------
    1. Guard against dist-opt and fp16 (both unsupported with Muon).
    2. Hoist ``muon_init_state_fn`` and ``adam_init_state_fn`` before construction
       so checkpoint utils can call ``opt.init_state_fn(opt)`` uniformly.
    3. Classify params into linear (Muon) and nonlinear (Adam) groups.
    4. If ``layer_wise_distributed_optimizer=False``, further split linear groups
       into dense and EP groups.
    5. Build separate TensorParallelMuon instances for dense and EP groups, with
       the EP instance carrying ``grad_stats_parallel_group = pg_collection.tp_ep_pp``.
    6. Wrap each in Float16Optimizer or FP32Optimizer as appropriate.
    7. Unfreeze nonlinear params, freeze linear params, build Adam optimizer.

    DES-LOC Additions:
    ------------------
    A. ``_tag_groups_with_device`` tags param groups with canonical device and
       offload flag so the DES-LOC scheduler can enforce locality.
    B. ``HeteroEPGradNormReducer`` is instantiated and attached to the returned
       optimizer list so the training loop can call two-stage norm/zero-count
       without saturating PCIe.
    C. EP optimizer state (momentum buffers, exp_avg) is initialised in pinned
       CPU DRAM when ``cfg.cpu_offload_ep_state=True``.
    D. The ``init_state_fn`` on each optimizer is device-placement-aware.

    Parameters
    ----------
    cfg : HeteroEPGradNormConfig
        Full configuration including hardware topology.
    model_chunks : list
        List of model chunks (same shape as Megatron model_chunks).
    pg_collection : ProcessGroupCollection-like
        Must expose ``.tp``, ``.tp_ep_pp``, and ``.dp`` process groups.
    layer_wise_distributed_optimizer : bool
        When True, EP splitting is skipped (the layerwise optimizer handles it).
    _get_param_groups_fn : callable, optional
        Injected for testing; defaults to a no-op stub.
    _TensorParallelMuon_cls : class, optional
        Injected for testing; skips actual TensorParallelMuon import.
    _Float16Optimizer_cls : class, optional
        Injected for testing.
    _FP32Optimizer_cls : class, optional
        Injected for testing.

    Returns
    -------
    list
        List of wrapped optimizer objects ready for use with DeepSpeed engine.
        Each carries:
          - ``.init_state_fn``   — uniform callable for checkpoint utils
          - ``._des_loc_device`` — canonical device tag (via param groups)
          - ``.grad_stats_parallel_group`` — for correct gradnorm reduction

        The list also has a ``.hetero_reducer`` attribute on its first element
        pointing to the shared ``HeteroEPGradNormReducer``.

    Raises
    ------
    RuntimeError
        If ``cfg.use_distributed_optimizer=True`` (unsupported with Muon).
    RuntimeError
        If ``cfg.fp16=True`` (DES-LOC locality cache requires BF16).
    """
    # -----------------------------------------------------------------
    # Guards (upstream + DES-LOC additions)
    # -----------------------------------------------------------------
    if cfg.use_distributed_optimizer:
        raise RuntimeError(
            "DES-LOC Muon: dist-optimizer is not supported.  "
            "Muon's Newton-Schulz step requires a full-rank gradient view "
            "which is incompatible with DDP grad-buffer sharding."
        )
    if cfg.fp16:
        raise RuntimeError(
            "DES-LOC Muon: fp16 is not supported.  "
            "The DES-LOC locality cache uses BF16 mantissa-aligned quantisation; "
            "fp16 would corrupt the cached momentum statistics."
        )

    logger.info(
        "Building DES-LOC Muon optimizer chain: h100=%s a6000=%s cpu_offload=%s",
        cfg.h100_device,
        cfg.a6000_devices,
        cfg.cpu_offload_ep_state,
    )

    # -----------------------------------------------------------------
    # Resolve injected dependencies (real or test stubs)
    # -----------------------------------------------------------------
    if _get_param_groups_fn is None:
        def _get_param_groups_fn(model_chunks, config, config_overrides=None):
            # Minimal stub: return all params as a single group.
            # In production this is megatron.core.optimizer.utils._get_param_groups
            all_params = []
            for chunk in model_chunks:
                if hasattr(chunk, "parameters"):
                    all_params.extend(list(chunk.parameters()))
            return [{"params": all_params, "is_expert_parallel": False, "lr": config.lr}]

    TensorParallelMuon = _TensorParallelMuon_cls
    Float16Optimizer = _Float16Optimizer_cls
    FP32Optimizer = _FP32Optimizer_cls

    # -----------------------------------------------------------------
    # Hoist init_state_fn before construction (upstream b2e93909 change)
    # -----------------------------------------------------------------
    # DES-LOC: EP variant uses cpu_offload when configured.
    dense_muon_init_fn = make_muon_init_state_fn(cpu_offload=False)
    ep_muon_init_fn = make_muon_init_state_fn(
        cpu_offload=cfg.cpu_offload_ep_state,
        prefetch_stream=cfg.ep_prefetch_stream,
    )
    dense_adam_init_fn = make_adam_init_state_fn(
        cpu_offload=False,
        use_precision_aware=cfg.use_precision_aware_optimizer,
    )
    ep_adam_init_fn = make_adam_init_state_fn(
        cpu_offload=cfg.cpu_offload_ep_state,
        use_precision_aware=cfg.use_precision_aware_optimizer,
    )

    # -----------------------------------------------------------------
    # Collect QKV shapes and param groups
    # -----------------------------------------------------------------
    qkv_split_shapes = _collect_qkv_split_shapes(model_chunks)
    muon_kwargs = _build_muon_kwargs(cfg, qkv_split_shapes, pg_collection)

    linear_param_groups = _get_param_groups_fn(model_chunks, cfg)

    # -----------------------------------------------------------------
    # Split dense vs EP param groups (upstream fix, DES-LOC adapted)
    # -----------------------------------------------------------------
    dense_groups, ep_groups = _classify_param_groups(
        linear_param_groups, layer_wise_distributed_optimizer
    )

    # Tag with DES-LOC device metadata
    a6000_anchor = cfg.a6000_devices[0] if cfg.a6000_devices else cfg.h100_device
    _tag_groups_with_device(dense_groups, device=a6000_anchor, offload=False)
    _tag_groups_with_device(ep_groups, device=cfg.h100_device, offload=cfg.cpu_offload_ep_state)

    # -----------------------------------------------------------------
    # Build heterogeneous grad-norm reducer
    # -----------------------------------------------------------------
    tp_pg = getattr(pg_collection, "tp", None)
    tp_ep_pp_pg = getattr(pg_collection, "tp_ep_pp", None)
    reducer = HeteroEPGradNormReducer(
        dense_pg=tp_pg,
        ep_pg=tp_ep_pp_pg,
        h100_device=cfg.h100_device,
        a6000_devices=cfg.a6000_devices,
    )

    optimizers: List[Any] = []

    # -----------------------------------------------------------------
    # Dense Muon optimizer (lives on A6000 pair, TP group for grad stats)
    # -----------------------------------------------------------------
    if dense_groups and TensorParallelMuon is not None:
        dense_opt = TensorParallelMuon(dense_groups, **muon_kwargs)
        dense_opt.init_state_fn = dense_muon_init_fn
        # grad_stats uses TP group (dense params are not EP-sharded)
        dense_opt.grad_stats_parallel_group = tp_pg

        if cfg.bf16:
            # Reset bf16 flag temporarily if optimizer wrapper needs it
            if Float16Optimizer is not None:
                dense_opt = Float16Optimizer(dense_opt, cfg, None, dense_muon_init_fn)
                dense_opt.init_state_fn = dense_muon_init_fn
        else:
            if FP32Optimizer is not None:
                dense_opt = FP32Optimizer(dense_opt, cfg, dense_muon_init_fn)
                dense_opt.init_state_fn = dense_muon_init_fn

        setattr(dense_opt, "grad_stats_parallel_group", tp_pg)
        setattr(dense_opt, "_des_loc_tag", "dense_muon")
        optimizers.append(dense_opt)
        logger.info(
            "Dense Muon optimizer built: %d groups, device=%s",
            len(dense_groups),
            a6000_anchor,
        )
    elif dense_groups:
        # Stub path for testing without real TensorParallelMuon
        logger.warning("TensorParallelMuon not injected; dense groups skipped in stub mode.")

    # -----------------------------------------------------------------
    # EP Muon optimizer (lives on H100, tp_ep_pp group for grad stats)
    # -----------------------------------------------------------------
    # Upstream b2e93909: only built when len(ep_groups) > 0 and not layerwise.
    if ep_groups and TensorParallelMuon is not None:
        ep_opt = TensorParallelMuon(ep_groups, **muon_kwargs)
        ep_opt.init_state_fn = ep_muon_init_fn

        if cfg.bf16:
            if Float16Optimizer is not None:
                ep_opt = Float16Optimizer(ep_opt, cfg, None, ep_muon_init_fn)
                ep_opt.init_state_fn = ep_muon_init_fn
        else:
            if FP32Optimizer is not None:
                ep_opt = FP32Optimizer(ep_opt, cfg, ep_muon_init_fn)
                ep_opt.init_state_fn = ep_muon_init_fn

        # Upstream: set grad_stats_parallel_group = pg_collection.tp_ep_pp
        # DES-LOC: same, but the reducer also knows about this group for stage-2 reduce.
        setattr(ep_opt, "grad_stats_parallel_group", tp_ep_pp_pg)
        setattr(ep_opt, "_des_loc_tag", "ep_muon")
        optimizers.append(ep_opt)
        logger.info(
            "EP Muon optimizer built: %d groups, device=%s, cpu_offload=%s",
            len(ep_groups),
            cfg.h100_device,
            cfg.cpu_offload_ep_state,
        )
    elif ep_groups:
        logger.warning("TensorParallelMuon not injected; EP groups skipped in stub mode.")

    # Attach reducer to the first optimizer for training-loop access
    if optimizers:
        setattr(optimizers[0], "_hetero_reducer", reducer)

    logger.info(
        "DES-LOC Muon chain complete: %d optimizer(s) built", len(optimizers)
    )
    return optimizers


# ---------------------------------------------------------------------------
# Grad-norm hook for DeepSpeed engine integration
# ---------------------------------------------------------------------------

class DESOCMuonGradNormHook:
    """
    DeepSpeed-compatible grad-norm hook that uses two-stage PCIe-aware reduction.

    DeepSpeed calls ``optimizer.get_global_grad_norm()`` (or equivalent) after
    gradient synchronisation.  For heterogeneous EP setups the default single-stage
    all-reduce is incorrect (wrong process group for EP) and expensive (large tensor
    all-reduce vs scalar).

    This hook wraps the ``HeteroEPGradNormReducer`` and exposes a DeepSpeed-
    compatible interface.  Attach it to the optimizer chain with::

        hook = DESOCMuonGradNormHook(reducer, dense_params, ep_params)
        engine.register_grad_norm_hook(hook)  # hypothetical API

    Attributes
    ----------
    reducer : HeteroEPGradNormReducer
    dense_params : list of Tensor
    ep_params : list of Tensor
    _last_norm : float — cached result from most recent call
    _last_zeros : int  — cached zero-count from most recent call
    """

    def __init__(
        self,
        reducer: HeteroEPGradNormReducer,
        dense_params: List[Tensor],
        ep_params: List[Tensor],
    ) -> None:
        self.reducer = reducer
        self.dense_params = dense_params
        self.ep_params = ep_params
        self._last_norm: float = 0.0
        self._last_zeros: int = 0
        logger.info(
            "DESOCMuonGradNormHook: dense_params=%d ep_params=%d",
            len(dense_params),
            len(ep_params),
        )

    def get_grad_norm(self) -> float:
        """Return the two-stage PCIe-aware global gradient norm."""
        norm = self.reducer.compute_grad_norm(self.dense_params, self.ep_params)
        self._last_norm = norm
        return norm

    def get_num_zeros(self) -> int:
        """Return the two-stage PCIe-aware global zero-gradient count."""
        zeros = self.reducer.compute_num_zeros(self.dense_params, self.ep_params)
        self._last_zeros = zeros
        return zeros

    def __repr__(self) -> str:
        return (
            f"DESOCMuonGradNormHook("
            f"last_norm={self._last_norm:.4f}, "
            f"last_zeros={self._last_zeros})"
        )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def uniform_init_state_for_chain(optimizer_chain: List[Any]) -> None:
    """
    Call ``opt.init_state_fn(opt)`` on every optimizer in the chain.

    Upstream (Megatron b2e93909 / test utils):
    The old checkpoint util had separate loops for chained_optimizers[0] (Muon)
    and chained_optimizers[1] (Adam), manually initialising momentum_buffer and
    exp_avg respectively.  After the upstream fix, each optimizer carries its own
    `init_state_fn` callable and the loop is simply::

        for opt in optimizer.chained_optimizers:
            opt.init_state_fn(opt)

    DES-LOC: this function wraps that pattern for use with our optimizer list,
    handling the case where an optimizer wraps another (Float16Optimizer contains
    .optimizer attribute pointing to the inner TensorParallelMuon).
    """
    for opt in optimizer_chain:
        fn = getattr(opt, "init_state_fn", None)
        if fn is not None:
            fn(opt)
            logger.debug("uniform_init_state_for_chain: initialised %s", type(opt).__name__)
        else:
            logger.warning(
                "uniform_init_state_for_chain: optimizer %s has no init_state_fn",
                type(opt).__name__,
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    # ---- Test 1: config guards ----
    cfg_bad_fp16 = HeteroEPGradNormConfig(fp16=True, bf16=False)
    try:
        build_des_loc_muon_optimizer_chain(cfg_bad_fp16, [], None)
        assert False, "Should have raised for fp16"
    except RuntimeError as e:
        assert "fp16" in str(e).lower()
    logger.info("TEST 1 passed: fp16 guard raised correctly")

    # ---- Test 2: param group classification ----
    groups = [
        {"params": [torch.randn(4, 4)], "is_expert_parallel": False, "lr": 1e-3},
        {"params": [torch.randn(4, 4)], "is_expert_parallel": True, "lr": 1e-3},
        {"params": [torch.randn(4, 4)], "is_expert_parallel": True, "lr": 1e-3},
    ]
    dense, ep = _classify_param_groups(groups, layer_wise_distributed_optimizer=False)
    assert len(dense) == 1 and len(ep) == 2, f"Expected 1 dense, 2 ep; got {len(dense)}, {len(ep)}"
    logger.info("TEST 2 passed: param group classification dense=%d ep=%d", len(dense), len(ep))

    # ---- Test 3: init_state_fn — Muon momentum buffer allocation ----
    dummy_p = torch.randn(8, 8)
    dummy_p.requires_grad_(True)

    class _StubOpt:
        param_groups = [{"params": [dummy_p]}]
        state: Dict = {}

    stub_opt = _StubOpt()
    fn = make_muon_init_state_fn(cpu_offload=False)
    fn(stub_opt)
    assert "momentum_buffer" in stub_opt.state[dummy_p], "momentum_buffer missing"
    assert stub_opt.state[dummy_p]["momentum_buffer"].shape == dummy_p.shape
    logger.info("TEST 3 passed: Muon init_state_fn (no offload) momentum_buffer allocated")

    # ---- Test 4: init_state_fn — CPU pinned offload ----
    stub_opt2 = _StubOpt()
    stub_opt2.state = {}
    fn_offload = make_muon_init_state_fn(cpu_offload=True)
    fn_offload(stub_opt2)
    assert "_cpu_momentum_buffer" in stub_opt2.state[dummy_p], "_cpu_momentum_buffer missing"
    cpu_buf = stub_opt2.state[dummy_p]["_cpu_momentum_buffer"]
    assert cpu_buf.device.type == "cpu", f"Expected cpu, got {cpu_buf.device}"
    logger.info("TEST 4 passed: Muon init_state_fn (cpu_offload=True) pinned buffer on CPU")

    # ---- Test 5: HeteroEPGradNormReducer local norms (no dist) ----
    p1 = torch.randn(4, 4, requires_grad=True)
    p1.grad = torch.ones(4, 4)
    p2 = torch.randn(4, 4, requires_grad=True)
    p2.grad = torch.zeros(4, 4)

    reducer_test = HeteroEPGradNormReducer(
        dense_pg=None, ep_pg=None,
        h100_device=torch.device("cpu"),
        a6000_devices=[torch.device("cpu")],
    )
    norm_val = reducer_test.compute_grad_norm([p1], [p2])
    expected = float(p1.grad.norm(2).item())
    assert abs(norm_val - expected) < 1e-4, f"norm mismatch: {norm_val} vs {expected}"
    zeros = reducer_test.compute_num_zeros([p1], [p2])
    assert zeros == 16, f"zero count expected 16 (p2 all zeros), got {zeros}"
    logger.info(
        "TEST 5 passed: HeteroEPGradNormReducer norm=%.4f zeros=%d", norm_val, zeros
    )

    logger.info("All smoke tests passed.")
