# ISSUE 3 — Activation Checkpointing & Offload: Tier-Aware Strategy

**Status:** Design proposal  
**Author:** OffloadLead  
**Target:** `desloc_engine.py` + `hetero_gdn_selective_recompute.py` + `fine_grained_activation_offload.py`  
**Cluster:** 2× RTX A6000 (48 GB VRAM, SM 8.6, PCIe ×16) + 1× H100 NVL (96 GB VRAM, SM 9.0, PCIe ×16)

---

## Background

Three separate defects prevent training at `seq_len=4096` on the A6000 nodes:

| # | File | Defect |
|---|------|--------|
| 1 | `desloc_engine.py` line 260 | `activation_checkpointing` defaults to `False`; `seq_len=2048` works but `seq_len=4096` OOMs |
| 2 | `hetero_gdn_selective_recompute.py` | `hetero_gdn_selective_recompute` module is not wired to produce per-tier strategies; it always runs `granularity="selective"` regardless of `TierClass` |
| 3 | `deepspeed/core/pipeline_parallel/fine_grained_activation_offload.py` | 1075-line `PipelineOffloadManager` / `FineGrainedActivationOffloadingInterface` stack exists, `maybe_enable_activation_offload()` in `core_adapters.py` wraps it correctly, but neither is called from the training path in `desloc_engine.py` |

These three defects interact: even if `activation_checkpointing=True` is set (fixing #1), GDN layers will still `selective`-recompute on A6000 (not full), and the activation tensor pool will still hit DRAM via autograd's default saved-tensor path rather than the async-pinned D2H path (fixing #3).

---

## 1. Per-Tier Activation Checkpointing: Full on A6000, Selective on H100

### Current state (line 1837 of `desloc_engine.py`)

```python
_ckpt_master_on = bool(config.activation_checkpointing)   # False by default
_ckpt_granularity = str(config.checkpoint_activations_granularity).lower()
```

The tier-dispatch logic already exists at lines 1862–1897 and does the right thing:

```python
elif tier == TierClass.A6000:
    apply_ckpt = True                          # every layer
    policy_label = "FULL (A6000 — every layer)"
elif tier == TierClass.H100:
    apply_ckpt = (layer_idx % 2 == 0)         # every other layer
    policy_label = "SELECTIVE (H100 — even layer)"
```

The only reason this doesn't fire is that `_ckpt_master_on` is `False`. The fix is therefore a default change plus a safe migration path.

### Fix 1a — Change the default and force-on for A6000 tiers

In `TrainingConfig` (line 260):

```python
# Before
activation_checkpointing: bool = False
checkpoint_activations_granularity: str = "full"

# After
activation_checkpointing: bool = True          # safe new default
checkpoint_activations_granularity: str = "full"
```

Changing the default to `True` is safe: the engine already guards all downstream callers with `_ckpt_master_on`, so no-op behaviour is preserved if the flag is explicitly set to `False` by a caller.

### Fix 1b — Tier-override bypass for A6000

Add a short-circuit that forces the master switch on whenever an A6000 tier is present, regardless of the config value. Insert after line 1837:

```python
_ckpt_master_on = bool(config.activation_checkpointing)

# Force-enable full checkpointing on A6000 tiers regardless of config.
# A6000 has 48 GB VRAM; seq_len=4096 activations exceed safe headroom.
_has_a6000 = any(
    spec.tier == TierClass.A6000 for spec in self.tiers
)
if _has_a6000 and not _ckpt_master_on:
    logger.warning(
        "[ActCkpt] A6000 tier detected with activation_checkpointing=False; "
        "forcing ON to prevent OOM at seq_len≥4096."
    )
    _ckpt_master_on = True
```

This addresses the immediate OOM without breaking callers that intentionally disable checkpointing on H100-only clusters.

### Fix 1c — Verify `hetero_gdn_selective_recompute` granularity aligns with tier policy

The `HeteroRecomputeConfig` built at line 1795 via `build_neuron_sp_config()` hardcodes `granularity="selective"` globally. This means GDN layers on A6000 nodes only recompute `norm_out`, not the full layer. For `seq_len=4096` the rest of the residual stream activations (attention QKV projections, MLP intermediate states) remain in VRAM.

The `torch.utils.checkpoint` wrapping at lines 1857–1922 already handles the full-vs-selective split for generic `TransformerBlock` layers. GDN layers need to additionally be told to do `full` recompute on A6000.

Extend `build_neuron_sp_config()` to accept a `granularity_override` argument:

```python
# hetero_gdn_selective_recompute.py — build_neuron_sp_config()

def build_neuron_sp_config(
    a6000_indices: Sequence[int] = (0, 1),
    h100_index: int = 2,
    recompute_threshold_gb: float = 0.5,
    loc_max_cpu_gb: float = 32.0,
    a6000_granularity: str = "full",      # <-- new param
) -> HeteroRecomputeConfig:
    ...
    return HeteroRecomputeConfig(
        granularity=a6000_granularity,    # "full" for A6000, "selective" for H100
        modules_per_device={
            DeviceClass.A6000: {"gdn_norm_out"},
            DeviceClass.H100_NVL: set(),
            DeviceClass.UNKNOWN: set(),
        },
        ...
    )
```

In `desloc_engine.py` line 1795, thread the tier presence check through:

```python
_a6000_indices = [
    spec.device_index for spec in self.tiers if spec.tier == TierClass.A6000
]
_h100_indices  = [
    spec.device_index for spec in self.tiers if spec.tier == TierClass.H100
]
_h100_idx = _h100_indices[0] if _h100_indices else 2

self.neuron_sp_config = build_neuron_sp_config(
    a6000_indices=_a6000_indices or (0, 1),
    h100_index=_h100_idx,
    # A6000 with seq_len>=4096: full recompute inside GDN layers.
    # H100 has headroom; keep selective (norm_out only).
    a6000_granularity="full" if _a6000_indices else "selective",
)
```

`HeteroRecomputeConfig.should_recompute_norm_out()` currently gates on `granularity == "selective"`. When `granularity="full"` it returns `False`, meaning `CheckpointWithoutOutput` is bypassed and `torch.utils.checkpoint` at the outer `TransformerBlock` level absorbs the entire GDN forward instead — which is the correct behaviour for A6000.

---

## 2. Integrating `fine_grained_activation_offload` into the Training Path

### Current gap

`maybe_enable_activation_offload()` exists in `core_adapters.py` (adapter #6, line 406) and correctly:
- Skips on H100/DATACENTER tiers (`offload_required_for_tier()` returns `False`)
- Calls `PipelineOffloadManager.reset_instance()` and pre-initialises a `ChunkOffloadHandler` for VPP stage 0
- Returns `FineGrainedActivationOffloadingInterface` or `None`

It is never called from `desloc_engine.py`. The `use_activation_offload` config flag (read in `maybe_enable_activation_offload`) is also absent from `TrainingConfig`.

### Fix 2a — Add config knobs to `TrainingConfig`

In `desloc_engine.py` around line 260:

```python
# Fine-grained CPU activation offload (A6000 only).
# use_activation_offload: enables PipelineOffloadManager D2H path for saved tensors.
# activation_offload_min_size: minimum tensor element count to offload (default 1 M).
# activation_offload_max_inflight: max concurrent D2H transfers per group (None = unlimited).
use_activation_offload: bool = False
activation_offload_min_size: int = 1_048_576   # 1 M elements ≈ 4 MB at fp32 / 2 MB at bf16
activation_offload_max_inflight: Optional[int] = 4
```

### Fix 2b — Call `maybe_enable_activation_offload` in `__init__` after tier classification

Insert after the `neuron_sp_config` build (after line 1800), using the dominant tier of rank-local GPU:

```python
from deepspeed.runtime.core_adapters import maybe_enable_activation_offload  # noqa: PLC0415
from deepspeed.core.desloc_config import TierType as _TierType               # noqa: PLC0415

# Map TierClass → TierType for the offload adapter.
_local_tier_class = _dev_tier.get(primary_idx, TierClass.UNKNOWN)
_tier_type_for_offload = {
    TierClass.A6000:          _TierType.PROFESSIONAL,
    TierClass.RTX_PRO_6000_BW: _TierType.PROFESSIONAL,
    TierClass.H100:           _TierType.DATACENTER,
    TierClass.UNKNOWN:        None,
}.get(_local_tier_class, None)

self._activation_offload_iface = maybe_enable_activation_offload(
    config, tier_type=_tier_type_for_offload
)
logger.info(
    "[FineGrainedOffload] iface=%s  tier=%s",
    "ACTIVE" if self._activation_offload_iface is not None else "SKIPPED",
    _local_tier_class.value,
)
```

`maybe_enable_activation_offload` reads `config.use_activation_offload`; if `False` it returns `None` immediately and is a complete no-op.

### Fix 2c — Wrap transformer layer forward passes in the offload context

In the training loop, the `PipelineOffloadManager.__enter__` / `__exit__` context manager replaces PyTorch's default `save_for_backward` with async D2H hooks via `torch._C._autograd._push_saved_tensors_default_hooks`. This must wrap each micro-batch forward call.

In `desloc_engine.py` inside the training loop (around where `self.model(batch)` is called):

```python
_offload_ctx = (
    self._activation_offload_iface.get_context(flag=True)
    if self._activation_offload_iface is not None
    else nullcontext()
)

with _offload_ctx:
    logits = self.model(tokens)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))

# Commit the offload group after forward (triggers bulk D2H of pending tensors).
if self._activation_offload_iface is not None:
    self._activation_offload_iface.group_commit(
        loss,
        name="micro_batch",
        delay_offload=False,
    )
```

`FineGrainedActivationOffloadingInterface.get_context(flag=True)` returns the live `PipelineOffloadManager` instance, which implements `__enter__`/`__exit__` to push/pop the saved-tensor hooks. The hooks only fire for tensors larger than `activation_offload_min_size` (screened inside `ChunkOffloadHandler.tensor_need_offloading_checker`).

### Fix 2d — Reset `PipelineOffloadManager` after eval (Megatron M3490 parity)

Megatron commit `b8e23d587` (M3490) fixes stale state in the offload manager after eval runs. Mirror this in `desloc_engine.py` wherever eval is performed:

```python
# After eval loop, before resuming training:
if self._activation_offload_iface is not None:
    self._activation_offload_iface.reset_instance()
    # Re-initialise with the same VPP stage 0 chunk handler.
    self._activation_offload_iface.init_chunk_handler(
        vp_size=getattr(config, "virtual_pipeline_model_parallel_size", None),
        vp_stage=0,
        min_offloaded_tensor_size=config.activation_offload_min_size,
        max_inflight_offloads=config.activation_offload_max_inflight,
    )
```

This mirrors the `PipelineOffloadManager.reset()` call inside `reset_instance()` and clears the backward-chunk deque that accumulates stale entries during eval inference (no backward pass to drain it).

### Integration summary

```
TrainingConfig
  use_activation_offload = True          (new)
  activation_offload_min_size = 1M       (new)
  activation_checkpointing = True        (default change)

desloc_engine.__init__
  ├── build_neuron_sp_config(a6000_granularity="full")   [MODIFIED]
  ├── per-layer torch.utils.checkpoint wrapping          [EXISTS, master switch fixed]
  └── maybe_enable_activation_offload(config, tier_type) [NEW CALL]
          └── PipelineOffloadManager.reset_instance()
          └── FineGrainedActivationOffloadingInterface.init_chunk_handler(vp_size=1, vp_stage=0)

desloc_engine.train() — inner loop
  └── with FineGrainedActivationOffloadingInterface.get_context(True):
          model forward  →  saved tensors intercepted by async D2H hooks
      group_commit()     →  bulk D2H flush

desloc_engine.eval()
  └── reset_instance() + re-init chunk handler           [NEW, M3490 parity]
```

---

## 3. Memory Savings Estimate for `seq_len=4096` on A6000

### Activation memory formula

For a transformer layer in standard seq-first layout `[S, B, H]` with `S = seq_len`, `B = micro_batch_size`, `H = hidden_size`, the dominant activation tensors saved during forward are:

| Tensor | Shape | Size (BF16) |
|--------|-------|-------------|
| Attention QKV projections (3×) | `[S, B, H]` each | `3 × S × B × H × 2` |
| Attention softmax scores | `[B, heads, S, S]` | `B × 32 × S² × 2` |
| MLP intermediate (×2 gates) | `[S, B, 4H]` each | `2 × S × B × 4H × 2` |
| Residual / input to each sub-layer | `[S, B, H]` | `S × B × H × 2` |
| GDN `norm_out_hp` | `[B, S_hp, H]` | `≈ S × B × H × 2` |

With the default config: `H=4096`, `B=2` (micro-batch), `heads=32`.

**Per-layer activation footprint at `seq_len=2048`** (no checkpointing):

```
QKV:       3 × 2048 × 2 × 4096 × 2 bytes =  96 MB
Attn mat:  2 × 32 × 2048² × 2 bytes      = 536 MB   (dominant term)
MLP:       2 × 2048 × 2 × 16384 × 2      = 256 MB
Residuals: 4 × 2048 × 2 × 4096 × 2       = 128 MB
─────────────────────────────────────────────────────
Per layer                                 ≈ 1016 MB ≈ 1.0 GB
32 layers                                 ≈ 32.5 GB
```

At `seq_len=4096` the attention matrix term scales as `S²`, everything else scales linearly with `S`:

```
QKV:       × 2 linear  →  192 MB
Attn mat:  × 4 quad    → 2144 MB   ← blows up
MLP:       × 2 linear  →  512 MB
Residuals: × 2 linear  →  256 MB
─────────────────────────────────────────────────────
Per layer                              ≈ 3.1 GB
32 layers                              ≈ 99 GB   >> 48 GB A6000 VRAM
```

Without any checkpointing, `seq_len=4096` is unrunnable on A6000.

### What `torch.utils.checkpoint` (full, every layer) saves

Full checkpointing discards **all** intermediate activations for each wrapped `TransformerBlock` during forward; only the block's input tensor is retained. On backward, the full block forward is rerun.

Saved after checkpointing per layer: only the block input `[S, B, H]`:
```
1 × 4096 × 2 × 4096 × 2 = 64 MB per layer
32 layers = 2.0 GB activation baseline
```

Activation memory freed relative to no checkpointing at `seq_len=4096`:
```
Freed = 99 GB − 2 GB ≈ 97 GB
```

In practice the A6000's full VRAM budget is also occupied by:
- Model parameters (BF16): `12 × 32 × 4096² × 2 ≈ 12.9 GB`
- Optimizer state (FP32 shards via `DistributedOptimizer`): `∼13 GB per A6000`
- Workspace / NCCL buffers: `∼1–2 GB`

Total non-activation overhead: `≈ 27 GB`, leaving `48 − 27 = 21 GB` for activations.

At `seq_len=4096` with full checkpointing (2 GB), the A6000 has `≈ 19 GB` headroom — comfortably within budget.

### What `fine_grained_activation_offload` adds on top

`PipelineOffloadManager` operates at a finer granularity than `torch.utils.checkpoint`: instead of rerunning the entire forward, it asynchronously D2H-copies specific large saved tensors (attention scores, MLP intermediates) during forward, then H2D-copies them back lazily during backward. This is only beneficial when the offload BW cost is cheaper than recompute cost.

For A6000 (PCIe BW ≈ 32 GB/s bidirectional, per card):

- Attention score tensor: `2144 MB` per layer → D2H at 32 GB/s ≈ **67 ms latency**
- Recompute equivalent: one full attention pass at A6000 bf16 TFLOPS ≈ 77.4 TF → ≈ **8 ms**

For attention activations, **recompute is cheaper than offload** on A6000 PCIe. Full checkpointing (`torch.utils.checkpoint`) is therefore the right primary strategy.

`fine_grained_activation_offload` is still useful for the residual/embedding tensors that are cheap to D2H but expensive to recompute (e.g. embedding lookup output, which is `S × B × H` = 64 MB and has a non-trivial backward through the embedding table). Suggested `activation_offload_min_size = 1_048_576` (1 M elements ≈ 4 MB bf16) will catch these while ignoring tiny tensors.

### Conservative headroom estimate at `seq_len=4096` after both fixes

| Component | A6000 VRAM |
|-----------|-----------|
| Model params (BF16) | 12.9 GB |
| Optimizer FP32 shards | 13.0 GB |
| NCCL / workspace | 1.5 GB |
| Activation baseline (full ckpt, 32 layers) | 2.0 GB |
| Offload residuals kept on-GPU | 0.5 GB |
| **Total estimated** | **29.9 GB** |
| **A6000 capacity** | **48.0 GB** |
| **Headroom** | **≈ 18 GB** |

This is a comfortable margin. Gradient accumulation over 8 micro-batches does not multiply activation memory (each micro-batch's checkpointed activations are freed after its backward pass); only the gradient buffers accumulate linearly (< 1 GB additional for BF16 grads).

### GDN `norm_out_hp` recompute saving (Megatron M4141 parity)

With `a6000_granularity="full"` in `build_neuron_sp_config`, GDN layers on A6000 are wrapped by `torch.utils.checkpoint` at the outer block level, so `norm_out_hp` is automatically discarded without needing `CheckpointWithoutOutput`. The M4141 selective-recompute path is still valuable on H100 (where full checkpoint is avoided): `norm_out_hp` shape `[B, S_hp, H] = [2, 2048, 4096]` = 32 MB saved per GDN layer at no compute cost on H100.

---

## Recommended Implementation Order

1. **`TrainingConfig` defaults** — change `activation_checkpointing=True`, add `use_activation_offload`, `activation_offload_min_size`, `activation_offload_max_inflight`. Low risk; all guarded by the master switch.

2. **Force-on guard for A6000** (Fix 1b) — 4-line insert, defensive, no behaviour change on H100-only clusters.

3. **`build_neuron_sp_config` granularity threading** (Fix 1c) — extend signature, thread `_a6000_indices` from engine init. GDN full-recompute on A6000 activates only when `granularity="full"`.

4. **`maybe_enable_activation_offload` wiring** (Fix 2b–2c) — call site in `__init__` and context-manager wrapper in training loop. Gated by `use_activation_offload=False` default; opt-in.

5. **Eval reset** (Fix 2d, M3490 parity) — add reset call after eval block.

All five changes are independently testable and can be landed in separate commits.

---

## References

- `desloc_engine.py` lines 255–261 (TrainingConfig activation_checkpointing fields)
- `desloc_engine.py` lines 1793–1932 (Phase 7: neuron_sp_config + per-tier ckpt wrapping)
- `deepspeed/runtime/hetero_gdn_selective_recompute.py` — `HeteroRecomputeConfig`, `build_neuron_sp_config`, `HeteroGDNNormOutRecompute`
- `deepspeed/core/pipeline_parallel/fine_grained_activation_offload.py` — `PipelineOffloadManager`, `FineGrainedActivationOffloadingInterface`, `offload_required_for_tier`
- `deepspeed/runtime/core_adapters.py` lines 403–498 (adapter #6, `maybe_enable_activation_offload`)
- Megatron commit M3490 `b8e23d587`: reset activation offload manager after eval
- Megatron commit M4141 `ff5264c33`: selective recompute for `norm_out` in GDN layers
