# M_AUDIT — Megatron PR Skip / Partial-Port Log

Each entry records a Megatron upstream commit that was **evaluated but not
fully ported** to Neuron_SP, with the reason.

---

## M3998 — Route non-Muon params through DistributedOptimizer
**Upstream commit:** `9e60da33` (Megatron PR #4771, internal ref `0044db1f2`)
**Audit date:** 2026-06-29
**Status:** ✅ SKIP — already handled; API surface pre-adapted

### What the upstream commit solves

Megatron's `Muon` optimizer (an emerging/orthogonal-gradient optimizer) only
manages a subset of model parameters — specifically 2-D weight matrices
(`param.dim() == 2`) that are not embedding or output-projection parameters.
All other parameters (biases, layernorms, embeddings, 1-D tensors) were
previously left inside `LayerWiseDistributedOptimizer`, whose shard-aligned
buffer layout is **wrong** for byte-level DistributedOptimizer sharding.

The upstream fix:

1. **`BufferKey.is_managed_by_layer_wise_optimizer`** field added to
   `param_layout.py` so DDP can split one DDP buffer into two: a
   LayerWise-owned shard-aligned buffer (Muon matrix params) and a
   DistOpt-owned byte-level buffer (everything else).

2. **`is_managed_by_layer_wise_optimizer` predicate** replaces the old
   `param.dim() == 2` inline check in `get_mup_config_overrides()` —
   correctness is now derived from the buffer tag, not a shape heuristic.

3. **`_get_megatron_emerging_optimizer()`** in `__init__.py` gains a routing
   split: Muon `param_groups` feed `LayerWiseDistributedOptimizer`; non-Muon
   (Adam) `param_groups` feed a separate `DistributedOptimizer` constructed
   from the non-LayerWise filtered buffers.

4. **`LayerWiseDistributedOptimizer`** gains `start_param_sync_for_bucket_group_subset()`
   (walks only LayerWise-tagged buckets) and `step_with_ready_grads()` (so the
   sibling DistOpt's chained step can call it without double-syncing).

5. **Guard added**: `overlap_param_gather_with_optimizer_step` is incompatible
   with `use_layer_wise` + emerging optimizer — an assert is inserted.

### Why Neuron_SP does not need this change

Neuron_SP does **not** integrate Megatron's `emerging_optimizers` package
(Muon / `TensorParallelMuon`).  Our optimizer stack is:

```
DistributedOptimizer          ← owns all params, byte-level ZeRO-3 sharding
  └─ DeslocFusedAdam          ← single param_group, no LayerWise split
```

There is no `LayerWiseDistributedOptimizer` in our codebase and no
`emerging_optimizer` param-group routing.  The structural precondition that
makes M3998 necessary — a mixed LayerWise+DistOpt setup with separate Muon vs
non-Muon groups — does not exist here.

### What was pre-adapted (API alignment only)

Two pieces of the M3998 API surface were already added to Neuron_SP as
forward-compatibility stubs during earlier sessions:

| Location | Symbol | Status |
|---|---|---|
| `deepspeed/core/optimizer/param_layout.py:114` | `BufferKey.is_managed_by_layer_wise_optimizer: bool = False` | ✅ field present, always `False` (single-buffer design) |
| `deepspeed/core/optimizer/distrib_optimizer.py:1662` | `start_param_sync_for_bucket_group_subset()` | ✅ stub present, delegates to `start_param_sync()` |
| `deepspeed/core/optimizer/distrib_optimizer.py` (step_with_ready_grads docstring) | M3998 evolution note | ✅ documented |

These stubs exist so that callers written against the Megatron M3998 API
continue to compile and run correctly against Neuron_SP without modification.
They do not change runtime behaviour because there is no LayerWise sibling
to coordinate with.

### Conditions that would reopen this task

This skip should be revisited if any of the following occur:

- Neuron_SP adopts Muon / `TensorParallelMuon` for matrix-weight updates
  while keeping Adam for embeddings / biases (mixed-optimizer regime).
- A `LayerWiseDistributedOptimizer` wrapper is introduced for pipeline or
  tensor-parallel overlap.
- `BufferKey.is_managed_by_layer_wise_optimizer` is ever set to `True` by
  any code path — that would mean non-DistOpt buffers exist and the routing
  split must be implemented.

### Verdict

**No optimizer code change required.**  The `BufferKey` field and
`start_param_sync_for_bucket_group_subset` stub already provide the necessary
API compatibility.  Full routing logic is not needed until a Muon integration
is planned.

---

## M4065 — Skip gradient updates when grad norm exceeds threshold
**Upstream commit:** `8fff54f8` (Megatron PR #3460, internal ref `180131620`)
**Audit date:** 2026-06-29
**Status:** ✅ SKIP — fully ported and extended; no further action required

### What the upstream commit does

Megatron adds a guard in `ChainedOptimizer.step()` that compares the
computed global gradient norm against a new config field
`grad_norm_skip_threshold` (default `inf`, i.e. disabled).  If the norm
exceeds the threshold the optimizer step is skipped entirely
(`update_successful = False`), preserving Adam's moment state from
corruption by a runaway gradient spike.  The change touches three files:

| File | Change |
|---|---|
| `optimizer_config.py` | `grad_norm_skip_threshold: float = float('inf')` added after `clip_grad` |
| `optimizer.py` | `should_skip_update` flag set when `grad_norm > threshold`; `step_with_ready_grads()` short-circuited |
| `clip_grads.py` | Whitespace / trailing-newline cleanup only (no logic change) |

### What Neuron_SP has

The feature is **fully present and extended** across four commits:

| Commit | What it does |
|---|---|
| `9caf71e0` | `deepspeed/runtime/hetero_grad_norm_skip.py` (1 343 lines) — DES-LOC reinterpretation as `HeteroGradNormSkipController`: per-device-class FP64 norm accumulation (A6000 vs H100), LOC-cache invalidation on skip, consecutive-skip detection, per-class thresholds |
| `e4c5f49b` | `grad_norm_skip_threshold: float = float('inf')` added to `OptimizerConfig` (the missing config field that would have caused `AttributeError` at runtime) |
| `d580f49a` | `step()` added to `DesLocEngine` to serve as the monkey-patch entry point for `HeteroGradNormSkipController` |
| `ecc5811a` | Engine-level wiring: `integrate_with_deepspeed_engine` plumbs `HeteroGradNormSkipController` into the training loop via `desloc_engine.py` |

Our `ChainedOptimizer.step()` (`optimizer.py:1575`) already has the
upstream skip guard, with one deliberate extension: the condition is
`grad_norm > threshold and main_params` (we gate on `main_params` to
avoid false skips when a stub optimizer with no real parameters is the
sole optimizer in the chain).

### Delta vs upstream

| Upstream Megatron | Neuron_SP |
|---|---|
| Single global `float` threshold | Per-device-class thresholds (`HeteroGradNormConfig`) |
| Skip decision in `ChainedOptimizer.step` | Skip decision delegated to `HeteroGradNormSkipController`, hooked via `DesLocEngine.step()` |
| No LOC coherence concern | `invalidate_loc_cache()` called on skip to flush stale PCIe-in-flight gradient fragments |
| No consecutive-skip tracking | Controller tracks consecutive skips; warns / raises after configurable limit |
| `clip_grads.py` whitespace only | Not ported (no-op) |

### Verdict

**No further action required.**  The upstream feature is fully present
(`grad_norm_skip_threshold` in config, skip guard in `ChainedOptimizer.step`)
and significantly extended with DES-LOC heterogeneous-hardware logic in
`hetero_grad_norm_skip.py`.  The `clip_grads.py` whitespace-only changes
from upstream are intentionally not ported.
