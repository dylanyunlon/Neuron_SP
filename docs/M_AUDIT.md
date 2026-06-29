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
