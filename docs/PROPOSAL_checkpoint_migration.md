# PROPOSAL: Checkpoint Migration — `torch.save` → `dist_checkpointing`

**Author:** InfraLead  
**Status:** RFC — awaiting review  
**Scope:** `deepspeed/runtime/desloc_engine.py` · `deepspeed/runtime/core_adapters.py`  
**Target module:** `deepspeed/core/dist_checkpointing/` (serialization + strategies layer)

---

## 1. Current State — What Gets Saved and Where

`save_checkpoint(path)` (line 2960) builds a single monolithic `payload` dict and
dispatches to one of three code paths. All three ultimately call `torch.save`.

### 1.1 Payload structure

```python
payload = {
    "global_step":     int,            # scalar
    "tokens_seen":     int,            # scalar
    "model_state":     dict,           # model.state_dict() — full parameters + buffers per rank
    "optimizer_state": dict,           # optimizer.state_dict() — Adam m/v, master FP32 weights
    "scheduler_state": dict,           # LR scheduler state
    "plan":            object,         # training plan object
    "config":          EngineConfig,   # dataclass
    "rng_state": {                     # best-effort, may be absent
        "cpu":       Tensor,
        "cuda":      Tensor | None,
        "shard_key": {"pp_rank", "tp_rank", "ep_rank", "ep_size"},
    },
}
```

### 1.2 The four `torch.save` call-sites

| # | Line | Call-site | What it saves | Notes |
|---|------|-----------|---------------|-------|
| **S1** | 3113 | CACHE-tier locality-cache stage (inside `_do_stage`) | Full `payload` → `locality_cache/step_N/hetero_metadata.pt` | Background thread; triggered only when `HeteroCheckpointConfig` is active **and** the currently-disabled hetero branch is re-enabled |
| **S2** | 3162 | WORKER-tier locality-cache stage (inside `_do_worker_stage`) | `{param_shard, global_step}` → `locality_cache/step_N/param_shard.pt` | Background thread; same gating as S1 |
| **S3** | 3241 | Synchronous single-GPU path (`_world == 1`) | Full `payload` → `path` (the caller-supplied `.pt` file) | Hot path for all single-device runs |
| **S4** | 3248 | Synchronous multi-GPU path (`_world > 1`) | Full `payload` → `ckpt_dir/rank_<R>.pt` per rank | Hot path for DP/TP/PP runs; every rank writes its own copy of the **entire** payload, including optimizer state that is largely redundant across DP ranks |

### 1.3 The dead adapter (`_dc_saver`)

Lines 3001–3003 build a `DistCheckpointAdapter` via
`build_dist_checkpoint_saver(config)` when `config.use_dist_checkpointing=True`,
but the returned object is **never called** — `_dc_saver` is assigned and then
falls off scope without routing S3/S4 through it. The adapter exists in
`core_adapters.py` and wraps `dist_checkpointing.save/load`, but the wire was
never completed in `desloc_engine.py`.

### 1.4 `load_checkpoint` counterpart (informational)

Three load paths mirror the save paths:

* **Stage 1** (line 3328): `torch.load(meta_file)` — reads `hetero_metadata.pt` from locality cache.
* **Stage 2** (line 3386): `torch.load(state_dict_meta)` — reads a hetero directory checkpoint via `HeteroAsyncCheckpointLoad`.
* **Stage 3** (line 3416): `torch.load(path)` — legacy `.pt` file fallback.

These must be updated in lockstep with the save-side changes.

---

## 2. Target State — dist_checkpointing API Mapping

### 2.1 Key APIs available in `deepspeed/core/dist_checkpointing/`

```
serialization.save(sharded_state_dict, checkpoint_dir,
                   sharded_strategy=None, common_strategy=None)
serialization.load(sharded_state_dict, checkpoint_dir,
                   sharded_strategy=None, common_strategy=None)
```

The `serialization` layer separates state into two buckets:

* **Common (non-sharded):** plain tensors, scalars, configs — saved once by rank 0 to `common.pt`.
* **Sharded:** `ShardedTensor` instances — each rank saves its own shard to `shard_NNNNN.pt`; resharding on load is handled automatically.

Topology is recorded in `metadata.json` via `CheckpointManifest`
(already integrated in `dist_checkpointing/__init__.py::save`).

The `async_save_checkpoint(save_fn, ...)` helper in `__init__.py` wraps
`save_fn` in a non-daemon `threading.Thread` with lowered nice/ionice priority
(M3407/M3461), which is the correct async primitive for daemon-process
environments like DES-LOC tier coordinators.

### 2.2 Proposed replacement mapping

| Current call-site | What replaces it |
|---|---|
| **S3** — single-GPU `torch.save(payload, path)` | `dist_checkpointing.save(sharded_payload, ckpt_dir)` synchronously; `path` becomes a directory |
| **S4** — per-rank `torch.save(payload, rank_path)` | Same `dist_checkpointing.save(sharded_payload, ckpt_dir)`; the library handles per-rank shard files internally, eliminating the manual `rank_<R>.pt` scheme and redundant optimizer state |
| **S1** — CACHE locality-cache `torch.save` | `async_save_checkpoint(dist_checkpointing.save, staged_sharded_payload, lc_dir)` — uses the thread-based async helper instead of a raw background thread |
| **S2** — WORKER locality-cache `torch.save` | `async_save_checkpoint(dist_checkpointing.save, worker_sharded_payload, lc_dir)` — same |
| `torch.load(meta_file)` (Stage 1 load) | `dist_checkpointing.load(sharded_state_dict, lc_dir)` |
| `torch.load(path)` (Stage 3 load) | `dist_checkpointing.load(sharded_state_dict, ckpt_dir)` with `check_is_distributed_checkpoint` guard; fall back to `torch.load` for legacy `.pt` files |

### 2.3 ShardedTensor wrapping strategy

`dist_checkpointing.save` needs a `ShardedStateDict` — a dict where model
parameter tensors are wrapped in `ShardedTensor`. The mapping layer already
exists in `dist_checkpointing/optimizer.py` (`make_sharded_optimizer_tensor`,
`optim_state_to_sharding_state`). For the initial migration we propose:

```
model_state      → wrapped via model.sharded_state_dict() if available,
                   else ShardedTensor(key=name, data=param, global_shape=param.shape,
                   global_offset=(0,...), axis_fragmentations=(1,...)) per param
                   (replica_id = dp_rank so only rank-0 of each DP group persists)

optimizer_state  → wrapped via optim_state_to_sharding_state(optim.state_dict(),
                   id_to_sharded_param_map) from dist_checkpointing/optimizer.py

scheduler_state  → passed through as common (non-sharded), rank 0 only
global_step,
tokens_seen,
plan, config,
rng_state        → common payload, rank 0 only (same semantics as current S4)
```

This eliminates the redundancy in S4 where every DP rank writes a full copy of
the optimizer state. After migration each DP shard writes only its own
`exp_avg` / `exp_avg_sq` slice.

---

## 3. Migration Steps

All steps are ordered to preserve backward compatibility — existing `.pt`
checkpoints must remain loadable throughout.

### Step 1 — Complete the `_dc_saver` wire-up (S3 + S4) behind the feature flag

**File:** `deepspeed/runtime/desloc_engine.py`  
**Scope:** synchronous save paths only (S3 and S4 — lines 3240–3253)

Replace the dead `_dc_saver` assignment with an actual dispatch:

```python
if _dc_saver is not None:
    # dist_checkpointing path: build sharded payload and delegate
    sharded_payload = _build_sharded_payload(payload, self)
    _dc_saver.save(sharded_payload, path)
    logger.info("dist_checkpoint saved: %s (step %d)", path, self.global_step)
else:
    # legacy torch.save path (S3 / S4)
    ...existing code...
```

`_build_sharded_payload` is a new private helper (see Step 2). The feature
flag `config.use_dist_checkpointing` gates the new path; the old path remains
fully functional.

`load_checkpoint` Stage 3 gets a symmetric change: if `path` is a directory
and `check_is_distributed_checkpoint(path)` returns True, use
`_dc_saver.load(path)`, otherwise fall through to `torch.load`.

**Gating:** `config.use_dist_checkpointing = True` (already defined, line 344).

---

### Step 2 — Implement `_build_sharded_payload` helper

**File:** `deepspeed/runtime/desloc_engine.py` (private method) or a new
`deepspeed/runtime/checkpoint_utils.py` module.

```python
def _build_sharded_payload(payload: dict, engine) -> ShardedStateDict:
    """Wrap model + optimizer tensors in ShardedTensor; pass scalars as-is."""
    from deepspeed.core.dist_checkpointing.mapping import ShardedTensor
    from deepspeed.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map, optim_state_to_sharding_state,
    )
    dp_rank = ...  # from parallel_state
    result = {}
    # Common (non-sharded) keys
    for k in ("global_step", "tokens_seen", "scheduler_state",
              "plan", "config", "rng_state"):
        if k in payload:
            result[k] = payload[k]
    # Model params: wrap each tensor in a ShardedTensor
    # (replica_id = dp_rank → only rank 0 of each DP group persists)
    model_sd = payload["model_state"]
    result["model_state"] = _wrap_model_state_dict(model_sd, dp_rank)
    # Optimizer state: use the existing dist_checkpointing/optimizer.py helpers
    id_map = get_param_id_to_sharded_param_map(
        result["model_state"], engine.model.parameters()
    )
    result["optimizer_state"] = optim_state_to_sharding_state(
        payload["optimizer_state"], id_map
    )
    return result
```

This step is prerequisite for Step 1 but can be reviewed independently.

---

### Step 3 — Update locality-cache staging (S1 + S2) to use `async_save_checkpoint`

**File:** `deepspeed/runtime/desloc_engine.py`  
**Scope:** `_do_stage` (line ~3110) and `_do_worker_stage` (line ~3150)  
**Gate:** only relevant once the `if cfg is not None and False:` guard (line 3055)
is re-enabled, which is a separate PR.

Replace raw `torch.save(payload_, meta_file)` inside `_do_stage` with
`dist_checkpointing.save(sharded_payload_, lc_path)`. Replace the manual
`self._cpu_stage_executor.submit(_do_stage)` with
`async_save_checkpoint(dist_checkpointing.save, ...)` so the thread management
is handled by the vetted M3407 helper rather than a bespoke executor.

This step depends on Step 1 being merged and stable.

---

### Step 4 — Update `load_checkpoint` (Stages 1 and 2)

**File:** `deepspeed/runtime/desloc_engine.py`  
**Scope:** `load_checkpoint`, Stages 1 and 2 (lines ~3328 and ~3386)

After Step 1 and 2 are merged:

* Stage 1 (locality-cache load): `torch.load(meta_file)` → `dist_checkpointing.load(expected_sharded_sd, lc_dir)`.
* Stage 2 (hetero async load): integrate `dist_checkpointing.load` as an
  alternative path alongside `HeteroAsyncCheckpointLoad`, controlled by
  `config.use_dist_checkpointing`.

`_apply_loaded_state` (called after load in both stages) does not change.

---

### Step 5 — Deprecation notice for `rank_<R>.pt` format

After Step 1 has been live for two release cycles, add a `DeprecationWarning`
in `load_checkpoint` Stage 3 when loading from a `rank_*.pt` directory,
directing users to re-save with `use_dist_checkpointing=True`. No removal yet.

---

## 4. Backward Compatibility Contract

| Scenario | Behaviour after migration |
|---|---|
| `use_dist_checkpointing=False` (default) | No change — all existing `torch.save` paths are untouched |
| Load of a legacy `.pt` file | `check_is_distributed_checkpoint` returns `False`; falls through to `torch.load` in Stage 3 |
| Load of a legacy `rank_*.pt` directory | Detected by absence of `metadata.json`; falls through to existing rank-file reconstruction in Stage 3 |
| Load of a new `dist_checkpointing` directory | `check_is_distributed_checkpoint` returns `True`; uses `dist_checkpointing.load` |
| Topology change on resume (TP/PP resize) | Handled natively by `dist_checkpointing.load` resharding; resharding of legacy formats is not supported and was not supported before |

---

## 5. Risk Points

### R1 — `_dc_saver` is built but the wire was never completed (confirmed bug)
The adapter in `core_adapters.py` is correct and tested in isolation, but
`desloc_engine.py` never calls `_dc_saver.save(...)`. Step 1 is purely a
wire-up — the adapter itself does not change. Low risk.

### R2 — `dist_checkpointing.save` requires an empty, pre-existing directory
The current `torch.save` paths create intermediate directories via
`mkdir(parents=True, exist_ok=True)` and write a single file. The
`serialization.save` entrypoint (line 110–115) **raises** if the directory is
non-empty. The engine must create the directory and verify it is empty before
calling `save`, or pass `exist_ok` semantics via a wrapper.
**Mitigation:** add a `_prepare_dist_ckpt_dir(path)` helper that creates the
directory or clears stale contents from the same step; validate in CI with a
repeated-save test.

### R3 — `ShardedTensor` wrapping requires TP/PP topology metadata
`_build_sharded_payload` needs to know `global_shape` and `axis_fragmentations`
for each model parameter. On single-GPU runs this is trivial (all ones). On
multi-GPU runs, parameters are already sharded locally; `global_shape` must be
reconstructed from `parallel_state`. If the model does not implement
`sharded_state_dict()`, a conservative fallback wraps each local param as a
replica shard (`axis_fragmentations=(1,...)`, `replica_id=dp_rank`), which is
correct but wastes storage on DP ranks > 0 (same as the current behaviour).
**Mitigation:** add a `model.sharded_state_dict()` interface check; only
promote to true sharding when the interface is available.

### R4 — Hetero async path has a known deadlock (line 3055 guard)
The `if cfg is not None and False:` guard in `save_checkpoint` disables the
entire hetero async branch because DCP's `dcp.save` calls `gather_object` →
`allgather` in a background thread while the main thread holds `all_reduce`
during gradient synchronisation. Steps 1–4 of this migration do **not** touch
the disabled branch. Step 3 (re-enabling with `async_save_checkpoint`) is
explicitly deferred and requires resolving this deadlock first (separate issue).

### R5 — Locality-cache `hetero_metadata.pt` consumed by `torch.load` in Stage 1
After Step 4, Stage 1 of `load_checkpoint` switches to `dist_checkpointing.load`.
Any locality-cache files written by the **old** `torch.save` path (S1/S2) will
not be recognisable as distributed checkpoints (`check_is_distributed_checkpoint`
will return `False`). The fallback to `torch.load` handles this gracefully, but
mixed-format caches (one step written with old code, next with new) could cause
confusion.
**Mitigation:** add a `format_version` field to the locality-cache directory
name or a sentinel file, and clear the cache on format change.

### R6 — `optimizer.state_dict()` size at checkpoint time
Currently S4 writes a full optimizer state to every rank. After migration, only
the primary DP replica writes its shard. This is a correctness improvement but
changes the checkpoint directory size expectation significantly. Operators
monitoring storage quotas need to be aware.

---

## 6. Files Changed (projected)

| File | Change |
|---|---|
| `deepspeed/runtime/desloc_engine.py` | Wire up `_dc_saver` dispatch in S3/S4 (Step 1); add `_build_sharded_payload` (Step 2); update locality-cache saves (Step 3); update load Stages 1/2 (Step 4) |
| `deepspeed/runtime/core_adapters.py` | No change needed for Steps 1–2; may add `.load_sharded` method to `DistCheckpointAdapter` in Step 4 |
| `deepspeed/runtime/checkpoint_utils.py` | New file: `_build_sharded_payload`, `_prepare_dist_ckpt_dir`, `_wrap_model_state_dict` (Step 2) |
| `tests/unit/runtime/test_checkpoint_migration.py` | New: round-trip tests with `use_dist_checkpointing=True`, legacy-format load fallback, topology-change load |

**`desloc_engine.py` is not modified in this PR.** This document is the review
gate before any code changes proceed.

---

*This proposal does not modify any source file. It is a pre-change architecture
analysis per NVIDIA PR process.*
