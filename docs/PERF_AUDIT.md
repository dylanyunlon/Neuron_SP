# PERF Commit Audit — GPU Tuning Code on Live Training Path

**Date:** 2026-06-29  
**Auditor:** automated grep + call-graph trace  
**Entry points traced:** `run_pretrain.py` → `DesLocEngine.__init__` + `DesLocEngine.train()`  
**Files audited:** `desloc_engine.py`, `zero3_hetero_shard.py`, `finalize_model_grads.py`, `distributed_data_parallel.py`, `param_and_grad_buffer.py`

---

## Summary Table

| Commit | Description | Files Changed | On Training Path? | Notes |
|--------|-------------|---------------|-------------------|-------|
| `c37ea95f` | 2497→9 NCCL calls — bucketed sync + local SGD | `zero3_hetero_shard.py` | ⚠️ **SUPERSEDED** | `ShardState._broadcast_model_params` is dead code; `ShardState` was replaced by `DistributedOptimizer` at `e48a9d8c` |
| `d89a62ba` | Fuse grad AllReduce for PCIe topology | `finalize_model_grads.py` | ✅ **LIVE** | `finalize_model_grads()` called every step; fused embedding path on critical path |
| `633ff4c4` | BF16 grad comm cast | `distributed_data_parallel.py` + `param_and_grad_buffer.py` | ⚠️ **NOT ACTIVATED** | Cast logic present but `megatron_fsdp_grad_comm_dtype` never set in `CoreDDPConfig` → branch is always False |
| `741d00d1` | Delay grad norm all_reduce to end of accumulation window | `desloc_engine.py` | ⚠️ **SUBSUMED** | `_is_last_micro` guard and `param_shard.grad` norm-sq branch removed by `e48a9d8c`; intent preserved by once-per-step `clip_grad_norm_` |
| `db3a925c` | Async `sync_shard_to_model` with CUDA stream overlap | `desloc_engine.py` + `zero3_hetero_shard.py` | ✅ **LIVE (upgraded)** | Mechanism fully preserved — secondary stream, fence, drain; stream now from `StreamManager` |

---

## Detailed Findings

### `c37ea95f` — Bucketed sync + local SGD (2497→9 NCCL calls)

**Files changed:** `deepspeed/runtime/zero3_hetero_shard.py`

**What it did:**
1. Removed per-param `all_reduce` in `ShardState` backward hooks (was 1816 NCCL/step) → local SGD.
2. Replaced 681 per-param `broadcast` calls in `_broadcast_model_params` with bucketed `all_reduce(SUM)` over 2 GB BF16 flat buffers (~7 calls/step for 7B params).

**Import chain:**
```
run_pretrain.py
  → DesLocEngine.__init__
      → zero3_hetero_shard.ZeRO3ForwardHook  (line 2209, lazy import — only for model→GPU move)
      # ShardState.build() is NOT called; _dist_optimizer = DistributedOptimizer (e48a9d8c)
```

**Verdict:** `ShardState.sync_shard_to_model()` and `_broadcast_model_params()` are not called anywhere in the current engine. Commit `e48a9d8c` (2026-06-27) replaced `ShardState` with `DistributedOptimizer`, which performs its own all-gather via `shard_to_model_broadcast()` in `distrib_optimizer.py`. `ZeRO3ForwardHook` is still imported and used, but only to move the full BF16 model to GPU once at startup — unrelated to the NCCL reduction logic this commit changed.

**The 2497→9 NCCL improvement is not active via this code path.**

---

### `d89a62ba` — Fuse grad AllReduce (M4149 pattern)

**Files changed:** `deepspeed/core/distributed/finalize_model_grads.py`

**What it did:**
1. Added `fuse_grad_reductions(tensors, pg)` — flattens N same-dtype grad tensors into one buffer, fires a single `all_reduce`, copies back.
2. Added `_allreduce_all_embedding_grads()` — fuses word-embedding + position-embedding grad reductions into one collective when they share a process group (2 calls → 1).
3. Replaced step 5 of `finalize_model_grads()` with the fused entry point.

**Import chain:**
```
desloc_engine.py line 57 (top-level, always imported):
    from deepspeed.core.distributed import finalize_model_grads

desloc_engine.train() line 2562 (every step):
    finalize_model_grads(
        model=_fmg_model, config=ModelParallelConfig(),
        skip_grad_sync=not _is_Kx_sync,
        force_all_reduce=self._dist_optimizer is not None,
    )

finalize_model_grads.py line 1204 (step 5, unconditional):
    _allreduce_all_embedding_grads(model, config, embd_group, pos_emb_group, pp_group)
      └─ fuse_grad_reductions(...)   when both embedding groups are the same
```

**Verdict:** ✅ **Fully live.** Both `fuse_grad_reductions` and `_allreduce_all_embedding_grads` are on the hot path every training step. The PCIe launch-overhead reduction for embedding grads is active.

---

### `633ff4c4` — BF16 grad comm cast (M3574)

**Files changed:** `deepspeed/core/distributed/distributed_data_parallel.py`, `deepspeed/core/distributed/param_and_grad_buffer.py`

**What it did:**
1. Added `megatron_fsdp_grad_comm_dtype: Optional[torch.dtype]` field to `DistributedDataParallelConfig`.
2. In `ParamAndGradBucketGroup.finish_grad_sync()`: reads `self.ddp_config.megatron_fsdp_grad_comm_dtype`; if set and mismatched, casts `grad_data` to comm dtype before `reduce_scatter`/`all_reduce`, then casts back.

**Import chain:**
```
desloc_engine.py line 55-56:
    from deepspeed.core.distributed import DistributedDataParallel as CoreDDP, \
                                           DistributedDataParallelConfig as CoreDDPConfig

desloc_engine.__init__ line 1935 — CoreDDPConfig construction:
    _ddp_cfg = CoreDDPConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        allow_skip_grad_sync=True,
        # megatron_fsdp_grad_comm_dtype NOT SET → defaults to None
    )
```

**Gate in param_and_grad_buffer.py:**
```python
_grad_comm_dtype = getattr(getattr(self, 'ddp_config', None), 'megatron_fsdp_grad_comm_dtype', None)
if _grad_comm_dtype is not None and bucket.grad_data.dtype != _grad_comm_dtype:
    # cast → always False when _grad_comm_dtype is None
```

**Additional blocker:** `CoreDDP` is only instantiated on the non-ZeRO-3 fallback path. When `DistributedOptimizer` is active (primary production path), `self._core_ddp` stays `None` and `CoreDDP.finish_grad_sync()` is never called regardless.

**Verdict:** ⚠️ **Code present, not activated.** Two independent blockers: (1) `megatron_fsdp_grad_comm_dtype` unset → cast always skipped; (2) `CoreDDP` not used on ZeRO-3 path. BF16 bandwidth saving is not realised.

**Fix:** Set `megatron_fsdp_grad_comm_dtype=torch.bfloat16` in `CoreDDPConfig` at `desloc_engine.py` ~L1935. Also evaluate whether `DistributedOptimizer`'s reduce-scatter path needs an equivalent cast.

---

### `741d00d1` — Delay grad norm all_reduce to end of accumulation window

**Files changed:** `deepspeed/runtime/desloc_engine.py`

**What it did:** Moved the async `param_shard.grad` norm-sq `all_reduce` from after *every* microbatch to after only the **last** microbatch (`_is_last_micro = (micro == num_microbatches - 1)`). Also removed dummy `all_reduce` padding from the heterogeneous microbatch-balancing loop, which had matched collective counts for the now-eliminated per-microbatch reductions.

**Current HEAD check:**
```bash
$ grep -n "_is_last_micro\|_micro_norm_sq\|_async_norm_tensor\|param_shard\.grad" desloc_engine.py
# → no output
```

The entire `param_shard_state.param_shard.grad` norm-sq branch was removed when `e48a9d8c` replaced `ShardState`. `DistributedOptimizer` does not expose a `param_shard.grad` tensor. Grad norm is now computed once per step at:
```python
# desloc_engine.py line 2578
gnorm = clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
```
This is inherently once-per-step; the per-microbatch overhead the commit eliminated no longer exists by any mechanism.

**Verdict:** ⚠️ **Logic subsumed.** The optimization intent (one norm collective per step) is achieved — but because the `ShardState` path that triggered N reductions was removed entirely, not because the `_is_last_micro` guard is active. The commit is a no-op in the current codebase; its correctness goal is preserved by a different mechanism.

---

### `db3a925c` — Async `sync_shard_to_model` with CUDA stream overlap

**Files changed:** `deepspeed/runtime/desloc_engine.py`, `deepspeed/runtime/zero3_hetero_shard.py`

**What it did:**
1. Added `ShardState.sync_shard_to_model_async(stream)` — issues `p.data.copy_(..., non_blocking=True)` on a secondary CUDA stream.
2. In `desloc_engine.train()`: allocated `_shard_sync_stream = torch.cuda.Stream()`, launched the async copy after `optimizer.step()`, waited before next forward, drained at loop exit.

**Current HEAD — mechanism is live (upgraded implementation):**

```python
# desloc_engine.py line 2331 — stream from StreamManager (I3 / M3724 upgrade)
_shard_sync_stream = StreamManager.get_shard_sync_stream(_shard_sync_gpu_type)
_shard_sync_pending = False

# After optimizer.step() (line 2625):
_shard_sync_stream.wait_stream(torch.cuda.current_stream())   # M3561 fence
with torch.cuda.stream(_shard_sync_stream):
    self._dist_optimizer.shard_to_model_broadcast()           # FP32→BF16 all-gather
_shard_sync_pending = True

# Before first forward of next step (line 2458):
if _shard_sync_pending and micro == 0:
    torch.cuda.current_stream().wait_stream(_shard_sync_stream)
    _shard_sync_pending = False

# At training loop exit (line 2774):
if _shard_sync_pending and _shard_sync_stream is not None:
    torch.cuda.current_stream().wait_stream(_shard_sync_stream)
```

Changes vs `db3a925c` original:
- Stream from `StreamManager.get_shard_sync_stream()` (framework-level pool) instead of ad-hoc `torch.cuda.Stream()`.
- Broadcast via `DistributedOptimizer.shard_to_model_broadcast()` instead of `ShardState.sync_shard_to_model_async()`.
- Added `_shard_sync_stream.wait_stream(current_stream)` fence before launch (Megatron M3561 — prevents stale-weight corruption when Adam writes lag behind the secondary stream).

Note: `ShardState.sync_shard_to_model_async()` at `zero3_hetero_shard.py:730` is now dead code (no callers in current engine).

**Verdict:** ✅ **Fully live.** The async stream-overlap pattern from `db3a925c` is active on every step of the production training loop, with correctness improvements.

---

## Live Path Diagram

```
run_pretrain.py  (--use-desloc flag)
  └─ run_desloc()
       └─ DesLocEngine(config, model, data_iter)
            ├─ __init__
            │    ├─ core.optimizer.DistributedOptimizer   [ZeRO-3 shard owner]
            │    ├─ ZeRO3ForwardHook (zero3_hetero_shard) — model→GPU once
            │    └─ CoreDDP (non-ZeRO-3 fallback only; inactive with DistributedOptimizer)
            └─ train()
                 ├─ per step:
                 │    ├─ wait_stream(_shard_sync_stream)        ← db3a925c ✅
                 │    ├─ forward / backward × num_microbatches
                 │    ├─ finalize_model_grads()                 ← d89a62ba ✅
                 │    │    └─ _allreduce_all_embedding_grads
                 │    │         └─ fuse_grad_reductions
                 │    ├─ clip_grad_norm_  (once/step, inherently)
                 │    ├─ optimizer.step()
                 │    └─ stream(_shard_sync_stream):
                 │         shard_to_model_broadcast()           ← db3a925c ✅
                 └─ drain _shard_sync_pending at exit
```

---

## Action Items

| Priority | Item |
|----------|------|
| 🔴 **High** | **`633ff4c4` not activated** — set `megatron_fsdp_grad_comm_dtype=torch.bfloat16` in `CoreDDPConfig` (`desloc_engine.py` ~L1935). Also verify whether `DistributedOptimizer`'s reduce-scatter path needs an equivalent BF16 cast. |
| 🟡 **Medium** | **`c37ea95f` bucketed broadcast dead** — `ShardState._broadcast_model_params` no longer called. If the 7-bucket NCCL strategy is still desired, port bucketing logic into `DistributedOptimizer.shard_to_model_broadcast()` or its reduce-scatter path. |
| 🟡 **Medium** | **`741d00d1` subsumed** — no action needed for correctness; note as subsumed by `e48a9d8c`. |
| 🟢 **Low** | **`db3a925c` `sync_shard_to_model_async`** (`zero3_hetero_shard.py:730`) is dead code with no callers. Add `# deprecated — use DistributedOptimizer.shard_to_model_broadcast()` or remove. |
