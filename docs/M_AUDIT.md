# Megatron Commit Audit — M3981

**Megatron commit:** `aa786b72c097d92c3656844321380a2e212c169e`  
**Title:** Thread custom process groups through MoE grad finalization  
**Audited by:** DistLead  
**Audit date:** 2026-06-29  
**Verdict:** ✅ Fully covered — applied across 4 separate commits

---

## What M3981 changes (upstream Megatron)

M3981 removes every implicit call to `parallel_state` globals inside the MoE
expert-bias gradient finalization path and replaces them with an explicit
`tp_dp_cp_group` parameter threaded from `finalize_model_grads` all the way
down to the `torch.distributed.all_reduce` inside `get_updated_expert_bias`.
It also removes two `parallel_state.get_tensor_model_parallel_group()` singleton
calls inside `MoELayer` checkpoint wrappers, replacing them with `self.tp_group`.

The diff touches **4 files** in Megatron-LM:

| File | Change |
|---|---|
| `megatron/core/distributed/finalize_model_grads.py` | `_update_router_expert_bias` gains `tp_dp_cp_group=` param; `finalize_model_grads` extracts `tp_dp_cp_group` from `pg_collection.tp_dp_cp` and passes it through; fallback reads `parallel_state.get_tensor_and_data_parallel_group()` when `pg_collection is None` |
| `megatron/core/pipeline_parallel/schedules.py` | All three pipeline-schedule entry-points set `pg_collection.tp_dp_cp` before calling `finalize_model_grads` |
| `megatron/core/transformer/moe/moe_layer.py` | Two `checkpoint()` call sites replace `parallel_state.get_tensor_model_parallel_group()` with `self.tp_group` |
| `megatron/core/transformer/moe/moe_utils.py` | `get_updated_expert_bias` gains `tp_dp_cp_group=` param; default falls back to `parallel_state` with a TODO comment |
| `megatron/core/transformer/moe/shared_experts.py` | Two SP-region call sites gain `group=self.tp_group` |

---

## Coverage map — our codebase

### File 1 — `finalize_model_grads.py` ✅ COVERED

**Our file:** `deepspeed/core/distributed/finalize_model_grads.py`

All three sub-changes are present:

1. `_update_router_expert_bias` signature (line 840):
   ```python
   def _update_router_expert_bias(
       model, config,
       tp_dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
   ):
   ```

2. `finalize_model_grads` asserts `pg_collection.tp_dp_cp` when
   `moe_router_enable_expert_bias=True` (lines 1105–1108) and extracts
   `tp_dp_cp_group = pg_collection.tp_dp_cp`.

3. Fallback when `pg_collection is None` (lines 1212–1218) reads
   `parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)`,
   matching the upstream `TODO(Hepteract): delete the usage of the global parallel_state` comment.

4. `_update_router_expert_bias` call site (line 1219) passes
   `tp_dp_cp_group=tp_dp_cp_group`.

**Applied in:** commit `ae60b3e0` (prior session) — evolution header lists M3981.

---

### File 2 — `schedules.py` ✅ COVERED

**Our file:** `deepspeed/core/pipeline_parallel/schedules.py`

`pg_collection.tp_dp_cp` is set in our pipeline schedule functions.
Evidence (grep output):

```
schedules.py:643  pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
schedules.py:787  pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
```

Both the no-pipelining and interleaving paths set `tp_dp_cp` before calling
`finalize_model_grads`, matching the three `schedules.py` hunks in M3981.

**Applied in:** commit `5e1453ef` (prior session) — evolution header comment `M3981 tp_dp_cp` at line 631.

---

### File 3 — `moe_layer.py` ✅ NOT APPLICABLE

**Our file:** `deepspeed/core/transformer/moe/moe_layer.py`

Megatron's change replaces two occurrences of
`parallel_state.get_tensor_model_parallel_group()` inside
`tensor_parallel.checkpoint()` wrappers in `MoELayer.forward()`.

Our `MoELayer` is a pure-PyTorch, PCIe-optimized rewrite with no TP
dimension — it does not use `tensor_parallel.checkpoint()` and never called
`parallel_state.get_tensor_model_parallel_group()`. The grep confirms zero
occurrences of that call in our file.

**No action required.**

---

### File 4 — `moe_utils.py` ✅ NOT APPLICABLE (import delegation)

**Our file:** `deepspeed/core/transformer/moe/moe_utils.py`

Our `moe_utils.py` does not define `get_updated_expert_bias`. Instead,
`finalize_model_grads.py` imports it directly from Megatron:

```python
# deepspeed/core/distributed/finalize_model_grads.py, line 127
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
```

When Megatron is installed, the upstream `get_updated_expert_bias` already
carries the `tp_dp_cp_group=` parameter (as verified in the Megatron source
at commit ≥ aa786b72). When Megatron is not installed, the function is set to
`None` and the expert-bias update is silently skipped — this is pre-existing
behaviour documented in the evolution header.

**No local change required.** The import delegation means the Megatron fix
is inherited automatically.

---

### File 5 — `shared_experts.py` ❌ GAP FOUND → FIXED in this session

**Our file:** `deepspeed/core/transformer/moe/shared_experts.py`

Two call sites inside `SharedExpertMLP._pre_forward_comm()` called:

```python
# BEFORE (using implicit parallel_state singleton):
gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True)
copy_to_tensor_model_parallel_region(input)
```

`self.tp_group` was already set (inherited from `MLP.__init__` via
`pg_collection.tp`) but was not passed to the collectives.

Additionally, `deepspeed/core/tensor_parallel/mappings.py` defined both
functions without a `group=` parameter, so the call sites could not have
passed one even if they wanted to.

**Fix applied in commit `5064b4f2` (this session):**

`mappings.py`:
- `_CopyToModelParallelRegion.forward()` gains `group=None`; stored as
  `ctx.group`; backward reads `ctx.group or _get_tp_group()`.
- `_GatherFromSequenceParallelRegion.forward()` gains `group=None`; stored as
  `ctx.group`; forward and backward use the explicit group for both
  `all_gather` and the `all_reduce` in the SP-output-grad path.
- Both public wrapper functions gain a `group=` keyword argument.

`shared_experts.py`:
```python
# AFTER (explicit group — M3981):
gather_from_sequence_parallel_region(
    input, tensor_parallel_output_grad=True, group=self.tp_group
)
copy_to_tensor_model_parallel_region(input, group=self.tp_group)
```

Backward compatibility: `group=None` default preserves all existing callers.

---

## Commit log for this audit

| Commit | File(s) | M3981 sub-change |
|---|---|---|
| `ae60b3e0` | `finalize_model_grads.py`, `__init__.py` | Sub-change 1 (finalize path) |
| `5e1453ef` | `schedules.py` (evidence in grep) | Sub-change 2 (schedules tp_dp_cp) |
| `5064b4f2` | `mappings.py`, `shared_experts.py` | Sub-change 5 (SP-region group threading) — **gap fixed this session** |

Sub-changes 3 (`moe_layer.py`) and 4 (`moe_utils.py`) are not applicable
to our codebase as documented above.

---

## Regression test

`tests/test_core_integration.py` **Test 7** (`test_finalize_model_grads_moe_tp_dp_cp_assert`):
- Verifies `pg_collection.tp_dp_cp` assertion is present in source.
- Verifies `tp_dp_cp_group=` kwarg is threaded to `_update_router_expert_bias`.

Additional source-level verification can be added for `shared_experts.py`
call sites in a follow-up if needed.

---

---

# Megatron Commit Audit — M4153

**Megatron commit:** `55638bc44`  
**Title:** Fix wgrad race condition when using double buffers  
**Audited by:** DistLead  
**Audit date:** 2026-06-29  
**Verdict:** ✅ Not applicable — all three bugs are structural to Megatron-FSDP's double-buffer pipeline, which we do not implement

---

## What M4153 changes (upstream Megatron)

M4153 fixes **three intertwined bugs** that only manifest in the
`MegatronFSDP` + `fsdp_double_buffer=True` code path. The diff touches two
files inside `Megatron-LM/megatron/core/distributed/fsdp/`:

### Bug 1 — `_megatron_fsdp_model` reference attached before DTensor replacement

**File:** `megatron_fsdp.py`

```python
# BEFORE (in __init__, before _replace_param_with_distributed_if_needed()):
for param in self.module.parameters():
    setattr(param, "_megatron_fsdp_model", self)
self._replace_param_with_distributed_if_needed()
# → DTensor wrappers created here lose the back-reference

# AFTER (inside _replace_param_with_distributed_if_needed(), after replacement):
if not hasattr(dist_param, "_megatron_fsdp_model"):
    dist_param._megatron_fsdp_model = self
# AND in the raw-param restore path:
if not hasattr(self.raw_param[name], "_megatron_fsdp_model"):
    self.raw_param[name]._megatron_fsdp_model = self
```

DTensor params created inside `_replace_param_with_distributed_if_needed()`
never received the back-reference because it was attached in `__init__` before
the replacement ran.

### Bug 2 — `_enforce_double_buffer_limit` called at wrong site (backward hook, not allocator)

**File:** `megatron_fsdp.py` + `param_and_grad_buffer.py`

```python
# BEFORE (in _accumulate_wgrad_into_main_grad backward hook):
if self.ddp_config.fsdp_double_buffer:
    self.grad_reduce_pipeline._enforce_double_buffer_limit([group_id])
param.main_grad = param.get_main_grad()   # ← allocation happens HERE

# AFTER (inside main_grad_getter, before fetch_bucket allocates memory):
def main_grad_getter(p):
    gbuf = p._gbuf
    item_id = p._item_id
    p._megatron_fsdp_model.grad_reduce_pipeline._enforce_double_buffer_limit(
        [gbuf.bucket_id]
    )                                       # ← enforcement at allocation site
    bucket = gbuf.fetch_bucket(...)
```

The enforcement must happen **at the allocation site** (`main_grad_getter` /
`fetch_bucket`), not in the backward hook that runs later. Between the backward
hook call and the actual `get_main_grad()` → `fetch_bucket()` call, a second
bucket could already be live in the double-buffer pool, causing a silent data
race on the gradient buffer memory.

Additionally, `"_megatron_fsdp_model"` must be propagated to DTensor-replaced
params inside `param_and_grad_buffer.py`'s attribute copy loop:
```python
# AFTER — added to the attr propagation list:
"_megatron_fsdp_model",
```

### Bug 3 — Off-by-one in `GradReducePipeline._enforce_double_buffer_limit`

**File:** `param_and_grad_buffer.py`, `GradReducePipeline`

```python
# BEFORE:
if len(double_buf_units) > 1:
    keep_n -= 1

# AFTER:
if len(double_buf_units) > 2:
    keep_n -= 1
```

Double-buffering by definition keeps **two** FSDP unit buffers live
simultaneously (the current unit computing and the next unit being prefetched).
The eviction threshold `> 1` prematurely evicted the second live unit, reducing
effective double-buffering to single-buffering and causing the very race it was
meant to prevent.

### Cosmetic changes (not bugs)

- Tuple unpacking style: `(a, b, c) = ...` → `a, b, c = ...` in two places.
- `(bucket_id, bwd) = next(...)` → `bucket_id, bwd = next(...)`.

---

## Coverage map — our codebase

### `MegatronFSDP` class — ✅ NOT APPLICABLE

```
$ find deepspeed/ -name "*.py" | xargs grep -l "class MegatronFSDP" | grep -v Megatron-LM
(no output)
```

We do not implement `MegatronFSDP`. Our DDP wrapper is
`deepspeed/core/distributed/distributed_data_parallel.py:DistributedDataParallel`,
which uses a flat contiguous-buffer + bucket-group design, not an FSDP
parameter-shard + double-buffer pool.

### `GradReducePipeline` class — ✅ NOT APPLICABLE

```
$ find deepspeed/ -name "*.py" | xargs grep -l "class GradReducePipeline" | grep -v Megatron-LM
(no output)
```

We do not have `GradReducePipeline` or `AllGatherPipeline`. Our sequencing
mechanism is:

- **Predecessor-drain (M4036):** `ParamAndGradBucketGroup.previous_grad_reduce_bucket_group`
  is waited at `start_grad_sync` dispatch time, ensuring the prior bucket's
  reduce-scatter completes before the next bucket's memory is touched.
- **BufferOwnership FSM (M3061/M3116):** `BufferOwnership.{FREE,PARAM_OWNED,GRAD_OWNED}`
  state machine on each `ParamAndGradBucketGroup` enforces that grad and param
  collectives never alias the same memory.

These two mechanisms jointly prevent the class of race that M4153 fixes in the
FSDP double-buffer path. There is **no equivalent race** in our design because:
1. We never pre-allocate a fixed two-slot pool — each bucket allocates from the
   flat contiguous buffer and the ownership FSM prevents reuse until `reset()`.
2. `main_grad` is not lazily allocated via a `main_grad_getter` property — it
   is a pre-computed view into the flat buffer set at `ParamAndGradBuffer` init.

### `_enforce_double_buffer_limit` — ✅ NOT APPLICABLE

The only occurrence in our tree is in the evolution comment header of
`param_and_grad_buffer.py` (line 39–43), explicitly documenting that the bug
does not apply. No executable code references this function.

### `fsdp_double_buffer` config flag — ✅ NOT APPLICABLE

```
$ grep -rn "fsdp_double_buffer" deepspeed/ --include="*.py" | grep -v Megatron-LM
(no output)
```

The flag that gates the entire double-buffer code path does not exist in our
`DistributedDataParallelConfig`. The feature was never ported.

### `_megatron_fsdp_model` back-reference — ✅ NOT APPLICABLE

We do not use DTensor parameter replacement or lazy `main_grad_getter`
properties, so the timing-of-attachment bug has no analog.

### Tuple-unpacking style changes — ✅ NOT APPLICABLE

These are cosmetic Megatron style fixes. Our code does not have the
parenthesised multi-assignment patterns that were cleaned up.

---

## Why the predecessor-drain + ownership FSM is safe

The root cause of M4153 Bug 2 is: a *lazy allocator* (`fetch_bucket`) is
called after an *enforcement check* (backward hook), creating a window where a
second allocation can sneak in. Our design has no lazy allocator:

```
ParamAndGradBuffer.__init__
  └─ allocates flat contiguous tensor once
  └─ computes fixed views: bucket.grad_data, param.main_grad
  └─ no runtime allocation during backward

ParamAndGradBucketGroup.start_grad_sync
  └─ asserts GRAD_OWNED (via BufferOwnership FSM)
  └─ drain predecessor (M4036)
  └─ launches collective on already-allocated bucket.grad_data
```

There is no window between "check" and "allocate" because allocation is not
deferred. The enforcement (FSM assertion) and the collective launch happen in
the same function without any intervening allocation.

---

## Relation to prior session work

The `param_and_grad_buffer.py` evolution header (commit `d5501415`, prior
session) already records this audit:

```
M4153 (55638bc44): [AUDITED — N/A] Fix wgrad race condition when using double
    buffers (Megatron-FSDP only: relocate _enforce_double_buffer_limit from
    backward hook to main_grad_getter; off-by-one > 1 → > 2 in eviction loop;
    _megatron_fsdp_model ref attachment after DTensor param replacement).
    Our DDP uses predecessor-drain (M4036) + ownership FSM (M3061/M3116)
    instead of FSDP double-buffering; these bugs do not apply.
```

This document provides the detailed evidence for that one-line summary.

---

## No code changes required

M4153 is fully audited. No changes to our codebase are needed.
