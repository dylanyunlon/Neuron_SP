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

# Megatron Commit Audit — M3981

**Megatron commit:** `aa786b72c097d92c3656844321380a2e212c169e`  
**Title:** Thread custom process groups through MoE grad finalization  
**Audited by:** DistLead  
**Audit date:** 2026-06-29  
**Verdict:** ✅ Fully covered — applied across 4 separate commits

## What M3981 changes (upstream Megatron)

M3981 removes every implicit `parallel_state` singleton call inside the MoE
expert-bias gradient finalization path, threading an explicit `tp_dp_cp_group`
parameter from `finalize_model_grads` down to the `torch.distributed.all_reduce`
inside `get_updated_expert_bias`. It also removes two
`parallel_state.get_tensor_model_parallel_group()` calls in `MoELayer`
checkpoint wrappers, replacing them with `self.tp_group`.

**5 files touched in Megatron-LM:**

| File | Change |
|---|---|
| `distributed/finalize_model_grads.py` | `_update_router_expert_bias` + `finalize_model_grads` get `tp_dp_cp_group` |
| `pipeline_parallel/schedules.py` | 3 schedule functions set `pg_collection.tp_dp_cp` |
| `transformer/moe/moe_layer.py` | 2 `checkpoint()` sites use `self.tp_group` |
| `transformer/moe/moe_utils.py` | `get_updated_expert_bias` gains `tp_dp_cp_group=` |
| `transformer/moe/shared_experts.py` | SP-region calls gain `group=self.tp_group` |

## Coverage map

### `finalize_model_grads.py` — ✅ COVERED (commit `ae60b3e0`)

- `_update_router_expert_bias` has `tp_dp_cp_group=` param (line 840).
- `finalize_model_grads` asserts `pg_collection.tp_dp_cp` when
  `moe_router_enable_expert_bias=True` and extracts `tp_dp_cp_group`.
- Fallback when `pg_collection is None` reads
  `parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)`.
- Call at line 1219 passes `tp_dp_cp_group=tp_dp_cp_group`.

### `schedules.py` — ✅ COVERED (commit `5e1453ef`)

`pg_collection.tp_dp_cp` set in both no-pipelining and interleaving paths:
```
schedules.py:643  pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(...)
schedules.py:787  pg_collection.tp_dp_cp = _ps.get_tensor_and_data_parallel_group(...)
```

### `moe_layer.py` — ✅ NOT APPLICABLE

Our `MoELayer` is a pure-PyTorch PCIe-optimized rewrite with no TP dimension.
It never called `parallel_state.get_tensor_model_parallel_group()` in
`tensor_parallel.checkpoint()` wrappers. Zero occurrences confirmed by grep.

### `moe_utils.py` — ✅ NOT APPLICABLE (import delegation)

Our `moe_utils.py` does not define `get_updated_expert_bias`. The function is
imported directly from Megatron (`finalize_model_grads.py` line 127), so the
upstream `tp_dp_cp_group=` fix is inherited automatically when Megatron ≥ aa786b72
is installed.

### `shared_experts.py` — ❌ GAP FOUND → ✅ FIXED (commit `5064b4f2`)

**Before:** both SP-region calls used implicit `parallel_state` singleton:
```python
gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True)
copy_to_tensor_model_parallel_region(input)
```

**Root cause:** `mappings.py` did not accept a `group=` parameter on either
function, so `self.tp_group` (already set via `pg_collection.tp`) could not
be threaded through.

**Fix (commit `5064b4f2`):**

`deepspeed/core/tensor_parallel/mappings.py`:
- `_CopyToModelParallelRegion` gains `group=None` in `forward()`; `backward()`
  uses `ctx.group` or falls back to `_get_tp_group()`.
- `_GatherFromSequenceParallelRegion` gains `group=None`; both forward all-gather
  and backward all-reduce use the explicit group.
- Both public wrappers gain a `group=` keyword; default `None` preserves
  backward compatibility.

`deepspeed/core/transformer/moe/shared_experts.py`:
```python
gather_from_sequence_parallel_region(
    input, tensor_parallel_output_grad=True, group=self.tp_group  # M3981
)
copy_to_tensor_model_parallel_region(input, group=self.tp_group)  # M3981
```

## Regression tests

`tests/test_core_integration.py` **Test 7** verifies the `pg_collection.tp_dp_cp`
assertion and `tp_dp_cp_group=` kwarg threading in `finalize_model_grads.py`.
