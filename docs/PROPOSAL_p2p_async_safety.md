# M3766 — Proposal: Wait for Async P2P Send Before Deallocating Output Tensor

**Task ID:** M3766  
**Reference:** Megatron-LM commit 260cba713 (PR #4047)  
**Author:** PipeLead  
**Date:** 2026-06-29  
**Status:** Proposed  

---

## 1. Problem Description

### Background

During pipeline-parallel training, each rank performs a forward pass and then
sends its `output_tensor` (activations) to the next stage via a point-to-point
(P2P) send.  When `config.overlap_p2p_comm=True` the send is issued as a
non-blocking `isend` (or a `batch_isend_irecv` in batched mode) and a *handle*
is returned so the CPU can proceed without waiting for the DMA/NCCL transfer
to finish.

### The Race

After the send is issued, the schedule calls:

```python
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

`deallocate_output_tensor` sets `output_tensor.data = torch.empty(0, ...)`,
which drops the reference to the original tensor storage.  If the Python
reference count falls to zero, PyTorch frees the underlying GPU buffer.

**If the `isend` has not yet completed when the buffer is freed**, NCCL/CUDA
continues DMAing stale or reused memory to the next rank.  The receiving rank
then reads corrupted activations, producing silent numerical errors or hard
crashes.  This class of bug is particularly dangerous because:

- It is non-deterministic: whether the race is hit depends on GPU load, PCIe
  bandwidth, NCCL kernel scheduling, and microbatch timing.
- It typically manifests as loss spikes or NaN/Inf divergence rather than an
  explicit error, making it difficult to attribute.
- The bug only triggers when **both** `overlap_p2p_comm=True` **and**
  `deallocate_pipeline_outputs=True` are set — a common production
  configuration for memory-efficient large-model training.

### Megatron's Fix (commit 260cba713)

Megatron addressed this in two symmetric locations inside
`forward_backward_pipelining_with_interleaving` (the interleaved 1F1B schedule
used with virtual pipeline stages).  Both are in the overlap path, immediately
before `deallocate_output_tensor`:

```python
# isend() copies asynchronously; wait until the copy is done before
# freeing the source buffer, otherwise the next PP stage gets corrupted data.
if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
    send_next_wait_handle.wait()
    send_next_wait_handle = None

deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

The guard is conditioned on `config.deallocate_pipeline_outputs` to avoid an
unnecessary synchronisation when the tensor is not being freed.

---

## 2. Is Our Code Affected?

### Current State: Already Patched

Inspection of `deepspeed/core/pipeline_parallel/schedules.py` shows that **both
fix sites from Megatron PR #4047 are present verbatim** in our codebase.

**Site 1 — warmup loop (line 1454–1460):**

```python
# isend() copies asynchronously; wait until the copy is done before
# freeing the source buffer, otherwise the next PP stage gets corrupted data.
if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
    send_next_wait_handle.wait()
    send_next_wait_handle = None

deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

**Site 2 — `pp_post_forward` closure (lines 1583–1585):**

```python
# isend() copies asynchronously; wait until the copy is done before
# freeing the source buffer, otherwise the next PP stage gets corrupted data.
if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
    send_next_wait_handle.wait()
    send_next_wait_handle = None
```

These match the Megatron diff exactly.

### Non-Overlap Path (line 1430) — Safe by Design

There is one additional `deallocate_output_tensor` call in the
`not config.overlap_p2p_comm_warmup_flush` branch (line 1430).  This path goes
through `p2p_communicator.send_forward_recv_forward(…)` **without**
`overlap_p2p_comm=True`, which calls `_communicate(…, wait_on_reqs=True)`.
Inside `_communicate`, when `wait_on_reqs=True`, every request handle is
`.wait()`-ed before the function returns:

```python
if wait_on_reqs and len(reqs) > 0:
    req_iter = reqs if isinstance(reqs, list) else reqs.values()
    for req in req_iter:
        req.wait()
```

The non-overlap path is therefore synchronous and **does not require a
pre-deallocation wait guard**.

### `batch_p2p_comm` Path

When `config.batch_p2p_comm=True`, `_communicate` asserts `wait_on_reqs=True`
(line 404), so `batch_isend_irecv` requests are also waited upon before
returning control to the schedule.  No additional guard is needed there either.

### Summary

| Path | overlap_p2p_comm | deallocate guard present? | Safe? |
|------|-----------------|--------------------------|-------|
| Non-overlap sync send | False | N/A (blocking) | ✅ |
| `batch_p2p_comm` | False | N/A (blocking) | ✅ |
| Overlap path — warmup loop | True | ✅ line 1456 | ✅ |
| Overlap path — `pp_post_forward` | True | ✅ line 1583 | ✅ |

**No additional code changes to `schedules.py` are required.**

---

## 3. Fix Plan

Because the fix is already applied, the remaining actions are:

### 3.1 Confirm Fix Origin

Verify via `git log` that the guard lines were introduced intentionally and are
not an accidental port without the accompanying comment, which would indicate
they might be rolled back or misunderstood.

```bash
git log -S 'isend() copies asynchronously' --oneline -- deepspeed/core/pipeline_parallel/schedules.py
```

### 3.2 Harden Against Regression

Add a code-review checklist item (in `CONTRIBUTING.md` or the PR template):

> Any new `deallocate_output_tensor` call inside an `overlap_p2p_comm` path
> **must** be preceded by a `send_next_wait_handle.wait()` guard when
> `config.deallocate_pipeline_outputs` may be True.

### 3.3 Future Risk: `send_prev_wait_handle`

Backward-pass gradient sends use `send_prev_wait_handle` (lines 1487, 1640).
These follow the same async pattern.  Current code waits on
`send_prev_wait_handle` before issuing the *next* backward send, but does **not**
explicitly guard before any backward-pass `deallocate_output_tensor` call.

Audit needed:
1. Confirm that `deallocate_output_tensor` is never called while
   `send_prev_wait_handle` is still in flight.
2. If such a call exists, apply the symmetric guard:

```python
if send_prev_wait_handle is not None and config.deallocate_pipeline_outputs:
    send_prev_wait_handle.wait()
    send_prev_wait_handle = None
```

This is a follow-up task and does not block the current M3766 closure.

---

## 4. Test Plan

### 4.1 Unit Test (new)

**File:** `tests/unit/core/pipeline_parallel/test_p2p_async_safety.py`

Scenario: mock `deallocate_output_tensor` to record the call timestamp; mock
the send handle to record when `.wait()` is called.  Assert that `.wait()` is
called **before** `deallocate_output_tensor` whenever
`config.deallocate_pipeline_outputs=True` and `config.overlap_p2p_comm=True`.

### 4.2 Integration Test

Run the existing interleaved-schedule integration tests with both flags
enabled:

```bash
pytest tests/functional/pipeline_parallel/ \
  -k "interleaved" \
  --extra-config "overlap_p2p_comm=True,deallocate_pipeline_outputs=True"
```

Confirm no numerical divergence over 100 steps at bf16 precision with a small
GPT-style model on 4 GPUs (2 PP × 2 DP, 2 virtual stages).

### 4.3 Stress Test

Run the same configuration for 1 000 steps under GPU memory pressure
(`--mem-fraction 0.95`) to force allocator reuse of freed buffers and maximise
the probability of observing any remaining race.  Compare loss curve against a
run with `overlap_p2p_comm=False` as reference.

### 4.4 Regression Guard

Add the unit test from 4.1 to CI so that any future change that removes the
wait guard will fail pre-merge.

---

## 5. References

- Megatron-LM PR #4047: "fix: wait for async P2P send before deallocating output tensor"
- Megatron commit `260cba713` (internal mirror: `25d9b55c`)
- PyTorch `dist.isend` docs: transfer completion is only guaranteed after `.wait()`
- `deepspeed/core/pipeline_parallel/schedules.py` lines 1454–1460, 1581–1585
- `deepspeed/core/pipeline_parallel/p2p_communication.py` `_communicate()` line 435–438
