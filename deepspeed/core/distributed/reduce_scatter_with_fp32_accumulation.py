# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""FP32-accumulation reduce-scatter primitive for DDP gradient synchronization.

Ported from Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py.

Evolution summary (Megatron-LM commit history):
  M2577 (bb216765d): Initial implementation — pluggable RS function to enable
      FP32 gradient accumulation during reduce-scatter without keeping a
      persistent FP32 grad buffer. Uses all-to-all + local fp32 sum + copy-back.
  M3834 (f2dcd421b): Added knob reduce_scatter_with_fp32_accumulation to
      DistributedDataParallelConfig; wired into ParamAndGradBucketGroup.

Algorithm:
  Standard reduce-scatter over BF16/FP16 gradients uses NCCL's built-in
  accumulation (lower precision).  This module replaces that with:
  1. all_to_all: collect all shards from all ranks in the original dtype.
  2. Local FP32 sum: sum across world_size slices in float32.
  3. Downcast + copy-back into the output shard tensor (original dtype).

  Result: every rank gets a reduce-scattered shard accumulated in FP32,
  at the cost of a 2× memory temporary (all_to_all_output_tensor).

DES-LOC notes:
  On PCIe-only heterogeneous topologies (A6000×2 + H100 NVL + Blackwell×2)
  all-to-all is routed over PCIe fabric.  The extra FP32 local sum is
  compute-only (no additional communication) and happens on-device.
  Enable via DistributedDataParallelConfig(reduce_scatter_with_fp32_accumulation=True)
  when BF16 gradient accumulation loss matters more than peak device memory.

Public API:
  _ReduceScatterWithFP32AccumulationWorkHandle
  reduce_scatter_with_fp32_accumulation()
"""

from __future__ import annotations

from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Work handle
# ---------------------------------------------------------------------------

class _ReduceScatterWithFP32AccumulationWorkHandle:
    """Work handle returned by reduce_scatter_with_fp32_accumulation when async_op=True.

    Callers invoke .wait() to complete the communication and associated FP32
    local accumulation before reading output_tensor.

    Args:
        all_to_all_handle: Async handle from torch.distributed.all_to_all_single,
            or None when the collective was issued synchronously.
        all_to_all_output_tensor: Temporary tensor holding all-to-all output
            in the original (low-precision) dtype.
        output_tensor: Destination tensor for the reduced shard (original dtype).
        world_size: Data-parallel world size (number of ranks in the group).
    """

    def __init__(
        self,
        all_to_all_handle: Any,
        all_to_all_output_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        world_size: int,
    ) -> None:
        self.all_to_all_handle = all_to_all_handle
        self.all_to_all_output_tensor = all_to_all_output_tensor
        self.output_tensor = output_tensor
        self.world_size = world_size

    def wait(self) -> None:
        """Complete the all-to-all communication and apply FP32 local accumulation.

        Steps:
          1. Wait for all_to_all_handle (if async).
          2. Accumulate all_to_all_output_tensor shards in FP32 via torch.sum.
          3. Copy downcasted FP32 result into output_tensor.
        """
        # 1. Wait for async all-to-all to complete (if it was dispatched asynchronously).
        if self.all_to_all_handle is not None:
            self.all_to_all_handle.wait()

        # 2. Local FP32 accumulation: reshape to (world_size, shard_size) then
        #    sum along dim=0 in float32 for higher-precision gradient accumulation.
        #    This avoids the precision loss of NCCL's built-in BF16/FP16 reduce.
        output_tensor_in_fp32 = torch.sum(
            self.all_to_all_output_tensor.view((self.world_size, -1)),
            dim=0,
            dtype=torch.float32,
        )
        assert output_tensor_in_fp32.dtype == torch.float32

        # 3. Downcast the FP32 sum back to the original tensor dtype and write
        #    into output_tensor in-place (copy_ handles dtype conversion).
        self.output_tensor.copy_(output_tensor_in_fp32)


# ---------------------------------------------------------------------------
# Main primitive
# ---------------------------------------------------------------------------

def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: Optional[torch.distributed.ProcessGroup],
    async_op: bool,
) -> Optional[_ReduceScatterWithFP32AccumulationWorkHandle]:
    """Reduce-scatter with FP32 local accumulation.

    Replaces a standard reduce_scatter_tensor call when higher-precision
    gradient accumulation is needed without a persistent FP32 grad buffer.

    Implementation:
      1. Allocate all_to_all_output_tensor (same shape/dtype as input_tensor).
      2. Issue all_to_all_single: every rank receives one shard from every rank.
      3. Locally accumulate world_size slices in FP32 via torch.sum(dim=0, dtype=float32).
      4. Downcast + copy into output_tensor.

    Only torch.distributed.ReduceOp.SUM is supported (FP32 sum semantics).

    Args:
        output_tensor: Output tensor for the reduce-scattered shard.
            Shape: [input_tensor.numel() // world_size] (same dtype as input_tensor).
        input_tensor: Input gradient tensor to reduce-scatter across the group.
            numel() must be divisible by world_size.
        op: Reduction operator — only ReduceOp.SUM is supported.
        group: Process group. None → uses the default (WORLD) group.
        async_op: If True, returns a _ReduceScatterWithFP32AccumulationWorkHandle;
            caller must call .wait() before reading output_tensor.
            If False, blocks until communication and accumulation complete.

    Returns:
        _ReduceScatterWithFP32AccumulationWorkHandle when async_op=True, else None.

    Raises:
        AssertionError: if op != ReduceOp.SUM, or if input_tensor.numel() % world_size != 0.
    """
    # Only SUM reduction is supported — FP32 accumulation is a sum operation.
    assert op == torch.distributed.ReduceOp.SUM, (
        f"reduce_scatter_with_fp32_accumulation only supports ReduceOp.SUM, got {op}"
    )

    # Resolve world size from the process group.
    if group is None:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = group.size()

    # Validate that input tensor is evenly divisible for sharding.
    assert input_tensor.numel() % world_size == 0, (
        f"input_tensor.numel()={input_tensor.numel()} must be divisible by "
        f"world_size={world_size}"
    )

    # Allocate output tensor for all-to-all (cannot be done in-place).
    # Same shape and dtype as input_tensor so every rank receives all shards.
    all_to_all_output_tensor = torch.empty_like(input_tensor)

    # Issue all_to_all_single: each rank sends equal-size chunks to all peers
    # and receives one chunk from each peer. After this collective, each rank
    # holds world_size chunks — one per source rank — for its own shard range.
    all_to_all_handle = torch.distributed.all_to_all_single(
        output=all_to_all_output_tensor,
        input=input_tensor,
        group=group,
        async_op=async_op,
    )

    # Construct the work handle that wraps the async communication and the
    # subsequent FP32 accumulation step.
    reduce_scatter_handle = _ReduceScatterWithFP32AccumulationWorkHandle(
        all_to_all_handle,
        all_to_all_output_tensor,
        output_tensor,
        world_size,
    )

    if async_op:
        # Return the handle; the caller is responsible for calling .wait()
        # before reading output_tensor.
        return reduce_scatter_handle
    else:
        # Synchronous path: complete communication and local FP32 accumulation now.
        reduce_scatter_handle.wait()
        return None
