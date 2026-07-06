# SPDX-License-Identifier: Apache-2.0
"""autosp_hook.py — connect AutoSP/Ulysses SP group to optimizer + grad reduction.

Async allreduce redesign
------------------------
Prior version used synchronous dist.all_reduce which blocked each
parameter backward hook until the SP allreduce completed, serialising
the backward pass with network communication.

New design: dist.all_reduce(async_op=True) returns a Work object stored in
self._pending_works. After the full backward pass, wait_all_reduces() calls
work.wait() on all pending works before the optimizer step. This fully overlaps
SP communication with the backward computation for subsequent layers.

Gradient scale
--------------
div_(sp_size) is applied in wait_all_reduces() after work.wait(), not inside
the hook. This avoids an in-place modification racing with the async collective.

SP variance scaling
-------------------
sp_aware_sync now scales per-param EMA variance by sp_size to account for
reduced effective batch size (each SP rank processes 1/sp_size tokens).
This prevents SP-sharded params from being under-classified in desloc_sync.

PCIe constraint
---------------
SP all-to-all deadlocks on PCIe-only topologies. wrap_grad_reduction checks
NEURON_SP_DISABLE_AUTOSP and skips hook installation when SP is disabled.
"""
from __future__ import annotations
import torch

import logging
import os
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    import torch
    from .tier_map import TierMap
    from .dist_opt_adapter import DistOptAdapter
    from .desloc_sync_policy import DesLocSyncPolicy

logger = logging.getLogger(__name__)

_ENV_DISABLE_AUTOSP = "NEURON_SP_DISABLE_AUTOSP"


class AutoSPHook:
    """Connects AutoSP/Ulysses sequence-parallel group to optimizer + grad sync.

    Args:
        sp_group: The sequence-parallel torch.distributed.ProcessGroup, or
                  None when SP is disabled.
        tier_map: Cluster GPU tier map (used for logging/PCIe-safe decisions).
    """

    def __init__(self, sp_group, tier_map: "TierMap") -> None:
        self.sp_group = sp_group
        self.tier_map = tier_map
        self._hooks: list = []
        # Pending async allreduce: List[(Work, grad_tensor, sp_size)]
        # Populated in backward hooks; drained by wait_all_reduces().
        self._pending_works: List[Tuple] = []

    def wrap_grad_reduction(self, adapter: "DistOptAdapter") -> None:
        """Install async SP-aware gradient reduction hooks on model parameters.

        Fires dist.all_reduce(async_op=True) in each grad hook, storing the
        Work object without blocking. Call wait_all_reduces() before optimizer.step().
        """
        import torch
        import torch.distributed as dist

        if os.environ.get(_ENV_DISABLE_AUTOSP, "0").strip() == "1":
            logger.info(
                "[AutoSPHook] wrap_grad_reduction skipped: %s=1 (PCIe-only; SP disabled).",
                _ENV_DISABLE_AUTOSP,
            )
            return

        if self.sp_group is None:
            logger.debug("[AutoSPHook] wrap_grad_reduction skipped: sp_group is None.")
            return

        if not dist.is_initialized():
            logger.debug("[AutoSPHook] wrap_grad_reduction skipped: dist not initialised.")
            return

        sp_size = dist.get_world_size(group=self.sp_group)
        if sp_size <= 1:
            logger.debug("[AutoSPHook] wrap_grad_reduction skipped: sp_size=%d.", sp_size)
            return

        model = adapter.model
        n_hooked = 0
        sp_group_ref = self.sp_group
        pending_ref = self._pending_works  # direct reference

        def _make_async_hook(p: "torch.nn.Parameter"):
            def hook(grad: "torch.Tensor") -> None:
                # Non-blocking SP all-reduce. Work object stored for later wait.
                work = dist.all_reduce(
                    grad,
                    op=dist.ReduceOp.SUM,
                    group=sp_group_ref,
                    async_op=True,
                )
                pending_ref.append((work, grad, sp_size))
                # Return None: grad modified in-place after work.wait() in wait_all_reduces.
                return None
            return hook

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            try:
                h = param.register_post_accumulate_grad_hook(_make_async_hook(param))
            except AttributeError:
                h = param.register_hook(_make_async_hook(param))
            self._hooks.append(h)
            n_hooked += 1

        logger.info(
            "[AutoSPHook] wrap_grad_reduction: installed async hooks on %d params "
            "(sp_size=%d). Call wait_all_reduces() before optimizer.step().",
            n_hooked, sp_size,
        )

    def wait_all_reduces(self) -> int:
        """Wait for all pending async SP all-reduce operations and scale gradients.

        Must be called after backward() and before optimizer.step().
        Applies div_(sp_size) after each work.wait() to convert sum to mean.

        Returns:
            int: Number of SP all-reduce operations completed.
        """
        n = 0
        for work, grad, sp_size in self._pending_works:
            work.wait()
            if sp_size > 1:
                grad.div_(sp_size)
            n += 1
        self._pending_works.clear()
        if n > 0:
            logger.debug("[AutoSPHook] wait_all_reduces: completed %d SP all-reduces.", n)
        return n

    def sp_aware_sync(self, policy: "DesLocSyncPolicy") -> None:
        """Propagate SP group membership into the DES-LOC sync policy.

        1. Upgrades all 'v'-class params to 'u' (prevents slow-sync demotion).
        2. Scales per-param EMA variance by sp_size to account for the reduced
           effective batch size seen by each SP rank (1/sp_size tokens).
           Larger EMA variance drives more aggressive sync in future classify().
        """
        if self.sp_group is None:
            logger.debug("[AutoSPHook] sp_aware_sync skipped: sp_group is None.")
            return

        if not policy._classified:
            logger.debug("[AutoSPHook] sp_aware_sync: policy not yet classified.")
            return

        import torch.distributed as dist
        sp_size = 1
        try:
            if dist.is_initialized():
                sp_size = dist.get_world_size(group=self.sp_group)
        except Exception:
            pass

        n_upgraded = n_scaled = 0
        for pid, cls in policy._classes.items():
            if cls == "v":
                policy._classes[pid] = "u"
                n_upgraded += 1
            if sp_size > 1 and pid in policy._var_ema:
                policy._var_ema[pid] *= sp_size
                n_scaled += 1

        logger.info(
            "[AutoSPHook] sp_aware_sync: upgraded %d params v->u, "
            "scaled %d EMA values by sp_size=%d.",
            n_upgraded, n_scaled, sp_size,
        )

    def remove_hooks(self) -> None:
        """Remove all gradient hooks. Waits for pending async reduces first."""
        if self._pending_works:
            logger.warning(
                "[AutoSPHook] remove_hooks: %d pending async reduces; waiting.",
                len(self._pending_works),
            )
            self.wait_all_reduces()
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        logger.debug("[AutoSPHook] gradient hooks removed.")
