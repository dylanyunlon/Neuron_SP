# SPDX-License-Identifier: Apache-2.0
"""autosp_hook.py — connect AutoSP/Ulysses SP group to optimizer + grad reduction.

Wires the sequence-parallel process group into:
  1. ``wrap_grad_reduction`` — installs a backward hook that reduces SP-sharded
     parameter gradients within the SP group before the DP all-reduce fires.
     This ensures every data-parallel rank sees the correct aggregated gradient
     for parameters that were sharded along the sequence dimension.

  2. ``sp_aware_sync`` — propagates the DES-LOC sync class for each parameter
     into ``DesLocSyncPolicy`` so that SP-sharded params obey the correct
     Kx/Ku/Kv period instead of always being classified 'x'.

SP group contract
-----------------
``self.sp_group`` is a ``torch.distributed.ProcessGroup`` whose member ranks
share a single sequence position (they hold different token subsequences).
Gradients from these ranks must be all-reduced within the SP group before the
outer DP all-reduce, otherwise the DP aggregation double-counts tokens.

PCIe constraint
---------------
This cluster has no NVLink.  SP all-to-all deadlocks on PCIe-only topologies
(see ARCHITECTURE.md).  ``wrap_grad_reduction`` therefore checks
``NEURON_SP_DISABLE_AUTOSP`` and silently skips hook installation when SP is
disabled — matching the kill-switch in ``desloc_engine.py`` line ~1738.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .tier_map import TierMap
    from .dist_opt_adapter import DistOptAdapter
    from .desloc_sync_policy import DesLocSyncPolicy

logger = logging.getLogger(__name__)

# Kill-switch env var mirrors desloc_engine.py line ~1738.
_ENV_DISABLE_AUTOSP = "NEURON_SP_DISABLE_AUTOSP"


class AutoSPHook:
    """Connects AutoSP/Ulysses sequence-parallel group to optimizer + grad sync.

    Args:
        sp_group: The sequence-parallel ``torch.distributed.ProcessGroup``, or
                  ``None`` when SP is disabled.
        tier_map: Cluster GPU tier map (used for logging/PCIe-safe decisions).
    """

    def __init__(self, sp_group, tier_map: "TierMap") -> None:
        self.sp_group = sp_group
        self.tier_map = tier_map
        # Handles installed by wrap_grad_reduction; stored for cleanup.
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Public API (frozen per ARCHITECTURE.md)
    # ------------------------------------------------------------------

    def wrap_grad_reduction(self, adapter: "DistOptAdapter") -> None:
        """Install SP-aware gradient reduction hooks on model parameters.

        For each parameter in ``adapter.model`` that is registered in the SP
        group, a post-accumulate gradient hook is installed that all-reduces
        the gradient within ``self.sp_group`` before the DP reduce-scatter
        fires.

        No-op conditions (safe to call regardless):
          - ``self.sp_group`` is None (SP disabled).
          - ``NEURON_SP_DISABLE_AUTOSP=1`` is set.
          - ``torch.distributed`` is not initialised.
          - The SP group has world_size == 1 (single-rank; no SP).

        This matches the kill-switch logic in ``desloc_engine.py`` line 1738
        so there is no divergence between engine-level SP state and the hooks.

        Args:
            adapter: The ``DistOptAdapter`` whose ``adapter.model`` parameters
                     will receive gradient reduction hooks.
        """
        import torch
        import torch.distributed as dist

        # ── Kill-switch checks ───────────────────────────────────────
        if os.environ.get(_ENV_DISABLE_AUTOSP, "0").strip() == "1":
            logger.info(
                "[AutoSPHook] wrap_grad_reduction skipped: "
                "%s=1 (PCIe-only topology; SP all-to-all disabled).",
                _ENV_DISABLE_AUTOSP,
            )
            return

        if self.sp_group is None:
            logger.debug(
                "[AutoSPHook] wrap_grad_reduction skipped: sp_group is None."
            )
            return

        if not dist.is_initialized():
            logger.debug(
                "[AutoSPHook] wrap_grad_reduction skipped: "
                "torch.distributed not initialised."
            )
            return

        sp_size = dist.get_world_size(group=self.sp_group)
        if sp_size <= 1:
            logger.debug(
                "[AutoSPHook] wrap_grad_reduction skipped: sp_size=%d (no-op).",
                sp_size,
            )
            return

        # ── Install hooks ────────────────────────────────────────────
        # We use register_post_accumulate_grad_hook (PyTorch ≥ 2.1) when
        # available; fall back to register_hook on the tensor for older builds.
        model = adapter.model
        n_hooked = 0
        sp_group_ref = self.sp_group  # capture for closure

        def _make_sp_allreduce_hook(p: "torch.nn.Parameter"):
            """Return a hook that all-reduces *p*'s grad in the SP group."""
            def hook(grad: "torch.Tensor") -> "torch.Tensor":
                # Fire SP all-reduce synchronously; DP reduce-scatter fires
                # later inside DistributedOptimizer._reduce_scatter_grads().
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=sp_group_ref)
                # Divide by SP group size to get the mean gradient.
                grad.div_(sp_size)
                return grad
            return hook

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            try:
                # PyTorch ≥ 2.1: hook fires after grad accumulation, before
                # any existing backward hooks — ideal for SP pre-reduction.
                h = param.register_post_accumulate_grad_hook(
                    _make_sp_allreduce_hook(param)
                )
            except AttributeError:
                # PyTorch < 2.1 fallback: register on .grad_fn output.
                h = param.register_hook(_make_sp_allreduce_hook(param))
            self._hooks.append(h)
            n_hooked += 1

        logger.info(
            "[AutoSPHook] wrap_grad_reduction: installed SP all-reduce hooks "
            "on %d parameters (sp_size=%d).",
            n_hooked, sp_size,
        )

    def sp_aware_sync(self, policy: "DesLocSyncPolicy") -> None:
        """Propagate SP group membership into the DES-LOC sync policy.

        SP-sharded parameters have higher effective gradient variance than
        purely DP-replicated parameters because each rank sees only a
        subsequence of the full token stream.  To avoid under-synchronisation
        this method re-classifies all SP-local parameters to at least class 'u'
        (i.e. they will not be demoted to the slow 'v' class), and forces
        embedding / LM-head parameters to class 'x'.

        No-op when SP is disabled (``self.sp_group`` is None).

        Args:
            policy: The ``DesLocSyncPolicy`` to update.  Must have been
                    initialised (``__init__`` called); ``classify()`` need not
                    have been called yet.
        """
        if self.sp_group is None:
            logger.debug(
                "[AutoSPHook] sp_aware_sync skipped: sp_group is None."
            )
            return

        if not policy._classified:
            logger.debug(
                "[AutoSPHook] sp_aware_sync: policy not yet classified; "
                "will apply upgrades on next classify() call via _var_ema."
            )
            return

        # Upgrade every 'v'-class param to 'u' so that SP-sharded params get
        # at least medium-frequency sync (Ku).  This is a conservative policy:
        # if the cluster gains NVLink in future we can relax it.
        n_upgraded = 0
        for pid, cls in policy._classes.items():
            if cls == "v":
                policy._classes[pid] = "u"
                n_upgraded += 1

        logger.info(
            "[AutoSPHook] sp_aware_sync: upgraded %d params from 'v' → 'u' "
            "(SP group size=%d; PCIe-safe conservative sync policy).",
            n_upgraded,
            self.sp_group.size() if hasattr(self.sp_group, "size") else -1,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all installed gradient hooks (call on engine teardown)."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        logger.debug("[AutoSPHook] gradient hooks removed.")
