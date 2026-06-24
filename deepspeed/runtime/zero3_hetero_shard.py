"""
ZeRO-3 Heterogeneous Parameter Sharding utilities for DesLocEngine.

Implements per-rank flat parameter sharding such that each rank holds only
``1/N`` of the total flattened parameter buffer (or a VRAM-proportional
fraction in the heterogeneous case), and provides on-demand all-gather
and reduce-scatter primitives for forward / backward.

Design notes
------------
* The model's parameters are conceptually concatenated into a single 1-D
  buffer of dtype FP32 (the master copy used for optimizer math). Each
  rank keeps a contiguous slice of that buffer in ``self.param_shard``.

* ``param_offsets`` records, for every original parameter, the
  ``(global_start, global_end, shape)`` triple inside the flat buffer so
  that we can scatter / gather back into the right ``nn.Parameter``.

* ``gather_full_params(module)`` is a context manager that materializes
  the full FP32 → BF16 buffer via ``dist.all_gather_into_tensor`` and
  rewrites the storage of every parameter to view into that buffer for
  the duration of the ``with`` block. On exit the storage is released.

* ``scatter_grads(model)`` reduces gradients across all ranks with
  ``dist.reduce_scatter_tensor`` and keeps only this rank's local slice,
  matching the layout of ``param_shard``.

* Heterogeneous shard sizing: when ``vram_weights`` is supplied the flat
  buffer is partitioned proportionally (an H100 with 96 GB receives
  twice the share of an A6000 with 48 GB, etc.). Otherwise the buffer
  is partitioned evenly.

* Backwards compatible: ``ShardState.build`` returns ``None`` for
  ``world_size == 1`` and callers must treat that as "no sharding".
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ParamSlice:
    """Locator for an ``nn.Parameter`` inside the flat parameter buffer."""
    name: str
    global_start: int
    global_end: int            # exclusive
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class ShardState:
    """
    Container for ZeRO-3 sharding state attached to the engine.

    Attributes:
        rank: This process's rank.
        world_size: Total number of ranks.
        total_numel: Total number of params across the whole model
                     (after any per-rank padding alignment).
        shard_sizes: List of length ``world_size``; ``shard_sizes[r]`` is
                     the number of FP32 elements held by rank ``r``.
        shard_offsets: Cumulative offsets ``[0, shard_sizes[0],
                       shard_sizes[0]+shard_sizes[1], ...]``.
        param_shard: This rank's local FP32 master shard (1-D tensor of
                     size ``shard_sizes[rank]``) living on CUDA.
        param_offsets: Mapping ``param-name -> ParamSlice``.
        param_order: Ordered list of (name, parameter) tuples used to
                     re-materialize the flat buffer.
        pad: Number of zero-padding elements appended to make
             ``total_numel`` evenly divisible across ranks when using
             ``all_gather_into_tensor`` (which requires equal shard
             sizes). For heterogeneous splits this is also non-zero so
             that the flat buffer length is the sum of ``shard_sizes``.
    """
    rank: int
    world_size: int
    total_numel: int
    shard_sizes: List[int]
    shard_offsets: List[int]
    param_shard: torch.Tensor
    param_offsets: Dict[str, ParamSlice]
    param_order: List[Tuple[str, nn.Parameter]] = field(default_factory=list)
    pad: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        model: nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        vram_weights: Optional[Sequence[float]] = None,
    ) -> Optional["ShardState"]:
        """
        Build the sharding plan for ``model``.

        Returns ``None`` when ``world_size <= 1`` (no sharding needed —
        backward-compatible single-GPU path).

        Args:
            model:        Module whose ``.parameters()`` will be sharded.
            rank:         This process's rank in [0, world_size).
            world_size:   Total number of ranks.
            device:       Local CUDA device.
            vram_weights: Optional per-rank positive weights. When given
                          the flat buffer is partitioned proportionally
                          (heterogeneous case: H100 gets a larger share).
                          Must have length ``world_size``.
        """
        if world_size is None or world_size <= 1:
            return None

        # ---- 1. Enumerate params in a deterministic order. ----------
        param_order: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        if not param_order:
            return None

        # ---- 2. Compute shard sizes. --------------------------------
        raw_total = sum(p.numel() for _, p in param_order)

        if vram_weights is not None:
            if len(vram_weights) != world_size:
                raise ValueError(
                    f"vram_weights length ({len(vram_weights)}) "
                    f"!= world_size ({world_size})"
                )
            if any(w <= 0 for w in vram_weights):
                raise ValueError("vram_weights must be strictly positive")
            wsum = float(sum(vram_weights))
            # First (N-1) ranks take floor(weight * total / sum); last
            # rank absorbs the remainder so the sum matches exactly.
            shard_sizes: List[int] = []
            assigned = 0
            for r in range(world_size - 1):
                size_r = int((vram_weights[r] / wsum) * raw_total)
                shard_sizes.append(size_r)
                assigned += size_r
            shard_sizes.append(max(0, raw_total - assigned))
            total_numel = raw_total
            pad = 0
        else:
            # Even split: pad up to a multiple of world_size so that
            # all_gather_into_tensor (equal shards) works cleanly.
            shard_size = (raw_total + world_size - 1) // world_size
            total_numel = shard_size * world_size
            pad = total_numel - raw_total
            shard_sizes = [shard_size] * world_size

        # Cumulative offsets — useful both for indexing and for
        # interpreting the layout after all-gather.
        shard_offsets: List[int] = [0]
        for s in shard_sizes:
            shard_offsets.append(shard_offsets[-1] + s)
        assert shard_offsets[-1] == total_numel, (
            f"shard layout mismatch: {shard_offsets[-1]} vs {total_numel}"
        )

        # ---- 3. Build param_offsets (flat layout map). --------------
        param_offsets: Dict[str, ParamSlice] = {}
        cursor = 0
        for name, p in param_order:
            n = p.numel()
            param_offsets[name] = ParamSlice(
                name=name,
                global_start=cursor,
                global_end=cursor + n,
                shape=p.shape,
                dtype=p.dtype,
            )
            cursor += n
        assert cursor == raw_total

        # ---- 4. Initialise this rank's FP32 master shard. -----------
        lo = shard_offsets[rank]
        hi = shard_offsets[rank + 1]
        local_size = hi - lo

        param_shard = torch.zeros(local_size, dtype=torch.float32, device=device)

        # Fill the slice by copying the relevant region of each param.
        # We walk the flat layout and intersect with [lo, hi).
        for name, p in param_order:
            sl = param_offsets[name]
            # Skip params entirely outside this rank's window.
            if sl.global_end <= lo or sl.global_start >= hi:
                continue
            # Intersection in global coordinates.
            g_start = max(sl.global_start, lo)
            g_end   = min(sl.global_end, hi)
            # Indices into the param's own flat view.
            p_start = g_start - sl.global_start
            p_end   = g_end   - sl.global_start
            # Indices into the local shard buffer.
            s_start = g_start - lo
            s_end   = g_end   - lo
            with torch.no_grad():
                flat = p.detach().reshape(-1).to(torch.float32)
                param_shard[s_start:s_end].copy_(flat[p_start:p_end])

        logger.info(
            "[zero3] rank=%d/%d shard=%d elems (of %d total, pad=%d, hetero=%s)",
            rank, world_size, local_size, total_numel, pad,
            vram_weights is not None,
        )

        # Pre-allocate the gradient buffer NOW, before forward all-gather
        # consumes GPU memory. Lazy allocation inside backward hooks would
        # OOM because forward already holds the full-layer gathered params.
        param_shard.grad = torch.zeros_like(param_shard)
        logger.info(
            "[zero3] rank=%d pre-allocated param_shard.grad: %.2f GB on %s",
            rank, param_shard.grad.nbytes / (1 << 30), device,
        )

        return cls(
            rank=rank,
            world_size=world_size,
            total_numel=total_numel,
            shard_sizes=shard_sizes,
            shard_offsets=shard_offsets,
            param_shard=param_shard,
            param_offsets=param_offsets,
            param_order=param_order,
            pad=pad,
        )

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------
    def _build_full_buffer(self, dtype: torch.dtype) -> torch.Tensor:
        """
        All-gather all rank shards into a single contiguous flat buffer.

        For even splits we can use ``dist.all_gather_into_tensor``
        directly. For heterogeneous (variable-size) splits we fall back
        to ``dist.all_gather`` into per-rank tensors and concat.
        """
        device = self.param_shard.device
        full = torch.empty(self.total_numel, dtype=dtype, device=device)

        # Cast our shard to the target gather dtype (typically BF16 for
        # forward, FP32 for diagnostic / debug).
        local = self.param_shard.to(dtype)

        even = all(s == self.shard_sizes[0] for s in self.shard_sizes)
        if even:
            dist.all_gather_into_tensor(full, local)
        else:
            chunks = [
                torch.empty(s, dtype=dtype, device=device)
                for s in self.shard_sizes
            ]
            dist.all_gather(chunks, local)
            for r, ch in enumerate(chunks):
                lo = self.shard_offsets[r]
                hi = self.shard_offsets[r + 1]
                full[lo:hi].copy_(ch)
        return full

    @contextmanager
    def gather_full_params(self, module: nn.Module) -> Iterator[None]:
        """
        Context manager: materialize full params via all-gather, rewrite
        each ``nn.Parameter`` to view into the gathered buffer, run the
        ``with`` block, then release the buffer and restore the local
        sharded storage.

        The gathered buffer's dtype matches the first parameter's dtype
        (usually BF16), so forward/backward run in low precision.
        """
        if self.world_size <= 1:
            yield
            return

        # Choose dtype to gather in — match the live parameter dtype.
        gather_dtype = self.param_order[0][1].dtype

        full = self._build_full_buffer(gather_dtype)

        # Save originals so we can restore after the block.
        saved: List[Tuple[nn.Parameter, torch.Tensor]] = []
        try:
            for name, p in self.param_order:
                sl = self.param_offsets[name]
                view = full[sl.global_start:sl.global_end].view(sl.shape)
                saved.append((p, p.data))
                p.data = view
            yield
        finally:
            for p, orig in saved:
                p.data = orig
            del full

    # ------------------------------------------------------------------
    # Per-parameter backward hooks (reduce-scatter on the fly)
    # ------------------------------------------------------------------
    def register_backward_hooks(
        self,
        fp32_grad_manager: Optional[object] = None,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register a post-accumulate-grad hook on every sharded parameter so
        that, the moment autograd finishes producing a parameter's
        ``.grad`` during the backward pass, that grad is reduce-scattered
        across ranks and only this rank's slice is retained.

        After the hook fires for parameter ``p``:
          * ``p.grad`` is overwritten by a per-rank grad shard tensor of
            shape ``(local_numel,)`` (i.e. the intersection of
            ``[lo, hi) = [shard_offsets[rank], shard_offsets[rank+1])``
            with this parameter's flat layout, possibly zero-length when
            the parameter lives entirely on other ranks).
          * The shard has already been ``SUM``-reduced across the world
            and divided by ``world_size`` (mean), matching standard DDP
            semantics.
          * When ``fp32_grad_manager`` is supplied, the shard is also
            accumulated into the corresponding FP32 ``main_grad`` slice
            (if one exists for this parameter) so the existing
            three-tier precision policy still applies — the only
            difference from the unsharded path is that ``main_grad`` now
            only ever sees its rank-local slice of the gradient.

        ``world_size <= 1`` is a no-op and returns an empty list.

        Returns the handles for later ``handle.remove()``.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []
        if self.world_size <= 1:
            return handles

        lo_rank = self.shard_offsets[self.rank]
        hi_rank = self.shard_offsets[self.rank + 1]
        world_size = self.world_size
        rank = self.rank
        shard_sizes = self.shard_sizes
        shard_offsets = self.shard_offsets
        even = all(s == shard_sizes[0] for s in shard_sizes)
        _param_shard = self.param_shard  # captured for closure

        def _make_hook(name: str, p: nn.Parameter):
            sl = self.param_offsets[name]
            shard_slices: List[Tuple[int, int]] = []
            for r in range(world_size):
                r_lo = shard_offsets[r]
                r_hi = shard_offsets[r + 1]
                g_lo = max(sl.global_start, r_lo)
                g_hi = min(sl.global_end,   r_hi)
                if g_hi <= g_lo:
                    shard_slices.append((0, 0))
                else:
                    shard_slices.append(
                        (g_lo - sl.global_start, g_hi - sl.global_start)
                    )

            # Pre-compute where this param's local grad lands in param_shard
            _g_start = max(sl.global_start, lo_rank)
            _g_end = min(sl.global_end, hi_rank)
            _shard_start = _g_start - lo_rank if _g_end > _g_start else 0
            _shard_end = _g_end - lo_rank if _g_end > _g_start else 0

            def _hook(param: nn.Parameter) -> None:
                grad = param.grad
                if grad is None:
                    return
                device = grad.device
                flat = grad.detach().reshape(-1)

                if even:
                    # Build a contiguous send buffer laid out as
                    # [rank0_slab | rank1_slab | ... | rankN-1_slab] so a
                    # single reduce_scatter_tensor produces our local slab.
                    shard = shard_sizes[0]
                    send = torch.zeros(
                        shard * world_size, dtype=flat.dtype, device=device,
                    )
                    for r in range(world_size):
                        p_lo, p_hi = shard_slices[r]
                        if p_hi > p_lo:
                            s_off = r * shard
                            send[s_off:s_off + (p_hi - p_lo)].copy_(
                                flat[p_lo:p_hi]
                            )
                    local_shard = torch.empty(
                        shard, dtype=flat.dtype, device=device,
                    )
                    dist.reduce_scatter_tensor(
                        local_shard, send, op=dist.ReduceOp.SUM,
                    )
                    local_shard.div_(world_size)
                    # Trim to the (possibly shorter) slice this param
                    # actually contributes on our rank — the rest is
                    # zero-padding belonging to other params.
                    p_lo, p_hi = shard_slices[rank]
                    take = p_hi - p_lo
                    local_grad = local_shard[:take].clone() if take > 0 else \
                        torch.empty(0, dtype=flat.dtype, device=device)
                else:
                    # Heterogeneous shard sizes — reduce_scatter_tensor
                    # requires equal-sized chunks, so fall back to an
                    # all-reduce + local slice.
                    full = flat.clone()
                    dist.all_reduce(full, op=dist.ReduceOp.SUM)
                    full.div_(world_size)
                    p_lo, p_hi = shard_slices[rank]
                    local_grad = full[p_lo:p_hi].clone() if p_hi > p_lo else \
                        torch.empty(0, dtype=flat.dtype, device=device)

                # Write reduced gradient directly into param_shard.grad at
                # the correct offset. Do NOT write param.grad — the param
                # shape doesn't match the 1-D reduced shard.
                if local_grad.numel() > 0 and _shard_end > _shard_start:
                    _param_shard.grad[_shard_start:_shard_end].add_(
                        local_grad[:_shard_end - _shard_start].to(
                            dtype=_param_shard.dtype, device=_param_shard.device
                        )
                    )
                # Clear param.grad to free memory (it's been consumed)
                param.grad = None

                # Optional FP32 accumulator integration. We only touch
                # ``main_grad`` if the manager actually allocated one for
                # this parameter (selective FP32 policy).
                if fp32_grad_manager is not None and hasattr(param, "main_grad"):
                    mg = param.main_grad
                    if mg is not None and local_grad.numel() > 0:
                        # ``main_grad`` may be full-shape FP32; flatten
                        # and add into the matching slice. If the
                        # manager pre-sharded it to local size, accept
                        # that too.
                        mg_flat = mg.reshape(-1)
                        if mg_flat.numel() == local_grad.numel():
                            mg_flat.add_(local_grad.float())
                        elif mg_flat.numel() == flat.numel():
                            # Full-shape main_grad: write only the local
                            # slice; the remaining entries are owned by
                            # other ranks and stay zero on this rank.
                            mg_flat[p_lo:p_hi].add_(local_grad.float())
                        # Any other shape mismatch is ignored — the
                        # manager's own accumulate()/after_backward()
                        # path will still run on whatever it owns.

            return p.register_post_accumulate_grad_hook(_hook)

        for name, p in self.param_order:
            if not p.requires_grad:
                continue
            handles.append(_make_hook(name, p))

        logger.info(
            "[zero3] rank=%d registered %d backward reduce-scatter hooks "
            "(fp32_grad_manager=%s)",
            self.rank, len(handles), fp32_grad_manager is not None,
        )
        return handles

    def scatter_grads(self, model: nn.Module) -> torch.Tensor:
        """
        Reduce-scatter gradients across all ranks.

        Returns the local FP32 gradient shard (1-D, same length as
        ``param_shard``). Each rank's returned tensor corresponds to its
        own slice of the flat parameter layout, so it can be used
        directly to update ``param_shard`` via the optimizer.

        After this call the per-parameter ``.grad`` fields are cleared
        to free memory.
        """
        device = self.param_shard.device

        # Assemble a flat gradient buffer in FP32 from each param's grad.
        flat_grad = torch.zeros(self.total_numel, dtype=torch.float32, device=device)
        for name, p in self.param_order:
            if p.grad is None:
                continue
            sl = self.param_offsets[name]
            flat_grad[sl.global_start:sl.global_end].copy_(
                p.grad.detach().reshape(-1).to(torch.float32)
            )
            p.grad = None

        local_grad = torch.empty(
            self.shard_sizes[self.rank], dtype=torch.float32, device=device
        )

        even = all(s == self.shard_sizes[0] for s in self.shard_sizes)
        if even:
            dist.reduce_scatter_tensor(local_grad, flat_grad, op=dist.ReduceOp.SUM)
            # Average across ranks (standard data-parallel reduction).
            local_grad.div_(self.world_size)
        else:
            # Variable-size fallback: all-reduce then slice.
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad.div_(self.world_size)
            lo = self.shard_offsets[self.rank]
            hi = self.shard_offsets[self.rank + 1]
            local_grad.copy_(flat_grad[lo:hi])

        return local_grad


# ---------------------------------------------------------------------------
# Per-layer forward all-gather hooks (Claude-128)
# ---------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Gradient collection & param sync for optimizer operating on param_shard
    # ------------------------------------------------------------------
    def collect_grads_into_shard(self) -> None:
        """Collect reduce-scattered per-param grads into param_shard.grad.

        After backward hooks fire, each param.grad contains only this rank's
        local slice of the reduced gradient. This method stitches them into
        a single 1-D FP32 gradient buffer aligned with param_shard.
        """
        device = self.param_shard.device
        if self.param_shard.grad is None:
            self.param_shard.grad = torch.zeros_like(self.param_shard)
        else:
            self.param_shard.grad.zero_()

        lo = self.shard_offsets[self.rank]
        hi = self.shard_offsets[self.rank + 1]

        for name, p in self.param_order:
            if p.grad is None:
                continue
            sl = self.param_offsets[name]
            # Intersection of this param's flat range with our shard window
            g_start = max(sl.global_start, lo)
            g_end = min(sl.global_end, hi)
            if g_end <= g_start:
                continue
            # Index into param_shard
            s_start = g_start - lo
            s_end = g_end - lo
            # The backward hook already reduce-scattered and stored the
            # local slice in param.grad. Its length should match (g_end - g_start).
            grad_flat = p.grad.detach().reshape(-1).to(device=device, dtype=torch.float32)
            take = min(grad_flat.numel(), s_end - s_start)
            if take > 0:
                self.param_shard.grad[s_start:s_start + take].copy_(grad_flat[:take])

    def sync_shard_to_model(self) -> None:
        """Copy updated FP32 param_shard back to model's BF16 parameters.

        Called after optimizer.step() updates param_shard. Each rank writes
        only the slice of each parameter that it owns.
        """
        lo = self.shard_offsets[self.rank]
        hi = self.shard_offsets[self.rank + 1]

        for name, p in self.param_order:
            sl = self.param_offsets[name]
            if sl.global_end <= lo or sl.global_start >= hi:
                continue
            g_start = max(sl.global_start, lo)
            g_end = min(sl.global_end, hi)
            p_start = g_start - sl.global_start
            p_end = g_end - sl.global_start
            s_start = g_start - lo
            s_end = g_end - lo
            with torch.no_grad():
                flat = p.data.reshape(-1)
                flat[p_start:p_end].copy_(
                    self.param_shard.data[s_start:s_end].to(p.dtype)
                )

class ZeRO3ForwardHook:
    """
    Layer-by-layer ZeRO-3 forward all-gather.

    Registers ``register_forward_pre_hook`` / ``register_forward_hook``
    on every ``nn.Module`` that directly owns trainable parameters. The
    pre-hook reconstructs the full BF16 params for *that layer only*
    via ``dist.all_gather_into_tensor`` (falling back to ``all_gather``
    when shard sizes vary across ranks). The post-hook restores each
    parameter's local sharded storage and frees the gathered buffer.

    Why per-layer (not all-at-once)
    -------------------------------
    Peak VRAM during forward becomes::

        model_shard + max(per_layer_full_params)

    instead of ``full_model``. For a 7B LLM with 32 transformer blocks
    that is roughly 1/32 of the model on top of the shard — making it
    feasible to train models that wouldn't otherwise fit.

    Notes
    -----
    * Only direct parameters of a module are gathered in that module's
      pre-hook (i.e. ``module._parameters``, not recursive). Each
      submodule will trigger its own hook.
    * Shared parameters (the same ``nn.Parameter`` referenced by
      multiple modules) are gathered once per forward call site; the
      restore step is keyed on ``id(parameter)`` so we always return
      the original local shard view.
    * ``world_size <= 1``: install no hooks (no-op).
    """

    def __init__(self, model: nn.Module, shard_state: "ShardState") -> None:
        self.model = model
        self.shard_state = shard_state
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # param identity (id) -> (param, original local-shard storage)
        # used as a transient per-layer save/restore map.
        self._saved: Dict[int, Tuple[nn.Parameter, torch.Tensor]] = {}
        # Map nn.Parameter id -> ParamSlice for O(1) lookup during hooks.
        self._param_to_slice: Dict[int, ParamSlice] = {}
        if shard_state is not None:
            for name, p in shard_state.param_order:
                sl = shard_state.param_offsets.get(name)
                if sl is not None:
                    self._param_to_slice[id(p)] = sl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self) -> int:
        """
        Walk the model and attach pre/post hooks to every module that
        directly owns at least one trainable, sharded parameter.

        Returns the number of modules hooked. Safe to call when
        ``world_size <= 1`` — installs nothing and returns 0.
        """
        if self.shard_state is None or self.shard_state.world_size <= 1:
            logger.info(
                "[zero3-hook] world_size<=1, skipping hook installation"
            )
            return 0

        hooked = 0
        for mod in self.model.modules():
            direct_params = [
                p for p in mod._parameters.values()
                if p is not None and id(p) in self._param_to_slice
            ]
            if not direct_params:
                continue
            pre = mod.register_forward_pre_hook(self._make_pre_hook(direct_params))
            post = mod.register_forward_hook(self._make_post_hook(direct_params))
            self._handles.extend([pre, post])
            hooked += 1

        logger.info(
            "[zero3-hook] installed forward all-gather hooks on %d modules "
            "(rank=%d/%d)",
            hooked, self.shard_state.rank, self.shard_state.world_size,
        )
        return hooked

    def remove(self) -> None:
        """Remove all installed hooks (e.g. for teardown / eval)."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        self._saved.clear()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------
    def _make_pre_hook(self, params: List[nn.Parameter]):
        def _pre(module: nn.Module, inputs):
            self._gather_params(params)
            return None
        return _pre

    def _make_post_hook(self, params: List[nn.Parameter]):
        def _post(module: nn.Module, inputs, output):
            # Register a one-shot full backward hook on the MODULE itself.
            # This fires AFTER all weight gradients for this module are
            # computed, so it's safe to release gathered params.
            # (register_hook on output tensor fires too early — before
            # weight grads are computed, causing device mismatch.)
            saved_params = list(params)
            hook_ref = [None]  # wrap in list for closure mutation

            def _module_backward_hook(module, grad_input, grad_output):
                self._release_params(saved_params)
                # Remove hook so it doesn't fire on next forward/backward
                if hook_ref[0] is not None:
                    hook_ref[0].remove()
                    hook_ref[0] = None

            hook_ref[0] = module.register_full_backward_hook(_module_backward_hook)
            return None
        return _post

    # ------------------------------------------------------------------
    # Per-layer gather / release
    # ------------------------------------------------------------------
    def _gather_params(self, params: List[nn.Parameter]) -> None:
        """All-gather each parameter in ``params`` and rewrite ``p.data``
        to view into the gathered buffer."""
        state = self.shard_state
        ws = state.world_size
        rank = state.rank
        device = state.param_shard.device

        for p in params:
            pid = id(p)
            if pid in self._saved:
                # Already gathered (re-entrant / shared param): skip.
                continue
            sl = self._param_to_slice.get(pid)
            if sl is None:
                continue

            gather_dtype = p.dtype  # typically BF16 during training
            numel = sl.global_end - sl.global_start

            # Determine each rank's contribution to this layer: the
            # intersection of [sl.global_start, sl.global_end) with that
            # rank's shard window [shard_offsets[r], shard_offsets[r+1]).
            per_rank_sizes: List[int] = []
            per_rank_local_offsets: List[Tuple[int, int]] = []  # (s_start, s_end) in our local shard
            for r in range(ws):
                lo = state.shard_offsets[r]
                hi = state.shard_offsets[r + 1]
                g0 = max(sl.global_start, lo)
                g1 = min(sl.global_end, hi)
                contrib = max(0, g1 - g0)
                per_rank_sizes.append(contrib)
                if r == rank and contrib > 0:
                    per_rank_local_offsets.append((g0 - lo, g1 - lo))
                else:
                    per_rank_local_offsets.append((0, 0))

            # This rank's slice of *this layer*, taken from param_shard.
            s_start, s_end = per_rank_local_offsets[rank]
            if s_end > s_start:
                local_piece = state.param_shard[s_start:s_end].to(gather_dtype)
            else:
                local_piece = torch.empty(0, dtype=gather_dtype, device=device)

            # Allocate the full per-layer buffer and run the collective.
            full = torch.empty(numel, dtype=gather_dtype, device=device)

            even = all(s == per_rank_sizes[0] for s in per_rank_sizes)
            if even and per_rank_sizes[0] > 0:
                # Fast path: equal contributions → all_gather_into_tensor.
                dist.all_gather_into_tensor(full, local_piece)
            else:
                # Variable-size fallback: per-rank tensors + concat.
                chunks = [
                    torch.empty(s, dtype=gather_dtype, device=device)
                    for s in per_rank_sizes
                ]
                # all_gather requires all chunks > 0 on every rank? In
                # practice torch tolerates zero-sized tensors here; if
                # not, fall back to padded all_gather_into_tensor.
                try:
                    dist.all_gather(chunks, local_piece)
                except Exception:  # noqa: BLE001
                    # Pad to max size and use the equal-size collective.
                    max_sz = max(per_rank_sizes) if per_rank_sizes else 0
                    padded_local = torch.zeros(
                        max_sz, dtype=gather_dtype, device=device
                    )
                    if local_piece.numel() > 0:
                        padded_local[:local_piece.numel()].copy_(local_piece)
                    padded_full = torch.empty(
                        max_sz * ws, dtype=gather_dtype, device=device
                    )
                    dist.all_gather_into_tensor(padded_full, padded_local)
                    for r in range(ws):
                        sz = per_rank_sizes[r]
                        if sz > 0:
                            chunks[r] = padded_full[r * max_sz : r * max_sz + sz]

                # Stitch chunks into ``full`` in rank order.
                cursor = 0
                for r in range(ws):
                    sz = per_rank_sizes[r]
                    if sz > 0:
                        full[cursor:cursor + sz].copy_(chunks[r])
                    cursor += sz

            # Save original storage and rewrite param to view into full.
            self._saved[pid] = (p, p.data)
            p.data = full.view(sl.shape)

    def _release_params(self, params: List[nn.Parameter]) -> None:
        """Restore each parameter's local sharded storage and drop the
        gathered buffer (freed when its last reference goes away)."""
        for p in params:
            pid = id(p)
            entry = self._saved.pop(pid, None)
            if entry is None:
                continue
            _, orig = entry
            p.data = orig


def install_zero3_forward_hooks(
    model: nn.Module, shard_state: Optional["ShardState"]
) -> Optional[ZeRO3ForwardHook]:
    """
    Convenience entry point used by ``DesLocEngine.train()`` to wire
    layer-by-layer all-gather hooks onto the model. Returns the hook
    manager (so callers can later ``.remove()`` it), or ``None`` when
    sharding is not active.
    """
    if shard_state is None or shard_state.world_size <= 1:
        return None
    hook = ZeRO3ForwardHook(model, shard_state)
    hook.register()
    return hook


# ---------------------------------------------------------------------------
# Heterogeneous helpers
# ---------------------------------------------------------------------------
def vram_weights_from_tiers(tiers: Sequence[object]) -> List[float]:
    """
    Build a per-rank ``vram_weights`` list from a list of ``TierSpec``
    objects. Each weight is the GPU's total memory in GB; the caller
    decides whether to pass these into ``ShardState.build`` for a
    heterogeneous split.

    Accepts any object exposing ``total_mem_gb`` and ``device_index``.
    """
    by_idx: Dict[int, float] = {}
    for t in tiers:
        idx = int(getattr(t, "device_index"))
        mem = float(getattr(t, "total_mem_gb"))
        by_idx[idx] = mem
    if not by_idx:
        return []
    # Order by device index so that rank r maps to the rth lowest GPU.
    ordered = [by_idx[k] for k in sorted(by_idx)]
    return ordered

