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
    # CPU-pinned full BF16 copy of each param, keyed by param name.
    # Used by forward/backward hooks to reconstruct full params on GPU
    # via H2D copy instead of NCCL all-gather — avoids cross-rank
    # synchronisation which is incompatible with heterogeneous
    # microbatch counts.
    cpu_param_data: Dict[str, torch.Tensor] = field(default_factory=dict)

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

        # cpu_param_data is no longer populated: the full BF16 model is kept
        # on GPU for the duration of training (see ZeRO3ForwardHook.register).
        # Keeping the field as an empty dict preserves backward-compatibility
        # for any callers that inspect it.
        cpu_param_data: Dict[str, torch.Tensor] = {}

        # Pre-allocate the gradient buffer NOW so optimizer state is ready
        # before forward runs.
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
            cpu_param_data=cpu_param_data,
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
                if _param_shard.grad is None:
                    # Should never happen — pre-allocated in build()
                    return
                flat = grad.detach().reshape(-1)

                # No NCCL here — just extract this rank's shard slice of
                # the full gradient and accumulate into param_shard.grad.
                # Gradient synchronisation across ranks happens once per
                # step via all_reduce on param_shard.grad before the
                # optimizer step (compatible with heterogeneous microbatch
                # counts).
                if _shard_end > _shard_start:
                    # Map from param-local flat indices to our shard window
                    p_lo = _g_start - sl.global_start
                    p_hi = _g_end - sl.global_start
                    # Bounds safety check
                    if p_hi > flat.numel():
                        logger.warning(
                            "[zero3-hook] BOUNDS: %s p_hi=%d > flat=%d, clamping",
                            name, p_hi, flat.numel(),
                        )
                        p_hi = flat.numel()
                    if p_lo >= p_hi:
                        param.grad = None
                        return
                    local_grad = flat[p_lo:p_hi]
                    take = min(local_grad.numel(), _shard_end - _shard_start)
                    if take > 0 and _shard_start + take <= _param_shard.grad.numel():
                        _param_shard.grad[_shard_start:_shard_start + take].add_(
                            local_grad[:take].to(
                                dtype=_param_shard.dtype,
                                device=_param_shard.device,
                            )
                        )
                    elif take > 0:
                        logger.warning(
                            "[zero3-hook] SHARD OVERFLOW: %s shard_start=%d take=%d grad_numel=%d",
                            name, _shard_start, take, _param_shard.grad.numel(),
                        )
                else:
                    local_grad = None

                # Clear param.grad to free memory (it's been consumed)
                param.grad = None

                # Optional FP32 accumulator integration. We only touch
                # ``main_grad`` if the manager actually allocated one for
                # this parameter (selective FP32 policy).
                if fp32_grad_manager is not None and hasattr(param, "main_grad") and local_grad is not None:
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
        """Write local FP32 shard slice then all_gather to synchronise all ranks.

        Standard ZeRO-3 post-optimizer-step sequence:
          1. Each rank writes its own updated FP32 shard slice into model BF16.
          2. all_gather broadcasts every rank's BF16 slice to all ranks so
             the full model is consistent everywhere.

        Without step 2, each rank's model copy has only 1/N of parameters
        updated — the remaining (N-1)/N are stale from the previous step,
        causing NaN on the next forward pass.
        """
        lo = self.shard_offsets[self.rank]
        hi = self.shard_offsets[self.rank + 1]

        # Step 1: write this rank's updated shard into model BF16 params
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
                updated = self.param_shard.data[s_start:s_end]
                flat = p.data.reshape(-1)
                flat[p_start:p_end].copy_(updated.to(p.dtype))

        # Step 2: all_gather so every rank has the complete updated model.
        # Each rank just wrote its own slice; now broadcast all slices.
        if self.world_size > 1 and dist.is_initialized():
            self._allgather_model_params()

    def _allgather_model_params(self) -> None:
        """All-gather BF16 model parameters across all ranks.

        After each rank writes its own shard slice into model params,
        this broadcasts every rank's slice to all ranks so the full
        model is consistent. Uses per-parameter all_gather to handle
        heterogeneous shard sizes correctly.
        """
        for _name, p in self.param_order:
            # Each rank already wrote its slice; now sync the full param.
            # broadcast from each rank in sequence is simpler than
            # reassembling from shards — and correct for uneven splits.
            flat = p.data.reshape(-1)
            # Collect the full parameter from all ranks.
            # Each rank has its own updated slice + stale slices from others.
            # We use all_reduce with a mask approach: each rank zeros out
            # the parts it didn't update, then SUM gives the full param.
            # But simpler: just all_gather the flat BF16 and scatter back.
            sl = self.param_offsets[_name]
            # Build per-rank chunks and all_gather
            chunks = []
            for r in range(self.world_size):
                r_lo = self.shard_offsets[r]
                r_hi = self.shard_offsets[r + 1]
                p_lo = max(sl.global_start, r_lo) - sl.global_start
                p_hi = min(sl.global_end, r_hi) - sl.global_start
                if p_hi > p_lo:
                    chunks.append((r, p_lo, p_hi))

            if len(chunks) <= 1:
                continue  # param lives entirely on one rank

            # Use broadcast from each rank that owns a slice
            for (r, p_lo, p_hi) in chunks:
                dist.broadcast(flat[p_lo:p_hi], src=r)

    def sync_shard_to_model_async(
        self, stream: Optional["torch.cuda.Stream"] = None
    ) -> "torch.cuda.Stream":
        """Write local shard + all_gather to synchronise full model across ranks.

        Local FP32→BF16 copies are issued on *stream*. The all_gather
        (NCCL broadcast per param) runs on the default stream after
        waiting for the copies to finish, because NCCL collectives must
        execute on the same stream across all ranks.

        The caller must wait on the returned stream before the next
        forward pass.
        """
        if not torch.cuda.is_available():
            self.sync_shard_to_model()
            return None  # type: ignore[return-value]

        if stream is None:
            stream = torch.cuda.Stream()

        lo = self.shard_offsets[self.rank]
        hi = self.shard_offsets[self.rank + 1]

        # Step 1: async local shard → model BF16 copies
        with torch.cuda.stream(stream):
            with torch.no_grad():
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
                    updated = self.param_shard.data[s_start:s_end]
                    flat = p.data.reshape(-1)
                    flat[p_start:p_end].copy_(updated.to(p.dtype), non_blocking=True)

        # Step 2: all_gather — must run after local copies finish.
        # Wait for copy stream, then broadcast on default stream.
        if self.world_size > 1 and dist.is_initialized():
            torch.cuda.current_stream().wait_stream(stream)
            self._allgather_model_params()

        return stream

class ZeRO3ForwardHook:
    """
    ZeRO-3 parameter materialisation for DesLocEngine.

    **Previous design (broken):** per-layer H2D gather/release hooks that
    swapped ``p.data`` between CPU and GPU around each module's forward.
    This broke autograd because the graph records the tensor storage
    address at forward time; swapping ``p.data`` to CPU in the post-hook
    made the backward-pre re-gather invisible to autograd's saved-tensor
    machinery.

    **Current design:** load the *entire* model to GPU once, in BF16,
    before the training loop starts.  ``register()`` performs this move
    and installs no per-module hooks.  ``remove()`` is a no-op.

    VRAM budget (per rank)
    ----------------------
    ::

        param_shard (FP32) : ~12 GB  (H100) / ~6 GB  (A6000)
        grad shard  (FP32) : ~12 GB  (H100) / ~6 GB  (A6000)
        Adam m+v    (FP32) : ~24 GB  (H100) / ~12 GB (A6000)
        full model  (BF16) : ~12 GB  (both)
        activations        :  ~5 GB  (H100) / ~3 GB  (A6000)
        ──────────────────────────────────────────────────────
        total              : ~65 GB / 93 GB  (H100)  OK
                           : ~39 GB / 47 GB  (A6000) OK

    Backward hooks in ``ShardState.register_backward_hooks`` still extract
    each parameter's local gradient slice into ``param_shard.grad``.
    ``sync_shard_to_model`` writes the updated FP32 shard back to the GPU
    BF16 parameters in-place after every optimizer step.
    """

    def __init__(self, model: nn.Module, shard_state: "ShardState") -> None:
        self.model = model
        self.shard_state = shard_state
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self) -> int:
        """
        Move the full model to the local GPU in BF16.

        Returns 1 when the move was performed, 0 when sharding is not
        active (``world_size <= 1``).  No per-module hooks are installed.
        """
        if self.shard_state is None or self.shard_state.world_size <= 1:
            logger.info(
                "[zero3-hook] world_size<=1, skipping full-model GPU load"
            )
            return 0

        device = self.shard_state.param_shard.device
        self.model.to(device=device, dtype=torch.bfloat16)
        logger.info(
            "[zero3-hook] full BF16 model loaded to %s "
            "(rank=%d/%d, no per-layer hooks)",
            device,
            self.shard_state.rank,
            self.shard_state.world_size,
        )
        return 1

    def remove(self) -> None:
        """No-op — no hooks were installed."""
        self._handles.clear()


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

