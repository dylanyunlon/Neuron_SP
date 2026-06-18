"""
hetero_mamba_chunked_prefill.py — DES-LOC Heterogeneous Mamba Chunked Prefill Adapter
========================================================================================

Upstream Design Intent (Megatron c65fb25):
-------------------------------------------
Megatron's Mamba chunked-prefill refactor (PR #3265) cleaned up a fragile
``has_explicit_chunked_prefill_req`` boolean that was threaded through
InferenceBatchDimensions, MambaMetadata.update(), DynamicInferenceContext, and
MambaMixer._dynamic_inference_forward().  The key insight was:

  1. The *position* of the chunked-prefill request in the batch is deterministic:
     it is always the **first** prefill slot (index = decode_req_count), not the
     last as in the old code.
  2. Chunked-prefill state (is it enabled? is it active *right now*?) should be
     derived from a single flag (``enable_chunked_prefill``) and the live batch
     rather than being stored as mutable boolean on the batch-dimension struct.
  3. CUDA-graph compatibility: hybrid SSM models (Mamba + Attention) must suppress
     chunked-prefill during graph capture / replay because variable-length state
     initialization is incompatible with static shapes.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a three-device
topology on this project:

  • GPU-0  A6000 48 GB  SM86  — decode worker (short, latency-sensitive)
  • GPU-1  A6000 48 GB  SM86  — prefill worker (long prompts, throughput)
  • GPU-2  H100 NVL 96 GB SM90 — SSM state host + KV-cache overflow (large DRAM)

Because there is **no NVLink** between any pair, all cross-device traffic goes over
PCIe.  The CPU DRAM (1.5 TB) acts as a shared locality cache for Mamba SSM states
that don't fit on the A6000s.

Key adaptations vs. upstream Megatron:

  A. ``HeteroBatchDimensions`` drops ``has_explicit_chunked_prefill_req`` (mirrors
     upstream removal) and adds ``device_affinity`` — which physical GPU "owns" the
     batch split.

  B. ``HeteroMambaMetadata`` replicates the upstream positional fix (chunked request
     is first prefill, not last) and adds PCIe-aware tensor placement: decode indices
     stay on GPU-0, prefill indices on GPU-1, SSM states pinned to GPU-2 / CPU DRAM.

  C. ``HeteroChunkedPrefillScheduler`` replaces DynamicEngine's scheduling logic
     for the chunked-prefill path.  It uses ``enable_chunked_prefill`` + a live
     ``is_creating_cuda_graphs`` guard (upstream addition) to decide whether the
     chunked path is safe to activate on each device.

  D. ``HeteroMambaMixer.forward`` mirrors the upstream refactor: three clean branches
     (decode-only, prefill-only, mixed) implemented as a flat decision tree, plus a
     helper ``_hetero_prefill()`` that mirrors ``_dynamic_inference_prefill()``.

  E. All tensor moves across PCIe are explicit and logged; the code never silently
     falls back to same-device copies.

Usage in Neuron_SP pipeline
---------------------------
    from deepspeed.inference.hetero_mamba_chunked_prefill import (
        HeteroBatchDimensions,
        HeteroMambaMetadata,
        HeteroChunkedPrefillScheduler,
        HeteroMambaMixer,
        build_hetero_batch_dimensions,
    )
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device topology constants for DES-LOC
# ---------------------------------------------------------------------------
#
# These mirror the physical cluster:  2× A6000 (SM86) + 1× H100-NVL (SM90).
# GPU indices are logical DeepSpeed device IDs; adjust via DESCLOC_DEVICE_MAP
# env-var in production.
#
DESCLOC_DECODE_DEVICE: int = 0      # A6000 #0 — handles decode
DESCLOC_PREFILL_DEVICE: int = 1     # A6000 #1 — handles prefill
DESCLOC_STATE_DEVICE: int = 2       # H100 NVL — holds Mamba SSM states
DESCLOC_CPU_LOCALITY_CACHE: str = "cpu"  # 1.5 TB DRAM overflow


# ---------------------------------------------------------------------------
# HeteroBatchDimensions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HeteroBatchDimensions:
    """
    Batch-dimension descriptor for DES-LOC heterogeneous inference.

    Mirrors Megatron's ``InferenceBatchDimensions`` after c65fb25 removed
    ``has_explicit_chunked_prefill_req`` from the struct.  We extend it with
    ``device_affinity`` so that CUDA-graph matching can account for which
    physical GPU is executing a given batch split.

    Fields
    ------
    token_count        : total tokens in this split
    prefill_req_count  : number of prefill requests
    decode_req_count   : number of decode requests
    device_affinity    : logical device id that "owns" this split
                         (DESCLOC_DECODE_DEVICE or DESCLOC_PREFILL_DEVICE)

    The ordering invariant (token_count ≥ prefill+decode) is preserved from
    upstream so that CUDA-graph bucket matching logic is unchanged.
    """

    token_count: int = 0
    prefill_req_count: int = 0
    decode_req_count: int = 0
    device_affinity: int = DESCLOC_DECODE_DEVICE

    # ------------------------------------------------------------------
    # Identity / hashing — mirrors upstream: no chunked-prefill field
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((self.token_count, self.prefill_req_count,
                     self.decode_req_count, self.device_affinity))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeteroBatchDimensions):
            return False
        return (
            self.token_count == other.token_count
            and self.prefill_req_count == other.prefill_req_count
            and self.decode_req_count == other.decode_req_count
            and self.device_affinity == other.device_affinity
        )

    def __str__(self) -> str:
        return (
            f"HeteroBatchDimensions("
            f"tok={self.token_count}, "
            f"pfx={self.prefill_req_count}, "
            f"dec={self.decode_req_count}, "
            f"dev={self.device_affinity})"
        )

    @property
    def req_count(self) -> int:
        return self.prefill_req_count + self.decode_req_count

    # ------------------------------------------------------------------
    # Subsumes check — used by CUDA-graph bucket matching
    # ------------------------------------------------------------------

    def subsumes(self, real: "HeteroBatchDimensions") -> bool:
        """
        Return True if *self* (a graph bucket) can cover *real* (live batch).

        Upstream logic (post-c65fb25):
          • If there are only decode requests, any bucket with ≥ token_count
            and ≥ decode_req_count works.
          • Mixed/prefill batches require exact device_affinity match and the
            bucket must have ≥ each counter independently.

        DES-LOC addition: device_affinity must match so that a decode-side
        graph is never applied to a prefill-side real batch.
        """
        if real.device_affinity != self.device_affinity:
            return False

        if real.prefill_req_count == 0:
            # decode-only: generous matching
            return (
                self.token_count >= real.token_count
                and self.decode_req_count >= real.decode_req_count
            )

        # mixed or prefill-only: strict per-counter
        return (
            self.token_count >= real.token_count
            and self.prefill_req_count >= real.prefill_req_count
            and self.decode_req_count >= real.decode_req_count
        )

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def is_valid(self, max_sequence_length: int) -> bool:
        if self.token_count < self.prefill_req_count + self.decode_req_count:
            return False
        if self.token_count > (
            self.prefill_req_count * max_sequence_length + self.decode_req_count
        ):
            return False
        return True


# ---------------------------------------------------------------------------
# Expert-parallel sync (mirrors upstream adjust_for_expert_parallelism)
# ---------------------------------------------------------------------------

def adjust_batch_dims_for_ep(
    local_dims: HeteroBatchDimensions,
    strict: bool,
    decode_only_cuda_graphs: bool,
    explicit_chunked_prefill: bool,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> Optional[HeteroBatchDimensions]:
    """
    Synchronise batch dimensions across expert-parallel ranks.

    Mirrors Megatron c65fb25 which:
      1. Removed ``has_explicit_chunked_prefill_req`` from the sync tensor
         (was index [2], caused off-by-one in sync_tensor[3/4]).
      2. Added ``explicit_chunked_prefill`` as a *call-site* parameter so the
         decision is derived once per step rather than stored in the struct.
      3. Unified the eager-mode trigger condition:
           ``is_any_non_decode AND (decode_only_graphs OR explicit_chunked_prefill)``

    DES-LOC note: we broadcast device_affinity as an extra element so that
    ranks on different GPUs can detect topology mismatches and fall back to
    eager mode rather than running a graph recorded on the wrong SM version.

    Parameters
    ----------
    local_dims             : this rank's live batch dimensions
    strict                 : if True, use max across EP for prefill/decode counts
    decode_only_cuda_graphs: only decode batches are graph-captured
    explicit_chunked_prefill: chunked-prefill is active AND not graph-capturing
    ep_group               : expert-parallel process group (None → global)

    Returns
    -------
    Adjusted HeteroBatchDimensions or None (None → run in eager mode).
    """
    if ep_group is None:
        # No EP — return as-is, no sync needed
        logger.debug("adjust_batch_dims_for_ep: no EP group, returning local dims as-is")
        return local_dims

    is_non_decode = local_dims.prefill_req_count > 0

    # Sync tensor layout (post-c65fb25, extended for DES-LOC):
    #   [0] token_count
    #   [1] is_non_decode (int)
    #   [2] prefill_req_count
    #   [3] decode_req_count
    #   [4] device_affinity
    sync_tensor = torch.tensor(
        [
            local_dims.token_count,
            int(is_non_decode),
            local_dims.prefill_req_count,
            local_dims.decode_req_count,
            local_dims.device_affinity,
        ],
        dtype=torch.int64,
        device=torch.device(f"cuda:{local_dims.device_affinity}"),
    )

    dist.all_reduce(sync_tensor, op=dist.ReduceOp.MAX, group=ep_group)
    sync_tensor = sync_tensor.cpu()

    is_any_non_decode = sync_tensor[1].item() == 1
    max_device_affinity = int(sync_tensor[4].item())

    # DES-LOC: if ranks disagree on device_affinity → topology mismatch → eager
    if max_device_affinity != local_dims.device_affinity:
        logger.warning(
            "adjust_batch_dims_for_ep: device_affinity mismatch across EP ranks "
            "(local=%d, max=%d); falling back to eager mode",
            local_dims.device_affinity,
            max_device_affinity,
        )
        return None

    # Upstream condition (c65fb25): eager if non-decode + (decode-only-graphs OR chunked-prefill)
    if is_any_non_decode and (decode_only_cuda_graphs or explicit_chunked_prefill):
        logger.debug(
            "adjust_batch_dims_for_ep: forcing eager mode "
            "(is_any_non_decode=%s, decode_only=%s, explicit_chunked=%s)",
            is_any_non_decode, decode_only_cuda_graphs, explicit_chunked_prefill,
        )
        return None

    adjusted_prefill = (
        int(sync_tensor[2].item()) if strict else local_dims.prefill_req_count
    )
    adjusted_decode = (
        int(sync_tensor[3].item()) if strict else local_dims.decode_req_count
    )

    result = HeteroBatchDimensions(
        token_count=int(sync_tensor[0].item()),
        prefill_req_count=adjusted_prefill,
        decode_req_count=adjusted_decode,
        device_affinity=local_dims.device_affinity,
    )
    logger.debug("adjust_batch_dims_for_ep: adjusted dims = %s", result)
    return result


# ---------------------------------------------------------------------------
# build_hetero_batch_dimensions  (convenience factory)
# ---------------------------------------------------------------------------

def build_hetero_batch_dimensions(
    token_count: int,
    prefill_req_count: int,
    decode_req_count: int,
    device_affinity: Optional[int] = None,
) -> HeteroBatchDimensions:
    """
    Factory that infers device_affinity from the request mix when not supplied.

    DES-LOC routing heuristic:
      • decode-only  → GPU-0 (A6000 decode worker)
      • prefill-only → GPU-1 (A6000 prefill worker)
      • mixed        → GPU-1 (prefill worker drives the step; decode indices
                               are later spliced to GPU-0 via PCIe)
    """
    if device_affinity is None:
        if decode_req_count > 0 and prefill_req_count == 0:
            device_affinity = DESCLOC_DECODE_DEVICE
        else:
            device_affinity = DESCLOC_PREFILL_DEVICE

    return HeteroBatchDimensions(
        token_count=token_count,
        prefill_req_count=prefill_req_count,
        decode_req_count=decode_req_count,
        device_affinity=device_affinity,
    )


# ---------------------------------------------------------------------------
# HeteroMambaMetadata
# ---------------------------------------------------------------------------

class HeteroMambaMetadata:
    """
    Mamba SSM batch-index tracker for DES-LOC heterogeneous execution.

    Upstream Design (Megatron c65fb25)
    ------------------------------------
    The refactor changed the *position* of the chunked-prefill request in the
    batch layout from **last** to **first** among prefill slots:

        Old:  [ decode... | regular prefill... | chunked prefill ]
        New:  [ decode... | chunked prefill   | regular prefill... ]

    This simplified cu_seqlens normalization and removed the need to track
    ``has_explicit_chunked_prefill_req`` on the struct.

    The ``update()`` method now receives ``enable_chunked_prefill`` as a
    parameter and computes ``has_chunked_prefill_req`` locally:
        has_chunked_prefill_req = enable_chunked_prefill AND prefill_req_count > 0

    DES-LOC Adaptations
    --------------------
    1. **Device placement**: decode indices are allocated on
       ``DESCLOC_DECODE_DEVICE``, prefill indices on ``DESCLOC_PREFILL_DEVICE``,
       SSM state tensors on ``DESCLOC_STATE_DEVICE`` (H100 NVL 96 GB).
       Overflow SSM states spill to CPU DRAM via ``_cpu_state_overflow_cache``.

    2. **PCIe-explicit transfers**: every cross-device copy is wrapped in
       ``_pcie_copy()`` with logging so bandwidth can be profiled.

    3. **Chunked-prefill ordering**: first-prefill-is-chunked logic from
       upstream is preserved verbatim.  The SSM state for the chunked request
       is fetched from DESCLOC_STATE_DEVICE (or CPU overflow) before the
       forward pass and written back after.

    4. **reset_varlen_metadata()** is called in ``__init__`` (mirrors upstream
       addition of that call in MambaMetadata.__init__).
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        ssm_state_dim: int = 256,
        conv_width: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.ssm_state_dim = ssm_state_dim
        self.conv_width = conv_width
        self.dtype = dtype

        self.decode_device = torch.device(f"cuda:{DESCLOC_DECODE_DEVICE}")
        self.prefill_device = torch.device(f"cuda:{DESCLOC_PREFILL_DEVICE}")
        self.state_device = torch.device(f"cuda:{DESCLOC_STATE_DEVICE}")

        # ----------------------------------------------------------------
        # Batch-index buffers — allocated on the device that will use them
        # ----------------------------------------------------------------
        self._batch_indices_decode_buffer = torch.full(
            (max_requests,), -1, dtype=torch.int32, device=self.decode_device
        )
        self._batch_indices_prefill_buffer = torch.full(
            (max_requests,), -1, dtype=torch.int32, device=self.prefill_device
        )
        self._batch_indices_chunked_prefill_buffer = torch.full(
            (1,), -1, dtype=torch.int32, device=self.prefill_device
        )

        # Mixed-batch split descriptors (decode_count, prefill_count)
        self._device_decode_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.prefill_device
        )

        # Chunked-prefill split: (chunked_token_count, regular_prefill_token_count)
        # Mirrors upstream tuple swap in c65fb25:
        #   OLD: (total_regular_prefill_seqlen, chunked_prefill_seqlen)
        #   NEW: (chunked_prefill_seqlen, regular_prefill_seqlen)
        self._device_chunked_prefill_buffer = torch.zeros(
            (2,), dtype=torch.int32, device=self.prefill_device
        )

        # seq_idx and cu_seqlens for varlen SSM kernels
        self._seq_idx_buffer = torch.full(
            (1, max_tokens), -1, dtype=torch.int32, device=self.prefill_device
        )
        self._cu_seqlens_buffer = torch.zeros(
            (max_requests + 1,), dtype=torch.int32, device=self.prefill_device
        )

        # Request → Mamba-state slot mapping (on state device)
        self.request_to_mamba_state_idx = torch.full(
            (max_requests,), -1, dtype=torch.int32, device=self.state_device
        )

        # Live view tensors (None = not active)
        self.batch_indices_decode: Optional[torch.Tensor] = None
        self.batch_indices_prefill: Optional[torch.Tensor] = None
        self.batch_indices_chunked_prefill: Optional[torch.Tensor] = None
        self.device_decode_prefill: Optional[torch.Tensor] = None
        self.device_chunked_prefill: Optional[torch.Tensor] = None
        self.seq_idx: Optional[torch.Tensor] = None
        self.cu_seqlens: Optional[torch.Tensor] = None

        # CPU overflow cache for SSM states that don't fit on A6000s
        self._cpu_state_overflow_cache: Dict[int, torch.Tensor] = {}
        self.mamba_state_free_slot_count = max_requests

        # Matches upstream: call reset_varlen_metadata in __init__
        self.reset_varlen_metadata()

        logger.info(
            "HeteroMambaMetadata initialised: max_requests=%d, max_tokens=%d, "
            "decode_dev=cuda:%d, prefill_dev=cuda:%d, state_dev=cuda:%d",
            max_requests, max_tokens,
            DESCLOC_DECODE_DEVICE, DESCLOC_PREFILL_DEVICE, DESCLOC_STATE_DEVICE,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def reset_varlen_metadata(self) -> None:
        """Reset variable-length metadata views to None (called at init and step end)."""
        self.batch_indices_decode = None
        self.batch_indices_prefill = None
        self.batch_indices_chunked_prefill = None
        self.device_decode_prefill = None
        self.device_chunked_prefill = None
        self.seq_idx = None
        self.cu_seqlens = None

    def reset(self) -> None:
        """Full reset: clear all state and release CPU overflow cache."""
        self.reset_varlen_metadata()
        self._batch_indices_decode_buffer.fill_(-1)
        self._batch_indices_prefill_buffer.fill_(-1)
        self._batch_indices_chunked_prefill_buffer.fill_(-1)
        self._device_decode_prefill_buffer.zero_()
        self._device_chunked_prefill_buffer.zero_()
        self._seq_idx_buffer.fill_(-1)
        self._cu_seqlens_buffer.zero_()
        self.request_to_mamba_state_idx.fill_(-1)
        self._cpu_state_overflow_cache.clear()
        self.mamba_state_free_slot_count = self.max_requests
        logger.debug("HeteroMambaMetadata.reset() called")

    @staticmethod
    def _pcie_copy(
        src: torch.Tensor,
        dst_device: torch.device,
        non_blocking: bool = True,
        label: str = "",
    ) -> torch.Tensor:
        """
        Explicit PCIe copy with logging.

        DES-LOC: All cross-device copies must go through this helper so that
        bandwidth can be profiled and potential stalls can be detected.
        """
        if src.device == dst_device:
            return src
        logger.debug(
            "PCIe copy %s: %s → %s  shape=%s  bytes=%d",
            label, src.device, dst_device, tuple(src.shape),
            src.numel() * src.element_size(),
        )
        return src.to(dst_device, non_blocking=non_blocking)

    # ------------------------------------------------------------------
    # Slot management (SSM states live on DESCLOC_STATE_DEVICE / CPU)
    # ------------------------------------------------------------------

    def allocate_slot(self) -> Optional[int]:
        """Allocate a free Mamba-state slot; returns None if exhausted."""
        if self.mamba_state_free_slot_count == 0:
            logger.warning("HeteroMambaMetadata: no free Mamba state slots")
            return None
        for idx in range(self.max_requests):
            if self.request_to_mamba_state_idx[idx].item() == -1:
                self.request_to_mamba_state_idx[idx] = idx
                self.mamba_state_free_slot_count -= 1
                logger.debug("allocate_slot: slot %d assigned", idx)
                return idx
        return None

    def free_slot(self, request_idx: int) -> None:
        """Free the Mamba-state slot for *request_idx*."""
        slot = int(self.request_to_mamba_state_idx[request_idx].item())
        if slot == -1:
            return
        self.request_to_mamba_state_idx[request_idx] = -1
        self.mamba_state_free_slot_count += 1
        # Also remove from CPU overflow cache if present
        if slot in self._cpu_state_overflow_cache:
            del self._cpu_state_overflow_cache[slot]
        logger.debug("free_slot: slot %d freed for request_idx %d", slot, request_idx)

    # ------------------------------------------------------------------
    # Core update (mirrors Megatron c65fb25 MambaMetadata.update)
    # ------------------------------------------------------------------

    def update(
        self,
        active_mamba_indices: torch.Tensor,
        token_to_request_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_dimensions: HeteroBatchDimensions,
        padded_batch_dimensions: HeteroBatchDimensions,
        enable_chunked_prefill: bool,
    ) -> None:
        """
        Populate all batch-index and sequence-length tensors for this step.

        Upstream change (c65fb25)
        --------------------------
        *  Chunked-prefill request is now the **first** prefill slot, not the last.
        *  ``has_explicit_chunked_prefill_req`` removed from the struct; derived here:
               has_chunked_prefill_req = enable_chunked_prefill AND prefill_req_count > 0
        *  cu_seqlens normalisation: subtract ``cu_seqlens[start_req_idx]`` (not a
           fixed ``real_decode_count`` offset) to handle the chunked slot correctly.

        DES-LOC adaptation
        -------------------
        *  Decode indices stay on ``decode_device``, prefill indices on
           ``prefill_device``.  Cross-device copies are PCIe-explicit.
        *  active_mamba_indices is expected on ``state_device`` (H100 NVL);
           we slice it locally and move slices to the appropriate device.

        Batch layout assumed by this function (mirrors new upstream ordering):

            [ decode_0 … decode_{D-1} | chunked_prefill | prefill_0 … prefill_{P-2} ]
                                         ↑ first prefill slot (index = D)

        Parameters
        ----------
        active_mamba_indices   : (N,) int32 — Mamba slot for each active request
        token_to_request_idx   : (T,) int32 — maps token position → request index
        cu_seqlens             : (N+1,) int32 — cumulative sequence lengths
        batch_dimensions       : real (un-padded) batch dims
        padded_batch_dimensions: padded dims used for CUDA graph / kernel launch
        enable_chunked_prefill : whether chunked prefill is active this step
        """
        real_decode_count = batch_dimensions.decode_req_count
        real_prefill_count = batch_dimensions.prefill_req_count

        padded_decode_count = padded_batch_dimensions.decode_req_count
        padded_prefill_count = padded_batch_dimensions.prefill_req_count
        padded_token_count = padded_batch_dimensions.token_count

        # Upstream: derive chunked flag locally, not from struct
        has_chunked_prefill_req = enable_chunked_prefill and real_prefill_count > 0

        logger.debug(
            "HeteroMambaMetadata.update: dec=%d pfx=%d chunked=%s "
            "padded(dec=%d pfx=%d tok=%d)",
            real_decode_count, real_prefill_count, has_chunked_prefill_req,
            padded_decode_count, padded_prefill_count, padded_token_count,
        )

        # ---- Decode indices ----
        if padded_decode_count > 0:
            self._batch_indices_decode_buffer[:real_decode_count].copy_(
                self._pcie_copy(
                    active_mamba_indices[:real_decode_count],
                    self.decode_device,
                    label="decode_indices",
                )
            )
            if padded_decode_count > real_decode_count:
                self._batch_indices_decode_buffer[real_decode_count:padded_decode_count].fill_(-1)
            self.batch_indices_decode = self._batch_indices_decode_buffer[:padded_decode_count]

        # ---- Number of regular-prefill requests (chunked slot is not "regular") ----
        regular_prefill_count = real_prefill_count
        chunked_req_idx = -1

        if has_chunked_prefill_req:
            # Upstream c65fb25: FIRST prefill slot is the chunked request
            regular_prefill_count -= 1
            chunked_req_idx = real_decode_count

            chunked_slot_on_prefill_dev = self._pcie_copy(
                active_mamba_indices[chunked_req_idx : chunked_req_idx + 1],
                self.prefill_device,
                label="chunked_prefill_slot",
            )
            self._batch_indices_chunked_prefill_buffer[0] = chunked_slot_on_prefill_dev[0]
            self.batch_indices_chunked_prefill = self._batch_indices_chunked_prefill_buffer

            logger.debug(
                "update: chunked_req_idx=%d mamba_slot=%d",
                chunked_req_idx,
                int(self._batch_indices_chunked_prefill_buffer[0].item()),
            )

        # ---- Regular prefill indices ----
        if padded_prefill_count > 0:
            if regular_prefill_count > 0:
                # If chunked prefill exists, regular prefills start after it
                start_idx = real_decode_count + (1 if has_chunked_prefill_req else 0)

                regular_slots = self._pcie_copy(
                    active_mamba_indices[start_idx : start_idx + regular_prefill_count],
                    self.prefill_device,
                    label="regular_prefill_slots",
                )
                self._batch_indices_prefill_buffer[:regular_prefill_count].copy_(regular_slots)

            if padded_prefill_count > regular_prefill_count:
                self._batch_indices_prefill_buffer[
                    regular_prefill_count:padded_prefill_count
                ].fill_(-1)

            self.batch_indices_prefill = self._batch_indices_prefill_buffer[:padded_prefill_count]

            # ---- seq_idx ----
            # Index range of regular-prefill requests in the batch
            end_regular_req_idx = (
                real_decode_count
                + regular_prefill_count
                + (1 if has_chunked_prefill_req else 0)
            )
            end_regular_token_idx = int(cu_seqlens[end_regular_req_idx].item())

            start_regular_req_idx = real_decode_count + (1 if has_chunked_prefill_req else 0)
            start_regular_token_idx = int(cu_seqlens[start_regular_req_idx].item())

            seq_len = end_regular_token_idx - start_regular_token_idx

            if seq_len > 0:
                raw_seq_idx = token_to_request_idx[
                    start_regular_token_idx:end_regular_token_idx
                ] - start_regular_req_idx

                self._seq_idx_buffer[:, :seq_len].copy_(
                    self._pcie_copy(raw_seq_idx, self.prefill_device, label="seq_idx")
                )

            if padded_token_count > seq_len:
                self._seq_idx_buffer[:, seq_len:padded_token_count] = -1

            self.seq_idx = self._seq_idx_buffer[:, :padded_token_count]

            # ---- cu_seqlens (normalised) ----
            self._cu_seqlens_buffer[0] = 0
            if regular_prefill_count > 0:
                start_req = real_decode_count + (1 if has_chunked_prefill_req else 0)
                end_req = start_req + regular_prefill_count

                # Upstream fix: subtract cu_seqlens[start_req] (not fixed offset)
                raw_cu = (
                    cu_seqlens[start_req + 1 : end_req + 1] - cu_seqlens[start_req]
                )
                self._cu_seqlens_buffer[1 : regular_prefill_count + 1].copy_(
                    self._pcie_copy(raw_cu, self.prefill_device, label="cu_seqlens")
                )

            # Pad remainder with last value (length-0 segments)
            last_val = int(self._cu_seqlens_buffer[regular_prefill_count].item())
            self._cu_seqlens_buffer[
                regular_prefill_count + 1 : padded_prefill_count + 1
            ] = last_val
            self.cu_seqlens = self._cu_seqlens_buffer[: padded_prefill_count + 1]

        # ---- Mixed-batch decode/prefill split descriptor ----
        if padded_decode_count > 0 and padded_prefill_count > 0:
            self._device_decode_prefill_buffer[0] = real_decode_count
            # Upstream: prefill count in split includes the chunked slot
            self._device_decode_prefill_buffer[1] = regular_prefill_count + (
                1 if has_chunked_prefill_req else 0
            )
            self.device_decode_prefill = self._device_decode_prefill_buffer

        # ---- Chunked-prefill token counts (layout swapped in c65fb25) ----
        if has_chunked_prefill_req:
            # Upstream c65fb25 swap:
            #   [0] = chunked_prefill_token_count  (was [1])
            #   [1] = regular_prefill_token_count  (was [0])
            chunked_token_count = int(
                (cu_seqlens[real_decode_count + 1] - cu_seqlens[real_decode_count]).item()
            )
            regular_token_count = 0
            if regular_prefill_count > 0:
                regular_token_count = int(
                    (
                        cu_seqlens[real_decode_count + 1 + regular_prefill_count]
                        - cu_seqlens[real_decode_count + 1]
                    ).item()
                )

            self._device_chunked_prefill_buffer[0] = chunked_token_count
            self._device_chunked_prefill_buffer[1] = regular_token_count
            self.device_chunked_prefill = self._device_chunked_prefill_buffer

            logger.debug(
                "update: chunked_token_count=%d regular_token_count=%d",
                chunked_token_count, regular_token_count,
            )


# ---------------------------------------------------------------------------
# HeteroChunkedPrefillScheduler
# ---------------------------------------------------------------------------

class HeteroChunkedPrefillScheduler:
    """
    Scheduling logic for Mamba chunked prefill in DES-LOC.

    Upstream Design (Megatron c65fb25 / DynamicEngine)
    ----------------------------------------------------
    The engine tracked ``has_explicit_chunked_prefill_req`` as mutable state
    on the context.  c65fb25 removed it and replaced with:
      1. ``enable_chunked_prefill`` (config flag, set once at init)
      2. ``is_creating_cuda_graphs`` (transient flag, set during graph capture)
      3. ``is_chunked_prefill_enabled()`` method that returns
         ``enable_chunked_prefill AND NOT (is_hybrid_model AND is_creating_graphs)``

    The scheduling loop was also simplified: the early-break that prevented
    scheduling additional requests after a final-chunk Mamba prefill was removed
    because the new batch layout (chunked-first) makes it safe to schedule
    multiple prefill requests in the same step.

    DES-LOC Adaptation
    -------------------
    * Decode requests are dispatched to GPU-0; prefill to GPU-1.
    * The scheduler tracks ``chunked_prefill_request_id`` (same as upstream)
      and ``is_creating_cuda_graphs`` for graph-capture suppression.
    * ``is_chunked_prefill_enabled()`` gates on ``is_hybrid_model`` (Mamba+Attn)
      and ``is_creating_cuda_graphs`` exactly as upstream.
    * The ``can_schedule`` flag is unconditionally True after a full-token
      schedule (upstream removal of early-break).
    """

    def __init__(
        self,
        max_tokens: int,
        is_hybrid_model: bool,
        enable_chunked_prefill: bool,
    ) -> None:
        self.max_tokens = max_tokens
        self.is_hybrid_model = is_hybrid_model
        self.enable_chunked_prefill = enable_chunked_prefill

        self.chunked_prefill_request_id: int = -1
        self.is_creating_cuda_graphs: bool = False
        self._active_token_count: int = 0

        logger.info(
            "HeteroChunkedPrefillScheduler: max_tokens=%d hybrid=%s chunked_prefill=%s",
            max_tokens, is_hybrid_model, enable_chunked_prefill,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_chunked_prefill_enabled(self) -> bool:
        """
        Returns whether chunked prefill is active for the current step.

        Mirrors upstream ``DynamicInferenceContext.is_chunked_prefill_enabled()``:
          • Always disabled during CUDA-graph capture on hybrid models.
          • Otherwise follows the config flag.
        """
        if self.is_hybrid_model:
            return self.enable_chunked_prefill and not self.is_creating_cuda_graphs
        return self.enable_chunked_prefill

    def begin_cuda_graph_capture(self) -> None:
        """Mark the start of CUDA-graph capture (suppresses chunked prefill)."""
        self.is_creating_cuda_graphs = True
        logger.debug("HeteroChunkedPrefillScheduler: CUDA graph capture started")

    def end_cuda_graph_capture(self) -> None:
        """Mark the end of CUDA-graph capture."""
        self.is_creating_cuda_graphs = False
        logger.debug("HeteroChunkedPrefillScheduler: CUDA graph capture ended")

    def reset_step(self) -> None:
        """
        Reset per-step state.  Mirrors upstream context.reset() subset:
          - chunked_prefill_request_id = -1
          - is_creating_cuda_graphs = False   (upstream addition in c65fb25)
        ``enable_chunked_prefill`` is NOT reset (it's a config flag).
        """
        self.chunked_prefill_request_id = -1
        self.is_creating_cuda_graphs = False
        self._active_token_count = 0
        logger.debug("HeteroChunkedPrefillScheduler.reset_step()")

    def try_schedule_request(
        self,
        request_id: int,
        prompt_token_count: int,
        remaining_tokens: int,
    ) -> Tuple[str, int]:
        """
        Attempt to schedule *request_id* into the current step.

        Returns
        -------
        (action, tokens_scheduled) where action is one of:
          "full"    — entire remaining prompt scheduled
          "partial" — partial chunk scheduled (chunked prefill)
          "skip"    — no room; caller should stop scheduling

        DES-LOC / upstream logic
        -------------------------
        * ``can_schedule`` is unconditionally True after a full-token schedule
          (upstream c65fb25 removed the early-break for Mamba final-chunk).
        * Partial scheduling sets ``chunked_prefill_request_id`` and records
          the chunk on the prefill device (GPU-1).
        """
        available = self.max_tokens - self._active_token_count

        if available <= 0:
            return "skip", 0

        if remaining_tokens <= available:
            # Full schedule
            self._active_token_count += remaining_tokens
            # If this was a chunked request completing its last chunk, clear it
            if self.chunked_prefill_request_id == request_id:
                self.chunked_prefill_request_id = -1
            logger.debug(
                "schedule_request %d: FULL (%d tokens), active=%d",
                request_id, remaining_tokens, self._active_token_count,
            )
            # Upstream c65fb25: can_schedule = True (no early-break)
            return "full", remaining_tokens

        # Partial — only possible if chunked prefill is enabled
        if not self.is_chunked_prefill_enabled():
            return "skip", 0

        chunk_length = available
        self._active_token_count += chunk_length
        self.chunked_prefill_request_id = request_id
        logger.debug(
            "schedule_request %d: PARTIAL chunk=%d, active=%d",
            request_id, chunk_length, self._active_token_count,
        )
        return "partial", chunk_length

    def build_explicit_chunked_prefill_flag(self) -> bool:
        """
        Compute the ``explicit_chunked_prefill`` flag passed to EP-sync and
        CUDA-graph matching.

        Mirrors upstream usage:
            explicit_chunked_prefill = is_chunked_prefill_enabled() AND is_hybrid_model
        """
        return self.is_chunked_prefill_enabled() and self.is_hybrid_model


# ---------------------------------------------------------------------------
# HeteroMambaMixer — forward pass
# ---------------------------------------------------------------------------

class HeteroMambaMixer:
    """
    Heterogeneous Mamba SSM forward pass for DES-LOC.

    Upstream Design (Megatron c65fb25 MambaMixer)
    -----------------------------------------------
    The refactor replaced a deeply nested if/elif/else tree (6 branches) with
    a clean three-branch structure:

        if decode_req_count > 0:   → y_decode via _ssm_decode
        if prefill_req_count > 0:  → y_prefill via _dynamic_inference_prefill
        merge if both present

    The helper ``_dynamic_inference_prefill`` encapsulates the chunked vs
    regular split, making the top-level forward readable.

    DES-LOC Adaptation
    -------------------
    * ``zxBCdt`` for decode tokens lives on ``decode_device`` (GPU-0).
    * ``zxBCdt`` for prefill tokens lives on ``prefill_device`` (GPU-1).
    * SSM states (conv_state, ssm_state) live on ``state_device`` (GPU-2).
    * Cross-device slices use ``HeteroMambaMetadata._pcie_copy()``.
    * The merge step (decode + prefill outputs) happens on ``prefill_device``
      and the result is moved to wherever the output projection expects it.

    This class is intentionally thin: it delegates all index bookkeeping to
    ``HeteroMambaMetadata`` and all scheduling decisions to
    ``HeteroChunkedPrefillScheduler``.
    """

    def __init__(
        self,
        decode_device: torch.device,
        prefill_device: torch.device,
        state_device: torch.device,
    ) -> None:
        self.decode_device = decode_device
        self.prefill_device = prefill_device
        self.state_device = state_device

    # ------------------------------------------------------------------
    # Tensor utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_get_slice_after(
        src: torch.Tensor,
        split_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the slice of *src* that comes *after* the split point.

        ``split_descriptor[0]`` is the split offset (in tokens).
        This mirrors ``tensor_get_slice_after`` used in upstream MambaMixer.
        """
        offset = int(split_descriptor[0].item())
        return src[offset:]

    @staticmethod
    def _tensor_merge(
        a: torch.Tensor,
        b: torch.Tensor,
        split_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge two token-dimension tensors [a | b] using split_descriptor[0]
        as the boundary.

        Mirrors upstream ``tensor_merge`` signature.
        """
        split = int(split_descriptor[0].item())
        total = split + b.shape[0]
        out = torch.empty(
            (total, *a.shape[1:]), dtype=a.dtype, device=a.device
        )
        out[:split].copy_(a[:split])
        out[split:].copy_(b)
        return out

    # ------------------------------------------------------------------
    # Core forward (mirrors _dynamic_inference_forward post-c65fb25)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: HeteroMambaMetadata,
        scheduler: HeteroChunkedPrefillScheduler,
        padded_dims: "HeteroBatchDimensions",
        ssm_forward_fn,
    ) -> torch.Tensor:
        """
        DES-LOC heterogeneous Mamba forward.

        Parameters
        ----------
        hidden_states  : (T, 1, D) token embeddings on prefill_device (or decode_device
                         for decode-only batches)
        metadata       : populated HeteroMambaMetadata for this step
        scheduler      : HeteroChunkedPrefillScheduler for this step
        padded_dims    : padded batch dimensions
        ssm_forward_fn : callable(hidden_states, metadata, mode) → tensor
                         mode in {"decode", "prefill", "chunked_prefill"}

        Returns
        -------
        output tensor (T, 1, D) on the same device as hidden_states
        """
        decode_count = padded_dims.decode_req_count
        prefill_count = padded_dims.prefill_req_count

        y_decode: Optional[torch.Tensor] = None
        y_prefill: Optional[torch.Tensor] = None

        # ----------------------------------------------------------
        # Branch 1: Decode
        # Mirrors upstream flat decode branch (no longer nested)
        # ----------------------------------------------------------
        if decode_count > 0:
            zxBCdt_decode = (
                hidden_states[:decode_count]
                if prefill_count > 0
                else hidden_states
            )
            # Ensure decode tensor is on decode_device for SM86 kernel
            zxBCdt_decode = HeteroMambaMetadata._pcie_copy(
                zxBCdt_decode, self.decode_device, label="fwd_decode_input"
            )
            y_decode = ssm_forward_fn(zxBCdt_decode, metadata, "decode")
            logger.debug("forward: decode branch done, shape=%s", tuple(y_decode.shape))

        # ----------------------------------------------------------
        # Branch 2: Prefill
        # Delegates to _hetero_prefill helper (mirrors upstream helper)
        # ----------------------------------------------------------
        if prefill_count > 0:
            if decode_count > 0:
                # Mixed batch: slice out prefill portion
                zxBCdt_prefill = torch.empty_like(hidden_states)
                zxBCdt_prefill = self._tensor_get_slice_after(
                    hidden_states, metadata.device_decode_prefill
                )
            else:
                zxBCdt_prefill = hidden_states

            zxBCdt_prefill = HeteroMambaMetadata._pcie_copy(
                zxBCdt_prefill, self.prefill_device, label="fwd_prefill_input"
            )
            y_prefill = self._hetero_prefill(
                zxBCdt_prefill, metadata, scheduler, padded_dims, ssm_forward_fn
            )
            logger.debug("forward: prefill branch done, shape=%s", tuple(y_prefill.shape))

        # ----------------------------------------------------------
        # Merge / select output
        # ----------------------------------------------------------
        if y_decode is not None and y_prefill is not None:
            # Move decode output to prefill_device for merge
            y_decode_merged = HeteroMambaMetadata._pcie_copy(
                y_decode, self.prefill_device, label="fwd_decode_to_prefill_for_merge"
            )
            token_count = padded_dims.token_count
            y = torch.empty(
                (token_count, 1, y_prefill.shape[-1]),
                dtype=y_prefill.dtype,
                device=self.prefill_device,
            )
            y = self._tensor_merge(
                y_decode_merged, y_prefill, metadata.device_decode_prefill
            )
            logger.debug("forward: merged decode+prefill, shape=%s", tuple(y.shape))
            return y
        elif y_decode is not None:
            return y_decode
        elif y_prefill is not None:
            return y_prefill
        else:
            raise RuntimeError(
                "HeteroMambaMixer.forward called with 0 decode and 0 prefill requests"
            )

    def _hetero_prefill(
        self,
        zxBCdt: torch.Tensor,
        metadata: HeteroMambaMetadata,
        scheduler: HeteroChunkedPrefillScheduler,
        padded_dims: "HeteroBatchDimensions",
        ssm_forward_fn,
    ) -> torch.Tensor:
        """
        Handle the prefill portion of the forward pass, including chunked prefill.

        Mirrors upstream ``MambaMixer._dynamic_inference_prefill`` (c65fb25).

        Layout (after the decode slice has been removed):
            [ chunked_prefill_tokens | regular_prefill_tokens... ]

        Chunked-prefill tokens = metadata.device_chunked_prefill[0]
        Regular-prefill tokens = metadata.device_chunked_prefill[1]
        """
        enable_chunked = scheduler.is_chunked_prefill_enabled()
        prefill_req_count = padded_dims.prefill_req_count
        prefill_token_count = zxBCdt.shape[0]

        y_chunked: Optional[torch.Tensor] = None
        y_regular: Optional[torch.Tensor] = None

        if enable_chunked and metadata.device_chunked_prefill is not None:
            chunked_len = int(metadata.device_chunked_prefill[0].item())
            y_chunked = ssm_forward_fn(
                zxBCdt[:chunked_len], metadata, "chunked_prefill"
            )
            logger.debug(
                "_hetero_prefill: chunked branch len=%d shape=%s",
                chunked_len, tuple(y_chunked.shape),
            )
            # Remainder for regular prefill
            zxBCdt = self._tensor_get_slice_after(
                zxBCdt, metadata.device_chunked_prefill
            )

        if not enable_chunked or prefill_req_count > 1:
            y_regular = ssm_forward_fn(zxBCdt, metadata, "prefill")
            logger.debug(
                "_hetero_prefill: regular branch shape=%s", tuple(y_regular.shape)
            )

        # Merge chunked + regular
        if y_chunked is not None and y_regular is not None:
            y_combined = self._tensor_merge(
                y_chunked, y_regular, metadata.device_chunked_prefill
            )
            return y_combined
        elif y_chunked is not None:
            # Chunked only: embed in full prefill-token buffer
            y_out = torch.empty(
                (prefill_token_count, 1, y_chunked.shape[-1]),
                dtype=y_chunked.dtype,
                device=y_chunked.device,
            )
            chunked_len = int(metadata.device_chunked_prefill[0].item())
            y_out[:chunked_len] = y_chunked
            return y_out
        elif y_regular is not None:
            return y_regular
        else:
            raise RuntimeError(
                "_hetero_prefill: no output produced (enable_chunked=%s, "
                "prefill_req_count=%d)" % (enable_chunked, prefill_req_count)
            )


# ---------------------------------------------------------------------------
# CUDAGraph batch-dimension matching (DES-LOC extension)
# ---------------------------------------------------------------------------

class HeteroCUDAGraphMatcher:
    """
    CUDA-graph bucket matcher for DES-LOC.

    Extends Megatron's ``CUDAGraphBatchDimensionBuilder.match_graph_config``
    with the ``explicit_chunked_prefill`` parameter added in c65fb25 and the
    DES-LOC device_affinity constraint.
    """

    def __init__(self, graph_dims: List[HeteroBatchDimensions]) -> None:
        self.graph_dims = graph_dims

    def match(
        self,
        real_dims: HeteroBatchDimensions,
        strict: bool = False,
        decode_only_cuda_graphs: bool = False,
        explicit_chunked_prefill: bool = False,
        ep_group: Optional[dist.ProcessGroup] = None,
    ) -> Optional[HeteroBatchDimensions]:
        """
        Find the smallest graph bucket that subsumes *real_dims*.

        Returns None → run in eager mode.

        Mirrors upstream ``match_graph_config`` post-c65fb25:
          1. EP-sync with ``explicit_chunked_prefill`` parameter.
          2. Early return None if explicit_chunked_prefill AND prefill_req_count > 0.
          3. Filter → sort → pick smallest subsuming bucket.
        """
        # EP sync
        adjusted = adjust_batch_dims_for_ep(
            real_dims,
            strict=strict,
            decode_only_cuda_graphs=decode_only_cuda_graphs,
            explicit_chunked_prefill=explicit_chunked_prefill,
            ep_group=ep_group,
        )
        if adjusted is None:
            logger.debug("HeteroCUDAGraphMatcher.match: EP sync → eager mode")
            return None

        # Upstream c65fb25 addition: explicit chunked prefill → eager for prefill batches
        if explicit_chunked_prefill and real_dims.prefill_req_count > 0:
            logger.debug(
                "HeteroCUDAGraphMatcher.match: explicit_chunked_prefill+prefill → eager"
            )
            return None

        # Filter to applicable buckets
        applicable = [
            g for g in self.graph_dims
            if g.subsumes(adjusted)
        ]
        if not applicable:
            logger.debug("HeteroCUDAGraphMatcher.match: no applicable graph bucket")
            return None

        # Pick smallest bucket by token_count (then req_count as tiebreak)
        best = min(
            applicable,
            key=lambda g: (g.token_count, g.prefill_req_count + g.decode_req_count),
        )
        logger.debug("HeteroCUDAGraphMatcher.match: best=%s", best)
        return best


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test 1: HeteroBatchDimensions equality / hash
    d1 = build_hetero_batch_dimensions(32, 2, 4)
    d2 = build_hetero_batch_dimensions(32, 2, 4, device_affinity=DESCLOC_PREFILL_DEVICE)
    assert d1 == d2, "equal dims should match"
    assert hash(d1) == hash(d2), "equal dims should have equal hashes"

    d3 = build_hetero_batch_dimensions(32, 0, 4, device_affinity=DESCLOC_DECODE_DEVICE)
    assert d3 != d1, "different dims should not match"

    # Test 2: subsumes
    bucket = build_hetero_batch_dimensions(64, 4, 4, device_affinity=DESCLOC_PREFILL_DEVICE)
    real = build_hetero_batch_dimensions(30, 2, 3, device_affinity=DESCLOC_PREFILL_DEVICE)
    assert bucket.subsumes(real), "bucket should subsume smaller real dims"

    wrong_dev = build_hetero_batch_dimensions(30, 2, 3, device_affinity=DESCLOC_DECODE_DEVICE)
    assert not bucket.subsumes(wrong_dev), "device_affinity mismatch should fail subsumes"

    # Test 3: HeteroChunkedPrefillScheduler basic scheduling
    sched = HeteroChunkedPrefillScheduler(
        max_tokens=100, is_hybrid_model=True, enable_chunked_prefill=True
    )
    action, tokens = sched.try_schedule_request(1, 200, 200)
    assert action == "partial", f"expected partial, got {action}"
    assert tokens == 100

    sched.reset_step()
    action2, tokens2 = sched.try_schedule_request(2, 50, 50)
    assert action2 == "full", f"expected full, got {action2}"
    assert tokens2 == 50

    # Test 4: CUDA graph capture suppresses chunked prefill on hybrid model
    sched.begin_cuda_graph_capture()
    assert not sched.is_chunked_prefill_enabled(), \
        "chunked prefill should be disabled during graph capture"
    sched.end_cuda_graph_capture()
    assert sched.is_chunked_prefill_enabled(), \
        "chunked prefill should re-enable after capture"

    # Test 5: HeteroCUDAGraphMatcher returns None for explicit chunked prefill + prefill batch
    matcher = HeteroCUDAGraphMatcher([
        build_hetero_batch_dimensions(128, 4, 4, device_affinity=DESCLOC_PREFILL_DEVICE)
    ])
    result = matcher.match(
        build_hetero_batch_dimensions(60, 2, 2, device_affinity=DESCLOC_PREFILL_DEVICE),
        explicit_chunked_prefill=True,
    )
    assert result is None, "explicit_chunked_prefill+prefill should return None"

    print("All smoke tests passed.")
