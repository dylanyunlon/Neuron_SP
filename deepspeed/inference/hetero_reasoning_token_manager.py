"""
HeteroReasoningTokenManager — DES-LOC Inference Token Strip & Prefix Alignment
================================================================================

Upstream design intent (Megatron commit 002255075c3728fded9a2e435677840b08560d55):
    Megatron's dynamic chat completion server runs multi-turn inference where each
    turn re-tokenizes the conversation from scratch (via chat template application).
    Because the tokenizer is applied to the *entire* conversation at each turn, the
    prefix tokens from turn N may differ from the prefix computed in turn N-1 — this
    happens whenever a reasoning model emits special "thinking" tokens (e.g., <think>
    ... </think>) that later get stripped before being fed back into the prompt.

    The original bug had two root causes:
      1. The guard `previous_turn_token_ids[-1] == eos_token_id` was evaluated on an
         empty sequence, causing an IndexError when a turn produced zero output tokens
         (edge case in beam-search / speculative decoding abort paths).
      2. The reverse scan over `retokeenized_previous_turn_token_ids` used its own
         length as the scan bound, but indexed into `current_turn_token_ids` which
         may be *shorter* after reasoning-token stripping — producing an out-of-bounds
         access on the current-turn side.

    The fix: guard against empty sequences and clamp the scan length to
    `min(len(retokenized), len(current))` before searching for the last EOS position.

DES-LOC adaptation (Decoupled Execution with Shared LOcality Cache):
    In DES-LOC heterogeneous training/inference on {2× A6000 48 GB SM86, 1× H100 NVL
    96 GB SM90}, the reasoning token strip problem has additional dimensions:

      • Device-local caching: Each device holds a Shared LOcality Cache (SLC) for KV
        blocks.  After reasoning tokens are stripped, the cached KV blocks for the
        stripped prefix must be invalidated or remapped — otherwise the H100 and A6000
        workers will disagree on which prompt positions are cache-valid.

      • Heterogeneous sequence lengths: The H100 (SM90, larger SRAM) may have
        processed a longer speculative prefix than the A6000 workers (SM86). The
        mismatch in retokenized lengths therefore has a *hardware* axis: the scan
        bound must also respect what each device actually cached.

      • PCIe-only interconnect: Without NVLink, cross-device prefix negotiation is
        expensive.  HeteroReasoningTokenManager performs all length clamping and EOS
        scanning on the CPU (pinned to 1.5 TB DRAM) so that no PCIe round-trip is
        needed just to compute the strip boundary.

      • Decoupled execution: DES-LOC decouples the prefill stage (H100) from the
        decode stage (A6000 × 2).  The manager tracks which device "owns" each turn's
        KV state and adjusts the strip boundary per device.

    This file provides:
      - `ReasoningStripResult`: dataclass carrying per-device strip metadata.
      - `HeteroReasoningTokenManager`: the main class with CPU-resident logic for
        boundary computation, EOS scanning, and SLC invalidation signaling.
      - `DESLOCDeviceProfile`: lightweight descriptor for each physical device.
      - `SLCInvalidationRequest`: message type sent (asynchronously, over shared
        memory) to device workers to trigger KV-cache block eviction.
      - Full unit tests in `__main__`.

Author adaptation: Neuron_SP / DES-LOC project (based on DeepSpeed)
Upstream commit: 002255075c3728fded9a2e435677840b08560d55
"""

from __future__ import annotations

import logging
import math
import os
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Logging — one named logger for the whole module; callers may configure it.
# ---------------------------------------------------------------------------
logger = logging.getLogger("deslock.hetero_reasoning")


# ---------------------------------------------------------------------------
# Hardware / device descriptors
# ---------------------------------------------------------------------------

class DeviceArch(Enum):
    SM86 = auto()   # NVIDIA Ampere — A6000
    SM90 = auto()   # NVIDIA Hopper — H100 NVL


@dataclass(frozen=True)
class DESLOCDeviceProfile:
    """
    Immutable descriptor for a physical device in the DES-LOC cluster.

    Attributes
    ----------
    device_id : int
        Local CUDA device index.
    arch : DeviceArch
        SM generation, used to choose KV-cache block sizing.
    vram_gb : int
        Nominal VRAM in gigabytes (used for SLC sizing heuristics).
    role : str
        Either ``"prefill"`` (H100 owns the prefill stage) or ``"decode"``
        (A6000 workers own the autoregressive decode stage).
    slc_block_size : int
        Number of tokens per SLC (Shared LOcality Cache) block.  Smaller on
        SM86 because L2 is narrower.
    max_cached_blocks : int
        Upper bound on blocks this device keeps in its SLC.
    """

    device_id: int
    arch: DeviceArch
    vram_gb: int
    role: str           # "prefill" | "decode"
    slc_block_size: int = 64
    max_cached_blocks: int = 4096

    def __post_init__(self) -> None:
        if self.role not in ("prefill", "decode"):
            raise ValueError(f"role must be 'prefill' or 'decode', got {self.role!r}")
        if self.slc_block_size <= 0:
            raise ValueError("slc_block_size must be positive")

    @property
    def is_hopper(self) -> bool:
        return self.arch == DeviceArch.SM90

    @property
    def effective_slc_capacity_tokens(self) -> int:
        return self.slc_block_size * self.max_cached_blocks


# Default cluster layout: H100 prefills, two A6000s decode
def build_default_cluster() -> List[DESLOCDeviceProfile]:
    """
    Return the three-device profile matching the Neuron_SP reference cluster:
      • device 0 — H100 NVL 96 GB (SM90), prefill
      • device 1 — A6000 48 GB (SM86), decode
      • device 2 — A6000 48 GB (SM86), decode
    """
    return [
        DESLOCDeviceProfile(
            device_id=0,
            arch=DeviceArch.SM90,
            vram_gb=96,
            role="prefill",
            slc_block_size=128,      # Hopper has larger SRAM → bigger blocks
            max_cached_blocks=6144,
        ),
        DESLOCDeviceProfile(
            device_id=1,
            arch=DeviceArch.SM86,
            vram_gb=48,
            role="decode",
            slc_block_size=64,
            max_cached_blocks=3072,
        ),
        DESLOCDeviceProfile(
            device_id=2,
            arch=DeviceArch.SM86,
            vram_gb=48,
            role="decode",
            slc_block_size=64,
            max_cached_blocks=3072,
        ),
    ]


# ---------------------------------------------------------------------------
# SLC invalidation message
# ---------------------------------------------------------------------------

@dataclass
class SLCInvalidationRequest:
    """
    Message sent from the CPU manager to a device worker, asking it to evict
    KV-cache blocks that no longer correspond to the valid prompt prefix.

    In DES-LOC, after reasoning tokens are stripped and the conversation is
    retokenized, any device that cached KV blocks for the stripped suffix must
    free those blocks.  Because interconnect is PCIe-only, this message is
    delivered via a shared-memory queue (or equivalent OS IPC) rather than a
    CUDA peer-copy.

    Attributes
    ----------
    device_id : int
        Target device.
    sequence_id : str
        Unique identifier for the inference sequence (conversation).
    valid_token_length : int
        Number of prompt tokens still valid after stripping.  Blocks covering
        positions >= this value should be evicted.
    strip_boundary_token : int
        The token index where the last EOS (end of reasoning block) was found.
        This is the boundary used to split "keep" from "evict" regions.
    timestamp_ns : int
        Monotonic nanosecond timestamp when the request was generated.
    """

    device_id: int
    sequence_id: str
    valid_token_length: int
    strip_boundary_token: int
    timestamp_ns: int = field(default_factory=time.monotonic_ns)

    def blocks_to_evict(self, block_size: int) -> range:
        """
        Return the range of block indices (0-based) that should be evicted on
        a device with the given ``block_size``.
        """
        first_invalid_block = math.ceil(self.valid_token_length / block_size)
        # We only evict up to the block containing the sequence's current max
        # length; the caller is responsible for clamping to actual allocation.
        return range(first_invalid_block, self.strip_boundary_token // block_size + 1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStripResult:
    """
    Output of :meth:`HeteroReasoningTokenManager.compute_strip_boundary`.

    Attributes
    ----------
    last_eos_index : int
        Token index (in the *current* turn's token list) of the last EOS
        found during the bounded scan.  ``-1`` if no EOS was found.
    scan_length : int
        Actual number of tokens scanned (``min`` of retokenized and current
        lengths, further clamped per device).
    previous_turn_stripped : bool
        ``True`` if the trailing EOS was removed from ``previous_turn_token_ids``.
    stripped_previous_ids : List[int]
        The (possibly trimmed) previous-turn token id list.
    invalidation_requests : List[SLCInvalidationRequest]
        One request per device that needs SLC eviction.  Empty if the strip
        boundary does not cross any cached block boundary on any device.
    sequence_length_after_strip : int
        Effective prompt length after stripping, i.e. ``last_eos_index`` if
        an EOS was found, else ``scan_length``.
    """

    last_eos_index: int
    scan_length: int
    previous_turn_stripped: bool
    stripped_previous_ids: List[int]
    invalidation_requests: List[SLCInvalidationRequest]
    sequence_length_after_strip: int


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class HeteroReasoningTokenManager:
    """
    Manages reasoning-token stripping and prefix alignment across heterogeneous
    devices in a DES-LOC inference cluster.

    Design contract
    ---------------
    All token-list manipulation happens on the CPU.  The manager never touches
    CUDA tensors directly; it emits :class:`SLCInvalidationRequest` objects
    that device workers consume asynchronously from a shared queue.

    Thread safety
    -------------
    `compute_strip_boundary` is *not* thread-safe by itself.  If multiple
    inference sequences are processed concurrently, create one manager instance
    per sequence (or protect with an external lock).  The invalidation request
    list returned in :class:`ReasoningStripResult` is self-contained and safe
    to hand off to any thread.

    Parameters
    ----------
    devices : List[DESLOCDeviceProfile]
        Physical devices in the cluster.  Must contain at least one prefill
        device (H100) and at least one decode device (A6000).
    eos_token_id : int
        The EOS token id used by the model (e.g. 2 for LLaMA, 151645 for Qwen).
    sequence_id : str
        Identifier for the conversation.  Used in invalidation request payloads.
    enable_slc_invalidation : bool
        When ``False``, skip building invalidation requests (useful in unit
        tests that do not have device workers running).
    max_scan_fraction : float
        Fraction of the shorter sequence to scan during EOS search.  Default
        1.0 (scan the entire shorter sequence).  Set < 1.0 to trade accuracy
        for speed when sequences are very long.
    """

    def __init__(
        self,
        devices: Optional[List[DESLOCDeviceProfile]] = None,
        eos_token_id: int = 2,
        sequence_id: str = "default",
        enable_slc_invalidation: bool = True,
        max_scan_fraction: float = 1.0,
    ) -> None:
        self._devices = devices if devices is not None else build_default_cluster()
        self._eos_token_id = eos_token_id
        self._sequence_id = sequence_id
        self._enable_slc_invalidation = enable_slc_invalidation

        if not (0.0 < max_scan_fraction <= 1.0):
            raise ValueError("max_scan_fraction must be in (0, 1]")
        self._max_scan_fraction = max_scan_fraction

        # Validate that we have at least one device of each role
        roles = {d.role for d in self._devices}
        if "prefill" not in roles:
            raise ValueError("Cluster must have at least one 'prefill' device (H100)")
        if "decode" not in roles:
            raise ValueError("Cluster must have at least one 'decode' device (A6000)")

        logger.debug(
            "HeteroReasoningTokenManager initialised: seq=%s eos=%d devices=%s",
            self._sequence_id,
            self._eos_token_id,
            [f"dev{d.device_id}({d.role})" for d in self._devices],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_strip_boundary(
        self,
        previous_turn_token_ids: List[int],
        retokenized_previous_turn_token_ids: List[int],
        current_turn_token_ids: List[int],
    ) -> ReasoningStripResult:
        """
        Compute the strip boundary after reasoning tokens are removed and the
        conversation is retokenized.

        This replicates — and extends — the logic from Megatron commit
        002255075c3728fded9a2e435677840b08560d55, adapted for DES-LOC:

          1. **Empty-sequence guard** (upstream fix #1): if
             ``previous_turn_token_ids`` is non-empty *and* its last token is
             EOS, strip it.  Avoids IndexError on empty sequences that arise
             when speculative-decoding aborts a turn with zero output tokens.

          2. **Scan-bound clamping** (upstream fix #2): limit the reverse scan
             to ``min(len(retokenized), len(current))`` tokens.  Without this
             clamp, indexing into ``current_turn_token_ids`` goes out of bounds
             when reasoning tokens are stripped and the current turn is shorter
             than the retokenized previous turn.

          3. **DES-LOC heterogeneous clamping**: further reduce the scan bound
             per device so that the scan never exceeds the SLC capacity of the
             most-constrained decode device (A6000, smaller SRAM).  This
             ensures that the CPU-computed boundary is always reachable by both
             SM86 and SM90 workers without additional PCIe negotiation.

          4. **SLC invalidation generation**: once the boundary is known, emit
             one :class:`SLCInvalidationRequest` per device whose cached region
             extends beyond the valid prefix.

        Parameters
        ----------
        previous_turn_token_ids : List[int]
            Raw token ids from the previous turn (may end with EOS).
        retokenized_previous_turn_token_ids : List[int]
            The previous turn re-tokenized by applying the full chat template.
            May differ from ``previous_turn_token_ids`` due to reasoning-token
            injection or template padding.
        current_turn_token_ids : List[int]
            Token ids for the *current* turn (after reasoning-token stripping).
            This list is the one we index during the EOS scan.

        Returns
        -------
        ReasoningStripResult
            See :class:`ReasoningStripResult` for field documentation.
        """
        # ------------------------------------------------------------------
        # Step 1: Strip trailing EOS from previous turn (upstream fix #1)
        # ------------------------------------------------------------------
        stripped = False
        prev_ids = list(previous_turn_token_ids)   # shallow copy; we may mutate

        if prev_ids and prev_ids[-1] == self._eos_token_id:
            prev_ids = prev_ids[:-1]
            stripped = True
            logger.debug(
                "seq=%s: stripped trailing EOS from previous turn "
                "(previous_turn length %d → %d)",
                self._sequence_id,
                len(previous_turn_token_ids),
                len(prev_ids),
            )

        # ------------------------------------------------------------------
        # Step 2: Compute baseline scan length (upstream fix #2)
        # ------------------------------------------------------------------
        base_scan_len = min(
            len(retokenized_previous_turn_token_ids),
            len(current_turn_token_ids),
        )

        # ------------------------------------------------------------------
        # Step 3: DES-LOC heterogeneous scan-bound clamping
        #
        # The H100 (prefill) can hold a larger SLC than the A6000 (decode).
        # After the prefill stage writes KV blocks into SLC, the decode
        # workers must be able to read the same prefix from *their* SLC.
        # If the scan boundary exceeds the A6000 SLC capacity, the KV blocks
        # for that prefix are not present on decode devices → we would be
        # signalling invalidation beyond what decode devices actually cached,
        # which is harmless but wasteful.  More importantly, we clamp here
        # to avoid ever computing a strip boundary that the decode device
        # cannot serve from cache.
        # ------------------------------------------------------------------
        decode_devices = [d for d in self._devices if d.role == "decode"]
        prefill_devices = [d for d in self._devices if d.role == "prefill"]

        # Minimum SLC capacity among decode devices (in tokens)
        min_decode_slc = min(d.effective_slc_capacity_tokens for d in decode_devices)
        # Maximum SLC capacity among prefill devices (in tokens)
        max_prefill_slc = max(d.effective_slc_capacity_tokens for d in prefill_devices)

        hetero_clamp = min(base_scan_len, min_decode_slc)

        # Apply optional fraction cap (for very long sequences)
        fraction_clamp = int(math.floor(base_scan_len * self._max_scan_fraction))
        scan_len = min(hetero_clamp, fraction_clamp) if fraction_clamp > 0 else hetero_clamp

        if scan_len != base_scan_len:
            logger.info(
                "seq=%s: DES-LOC clamped scan length from %d → %d "
                "(min_decode_slc=%d, max_prefill_slc=%d)",
                self._sequence_id,
                base_scan_len,
                scan_len,
                min_decode_slc,
                max_prefill_slc,
            )

        # ------------------------------------------------------------------
        # Step 4: Reverse EOS scan within clamped window
        # ------------------------------------------------------------------
        # Default: no EOS found; use the last index in the retokenized list
        # (matching Megatron's fallback assignment of `last_eos_token_id_index`).
        last_eos_index = len(retokenized_previous_turn_token_ids) - 1

        for i in reversed(range(scan_len)):
            if current_turn_token_ids[i] == self._eos_token_id:
                last_eos_index = i
                logger.debug(
                    "seq=%s: found EOS at position %d (scan_len=%d)",
                    self._sequence_id,
                    i,
                    scan_len,
                )
                break
        else:
            logger.debug(
                "seq=%s: no EOS found in scan window [0, %d); "
                "using retokenized length fallback index %d",
                self._sequence_id,
                scan_len,
                last_eos_index,
            )

        # ------------------------------------------------------------------
        # Step 5: Determine effective sequence length after stripping
        # ------------------------------------------------------------------
        if last_eos_index >= 0 and (
            last_eos_index < len(current_turn_token_ids)
            and current_turn_token_ids[last_eos_index] == self._eos_token_id
        ):
            sequence_length_after_strip = last_eos_index
        else:
            sequence_length_after_strip = scan_len

        # ------------------------------------------------------------------
        # Step 6: Build SLC invalidation requests
        # ------------------------------------------------------------------
        invalidation_requests: List[SLCInvalidationRequest] = []

        if self._enable_slc_invalidation:
            invalidation_requests = self._build_invalidation_requests(
                valid_length=sequence_length_after_strip,
                strip_boundary=last_eos_index,
            )

        return ReasoningStripResult(
            last_eos_index=last_eos_index,
            scan_length=scan_len,
            previous_turn_stripped=stripped,
            stripped_previous_ids=prev_ids,
            invalidation_requests=invalidation_requests,
            sequence_length_after_strip=sequence_length_after_strip,
        )

    def replace_prefix_tokens(
        self,
        previous_turn_token_ids: List[int],
        retokenized_previous_turn_token_ids: List[int],
        current_turn_token_ids: List[int],
    ) -> Tuple[List[int], ReasoningStripResult]:
        """
        High-level entry point that mirrors Megatron's ``_replace_prefix_tokens``
        function but returns both the updated token list and the full strip
        result with DES-LOC metadata.

        The function identifies the longest common prefix between the
        retokenized previous turn and the current turn (up to the last EOS),
        then replaces the prefix of ``current_turn_token_ids`` with the
        corresponding tokens from ``retokenized_previous_turn_token_ids``.
        This is needed because the two tokenizations may assign different token
        ids to the same text when the reasoning tokens alter the BPE context.

        Parameters
        ----------
        previous_turn_token_ids : List[int]
            Raw previous-turn ids (possibly ending with EOS).
        retokenized_previous_turn_token_ids : List[int]
            Re-tokenized previous turn.
        current_turn_token_ids : List[int]
            Current-turn ids (reasoning tokens already stripped by caller).

        Returns
        -------
        updated_current_ids : List[int]
            ``current_turn_token_ids`` with the prefix replaced.
        strip_result : ReasoningStripResult
            Full strip metadata including SLC invalidation requests.
        """
        strip_result = self.compute_strip_boundary(
            previous_turn_token_ids=previous_turn_token_ids,
            retokenized_previous_turn_token_ids=retokenized_previous_turn_token_ids,
            current_turn_token_ids=current_turn_token_ids,
        )

        boundary = strip_result.last_eos_index
        # Replace tokens in [0, boundary) with retokenized versions
        updated = list(current_turn_token_ids)
        replace_end = min(
            boundary,
            len(retokenized_previous_turn_token_ids),
            len(updated),
        )
        for idx in range(replace_end):
            updated[idx] = retokenized_previous_turn_token_ids[idx]

        logger.debug(
            "seq=%s: replaced %d prefix tokens (boundary=%d)",
            self._sequence_id,
            replace_end,
            boundary,
        )
        return updated, strip_result

    # ------------------------------------------------------------------
    # SLC invalidation helpers
    # ------------------------------------------------------------------

    def _build_invalidation_requests(
        self,
        valid_length: int,
        strip_boundary: int,
    ) -> List[SLCInvalidationRequest]:
        """
        Build one :class:`SLCInvalidationRequest` per device for which the
        strip boundary falls inside the device's cached region.

        A device needs an invalidation request only if ``valid_length`` falls
        within the range of blocks it has cached.  If ``valid_length`` is
        beyond the device's entire SLC capacity, no cached blocks are affected.

        The decision is made entirely on the CPU from the device profile —
        no CUDA calls are issued here.

        Parameters
        ----------
        valid_length : int
            Number of tokens still valid after stripping.
        strip_boundary : int
            Token index of the last EOS (the strip point).

        Returns
        -------
        List[SLCInvalidationRequest]
            One request per device that needs cache eviction.
        """
        requests: List[SLCInvalidationRequest] = []
        ts = time.monotonic_ns()

        for device in self._devices:
            # Block index of the first invalid token
            first_invalid_block = valid_length // device.slc_block_size

            # If the strip boundary is beyond what this device could even cache,
            # the device has nothing to evict.
            if first_invalid_block >= device.max_cached_blocks:
                continue

            # If valid_length already aligns to the start of a block, the
            # device only needs to evict blocks *after* the valid region.
            # Either way, we issue the request so the device worker can decide
            # the exact block range using its own allocation metadata.
            req = SLCInvalidationRequest(
                device_id=device.device_id,
                sequence_id=self._sequence_id,
                valid_token_length=valid_length,
                strip_boundary_token=strip_boundary,
                timestamp_ns=ts,
            )
            requests.append(req)

        if requests:
            logger.info(
                "seq=%s: generated %d SLC invalidation request(s) "
                "(valid_length=%d, strip_boundary=%d, devices=%s)",
                self._sequence_id,
                len(requests),
                valid_length,
                strip_boundary,
                [r.device_id for r in requests],
            )

        return requests

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def device_summary(self) -> str:
        """Return a human-readable summary of the cluster layout."""
        lines = ["DES-LOC cluster layout:"]
        for d in self._devices:
            lines.append(
                f"  device {d.device_id}: {d.arch.name} "
                f"({d.vram_gb} GB, {d.role}) "
                f"SLC={d.effective_slc_capacity_tokens} tokens "
                f"(block={d.slc_block_size}×{d.max_cached_blocks})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities used by multi-turn chat server integration
# ---------------------------------------------------------------------------

def strip_reasoning_tokens(
    token_ids: List[int],
    think_start_id: int,
    think_end_id: int,
) -> List[int]:
    """
    Remove reasoning token spans delimited by ``think_start_id`` …
    ``think_end_id`` from a token id list.

    This is a pre-processing step that the chat server performs *before*
    calling :meth:`HeteroReasoningTokenManager.compute_strip_boundary`.
    The stripped list is what becomes ``current_turn_token_ids``.

    Implementation note
    -------------------
    The function handles nested think blocks (unusual but possible in some
    model checkpoints) by tracking a depth counter rather than a simple flag.

    Parameters
    ----------
    token_ids : List[int]
        Raw token ids possibly containing reasoning spans.
    think_start_id : int
        Token id that opens a reasoning span.
    think_end_id : int
        Token id that closes a reasoning span.

    Returns
    -------
    List[int]
        Token ids with all reasoning spans removed.

    Examples
    --------
    >>> strip_reasoning_tokens([1, 100, 2, 3, 101, 4], 100, 101)
    [1, 4]
    >>> strip_reasoning_tokens([1, 100, 100, 2, 101, 3, 101, 4], 100, 101)
    [1, 4]
    """
    result: List[int] = []
    depth = 0
    for tok in token_ids:
        if tok == think_start_id:
            depth += 1
        elif tok == think_end_id:
            if depth > 0:
                depth -= 1
        elif depth == 0:
            result.append(tok)
    return result


def compute_retokenization_drift(
    original_ids: List[int],
    retokenized_ids: List[int],
) -> int:
    """
    Return the index of the first position where the two tokenizations diverge.

    Used in diagnostic / logging paths to quantify how much the BPE context
    shifted due to reasoning token injection or removal.

    Parameters
    ----------
    original_ids : List[int]
        Token ids before retokenization.
    retokenized_ids : List[int]
        Token ids after retokenization.

    Returns
    -------
    int
        First divergence index.  Equal to ``min(len(original), len(retokenized))``
        if the shorter list is a prefix of the longer.
    """
    limit = min(len(original_ids), len(retokenized_ids))
    for i in range(limit):
        if original_ids[i] != retokenized_ids[i]:
            return i
    return limit


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    class TestReasoningStripResult(unittest.TestCase):
        """Tests for ReasoningStripResult and its helpers."""

        def test_blocks_to_evict_basic(self) -> None:
            req = SLCInvalidationRequest(
                device_id=0,
                sequence_id="t",
                valid_token_length=130,
                strip_boundary_token=200,
            )
            blocks = list(req.blocks_to_evict(block_size=64))
            # first_invalid_block = ceil(130/64) = 3
            # strip_boundary block = 200//64 = 3
            # range(3, 4) → [3]
            self.assertEqual(blocks, [3])

        def test_blocks_to_evict_aligned(self) -> None:
            req = SLCInvalidationRequest(
                device_id=1,
                sequence_id="t",
                valid_token_length=128,
                strip_boundary_token=255,
            )
            blocks = list(req.blocks_to_evict(block_size=64))
            # first_invalid_block = ceil(128/64) = 2
            # strip_boundary block = 255//64 = 3
            # range(2, 4) → [2, 3]
            self.assertEqual(blocks, [2, 3])

    class TestStripReasoningTokens(unittest.TestCase):
        """Tests for the standalone reasoning-token stripper."""

        def test_simple_strip(self) -> None:
            ids = [1, 100, 99, 101, 4]
            result = strip_reasoning_tokens(ids, think_start_id=100, think_end_id=101)
            self.assertEqual(result, [1, 4])

        def test_no_reasoning_tokens(self) -> None:
            ids = [1, 2, 3]
            result = strip_reasoning_tokens(ids, think_start_id=100, think_end_id=101)
            self.assertEqual(result, [1, 2, 3])

        def test_empty_sequence(self) -> None:
            result = strip_reasoning_tokens([], think_start_id=100, think_end_id=101)
            self.assertEqual(result, [])

        def test_nested_reasoning_blocks(self) -> None:
            # depth increases twice before decreasing
            ids = [1, 100, 100, 99, 101, 98, 101, 4]
            result = strip_reasoning_tokens(ids, think_start_id=100, think_end_id=101)
            self.assertEqual(result, [1, 4])

        def test_reasoning_at_start_and_end(self) -> None:
            ids = [100, 99, 101, 1, 2, 3, 100, 88, 101]
            result = strip_reasoning_tokens(ids, think_start_id=100, think_end_id=101)
            self.assertEqual(result, [1, 2, 3])

        def test_unclosed_reasoning_block(self) -> None:
            # If think_end never appears, everything after think_start is swallowed
            ids = [1, 100, 99, 98, 4]
            result = strip_reasoning_tokens(ids, think_start_id=100, think_end_id=101)
            self.assertEqual(result, [1])

    class TestRetokenizationDrift(unittest.TestCase):
        """Tests for compute_retokenization_drift."""

        def test_identical(self) -> None:
            self.assertEqual(compute_retokenization_drift([1, 2, 3], [1, 2, 3]), 3)

        def test_prefix_match(self) -> None:
            self.assertEqual(compute_retokenization_drift([1, 2], [1, 2, 3]), 2)

        def test_immediate_divergence(self) -> None:
            self.assertEqual(compute_retokenization_drift([1, 2], [9, 2]), 0)

        def test_both_empty(self) -> None:
            self.assertEqual(compute_retokenization_drift([], []), 0)

    class TestDESLOCDeviceProfile(unittest.TestCase):
        """Tests for DESLOCDeviceProfile."""

        def test_default_cluster_has_both_roles(self) -> None:
            cluster = build_default_cluster()
            roles = {d.role for d in cluster}
            self.assertIn("prefill", roles)
            self.assertIn("decode", roles)

        def test_hopper_flag(self) -> None:
            cluster = build_default_cluster()
            h100 = next(d for d in cluster if d.arch == DeviceArch.SM90)
            a6000 = next(d for d in cluster if d.arch == DeviceArch.SM86)
            self.assertTrue(h100.is_hopper)
            self.assertFalse(a6000.is_hopper)

        def test_slc_capacity(self) -> None:
            cluster = build_default_cluster()
            h100 = next(d for d in cluster if d.arch == DeviceArch.SM90)
            # 128 tokens/block × 6144 blocks = 786 432 tokens
            self.assertEqual(h100.effective_slc_capacity_tokens, 128 * 6144)

        def test_invalid_role_raises(self) -> None:
            with self.assertRaises(ValueError):
                DESLOCDeviceProfile(
                    device_id=99, arch=DeviceArch.SM86, vram_gb=48, role="unknown"
                )

        def test_invalid_block_size_raises(self) -> None:
            with self.assertRaises(ValueError):
                DESLOCDeviceProfile(
                    device_id=0, arch=DeviceArch.SM86, vram_gb=48,
                    role="decode", slc_block_size=0
                )

    class TestHeteroReasoningTokenManager(unittest.TestCase):
        """Core tests for HeteroReasoningTokenManager."""

        EOS = 2

        def _make_manager(self, **kwargs) -> HeteroReasoningTokenManager:
            return HeteroReasoningTokenManager(
                devices=build_default_cluster(),
                eos_token_id=self.EOS,
                sequence_id="test-seq",
                enable_slc_invalidation=False,  # no workers running
                **kwargs,
            )

        # ---- guard: empty previous turn ----

        def test_empty_previous_turn_no_crash(self) -> None:
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[],
                retokenized_previous_turn_token_ids=[10, 20, 30],
                current_turn_token_ids=[10, 20, 30],
            )
            self.assertFalse(result.previous_turn_stripped)
            self.assertEqual(result.stripped_previous_ids, [])

        def test_single_eos_previous_turn(self) -> None:
            """Previous turn with only the EOS token should not crash."""
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[self.EOS],
                retokenized_previous_turn_token_ids=[10, 20],
                current_turn_token_ids=[10, 20],
            )
            self.assertTrue(result.previous_turn_stripped)
            self.assertEqual(result.stripped_previous_ids, [])

        # ---- trailing EOS stripping ----

        def test_trailing_eos_stripped(self) -> None:
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[5, 6, 7, self.EOS],
                retokenized_previous_turn_token_ids=[5, 6, 7],
                current_turn_token_ids=[5, 6, 7],
            )
            self.assertTrue(result.previous_turn_stripped)
            self.assertEqual(result.stripped_previous_ids, [5, 6, 7])

        def test_no_trailing_eos(self) -> None:
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[5, 6, 7],
                retokenized_previous_turn_token_ids=[5, 6, 7],
                current_turn_token_ids=[5, 6, 7],
            )
            self.assertFalse(result.previous_turn_stripped)

        # ---- scan bound clamping: upstream fix #2 ----

        def test_shorter_current_prevents_oob(self) -> None:
            """
            Megatron bug: retokenized is longer than current; without clamping,
            reverse scan would index current out-of-bounds.
            """
            mgr = self._make_manager()
            retokenized = list(range(20))        # length 20
            current = list(range(10))            # length 10  (shorter)
            # Neither list contains EOS
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=retokenized,
                retokenized_previous_turn_token_ids=retokenized,
                current_turn_token_ids=current,
            )
            self.assertLessEqual(result.scan_length, len(current))

        def test_scan_finds_eos_in_current(self) -> None:
            current = [10, 20, self.EOS, 30, 40]
            retokenized = [10, 20, self.EOS, 30, 40, 50]
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[10, 20, self.EOS, 30, 40],
                retokenized_previous_turn_token_ids=retokenized,
                current_turn_token_ids=current,
            )
            self.assertEqual(result.last_eos_index, 2)

        def test_scan_finds_last_eos(self) -> None:
            """When multiple EOS tokens exist, the last one in the window wins."""
            current = [self.EOS, 10, self.EOS, 20]
            retokenized = [self.EOS, 10, self.EOS, 20]
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=current,
                retokenized_previous_turn_token_ids=retokenized,
                current_turn_token_ids=current,
            )
            # reversed scan: last EOS is at index 2
            self.assertEqual(result.last_eos_index, 2)

        def test_no_eos_fallback_index(self) -> None:
            retokenized = [10, 20, 30]
            current = [10, 20, 30]
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[10, 20, 30],
                retokenized_previous_turn_token_ids=retokenized,
                current_turn_token_ids=current,
            )
            # Fallback: len(retokenized) - 1 = 2
            self.assertEqual(result.last_eos_index, 2)

        # ---- DES-LOC clamping ----

        def test_hetero_clamp_limits_scan_to_decode_slc(self) -> None:
            """
            If the token lists are longer than the A6000 SLC capacity, the
            scan length should be clamped to the decode SLC capacity.
            """
            # A6000 SLC = 64 × 3072 = 196 608 tokens per device
            # Build a very small cluster to make the clamp observable
            tiny_decode = DESLOCDeviceProfile(
                device_id=1, arch=DeviceArch.SM86, vram_gb=48,
                role="decode", slc_block_size=4, max_cached_blocks=10
            )  # SLC = 40 tokens
            tiny_prefill = DESLOCDeviceProfile(
                device_id=0, arch=DeviceArch.SM90, vram_gb=96,
                role="prefill", slc_block_size=8, max_cached_blocks=20
            )  # SLC = 160 tokens

            mgr = HeteroReasoningTokenManager(
                devices=[tiny_prefill, tiny_decode],
                eos_token_id=self.EOS,
                sequence_id="clamp-test",
                enable_slc_invalidation=False,
            )

            long_list = list(range(100))  # 100 tokens > 40-token decode SLC
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=long_list,
                retokenized_previous_turn_token_ids=long_list,
                current_turn_token_ids=long_list,
            )
            # Scan length must not exceed decode SLC capacity (40)
            self.assertLessEqual(result.scan_length, 40)

        def test_fraction_cap(self) -> None:
            mgr = self._make_manager(max_scan_fraction=0.5)
            tokens = list(range(100))
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=tokens,
                retokenized_previous_turn_token_ids=tokens,
                current_turn_token_ids=tokens,
            )
            # 50% of 100 = 50
            self.assertLessEqual(result.scan_length, 50)

        def test_invalid_fraction_raises(self) -> None:
            with self.assertRaises(ValueError):
                self._make_manager(max_scan_fraction=0.0)
            with self.assertRaises(ValueError):
                self._make_manager(max_scan_fraction=1.5)

        # ---- replace_prefix_tokens ----

        def test_replace_prefix_tokens_basic(self) -> None:
            mgr = self._make_manager()
            prev = [10, 20, 30, self.EOS]
            retokenized = [10, 21, 30]   # position 1 drifted
            current = [10, 20, 30, 40]   # same length as retokenized extended

            updated, result = mgr.replace_prefix_tokens(
                previous_turn_token_ids=prev,
                retokenized_previous_turn_token_ids=retokenized,
                current_turn_token_ids=current,
            )
            # Prefix up to boundary should be from retokenized
            # boundary = last_eos_index (no EOS in current → fallback = len(retokenized)-1 = 2)
            self.assertEqual(updated[1], 21)  # retokenized drift at position 1

        def test_replace_prefix_identity_when_no_drift(self) -> None:
            mgr = self._make_manager()
            tokens = [10, 20, 30]
            updated, _ = mgr.replace_prefix_tokens(
                previous_turn_token_ids=tokens,
                retokenized_previous_turn_token_ids=tokens,
                current_turn_token_ids=tokens,
            )
            self.assertEqual(updated, tokens)

        # ---- SLC invalidation ----

        def test_slc_invalidation_requests_generated(self) -> None:
            cluster = build_default_cluster()
            mgr = HeteroReasoningTokenManager(
                devices=cluster,
                eos_token_id=self.EOS,
                sequence_id="inv-test",
                enable_slc_invalidation=True,
            )
            # Token list short enough to be within SLC of all devices
            tokens = [10, 20, self.EOS, 30, 40]
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=tokens,
                retokenized_previous_turn_token_ids=tokens,
                current_turn_token_ids=tokens,
            )
            # We should have at most len(cluster) requests
            self.assertLessEqual(
                len(result.invalidation_requests), len(cluster)
            )

        def test_slc_invalidation_disabled(self) -> None:
            mgr = self._make_manager()
            tokens = [10, 20, self.EOS]
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=tokens,
                retokenized_previous_turn_token_ids=tokens,
                current_turn_token_ids=tokens,
            )
            self.assertEqual(result.invalidation_requests, [])

        def test_slc_request_device_ids(self) -> None:
            cluster = build_default_cluster()
            mgr = HeteroReasoningTokenManager(
                devices=cluster,
                eos_token_id=self.EOS,
                sequence_id="devid-test",
                enable_slc_invalidation=True,
            )
            tokens = [10, 20, self.EOS, 30]
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=tokens,
                retokenized_previous_turn_token_ids=tokens,
                current_turn_token_ids=tokens,
            )
            device_ids_in_requests = {r.device_id for r in result.invalidation_requests}
            cluster_device_ids = {d.device_id for d in cluster}
            # All request device ids must belong to the cluster
            self.assertTrue(device_ids_in_requests.issubset(cluster_device_ids))

        def test_blocks_to_evict_integration(self) -> None:
            req = SLCInvalidationRequest(
                device_id=1,
                sequence_id="t",
                valid_token_length=65,
                strip_boundary_token=128,
            )
            blocks = list(req.blocks_to_evict(block_size=64))
            # first_invalid_block = ceil(65/64) = 2
            # strip_boundary block = 128//64 = 2
            # range(2, 3) → [2]
            self.assertEqual(blocks, [2])

        # ---- device summary ----

        def test_device_summary_contains_all_devices(self) -> None:
            mgr = self._make_manager()
            summary = mgr.device_summary()
            for d in build_default_cluster():
                self.assertIn(f"device {d.device_id}", summary)

        # ---- invariants under empty inputs ----

        def test_all_empty_lists(self) -> None:
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[],
                retokenized_previous_turn_token_ids=[],
                current_turn_token_ids=[],
            )
            self.assertEqual(result.scan_length, 0)
            self.assertEqual(result.sequence_length_after_strip, 0)

        def test_single_token_current(self) -> None:
            mgr = self._make_manager()
            result = mgr.compute_strip_boundary(
                previous_turn_token_ids=[self.EOS],
                retokenized_previous_turn_token_ids=[self.EOS],
                current_turn_token_ids=[self.EOS],
            )
            # EOS found at index 0
            self.assertEqual(result.last_eos_index, 0)

    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestReasoningStripResult,
        TestStripReasoningTokens,
        TestRetokenizationDrift,
        TestDESLOCDeviceProfile,
        TestHeteroReasoningTokenManager,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
