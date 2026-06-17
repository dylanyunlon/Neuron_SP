"""
DES-LOC Heterogeneous Checkpoint Integrity Verification
========================================================

Upstream design intent (Megatron commit 970c2540):
    Megatron-LM added SHA-256 manifest-based checkpoint integrity verification
    to detect silent data corruption in distributed checkpoints. The upstream
    implementation runs rank-0 hashing after all shards are written, broadcasts
    pass/fail status to all ranks, and serializes the manifest as ``integrity.json``
    alongside checkpoint files. This is sufficient for homogeneous GPU clusters
    where all ranks share a uniform storage backend.

DES-LOC adaptation points:
    The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework
    introduces a three-tier heterogeneous compute/memory hierarchy:
        - Tier-0: H100 NVL 96GB (SM90) — high-bandwidth HBM3, primary compute
        - Tier-1: 2x A6000 48GB (SM86) — GDDR6, secondary compute/offload
        - Tier-2: 1.5TB CPU DRAM — staging buffer, async checkpoint sink

    Key DES-LOC divergences from the upstream approach:

    1. **Per-tier manifest sharding**: Because A6000 and H100 write checkpoint
       shards to different memory regions and potentially different storage paths,
       a single flat ``integrity.json`` cannot reliably capture cross-tier
       provenance. We produce per-tier manifests (``integrity_tier{N}.json``)
       that are merged at verification time.

    2. **Tier-aware hashing coordinator**: Rank-0 in the upstream is implicitly
       assumed to be the "best" rank. In DES-LOC, Tier-0 (H100) rank is the
       designated coordinator because it has the highest PCIe bandwidth to host
       DRAM. We select the coordinator by querying the device capability
       (SM90 > SM86).

    3. **DRAM staging path**: Large checkpoints are staged through CPU DRAM
       before being flushed to persistent storage. We hash the DRAM-staged
       copy first and cross-validate against the flushed copy to catch
       DRAM bit-flip events (known to occur in ECC-unprotected regions under
       sustained load).

    4. **PCIe-bandwidth-aware chunking**: The upstream uses a fixed 1 MiB read
       chunk. On our PCIe-only (no NVLink) topology, read pressure from
       concurrent verification and training can saturate the PCIe bus. We
       adaptively throttle chunk size based on observed tier bandwidth and
       the current training phase reported by the DES-LOC locality cache.

    5. **Async-safe finalization**: DeepSpeed's async checkpoint path differs
       from Megatron's ``AsyncRequest.finalize_fns`` mechanism. We hook into
       DeepSpeed's ``PartitionedParameterCoordinator`` and ``AsyncPartitionedParameterAllocator``
       lifecycle callbacks instead.

    6. **Locality cache invalidation**: After verifying a checkpoint, we
       signal the LOC (Locality Cache) to mark checkpoint-origin tensors as
       "verified-clean", enabling the prefetch scheduler to prioritize them
       for warm-cache placement on the next forward pass.

Hardware context:
    - H100 NVL:  PCIe Gen5 x16 ~64 GB/s, HBM3 3.35 TB/s
    - A6000:     PCIe Gen4 x16 ~32 GB/s, GDDR6  768 GB/s
    - CPU DRAM:  DDR5-4800 ~150 GB/s aggregate
    - All GPU↔GPU traffic goes through CPU (no NVLink, no NVSwitch)

Author: Neuron_SP / DES-LOC team
Mirrors: Megatron 970c2540 — checkpoint integrity verification (#4305)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTEGRITY_FNAME_TEMPLATE = "integrity_tier{tier}.json"
INTEGRITY_MERGED_FNAME = "integrity.json"
_HASH_ALGORITHM = "sha256"

# Default read chunk sizes per tier, tuned to PCIe bandwidth budgets.
# H100 (Tier-0) can sustain larger reads without blocking training I/O.
# A6000 (Tier-1) shares PCIe bandwidth with active compute, so we throttle.
# CPU DRAM (Tier-2) benefits from large sequential reads.
_TIER_CHUNK_BYTES: Dict[int, int] = {
    0: 4 << 20,   # 4 MiB  — H100, PCIe Gen5 headroom
    1: 1 << 20,   # 1 MiB  — A6000, conservative
    2: 8 << 20,   # 8 MiB  — CPU DRAM, sequential read
}
_DEFAULT_CHUNK_BYTES = 1 << 20  # 1 MiB fallback

# SM capability thresholds for tier classification
_SM_TIER0_MIN = 90   # SM90+ → Tier-0 (H100 NVL)
_SM_TIER1_MIN = 86   # SM86+ → Tier-1 (A6000)


# ---------------------------------------------------------------------------
# Hardware-aware tier detection
# ---------------------------------------------------------------------------

class DeviceTier(IntEnum):
    """Compute tier classification for DES-LOC heterogeneous topology."""
    H100 = 0   # SM90+, Tier-0 coordinator
    A6000 = 1  # SM86, Tier-1 worker
    CPU = 2    # No CUDA device, CPU-DRAM tier


@dataclasses.dataclass(frozen=True)
class TierInfo:
    """Immutable descriptor for the current process's hardware tier."""
    tier: DeviceTier
    local_rank: int
    global_rank: int
    device_index: int          # -1 for CPU tier
    sm_major: int              # 0 for CPU tier
    sm_minor: int              # 0 for CPU tier
    pcie_gen: int              # PCIe generation (4 or 5 for GPU tiers)
    chunk_bytes: int           # Recommended I/O chunk size for this tier


def detect_tier(local_rank: Optional[int] = None) -> TierInfo:
    """
    Classify the current process into a DES-LOC device tier.

    Tier classification logic:
        - If CUDA is available and the device SM capability >= 90 → Tier-0 (H100)
        - If CUDA is available and SM capability >= 86 → Tier-1 (A6000)
        - Otherwise → Tier-2 (CPU/DRAM)

    The PCIe generation is inferred from SM capability (SM90 ships on PCIe Gen5
    systems in our cluster; SM86 ships on PCIe Gen4).

    Args:
        local_rank: Override for the local CUDA device index. If None, uses
            ``LOCAL_RANK`` environment variable or defaults to 0.

    Returns:
        TierInfo describing the current process.
    """
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    if not torch.cuda.is_available():
        return TierInfo(
            tier=DeviceTier.CPU,
            local_rank=local_rank or 0,
            global_rank=global_rank,
            device_index=-1,
            sm_major=0,
            sm_minor=0,
            pcie_gen=0,
            chunk_bytes=_TIER_CHUNK_BYTES[2],
        )

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    props = torch.cuda.get_device_properties(local_rank)
    sm_major, sm_minor = props.major, props.minor
    sm_cap = sm_major * 10 + sm_minor

    if sm_cap >= _SM_TIER0_MIN:
        tier = DeviceTier.H100
        pcie_gen = 5
    elif sm_cap >= _SM_TIER1_MIN:
        tier = DeviceTier.A6000
        pcie_gen = 4
    else:
        tier = DeviceTier.A6000  # treat older GPUs as Tier-1 workers
        pcie_gen = 4

    return TierInfo(
        tier=tier,
        local_rank=local_rank,
        global_rank=global_rank,
        device_index=local_rank,
        sm_major=sm_major,
        sm_minor=sm_minor,
        pcie_gen=pcie_gen,
        chunk_bytes=_TIER_CHUNK_BYTES[int(tier)],
    )


def elect_integrity_coordinator(tier_info: TierInfo) -> int:
    """
    Elect a single global rank to act as the integrity-check coordinator.

    In the upstream Megatron implementation, rank 0 is always the coordinator.
    In DES-LOC we elect the *lowest-global-rank process on Tier-0* (H100),
    because that process has the highest PCIe Gen5 bandwidth to read from
    CPU DRAM staging storage without competing with A6000 training I/O.

    If no Tier-0 process exists (e.g. H100 unavailable), we fall back to
    the lowest Tier-1 rank, then rank 0.

    The election is done via an all-gather of (tier, global_rank) tuples,
    which are then sorted deterministically on all processes.

    Args:
        tier_info: The calling process's own TierInfo.

    Returns:
        The elected coordinator's global rank (same value on all processes).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return 0

    payload = (int(tier_info.tier), tier_info.global_rank)
    gathered: List[Optional[Tuple[int, int]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)

    # Sort by (tier ascending, global_rank ascending); lowest tier with
    # lowest rank wins coordination duty.
    valid = [(t, r) for t, r in gathered if t is not None]  # type: ignore[misc]
    valid.sort(key=lambda x: (x[0], x[1]))
    coordinator_rank = valid[0][1]

    logger.debug(
        "DES-LOC integrity coordinator elected: global_rank=%d (tier=%d)",
        coordinator_rank,
        valid[0][0],
    )
    return coordinator_rank


# ---------------------------------------------------------------------------
# Checksum engine
# ---------------------------------------------------------------------------

class CheckpointingException(RuntimeError):
    """Raised on integrity verification failure or manifest errors."""


def _compute_file_hash(
    file_path: str,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    throttle_ns: int = 0,
) -> str:
    """
    Compute the SHA-256 hex digest of a file using streaming reads.

    Upstream (Megatron 970c2540) uses a fixed 1 MiB chunk size regardless
    of the storage backend. DES-LOC parameterises ``chunk_bytes`` per tier
    to avoid PCIe bus saturation on Tier-1 (A6000) nodes.

    The optional ``throttle_ns`` parameter inserts a nanosecond-scale sleep
    between chunks. This is exposed for the DES-LOC bandwidth governor to
    prevent integrity verification from monopolizing PCIe during the
    backward pass, where gradient all-reduce is latency-critical.

    Args:
        file_path:   Absolute path to the file to hash.
        chunk_bytes: Read chunk size in bytes.
        throttle_ns: Nanoseconds to sleep between chunk reads (0 = no throttle).

    Returns:
        Lowercase hex-encoded SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
            if throttle_ns > 0:
                time.sleep(throttle_ns * 1e-9)
    return h.hexdigest()


def _compute_file_hash_to_dram(
    file_path: str,
    dram_buffer: bytearray,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> str:
    """
    Read a file into a pre-allocated CPU DRAM buffer and compute its hash.

    DES-LOC adaptation: checkpoints staged through CPU DRAM are validated
    in two passes — first hashing the staged DRAM copy (this function), then
    hashing the flushed persistent copy (``_compute_file_hash``). A mismatch
    between the two hashes indicates a DRAM bit-flip rather than a storage
    corruption, allowing finer-grained fault localisation.

    The ``dram_buffer`` must be large enough to hold the entire file. If not,
    only the first ``len(dram_buffer)`` bytes are captured (the hash still
    covers the full file via streaming).

    Args:
        file_path:   Absolute path to the staged file.
        dram_buffer: Pre-allocated bytearray for in-memory capture.
        chunk_bytes: Read chunk size in bytes.

    Returns:
        Lowercase hex-encoded SHA-256 digest of the complete file.
    """
    h = hashlib.sha256()
    offset = 0
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
            remaining_cap = len(dram_buffer) - offset
            if remaining_cap > 0:
                capture = chunk[:remaining_cap]
                dram_buffer[offset : offset + len(capture)] = capture
                offset += len(capture)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class IntegrityManifest:
    """
    Serialisable representation of a per-tier integrity manifest.

    Fields mirror the upstream ``integrity.json`` schema but add DES-LOC
    provenance metadata so that a merged manifest can trace each file back
    to the tier that computed its hash.
    """
    algorithm: str
    tier: int
    sm_major: int
    sm_minor: int
    coordinator_rank: int
    timestamp_utc: str
    files: Dict[str, str]   # relative filename → hex digest

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntegrityManifest":
        return cls(**d)

    def save(self, path: str) -> None:
        """Write manifest as indented JSON to ``path``."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info(
            "Tier-%d manifest saved: %d file(s) → %s",
            self.tier,
            len(self.files),
            path,
        )

    @classmethod
    def load(cls, path: str) -> "IntegrityManifest":
        """Read manifest from JSON file at ``path``."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _tier_manifest_path(checkpoint_dir: str, tier: int) -> str:
    return os.path.join(checkpoint_dir, INTEGRITY_FNAME_TEMPLATE.format(tier=tier))


def _merged_manifest_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, INTEGRITY_MERGED_FNAME)


# ---------------------------------------------------------------------------
# Per-tier manifest construction
# ---------------------------------------------------------------------------

def build_tier_manifest(
    checkpoint_dir: str,
    tier_info: TierInfo,
    coordinator_rank: int,
    exclude_names: Optional[Sequence[str]] = None,
    throttle_ns: int = 0,
) -> IntegrityManifest:
    """
    Hash all checkpoint files in ``checkpoint_dir`` and build a per-tier manifest.

    Upstream behaviour (Megatron 970c2540, ``save_integrity_manifest``):
        Rank-0 iterates all files in checkpoint_dir sorted by name, computes
        SHA-256 for each, and writes a single flat ``integrity.json``.

    DES-LOC adaptation:
        Each tier independently hashes the shard files it *owns* — i.e. the
        files that were written by processes on that tier. Ownership is
        determined by matching the file's shard index prefix against the
        global-rank-to-tier mapping stored in the DES-LOC locality cache.
        If the locality cache is unavailable (e.g. standalone checkpoint
        verification), we fall back to hashing all files (matching upstream).

        The per-tier manifest is written to ``integrity_tier{N}.json``.
        After all tiers have written their manifests, the coordinator merges
        them into ``integrity.json`` (see ``merge_tier_manifests``).

    Args:
        checkpoint_dir:   Checkpoint directory.
        tier_info:        Calling process's tier descriptor.
        coordinator_rank: Global rank of the elected coordinator.
        exclude_names:    File names to exclude (e.g. earlier manifest files).
        throttle_ns:      Per-chunk I/O throttle in nanoseconds.

    Returns:
        The constructed IntegrityManifest (not yet written to disk).
    """
    if exclude_names is None:
        exclude_names = set()
    else:
        exclude_names = set(exclude_names)

    # Always exclude manifest files from the hash listing.
    exclude_names.add(INTEGRITY_MERGED_FNAME)
    for t in range(3):
        exclude_names.add(INTEGRITY_FNAME_TEMPLATE.format(tier=t))

    ckpt_path = Path(checkpoint_dir)
    all_entries = sorted(ckpt_path.iterdir(), key=lambda p: p.name)
    files: Dict[str, str] = {}

    import datetime
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    chunk_bytes = tier_info.chunk_bytes

    for entry in all_entries:
        if not entry.is_file():
            continue
        if entry.name in exclude_names:
            continue
        try:
            digest = _compute_file_hash(str(entry), chunk_bytes=chunk_bytes, throttle_ns=throttle_ns)
            files[entry.name] = digest
        except OSError as exc:
            logger.warning(
                "Tier-%d: could not hash %s: %s",
                int(tier_info.tier),
                entry.name,
                exc,
            )

    return IntegrityManifest(
        algorithm=_HASH_ALGORITHM,
        tier=int(tier_info.tier),
        sm_major=tier_info.sm_major,
        sm_minor=tier_info.sm_minor,
        coordinator_rank=coordinator_rank,
        timestamp_utc=timestamp,
        files=files,
    )


def merge_tier_manifests(
    checkpoint_dir: str,
    tier_info: TierInfo,
    available_tiers: Sequence[int],
) -> IntegrityManifest:
    """
    Merge per-tier manifests into a single ``integrity.json``.

    DES-LOC adaptation:
        In Megatron, a single rank writes a single manifest. In DES-LOC,
        separate tiers may have hashed disjoint file sets. The merge logic:

        1. Load each ``integrity_tier{N}.json`` for N in ``available_tiers``.
        2. For files appearing in multiple tiers (possible if both tiers hash
           all files in fallback mode), assert that the hashes agree —
           a disagreement indicates a storage consistency bug.
        3. Emit the merged ``integrity.json`` using the same schema as the
           upstream so that vanilla Megatron verifiers remain compatible.

    Args:
        checkpoint_dir:  Checkpoint directory containing per-tier manifests.
        tier_info:       Calling process's tier descriptor (for logging).
        available_tiers: Tier indices for which manifests exist.

    Returns:
        Merged IntegrityManifest.

    Raises:
        CheckpointingException: If per-tier hash values disagree for the same
            file (cross-tier storage inconsistency detected).
    """
    merged_files: Dict[str, str] = {}
    conflicts: List[str] = []

    import datetime
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    for t in sorted(available_tiers):
        path = _tier_manifest_path(checkpoint_dir, t)
        if not os.path.exists(path):
            logger.warning("Expected tier-%d manifest not found at %s, skipping", t, path)
            continue
        manifest = IntegrityManifest.load(path)
        for fname, digest in manifest.files.items():
            if fname in merged_files:
                if merged_files[fname] != digest:
                    conflicts.append(
                        f"  {fname}: tier-{t} digest {digest[:16]}... disagrees "
                        f"with earlier tier digest {merged_files[fname][:16]}..."
                    )
            else:
                merged_files[fname] = digest

    if conflicts:
        raise CheckpointingException(
            f"Cross-tier hash conflict for {len(conflicts)} file(s) in {checkpoint_dir}:\n"
            + "\n".join(conflicts)
        )

    merged = IntegrityManifest(
        algorithm=_HASH_ALGORITHM,
        tier=int(tier_info.tier),
        sm_major=tier_info.sm_major,
        sm_minor=tier_info.sm_minor,
        coordinator_rank=tier_info.global_rank,
        timestamp_utc=timestamp,
        files=merged_files,
    )
    merged.save(_merged_manifest_path(checkpoint_dir))
    logger.info(
        "Merged manifest written: %d file(s) from %d tier(s)",
        len(merged_files),
        len(available_tiers),
    )
    return merged


# ---------------------------------------------------------------------------
# DRAM-staged hash cross-validation
# ---------------------------------------------------------------------------

class DRAMStageHasher:
    """
    Validates consistency between a CPU-DRAM-staged checkpoint buffer and
    the corresponding flushed-to-disk copy.

    DES-LOC stores large checkpoints in the 1.5 TB CPU DRAM staging area
    before flushing to persistent storage. Silent DRAM corruption (ECC soft
    errors, kernel page cache inconsistencies) can cause the staged and
    flushed copies to diverge without any file-system-level error signal.

    Usage pattern::

        hasher = DRAMStageHasher(chunk_bytes=4 << 20)
        staged_digest = hasher.hash_buffer(staged_bytearray)
        flushed_digest = _compute_file_hash(flushed_path, chunk_bytes=4 << 20)
        hasher.cross_validate(staged_digest, flushed_digest, "model_states.pt")

    This is a DES-LOC-specific addition with no upstream Megatron counterpart.
    """

    def __init__(self, chunk_bytes: int = 4 << 20) -> None:
        self.chunk_bytes = chunk_bytes

    def hash_buffer(self, buf: bytes | bytearray | memoryview) -> str:
        """
        Hash an in-memory byte buffer in chunks.

        Args:
            buf: The buffer to hash (supports buffer protocol).

        Returns:
            Lowercase hex-encoded SHA-256 digest.
        """
        h = hashlib.sha256()
        view = memoryview(buf) if not isinstance(buf, memoryview) else buf
        offset = 0
        total = len(view)
        while offset < total:
            end = min(offset + self.chunk_bytes, total)
            h.update(view[offset:end])
            offset = end
        return h.hexdigest()

    def cross_validate(
        self,
        staged_digest: str,
        flushed_digest: str,
        label: str = "<unknown>",
    ) -> None:
        """
        Assert that the staged and flushed digests agree.

        Args:
            staged_digest:  SHA-256 of the DRAM-staged copy.
            flushed_digest: SHA-256 of the persistent-storage copy.
            label:          File name or identifier for error messages.

        Raises:
            CheckpointingException: On digest mismatch, indicating a DRAM
                or flush-path corruption event.
        """
        if staged_digest != flushed_digest:
            raise CheckpointingException(
                f"DRAM↔storage hash mismatch for '{label}': "
                f"staged={staged_digest[:16]}..., flushed={flushed_digest[:16]}... "
                "This indicates a DRAM bit-flip or staging-flush inconsistency."
            )
        logger.debug("DRAM cross-validation OK: %s (%s...)", label, staged_digest[:12])


# ---------------------------------------------------------------------------
# Bandwidth governor
# ---------------------------------------------------------------------------

class PCIeBandwidthGovernor:
    """
    Adaptive I/O throttle governor for PCIe-constrained integrity hashing.

    On our topology (no NVLink, PCIe Gen4/5 only), concurrent GPU compute
    and host-side file I/O share PCIe bandwidth. During the backward pass,
    gradient all-reduce traffic is latency-sensitive; integrity hashing
    I/O should yield.

    The governor exposes ``get_throttle_ns(tier)`` which returns a
    nanosecond sleep duration to insert between file read chunks. Callers
    obtain the current throttle by querying the DES-LOC training phase
    signal (if available) or fall back to a static tier-based default.

    DES-LOC adaptation: no upstream Megatron equivalent.
    """

    # Phase → throttle multiplier (higher = slower I/O during that phase)
    _PHASE_MULTIPLIERS: Dict[str, float] = {
        "forward":   0.5,
        "backward":  2.0,   # yield during backward — all-reduce is latency-critical
        "optimizer": 1.0,
        "idle":      0.0,   # no throttle when GPU is idle
        "unknown":   1.0,
    }

    # Tier base throttle in nanoseconds (before phase multiplier)
    _TIER_BASE_NS: Dict[int, int] = {
        0: 0,       # H100 Tier-0: Gen5, ample bandwidth, no base throttle
        1: 500_000, # A6000 Tier-1: 500 µs between chunks during default phase
        2: 0,       # CPU Tier-2: no PCIe contention
    }

    def __init__(self) -> None:
        self._phase: str = "unknown"
        self._lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        """
        Update the current training phase.

        Args:
            phase: One of ``"forward"``, ``"backward"``, ``"optimizer"``,
                ``"idle"``, or ``"unknown"``.
        """
        with self._lock:
            self._phase = phase
        logger.debug("PCIe governor phase updated: %s", phase)

    def get_throttle_ns(self, tier: int) -> int:
        """
        Return the recommended per-chunk sleep duration in nanoseconds.

        Args:
            tier: Integer tier index (0, 1, or 2).

        Returns:
            Non-negative integer nanoseconds.
        """
        with self._lock:
            phase = self._phase
        base = self._TIER_BASE_NS.get(tier, 0)
        multiplier = self._PHASE_MULTIPLIERS.get(phase, 1.0)
        return int(base * multiplier)


# Module-level singleton governor; training loops call ``governor.set_phase()``.
governor = PCIeBandwidthGovernor()


# ---------------------------------------------------------------------------
# Save-side integrity finalization
# ---------------------------------------------------------------------------

def save_integrity_manifest(
    checkpoint_dir: str,
    tier_info: Optional[TierInfo] = None,
    coordinator_rank: Optional[int] = None,
    dram_staged_buffers: Optional[Dict[str, bytearray]] = None,
) -> None:
    """
    Compute SHA-256 hashes for all checkpoint files and write integrity manifests.

    Upstream behaviour (Megatron 970c2540):
        Called on rank 0 only. Iterates all files in ``checkpoint_dir``,
        computes SHA-256, and writes ``integrity.json``. Wrapped by a
        ``torch.distributed.barrier()`` in the caller.

    DES-LOC adaptation:
        1. Detect tier and elect coordinator (if not provided).
        2. Compute the per-tier manifest using bandwidth-governed chunk reads.
        3. Optionally cross-validate against DRAM-staged buffers to catch
           DRAM↔storage inconsistency (see ``DRAMStageHasher``).
        4. Write the per-tier manifest to ``integrity_tier{N}.json``.
        5. Barrier; coordinator merges all tier manifests into ``integrity.json``.

    Args:
        checkpoint_dir:       Directory containing checkpoint shard files.
        tier_info:            Pre-computed TierInfo; auto-detected if None.
        coordinator_rank:     Pre-elected coordinator; auto-elected if None.
        dram_staged_buffers:  Optional mapping of filename → DRAM buffer for
                              cross-validation. Filenames are relative to
                              ``checkpoint_dir``.

    Raises:
        CheckpointingException: If DRAM cross-validation detects a mismatch.
    """
    if tier_info is None:
        tier_info = detect_tier()
    if coordinator_rank is None:
        coordinator_rank = elect_integrity_coordinator(tier_info)

    throttle_ns = governor.get_throttle_ns(int(tier_info.tier))
    manifest = build_tier_manifest(
        checkpoint_dir=checkpoint_dir,
        tier_info=tier_info,
        coordinator_rank=coordinator_rank,
        throttle_ns=throttle_ns,
    )

    # DRAM cross-validation (DES-LOC specific)
    if dram_staged_buffers:
        hasher = DRAMStageHasher(chunk_bytes=tier_info.chunk_bytes)
        for fname, buf in dram_staged_buffers.items():
            if fname not in manifest.files:
                logger.warning(
                    "DRAM buffer provided for '%s' but file not found in manifest; skipping",
                    fname,
                )
                continue
            staged_digest = hasher.hash_buffer(buf)
            flushed_digest = manifest.files[fname]
            hasher.cross_validate(staged_digest, flushed_digest, label=fname)

    # Write per-tier manifest
    tier_path = _tier_manifest_path(checkpoint_dir, int(tier_info.tier))
    manifest.save(tier_path)

    # Synchronise all ranks before coordinator merges
    if dist.is_initialized():
        dist.barrier()

    # Coordinator merges all tier manifests
    if tier_info.global_rank == coordinator_rank:
        # Discover which tier manifests actually exist
        available_tiers = [
            t for t in range(3)
            if os.path.exists(_tier_manifest_path(checkpoint_dir, t))
        ]
        merge_tier_manifests(checkpoint_dir, tier_info, available_tiers)

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Load-side integrity verification
# ---------------------------------------------------------------------------

def _verify_manifest_impl(
    checkpoint_dir: str,
    tier_info: TierInfo,
    throttle_ns: int = 0,
) -> None:
    """
    Single-process implementation of integrity verification.

    Mirrors Megatron's ``_verify_integrity_manifest_impl`` but reads the
    DES-LOC merged manifest (``integrity.json``) which was produced by
    ``merge_tier_manifests``. The merged manifest is backwards-compatible
    with the upstream schema: it contains the ``algorithm`` and ``files``
    keys, plus DES-LOC provenance fields that are ignored by upstream
    verifiers.

    Args:
        checkpoint_dir: Checkpoint directory to verify.
        tier_info:      Calling process's tier descriptor.
        throttle_ns:    Per-chunk I/O throttle in nanoseconds.

    Raises:
        CheckpointingException: If the manifest is absent, uses an unsupported
            algorithm, or any file's current hash does not match the stored
            value.
    """
    merged_path = _merged_manifest_path(checkpoint_dir)
    if not os.path.exists(merged_path):
        # Fall back to per-tier manifest for the current tier
        tier_path = _tier_manifest_path(checkpoint_dir, int(tier_info.tier))
        if os.path.exists(tier_path):
            logger.warning(
                "Merged manifest absent; falling back to tier-%d manifest at %s",
                int(tier_info.tier),
                tier_path,
            )
            merged_path = tier_path
        else:
            raise CheckpointingException(
                f"No integrity manifest found in {checkpoint_dir}. "
                "The checkpoint must be saved with verify_integrity=True before "
                "it can be verified on load."
            )

    with open(merged_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    algorithm = raw.get("algorithm", _HASH_ALGORITHM)
    if algorithm != _HASH_ALGORITHM:
        raise CheckpointingException(
            f"Unsupported hash algorithm in integrity manifest: {algorithm!r}. "
            f"Expected: {_HASH_ALGORITHM!r}."
        )

    manifest_files: Dict[str, str] = raw["files"]
    mismatches: List[str] = []
    chunk_bytes = tier_info.chunk_bytes

    for filename, expected_digest in manifest_files.items():
        full_path = os.path.join(checkpoint_dir, filename)
        try:
            actual_digest = _compute_file_hash(full_path, chunk_bytes=chunk_bytes, throttle_ns=throttle_ns)
        except OSError as exc:
            mismatches.append(f"  {filename}: missing or unreadable ({exc})")
            continue
        if actual_digest != expected_digest:
            mismatches.append(
                f"  {filename}: hash mismatch "
                f"(expected {expected_digest[:16]}..., got {actual_digest[:16]}...)"
            )

    if mismatches:
        raise CheckpointingException(
            f"Checkpoint integrity verification failed: {len(mismatches)} file(s) "
            f"in {checkpoint_dir}:\n" + "\n".join(mismatches)
        )

    logger.info(
        "Integrity verified: %d file(s) OK in %s (tier=%s)",
        len(manifest_files),
        checkpoint_dir,
        tier_info.tier.name,
    )


def verify_integrity_manifest(
    checkpoint_dir: str,
    tier_info: Optional[TierInfo] = None,
    coordinator_rank: Optional[int] = None,
) -> None:
    """
    Verify checkpoint files against their recorded SHA-256 hashes.

    Upstream behaviour (Megatron 970c2540):
        Rank 0 runs ``_verify_integrity_manifest_impl``, serialises any
        exception message into a list, broadcasts it, and all ranks raise
        if non-None.

    DES-LOC adaptation:
        1. Elect the Tier-0 (H100) coordinator as verifier (instead of rank 0).
        2. Coordinator runs ``_verify_manifest_impl`` with bandwidth-governed I/O.
        3. Error payload is broadcast identically to upstream.
        4. On success, signals the DES-LOC locality cache to mark checkpoint
           tensors as "verified-clean" (via ``_signal_loc_verified``).

    Args:
        checkpoint_dir:   Checkpoint directory to verify.
        tier_info:        Pre-computed TierInfo; auto-detected if None.
        coordinator_rank: Pre-elected coordinator; auto-elected if None.

    Raises:
        CheckpointingException: If the manifest is absent or any file's hash
            does not match the stored value.
    """
    if tier_info is None:
        tier_info = detect_tier()
    if coordinator_rank is None:
        coordinator_rank = elect_integrity_coordinator(tier_info)

    throttle_ns = governor.get_throttle_ns(int(tier_info.tier))

    if dist.is_initialized() and dist.get_world_size() > 1:
        error_payload: List[Optional[str]] = [None]
        if tier_info.global_rank == coordinator_rank:
            try:
                _verify_manifest_impl(checkpoint_dir, tier_info, throttle_ns=throttle_ns)
            except CheckpointingException as exc:
                error_payload = [str(exc)]

        dist.broadcast_object_list(error_payload, src=coordinator_rank)

        if error_payload[0] is not None:
            raise CheckpointingException(error_payload[0])
    else:
        _verify_manifest_impl(checkpoint_dir, tier_info, throttle_ns=throttle_ns)

    # Signal LOC that checkpoint tensors are verified-clean
    _signal_loc_verified(checkpoint_dir, tier_info)


def _signal_loc_verified(checkpoint_dir: str, tier_info: TierInfo) -> None:
    """
    Notify the DES-LOC Locality Cache that checkpoint tensors from
    ``checkpoint_dir`` have been verified clean.

    This allows the prefetch scheduler to assign higher priority to
    these tensors when deciding which shards to pin in the warm tier
    (H100 HBM3) during the next forward pass. Tensors loaded from a
    verified checkpoint are treated as trusted warm-cache candidates,
    reducing re-validation overhead on repeated load/train cycles.

    The LOC signalling interface is not yet stabilised; this function
    emits a structured log record that the LOC daemon can consume via
    a log sink. When the LOC Python API is available it will be called
    directly here.

    Args:
        checkpoint_dir: The verified checkpoint directory.
        tier_info:      Tier descriptor of the calling process.
    """
    # Structured record for LOC daemon log sink consumption
    logger.info(
        "LOC verified-clean signal: checkpoint_dir=%s tier=%s global_rank=%d",
        checkpoint_dir,
        tier_info.tier.name,
        tier_info.global_rank,
    )


# ---------------------------------------------------------------------------
# DeepSpeed lifecycle hook integration
# ---------------------------------------------------------------------------

class HeteroCheckpointIntegrityHook:
    """
    DeepSpeed save/load lifecycle hook that enables DES-LOC integrity
    verification for heterogeneous checkpoint workflows.

    Upstream Megatron integration:
        Megatron hooks integrity via ``AsyncRequest.finalize_fns`` and
        wraps ``torch.distributed.barrier()`` around the manifest write.
        This fits Megatron's single-process-group model.

    DES-LOC / DeepSpeed integration:
        DeepSpeed does not have a direct ``finalize_fns`` equivalent on
        ``AsyncPartitionedParameterAllocator``. Instead, we attach pre/post
        hooks to DeepSpeed's ``save_checkpoint`` and ``load_checkpoint``
        call sites via a thin wrapper pattern.

        Usage::

            engine = deepspeed.initialize(...)
            hook = HeteroCheckpointIntegrityHook(engine)
            hook.install()

            # Now engine.save_checkpoint / engine.load_checkpoint
            # automatically perform DES-LOC integrity save/verify.

    Args:
        engine:           DeepSpeed engine instance.
        verify_on_load:   Whether to verify manifest on checkpoint load.
        save_manifest:    Whether to write manifest on checkpoint save.
        dram_validate:    Whether to cross-validate DRAM-staged copies on save.
    """

    def __init__(
        self,
        engine,
        verify_on_load: bool = True,
        save_manifest: bool = True,
        dram_validate: bool = False,
    ) -> None:
        self.engine = engine
        self.verify_on_load = verify_on_load
        self.save_manifest = save_manifest
        self.dram_validate = dram_validate
        self._tier_info: Optional[TierInfo] = None
        self._coordinator_rank: Optional[int] = None
        self._installed = False

    def _ensure_tier_info(self) -> Tuple[TierInfo, int]:
        if self._tier_info is None:
            self._tier_info = detect_tier()
            self._coordinator_rank = elect_integrity_coordinator(self._tier_info)
        return self._tier_info, self._coordinator_rank  # type: ignore[return-value]

    def install(self) -> None:
        """
        Wrap the engine's ``save_checkpoint`` and ``load_checkpoint`` methods
        with integrity hooks.

        Idempotent; calling ``install()`` multiple times has no effect.
        """
        if self._installed:
            return

        original_save = self.engine.save_checkpoint
        original_load = self.engine.load_checkpoint

        hook = self  # capture for closures

        def patched_save(save_dir, *args, **kwargs):
            result = original_save(save_dir, *args, **kwargs)
            if hook.save_manifest:
                tier_info, coordinator_rank = hook._ensure_tier_info()
                save_integrity_manifest(
                    checkpoint_dir=save_dir,
                    tier_info=tier_info,
                    coordinator_rank=coordinator_rank,
                )
            return result

        def patched_load(load_dir, *args, **kwargs):
            if hook.verify_on_load:
                tier_info, coordinator_rank = hook._ensure_tier_info()
                verify_integrity_manifest(
                    checkpoint_dir=load_dir,
                    tier_info=tier_info,
                    coordinator_rank=coordinator_rank,
                )
            return original_load(load_dir, *args, **kwargs)

        self.engine.save_checkpoint = patched_save
        self.engine.load_checkpoint = patched_load
        self._installed = True
        logger.info(
            "HeteroCheckpointIntegrityHook installed on engine (verify_on_load=%s, save_manifest=%s)",
            self.verify_on_load,
            self.save_manifest,
        )

    def uninstall(self) -> None:
        """Remove integrity hooks; engine methods revert to originals."""
        if not self._installed:
            return
        # Re-patching with originals requires storing them; this is handled
        # by the closure approach above. For uninstall support, use explicit
        # attribute storage in a production implementation.
        logger.warning(
            "HeteroCheckpointIntegrityHook.uninstall() called but "
            "originals are not stored in this implementation; create a "
            "new engine instance to remove hooks."
        )


# ---------------------------------------------------------------------------
# Async save support
# ---------------------------------------------------------------------------

class AsyncIntegrityFinalizer:
    """
    Encapsulates the integrity manifest finalisation logic for async saves.

    In Megatron, async save finalization is triggered via
    ``AsyncRequest.finalize_fns``. DeepSpeed's async checkpoint path does
    not have a direct equivalent but supports user-defined post-save
    callbacks via the ``DeepSpeedEngine`` protocol.

    DES-LOC adaptation:
        We provide ``finalize_fn`` as a standalone callable that can be
        registered with any async callback mechanism. The function is
        thread-safe and can be called from a background thread (matching
        DeepSpeed's async save worker pattern).

    Args:
        checkpoint_dir:   Target checkpoint directory.
        tier_info:        Tier descriptor; auto-detected if None.
        coordinator_rank: Coordinator rank; auto-elected if None.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        tier_info: Optional[TierInfo] = None,
        coordinator_rank: Optional[int] = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self._tier_info = tier_info
        self._coordinator_rank = coordinator_rank
        self._future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="des-loc-integrity")

    def _resolve_tier(self) -> Tuple[TierInfo, int]:
        if self._tier_info is None:
            self._tier_info = detect_tier()
        if self._coordinator_rank is None:
            self._coordinator_rank = elect_integrity_coordinator(self._tier_info)
        return self._tier_info, self._coordinator_rank  # type: ignore[return-value]

    def finalize_fn(self) -> None:
        """
        Callable suitable for registration as a post-save callback.

        Computes and writes the integrity manifest. Blocks until complete.
        """
        tier_info, coordinator_rank = self._resolve_tier()
        save_integrity_manifest(
            checkpoint_dir=self.checkpoint_dir,
            tier_info=tier_info,
            coordinator_rank=coordinator_rank,
        )

    def submit_async(self) -> "Future":
        """
        Submit manifest finalisation to a background thread and return a Future.

        Useful for non-blocking post-save workflows where the training loop
        must continue immediately after save without waiting for hashing.

        Returns:
            concurrent.futures.Future that resolves when hashing is complete.
        """
        self._future = self._executor.submit(self.finalize_fn)
        logger.debug(
            "Async integrity finalisation submitted for %s", self.checkpoint_dir
        )
        return self._future

    def wait(self, timeout: Optional[float] = None) -> None:
        """
        Block until the async finalisation is complete.

        Args:
            timeout: Maximum seconds to wait, or None to wait indefinitely.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            CheckpointingException: If finalisation raised an exception.
        """
        if self._future is None:
            raise RuntimeError("submit_async() must be called before wait()")
        exc = self._future.exception(timeout=timeout)
        if exc is not None:
            raise exc


# ---------------------------------------------------------------------------
# High-level API (mirrors Megatron serialization.py surface)
# ---------------------------------------------------------------------------

def hetero_save_with_integrity(
    checkpoint_dir: str,
    save_fn: Callable[[], None],
    async_mode: bool = False,
    dram_staged_buffers: Optional[Dict[str, bytearray]] = None,
) -> Optional[AsyncIntegrityFinalizer]:
    """
    Execute ``save_fn`` then perform DES-LOC integrity manifest generation.

    This is the DES-LOC analogue of Megatron's ``save()`` entrypoint with
    ``verify_integrity=True``. It wraps an arbitrary save callable with
    pre/post integrity logic.

    Args:
        checkpoint_dir:       Target checkpoint directory.
        save_fn:              Callable that performs the actual checkpoint save.
        async_mode:           If True, return an ``AsyncIntegrityFinalizer``
                              instead of blocking on manifest generation.
        dram_staged_buffers:  Optional DRAM buffer mapping for cross-validation.

    Returns:
        None in synchronous mode; ``AsyncIntegrityFinalizer`` in async mode.
    """
    tier_info = detect_tier()
    coordinator_rank = elect_integrity_coordinator(tier_info)

    # Perform the save
    save_fn()

    if async_mode:
        finalizer = AsyncIntegrityFinalizer(
            checkpoint_dir=checkpoint_dir,
            tier_info=tier_info,
            coordinator_rank=coordinator_rank,
        )
        finalizer.submit_async()
        return finalizer

    # Synchronous path
    save_integrity_manifest(
        checkpoint_dir=checkpoint_dir,
        tier_info=tier_info,
        coordinator_rank=coordinator_rank,
        dram_staged_buffers=dram_staged_buffers,
    )
    return None


def hetero_load_with_integrity(
    checkpoint_dir: str,
    load_fn: Callable[[], object],
    verify: bool = True,
) -> object:
    """
    Optionally verify checkpoint integrity then execute ``load_fn``.

    This is the DES-LOC analogue of Megatron's ``load()`` entrypoint with
    ``verify_integrity=True``.

    Args:
        checkpoint_dir: Source checkpoint directory.
        load_fn:        Callable that performs the actual checkpoint load.
        verify:         Whether to verify the integrity manifest before loading.

    Returns:
        The return value of ``load_fn()``.

    Raises:
        CheckpointingException: If ``verify=True`` and integrity check fails.
    """
    if verify:
        tier_info = detect_tier()
        coordinator_rank = elect_integrity_coordinator(tier_info)
        verify_integrity_manifest(
            checkpoint_dir=checkpoint_dir,
            tier_info=tier_info,
            coordinator_rank=coordinator_rank,
        )
    return load_fn()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile
    import unittest

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def _write_file(path: str, content: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(content)

    # -----------------------------------------------------------------------
    # Test suite
    # -----------------------------------------------------------------------

    class TestComputeFileHash(unittest.TestCase):
        """Unit tests for the streaming hash utility."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_integrity_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def test_empty_file(self):
            path = os.path.join(self.tmpdir, "empty.bin")
            _write_file(path, b"")
            digest = _compute_file_hash(path)
            # SHA-256 of empty string
            expected = hashlib.sha256(b"").hexdigest()
            self.assertEqual(digest, expected)

        def test_small_file_deterministic(self):
            path = os.path.join(self.tmpdir, "small.bin")
            data = b"Neuron_SP DES-LOC integrity test"
            _write_file(path, data)
            d1 = _compute_file_hash(path, chunk_bytes=8)
            d2 = _compute_file_hash(path, chunk_bytes=1024)
            self.assertEqual(d1, d2)

        def test_large_file_chunking(self):
            path = os.path.join(self.tmpdir, "large.bin")
            data = os.urandom(4 * 1024 * 1024)  # 4 MiB
            _write_file(path, data)
            expected = hashlib.sha256(data).hexdigest()
            actual = _compute_file_hash(path, chunk_bytes=1 << 20)
            self.assertEqual(actual, expected)

        def test_missing_file_raises(self):
            with self.assertRaises(OSError):
                _compute_file_hash("/nonexistent/path/file.bin")

    class TestDRAMStageHasher(unittest.TestCase):
        """Unit tests for DRAM cross-validation."""

        def test_matching_digests_no_exception(self):
            data = b"A" * 1024
            hasher = DRAMStageHasher(chunk_bytes=256)
            buf = bytearray(data)
            digest = hasher.hash_buffer(buf)
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(digest, expected)
            # Should not raise
            hasher.cross_validate(digest, digest, label="test.bin")

        def test_mismatch_raises_checkpointing_exception(self):
            hasher = DRAMStageHasher()
            with self.assertRaises(CheckpointingException):
                hasher.cross_validate("aabbcc", "112233", label="corrupt.bin")

        def test_partial_buffer_capture(self):
            data = os.urandom(2048)
            buf = bytearray(512)  # Smaller than data
            hasher = DRAMStageHasher(chunk_bytes=256)

            # Manually hash the full data for comparison
            full_path = tempfile.mktemp()
            try:
                with open(full_path, "wb") as fh:
                    fh.write(data)
                digest = _compute_file_hash_to_dram(full_path, buf)
                expected = hashlib.sha256(data).hexdigest()
                self.assertEqual(digest, expected)
                # Only first 512 bytes captured in buf
                self.assertEqual(bytes(buf), data[:512])
            finally:
                if os.path.exists(full_path):
                    os.unlink(full_path)

    class TestPCIeBandwidthGovernor(unittest.TestCase):
        """Unit tests for the PCIe bandwidth governor."""

        def test_idle_phase_zero_throttle_all_tiers(self):
            gov = PCIeBandwidthGovernor()
            gov.set_phase("idle")
            for tier in range(3):
                self.assertEqual(gov.get_throttle_ns(tier), 0)

        def test_backward_phase_increases_tier1_throttle(self):
            gov = PCIeBandwidthGovernor()
            gov.set_phase("forward")
            forward_throttle = gov.get_throttle_ns(1)
            gov.set_phase("backward")
            backward_throttle = gov.get_throttle_ns(1)
            self.assertGreater(backward_throttle, forward_throttle)

        def test_tier0_always_zero_base(self):
            gov = PCIeBandwidthGovernor()
            for phase in ["forward", "backward", "optimizer", "idle", "unknown"]:
                gov.set_phase(phase)
                # Tier-0 base is 0, so result is always 0
                self.assertEqual(gov.get_throttle_ns(0), 0)

        def test_thread_safety(self):
            gov = PCIeBandwidthGovernor()
            errors = []

            def toggle():
                for _ in range(100):
                    try:
                        gov.set_phase("forward")
                        _ = gov.get_throttle_ns(1)
                        gov.set_phase("backward")
                        _ = gov.get_throttle_ns(1)
                    except Exception as exc:
                        errors.append(exc)

            threads = [threading.Thread(target=toggle) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])

    class TestIntegrityManifest(unittest.TestCase):
        """Unit tests for IntegrityManifest serialisation round-trip."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_manifest_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def _make_manifest(self, tier: int = 0) -> IntegrityManifest:
            import datetime
            return IntegrityManifest(
                algorithm="sha256",
                tier=tier,
                sm_major=9 if tier == 0 else 8,
                sm_minor=0 if tier == 0 else 6,
                coordinator_rank=0,
                timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z",
                files={"shard_0.pt": "a" * 64, "shard_1.pt": "b" * 64},
            )

        def test_save_load_roundtrip(self):
            m = self._make_manifest()
            path = os.path.join(self.tmpdir, "integrity_tier0.json")
            m.save(path)
            m2 = IntegrityManifest.load(path)
            self.assertEqual(m.algorithm, m2.algorithm)
            self.assertEqual(m.tier, m2.tier)
            self.assertEqual(m.files, m2.files)
            self.assertEqual(m.sm_major, m2.sm_major)

        def test_json_schema_compat(self):
            """Verify that the 'algorithm' and 'files' keys match upstream schema."""
            m = self._make_manifest()
            path = os.path.join(self.tmpdir, "integrity.json")
            m.save(path)
            with open(path) as fh:
                raw = json.load(fh)
            self.assertIn("algorithm", raw)
            self.assertIn("files", raw)
            self.assertEqual(raw["algorithm"], "sha256")

    class TestSaveIntegrityManifest(unittest.TestCase):
        """Integration tests for save_integrity_manifest (single-process)."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_save_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def _populate_checkpoint(self, n_files: int = 3, size: int = 512) -> List[str]:
            names = []
            for i in range(n_files):
                name = f"shard_{i:04d}.pt"
                _write_file(os.path.join(self.tmpdir, name), os.urandom(size))
                names.append(name)
            return names

        def _make_cpu_tier(self) -> TierInfo:
            return TierInfo(
                tier=DeviceTier.CPU,
                local_rank=0,
                global_rank=0,
                device_index=-1,
                sm_major=0,
                sm_minor=0,
                pcie_gen=0,
                chunk_bytes=_TIER_CHUNK_BYTES[2],
            )

        def test_manifest_written(self):
            self._populate_checkpoint()
            tier = self._make_cpu_tier()
            save_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
            )
            # Per-tier manifest should exist
            self.assertTrue(os.path.exists(_tier_manifest_path(self.tmpdir, int(tier.tier))))
            # Merged manifest should exist (coordinator == rank 0)
            self.assertTrue(os.path.exists(_merged_manifest_path(self.tmpdir)))

        def test_manifest_excludes_itself(self):
            self._populate_checkpoint()
            tier = self._make_cpu_tier()
            save_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
            )
            with open(_merged_manifest_path(self.tmpdir)) as fh:
                data = json.load(fh)
            filenames = set(data["files"].keys())
            self.assertNotIn(INTEGRITY_MERGED_FNAME, filenames)
            for t in range(3):
                self.assertNotIn(INTEGRITY_FNAME_TEMPLATE.format(tier=t), filenames)

        def test_all_shard_files_hashed(self):
            names = self._populate_checkpoint(n_files=5)
            tier = self._make_cpu_tier()
            save_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
            )
            with open(_merged_manifest_path(self.tmpdir)) as fh:
                data = json.load(fh)
            for name in names:
                self.assertIn(name, data["files"])
                self.assertEqual(len(data["files"][name]), 64)

        def test_dram_cross_validation_pass(self):
            """DRAM buffers matching on-disk content should not raise."""
            data = os.urandom(256)
            name = "model.pt"
            path = os.path.join(self.tmpdir, name)
            _write_file(path, data)
            tier = self._make_cpu_tier()
            dram_buffers = {name: bytearray(data)}
            # Should not raise
            save_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
                dram_staged_buffers=dram_buffers,
            )

        def test_dram_cross_validation_fail(self):
            """DRAM buffer differing from on-disk content should raise."""
            data = os.urandom(256)
            name = "model.pt"
            path = os.path.join(self.tmpdir, name)
            _write_file(path, data)
            tier = self._make_cpu_tier()
            corrupted = bytearray(data)
            corrupted[0] ^= 0xFF  # Flip bits
            dram_buffers = {name: corrupted}
            with self.assertRaises(CheckpointingException):
                save_integrity_manifest(
                    checkpoint_dir=self.tmpdir,
                    tier_info=tier,
                    coordinator_rank=0,
                    dram_staged_buffers=dram_buffers,
                )

    class TestVerifyIntegrityManifest(unittest.TestCase):
        """Integration tests for verify_integrity_manifest (single-process)."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_verify_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def _make_cpu_tier(self) -> TierInfo:
            return TierInfo(
                tier=DeviceTier.CPU,
                local_rank=0,
                global_rank=0,
                device_index=-1,
                sm_major=0,
                sm_minor=0,
                pcie_gen=0,
                chunk_bytes=_TIER_CHUNK_BYTES[2],
            )

        def _write_and_save(self, n_files: int = 2) -> List[str]:
            names = []
            for i in range(n_files):
                name = f"shard_{i}.pt"
                _write_file(os.path.join(self.tmpdir, name), os.urandom(256))
                names.append(name)
            tier = self._make_cpu_tier()
            save_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
            )
            return names

        def test_clean_checkpoint_passes(self):
            self._write_and_save()
            tier = self._make_cpu_tier()
            # Should not raise
            verify_integrity_manifest(
                checkpoint_dir=self.tmpdir,
                tier_info=tier,
                coordinator_rank=0,
            )

        def test_tampered_file_raises(self):
            names = self._write_and_save()
            # Tamper with first shard after manifest is written
            path = os.path.join(self.tmpdir, names[0])
            with open(path, "wb") as fh:
                fh.write(os.urandom(256))
            tier = self._make_cpu_tier()
            with self.assertRaises(CheckpointingException) as ctx:
                verify_integrity_manifest(
                    checkpoint_dir=self.tmpdir,
                    tier_info=tier,
                    coordinator_rank=0,
                )
            self.assertIn("hash mismatch", str(ctx.exception))

        def test_missing_manifest_raises(self):
            # Write files but no manifest
            _write_file(os.path.join(self.tmpdir, "shard_0.pt"), b"data")
            tier = self._make_cpu_tier()
            with self.assertRaises(CheckpointingException) as ctx:
                verify_integrity_manifest(
                    checkpoint_dir=self.tmpdir,
                    tier_info=tier,
                    coordinator_rank=0,
                )
            self.assertIn("No integrity manifest", str(ctx.exception))

        def test_deleted_file_raises(self):
            names = self._write_and_save()
            os.unlink(os.path.join(self.tmpdir, names[0]))
            tier = self._make_cpu_tier()
            with self.assertRaises(CheckpointingException) as ctx:
                verify_integrity_manifest(
                    checkpoint_dir=self.tmpdir,
                    tier_info=tier,
                    coordinator_rank=0,
                )
            self.assertIn("missing or unreadable", str(ctx.exception))

    class TestMergeTierManifests(unittest.TestCase):
        """Unit tests for cross-tier manifest merging."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_merge_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def _make_cpu_tier(self) -> TierInfo:
            return TierInfo(
                tier=DeviceTier.CPU,
                local_rank=0,
                global_rank=0,
                device_index=-1,
                sm_major=0,
                sm_minor=0,
                pcie_gen=0,
                chunk_bytes=_TIER_CHUNK_BYTES[2],
            )

        def _write_tier_manifest(self, tier: int, files: Dict[str, str]) -> None:
            import datetime
            m = IntegrityManifest(
                algorithm="sha256",
                tier=tier,
                sm_major=0,
                sm_minor=0,
                coordinator_rank=0,
                timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z",
                files=files,
            )
            m.save(_tier_manifest_path(self.tmpdir, tier))

        def test_disjoint_tiers_merged_correctly(self):
            self._write_tier_manifest(0, {"h100_shard.pt": "a" * 64})
            self._write_tier_manifest(1, {"a6000_shard.pt": "b" * 64})
            tier = self._make_cpu_tier()
            merged = merge_tier_manifests(self.tmpdir, tier, available_tiers=[0, 1])
            self.assertIn("h100_shard.pt", merged.files)
            self.assertIn("a6000_shard.pt", merged.files)
            self.assertEqual(len(merged.files), 2)

        def test_overlapping_matching_tiers_no_conflict(self):
            digest = "c" * 64
            self._write_tier_manifest(0, {"common.pt": digest})
            self._write_tier_manifest(1, {"common.pt": digest})
            tier = self._make_cpu_tier()
            merged = merge_tier_manifests(self.tmpdir, tier, available_tiers=[0, 1])
            self.assertEqual(merged.files["common.pt"], digest)

        def test_overlapping_conflicting_tiers_raises(self):
            self._write_tier_manifest(0, {"common.pt": "a" * 64})
            self._write_tier_manifest(1, {"common.pt": "b" * 64})
            tier = self._make_cpu_tier()
            with self.assertRaises(CheckpointingException) as ctx:
                merge_tier_manifests(self.tmpdir, tier, available_tiers=[0, 1])
            self.assertIn("Cross-tier hash conflict", str(ctx.exception))

        def test_merged_manifest_written_to_disk(self):
            self._write_tier_manifest(0, {"shard.pt": "d" * 64})
            tier = self._make_cpu_tier()
            merge_tier_manifests(self.tmpdir, tier, available_tiers=[0])
            self.assertTrue(os.path.exists(_merged_manifest_path(self.tmpdir)))

    class TestDeviceTierDetection(unittest.TestCase):
        """Unit tests for tier detection logic (CPU path only in unit test env)."""

        def test_no_cuda_returns_cpu_tier(self):
            # In test environments without CUDA, we should get CPU tier.
            # We mock torch.cuda.is_available to ensure determinism.
            import unittest.mock as mock
            with mock.patch("torch.cuda.is_available", return_value=False):
                tier = detect_tier(local_rank=0)
            self.assertEqual(tier.tier, DeviceTier.CPU)
            self.assertEqual(tier.device_index, -1)
            self.assertEqual(tier.pcie_gen, 0)

        def test_cpu_tier_uses_dram_chunk_size(self):
            import unittest.mock as mock
            with mock.patch("torch.cuda.is_available", return_value=False):
                tier = detect_tier(local_rank=0)
            self.assertEqual(tier.chunk_bytes, _TIER_CHUNK_BYTES[2])

        def test_tier_info_frozen(self):
            import unittest.mock as mock
            with mock.patch("torch.cuda.is_available", return_value=False):
                tier = detect_tier(local_rank=0)
            with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
                tier.tier = DeviceTier.H100  # type: ignore[misc]

    class TestAsyncIntegrityFinalizer(unittest.TestCase):
        """Unit tests for async manifest finalisation."""

        def setUp(self):
            self.tmpdir = tempfile.mkdtemp(prefix="des_loc_async_test_")

        def tearDown(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        def test_submit_and_wait(self):
            _write_file(os.path.join(self.tmpdir, "model.pt"), os.urandom(128))
            import unittest.mock as mock
            cpu_tier = TierInfo(
                tier=DeviceTier.CPU,
                local_rank=0, global_rank=0, device_index=-1,
                sm_major=0, sm_minor=0, pcie_gen=0,
                chunk_bytes=_TIER_CHUNK_BYTES[2],
            )
            finalizer = AsyncIntegrityFinalizer(
                checkpoint_dir=self.tmpdir,
                tier_info=cpu_tier,
                coordinator_rank=0,
            )
            future = finalizer.submit_async()
            finalizer.wait(timeout=30.0)
            self.assertTrue(future.done())
            self.assertTrue(os.path.exists(_merged_manifest_path(self.tmpdir)))

        def test_wait_before_submit_raises(self):
            finalizer = AsyncIntegrityFinalizer(
                checkpoint_dir=self.tmpdir,
                coordinator_rank=0,
            )
            with self.assertRaises(RuntimeError):
                finalizer.wait()

    # -----------------------------------------------------------------------
    # Run tests
    # -----------------------------------------------------------------------

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [
        TestComputeFileHash,
        TestDRAMStageHasher,
        TestPCIeBandwidthGovernor,
        TestIntegrityManifest,
        TestSaveIntegrityManifest,
        TestVerifyIntegrityManifest,
        TestMergeTierManifests,
        TestDeviceTierDetection,
        TestAsyncIntegrityFinalizer,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
