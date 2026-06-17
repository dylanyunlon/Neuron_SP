# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""RetrievalBudgetManager — retro-pipeline removal and reclaimed-VRAM batch scaling for DES-LOC.

Mirrors Megatron fc6969fbb — remove retro, which deletes the entire RETRO
(Retrieval-Enhanced Transformer) codebase: retrieval database (FAISS index,
chunk DB), retrieval pipeline (neighbor querying, sequence construction), and
the RETRO model architecture (encoder/decoder cross-attention modules).

Design intent (upstream fc6969fbb)
-------------------------------------
RETRO's retrieval pipeline required three categories of persistent GPU memory
that were allocated at model-init time and held for the entire training run:

1. **Retrieval sequence buffers** — neighbor tokens fetched from the FAISS
   index were materialized as dense GPU tensors of shape
   ``(batch, n_neighbors, chunk_len, hidden_dim)`` before being fed to the
   RETRO cross-attention encoder.  For a 2B RETRO model with 2 neighbors and
   chunk_len=64, this consumed ~800 MB per GPU at bf16.

2. **BERT embedding buffers** — during preprocessing (and in some in-training
   online variants), a BERT encoder ran concurrently with the GPT decoder,
   requiring its own KV-cache and activation workspace (~400 MB).

3. **FAISS index GPU mirror** — the IVF/HNSW index was partially mirrored to
   GPU for fast distance computation (~200–600 MB depending on index size).

Removing RETRO frees this budget unconditionally.  The correct engineering
response is NOT to simply remove the allocations and leave the memory idle:
freed budget should be re-invested into a larger effective batch size, which
directly improves gradient quality, communication efficiency (fewer syncs per
token of data), and loss convergence rate.

DES-LOC adaptation
--------------------
``RetrievalBudgetManager`` implements the freed-budget reinvestment policy:

1. At engine-init time, it audits whether any ``retro_*`` config keys are
   present in the DeepSpeed config dict and emits a deprecation warning if so,
   since the upstream has removed them.

2. It estimates the GPU memory that would have been consumed by the RETRO
   pipeline given the model's hidden dimension, sequence length, and retrieval
   config (if any were specified).

3. It computes the maximum allowable micro-batch-size increase that fits
   within the reclaimed budget, subject to a conservative cap
   (``RETRO_BUDGET_BATCH_CAP``) to avoid OOM from other fluctuations.

4. It exposes ``suggest_micro_batch_size(current_micro_bs)`` and
   ``suggest_grad_accum_steps(current_gas, target_global_bs)`` which can be
   called from the engine or launcher to transparently upscale throughput.

5. At step-boundary (``on_step_begin``), it logs a compact one-liner when the
   utilisation of the reclaimed budget crosses diagnostic thresholds — this
   is the M451 GREW pattern: diagnose at decision boundaries, not per-step
   noise.

Diagnostic prefix: ``[RBM]`` (Retrieval Budget Manager)

Usage::

    rbm = RetrievalBudgetManager.from_ds_config(ds_config_dict, model_config)
    rbm.attach_to_engine(engine)

    # In training loop:
    new_mbs = rbm.suggest_micro_batch_size(current_micro_bs=4)
    # -> e.g. 6, if 1.5 GB was reclaimed and each extra sample costs ~500 MB

Architecture note
------------------
No ``megatron.core`` imports.  No RETRO code is replicated.  This module
knows only the *memory footprint model* of what RETRO would have consumed,
so it can calculate reclaimed budget arithmetic.  The actual RETRO model
objects are never instantiated — their absence is the whole point.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_LOG_PREFIX = "[RBM]"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Conservative fraction of reclaimed budget we actually reinvest into batch.
# Keeping at 0.7 leaves 30% headroom for activation growth and fragmentation.
RETRO_BUDGET_REINVEST_FRACTION: float = 0.70

# Hard cap on micro-batch-size multiplier (never more than 4× original).
RETRO_BUDGET_BATCH_CAP: float = 4.0

# Per-token activation memory overhead coefficient at bf16 (empirical, ~12
# bytes × hidden_dim per transformer layer per token as bf16 pairs).
# Used to estimate marginal cost of one extra sample in a micro-batch.
_BF16_ACT_BYTES_PER_TOKEN_PER_LAYER: int = 24  # 2 × 12 for fwd+bwd

# RETRO retrieval sequence buffer memory model:
# shape (batch, n_neighbors, chunk_len, hidden_dim), bf16 = 2 bytes.
_RETRO_NEIGHBOR_BUF_COEFF: int = 2  # bytes per element (bf16)

# Typical BERT in-parallel memory overhead (static allocation share).
_RETRO_BERT_STATIC_MB: int = 400  # MB, rough empirical

# Typical FAISS GPU mirror fraction of total index size.
_RETRO_FAISS_GPU_FRACTION: float = 0.4

# Diagnostic thresholds for GREW-pattern boundary logging.
_DIAG_STEP_PERIOD: int = 200   # emit at multiples of this step count
_DIAG_BUDGET_WARN_GB: float = 1.0  # warn if reclaimed budget is < 1 GB

# Deprecation keys that upstream fc6969fbb removed from the config contract.
_DEPRECATED_RETRO_CONFIG_KEYS: Tuple[str, ...] = (
    "retro_project_dir",
    "retro_tasks",
    "retro_task_validate",
    "retro_block_size",
    "retro_doc_block_size",
    "retro_gpt_seed",
    "retro_gpt_data_path",
    "retro_gpt_data_cache_path",
    "retro_gpt_split",
    "retro_gpt_train_samples",
    "retro_gpt_eval_interval",
    "retro_gpt_eval_iters",
    "retro_gpt_tokenizer_type",
    "retro_gpt_tokenizer_model",
    "retro_gpt_vocab_file",
    "retro_gpt_merge_file",
    "retro_gpt_seq_length",
    "retro_gpt_global_batch_size",
    "retro_gpt_chunk_length",
    "retro_bert_tokenizer_type",
    "retro_bert_vocab_file",
    "retro_bert_batch_size",
    "retro_bert_max_chunk_length",
    "retro_index_type",
    "retro_index_str",
    "retro_index_ntrain",
    "retro_index_train_load_fraction",
    "retro_index_add_load_fraction",
    "retro_index_delete_training_embeddings",
    "retro_index_delete_added_codes",
    "retro_query_ef_search",
    "retro_query_nprobe",
    "retro_query_num_neighbors_query",
    "retro_query_num_neighbors_save",
    # Training-time keys from megatron/training/arguments.py
    "retro_add_retriever",
    "retriever_seq_length",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetroMemoryProfile:
    """Estimated GPU memory footprint of the RETRO pipeline, in MB.

    All fields are *per GPU* estimates at bf16 precision.  The total is the
    budget freed when RETRO is removed (upstream fc6969fbb).

    Attributes:
        neighbor_buffer_mb: Dense neighbor-token tensors fed to cross-attn.
        bert_static_mb: BERT encoder static allocation share.
        faiss_gpu_mb: FAISS index GPU mirror share.
        total_mb: Sum of the above; the reclaimed budget.
    """

    neighbor_buffer_mb: float = 0.0
    bert_static_mb: float = 0.0
    faiss_gpu_mb: float = 0.0

    @property
    def total_mb(self) -> float:
        return self.neighbor_buffer_mb + self.bert_static_mb + self.faiss_gpu_mb

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024.0

    def __str__(self) -> str:
        return (
            f"RetroMemoryProfile("
            f"neighbor={self.neighbor_buffer_mb:.1f}MB "
            f"bert={self.bert_static_mb:.1f}MB "
            f"faiss={self.faiss_gpu_mb:.1f}MB "
            f"→ total={self.total_mb:.1f}MB / {self.total_gb:.2f}GB)"
        )


@dataclass
class BatchScaleRecommendation:
    """Recommendation produced by ``RetrievalBudgetManager.suggest_*``.

    Attributes:
        suggested_micro_bs: Recommended micro-batch-size after reinvestment.
        suggested_grad_accum: Recommended gradient accumulation steps.
        reclaimed_mb: Total MB freed by RETRO removal.
        reinvested_mb: MB actually assigned to the batch increase.
        headroom_mb: MB kept as safety headroom (reclaimed - reinvested).
        scale_factor: Ratio suggested_micro_bs / original_micro_bs.
    """

    original_micro_bs: int
    suggested_micro_bs: int
    original_grad_accum: int
    suggested_grad_accum: int
    reclaimed_mb: float
    reinvested_mb: float
    headroom_mb: float

    @property
    def scale_factor(self) -> float:
        if self.original_micro_bs == 0:
            return 1.0
        return self.suggested_micro_bs / self.original_micro_bs

    def __str__(self) -> str:
        return (
            f"BatchScaleRecommendation("
            f"mbs {self.original_micro_bs}→{self.suggested_micro_bs} "
            f"gas {self.original_grad_accum}→{self.suggested_grad_accum} "
            f"scale={self.scale_factor:.2f}x "
            f"reclaimed={self.reclaimed_mb:.0f}MB "
            f"reinvested={self.reinvested_mb:.0f}MB "
            f"headroom={self.headroom_mb:.0f}MB)"
        )


@dataclass
class RetrievalBudgetConfig:
    """Configuration for RetrievalBudgetManager.

    Attributes:
        hidden_dim: Model hidden dimension (d_model).
        n_layers: Number of transformer layers.
        seq_len: Training sequence length (tokens).
        micro_batch_size: Current micro-batch-size per GPU.
        n_neighbors: RETRO neighbor count (default 2; upstream used 2).
        chunk_len: RETRO chunk length in tokens (default 64).
        faiss_index_size_mb: Approximate total FAISS index size in MB.
            Set to 0 if no FAISS was used.
        bert_enabled: Whether BERT in-parallel embedder was active.
        reinvest_fraction: Fraction of reclaimed budget to reinvest.
        batch_cap_multiplier: Hard upper bound on mbs scaling factor.
    """

    hidden_dim: int = 1024
    n_layers: int = 24
    seq_len: int = 2048
    micro_batch_size: int = 4
    n_neighbors: int = 2
    chunk_len: int = 64
    faiss_index_size_mb: float = 0.0
    bert_enabled: bool = True
    reinvest_fraction: float = RETRO_BUDGET_REINVEST_FRACTION
    batch_cap_multiplier: float = RETRO_BUDGET_BATCH_CAP


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class RetrievalBudgetManager:
    """Manages freed GPU memory budget after RETRO pipeline removal.

    Mirrors Megatron fc6969fbb — remove retro.

    The class has two responsibilities:

    A. **Audit** existing DeepSpeed configs for deprecated ``retro_*`` keys
       and emit structured deprecation warnings (matching the upstream intent
       of hard-removing these config fields).

    B. **Reinvest** the GPU memory that RETRO's pipeline would have consumed
       into a larger effective batch size, exposing ``suggest_micro_batch_size``
       and ``suggest_grad_accum_steps`` for use by the engine or launcher.

    Diagnostic events (GREW-style, at boundaries not per-step):
    - ``INIT``     — emitted once at construction with full profile summary.
    - ``DEPREC``   — emitted per deprecated config key found.
    - ``SUGGEST``  — emitted when suggest_* is called.
    - ``STEP``     — emitted every ``_DIAG_STEP_PERIOD`` steps.
    - ``ATTACH``   — emitted when hooked into a DeepSpeed engine.
    """

    def __init__(self, config: RetrievalBudgetConfig) -> None:
        self.config = config
        self._step: int = 0
        self._attached_engine = None
        self._deprecated_keys_found: List[str] = []
        self._last_recommendation: Optional[BatchScaleRecommendation] = None

        self._profile = self._estimate_retro_memory()
        self._emit("INIT", str(self._profile))

        if self._profile.total_mb < _DIAG_BUDGET_WARN_GB * 1024:
            self._emit(
                "INIT",
                f"reclaimed budget {self._profile.total_mb:.1f}MB is below "
                f"{_DIAG_BUDGET_WARN_GB:.1f}GB threshold — "
                "RETRO may not have been active, or model is very small"
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_ds_config(
        cls,
        ds_config: dict,
        model_config: Optional[dict] = None,
    ) -> "RetrievalBudgetManager":
        """Construct from a raw DeepSpeed config dict.

        Also audits ``ds_config`` for deprecated ``retro_*`` keys and emits
        deprecation warnings immediately (before returning the instance).

        Args:
            ds_config: Raw DeepSpeed config dict (from ds_config.json / init).
            model_config: Optional dict with ``hidden_dim``, ``n_layers``,
                ``seq_len``, ``micro_batch_size`` keys.  Missing keys fall
                back to defaults in ``RetrievalBudgetConfig``.

        Returns:
            Configured ``RetrievalBudgetManager`` instance.
        """
        mc = model_config or {}

        # Extract RETRO-specific config fields if present (they should not be,
        # but if someone is migrating from a pre-fc6969fbb config file we want
        # to parse them for the memory model before warning about deprecation).
        n_neighbors = _safe_int(ds_config.get("retro_query_num_neighbors_query"), default=2)
        chunk_len   = _safe_int(ds_config.get("retro_gpt_chunk_length"), default=64)
        bert_on     = bool(ds_config.get("retro_bert_tokenizer_type", None))

        # Estimate FAISS size from index string if present.
        faiss_mb = _estimate_faiss_size_mb(
            ds_config.get("retro_index_str", ""),
            ds_config.get("retro_index_ntrain", 0),
        )

        cfg = RetrievalBudgetConfig(
            hidden_dim=_safe_int(mc.get("hidden_dim"), default=1024),
            n_layers=_safe_int(mc.get("n_layers"), default=24),
            seq_len=_safe_int(mc.get("seq_len"), default=2048),
            micro_batch_size=_safe_int(
                ds_config.get("train_micro_batch_size_per_gpu",
                              mc.get("micro_batch_size")),
                default=4,
            ),
            n_neighbors=n_neighbors,
            chunk_len=chunk_len,
            faiss_index_size_mb=faiss_mb,
            bert_enabled=bert_on,
        )

        instance = cls(cfg)

        # Audit deprecated keys.
        found = [k for k in _DEPRECATED_RETRO_CONFIG_KEYS if k in ds_config]
        for key in found:
            instance._deprecated_keys_found.append(key)
            instance._emit(
                "DEPREC",
                f"config key '{key}' was removed in Megatron fc6969fbb; "
                "it has no effect and should be removed from ds_config"
            )

        if found:
            instance._emit(
                "DEPREC",
                f"{len(found)} deprecated retro_* key(s) found in ds_config "
                f"({', '.join(found[:5])}{'...' if len(found) > 5 else ''}). "
                "Upstream fc6969fbb hard-deleted RETRO; remove these keys."
            )

        return instance

    # ------------------------------------------------------------------
    # Memory model
    # ------------------------------------------------------------------

    def _estimate_retro_memory(self) -> RetroMemoryProfile:
        """Compute per-GPU RETRO memory footprint estimate.

        Three components (all at bf16 / 2 bytes per element):

        1. Neighbor buffers: batch × n_neighbors × chunk_len × hidden_dim × 2
           (the dense retrieval sequences materialised before cross-attention).

        2. BERT static allocation: empirical constant ``_RETRO_BERT_STATIC_MB``
           if BERT was enabled.

        3. FAISS GPU mirror: ``_RETRO_FAISS_GPU_FRACTION`` of total index.

        Returns:
            ``RetroMemoryProfile`` with per-component and total MB.
        """
        cfg = self.config

        # Component 1: neighbor token buffers.
        neighbor_elements = (
            cfg.micro_batch_size *
            cfg.n_neighbors *
            cfg.chunk_len *
            cfg.hidden_dim
        )
        neighbor_mb = neighbor_elements * _RETRO_NEIGHBOR_BUF_COEFF / (1024 ** 2)

        # Component 2: BERT static overhead.
        bert_mb = float(_RETRO_BERT_STATIC_MB) if cfg.bert_enabled else 0.0

        # Component 3: FAISS GPU mirror.
        faiss_mb = cfg.faiss_index_size_mb * _RETRO_FAISS_GPU_FRACTION

        return RetroMemoryProfile(
            neighbor_buffer_mb=round(neighbor_mb, 2),
            bert_static_mb=round(bert_mb, 2),
            faiss_gpu_mb=round(faiss_mb, 2),
        )

    # ------------------------------------------------------------------
    # Batch scaling recommendations
    # ------------------------------------------------------------------

    def suggest_micro_batch_size(
        self,
        current_micro_bs: Optional[int] = None,
        grad_accum_steps: int = 1,
        target_global_bs: Optional[int] = None,
    ) -> BatchScaleRecommendation:
        """Recommend a micro-batch-size using the reclaimed RETRO budget.

        The freed VRAM is reinvested into additional samples per micro-batch.
        The marginal cost of one extra sample is estimated from transformer
        activation memory (O(seq_len × hidden_dim × n_layers) at bf16).

        If ``target_global_bs`` is supplied, the recommendation also adjusts
        gradient accumulation steps to maintain the same global batch size
        while maximising per-step throughput.

        Args:
            current_micro_bs: Current micro-batch-size per GPU.  Falls back
                to ``self.config.micro_batch_size`` if None.
            grad_accum_steps: Current gradient accumulation step count.
            target_global_bs: Desired global batch size (tokens × world-size).
                If None, no grad-accum adjustment is made.

        Returns:
            ``BatchScaleRecommendation`` with all relevant fields populated.

        Diagnostic: [RBM] SUGGEST — one line with scale factor.
        """
        mbs = current_micro_bs if current_micro_bs is not None else self.config.micro_batch_size
        cfg = self.config

        reclaimed_mb = self._profile.total_mb
        reinvest_mb  = reclaimed_mb * cfg.reinvest_fraction
        headroom_mb  = reclaimed_mb - reinvest_mb

        # Marginal activation cost of one sample (MB) at bf16.
        # Rough: seq_len × hidden_dim × n_layers × bytes_per_tok / 1M
        sample_act_mb = (
            cfg.seq_len * cfg.hidden_dim * cfg.n_layers *
            _BF16_ACT_BYTES_PER_TOKEN_PER_LAYER / (1024 ** 2)
        )
        sample_act_mb = max(sample_act_mb, 1.0)  # never divide by zero

        # How many extra samples fit in the reinvested budget?
        extra_samples = max(0, int(reinvest_mb / sample_act_mb))

        new_mbs = mbs + extra_samples
        cap = int(math.floor(mbs * cfg.batch_cap_multiplier))
        new_mbs = min(new_mbs, cap)
        new_mbs = max(new_mbs, mbs)  # never shrink

        # Grad accum adjustment: keep global_bs constant if target supplied.
        new_gas = grad_accum_steps
        if target_global_bs is not None and target_global_bs > 0 and new_mbs > 0:
            # global_bs ≈ mbs × gas × world_size; we don't know world_size here
            # so we work with the mbs×gas product.
            old_product = mbs * grad_accum_steps
            new_gas = max(1, int(math.ceil(old_product / new_mbs)))

        rec = BatchScaleRecommendation(
            original_micro_bs=mbs,
            suggested_micro_bs=new_mbs,
            original_grad_accum=grad_accum_steps,
            suggested_grad_accum=new_gas,
            reclaimed_mb=reclaimed_mb,
            reinvested_mb=reinvest_mb,
            headroom_mb=headroom_mb,
        )

        self._last_recommendation = rec
        self._emit("SUGGEST", str(rec))
        return rec

    def suggest_grad_accum_steps(
        self,
        current_gas: int,
        target_global_bs: int,
        world_size: int = 1,
    ) -> int:
        """Recommend gradient accumulation steps after micro-batch upscaling.

        If ``suggest_micro_batch_size`` has been called, uses its result;
        otherwise calls it with current config defaults.

        Args:
            current_gas: Current gradient accumulation step count.
            target_global_bs: Target global batch size (samples × world_size).
            world_size: Data-parallel world size.

        Returns:
            Recommended grad accum steps (int ≥ 1).
        """
        if self._last_recommendation is None:
            self.suggest_micro_batch_size(grad_accum_steps=current_gas)

        rec = self._last_recommendation
        new_mbs = rec.suggested_micro_bs

        # Recompute based on actual world size.
        new_gas = max(1, int(math.ceil(target_global_bs / (new_mbs * max(world_size, 1)))))
        self._emit(
            "SUGGEST",
            f"grad_accum: gas {current_gas}→{new_gas} "
            f"mbs={new_mbs} world_size={world_size} "
            f"target_global_bs={target_global_bs}"
        )
        return new_gas

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def attach_to_engine(self, engine) -> None:
        """Hook into a DeepSpeed engine for step-boundary diagnostics.

        Wraps ``engine.train_batch()`` (if available) to call
        ``self.on_step_begin()`` / ``self.on_step_end()`` at training step
        boundaries.  If ``train_batch`` is not available (pipeline engine,
        custom engine), logs a warning and leaves integration to the caller.

        The hook is intentionally lightweight: it only records step count
        and emits periodic GREW-style diagnostics.  It does NOT modify any
        training tensors.

        Args:
            engine: A DeepSpeedEngine-compatible object.

        Diagnostic: [RBM] ATTACH — confirms hook insertion.
        """
        self._attached_engine = engine

        train_batch_fn = getattr(engine, "train_batch", None)
        if train_batch_fn is None:
            self._emit(
                "ATTACH",
                "engine has no train_batch(); step hooks not installed. "
                "Call on_step_begin()/on_step_end() manually."
            )
            return

        mgr = self  # avoid closure capture issues

        def _wrapped_train_batch(*args, **kwargs):
            mgr.on_step_begin()
            result = train_batch_fn(*args, **kwargs)
            mgr.on_step_end()
            return result

        engine.train_batch = _wrapped_train_batch
        self._emit(
            "ATTACH",
            f"step hooks installed on engine.train_batch() "
            f"(engine type: {type(engine).__name__})"
        )

    def on_step_begin(self) -> None:
        """Called at the beginning of each training step.

        Emits a compact diagnostic at ``_DIAG_STEP_PERIOD`` multiples,
        following the M451 GREW boundary pattern.
        """
        self._step += 1
        if self._step % _DIAG_STEP_PERIOD == 1:
            self._emit(
                "STEP",
                f"step={self._step} "
                f"reclaimed={self._profile.total_mb:.0f}MB "
                f"({self._profile.total_gb:.2f}GB) "
                f"deprecated_keys={len(self._deprecated_keys_found)} "
                f"last_rec={self._last_recommendation}"
            )

    def on_step_end(self) -> None:
        """Called at the end of each training step (currently a no-op).

        Reserved for future per-step budget monitoring (e.g., tracking
        actual peak memory to validate the reclaimed budget estimate).
        """
        pass

    # ------------------------------------------------------------------
    # Convenience: config scrubber
    # ------------------------------------------------------------------

    @staticmethod
    def scrub_retro_keys(ds_config: dict) -> Tuple[dict, List[str]]:
        """Remove all deprecated ``retro_*`` keys from a DeepSpeed config dict.

        Returns a *new* dict (does not modify in-place) and a list of keys
        that were removed.

        This is the DeepSpeed equivalent of the upstream fc6969fbb operation
        of deleting the entire ``megatron/core/datasets/retro/`` and
        ``megatron/core/models/retro/`` trees — applied to the config layer.

        Args:
            ds_config: Raw DeepSpeed config dict.

        Returns:
            Tuple of (scrubbed_config, removed_keys).

        Diagnostic: [RBM] SCRUB — reports count of removed keys.
        """
        removed = [k for k in _DEPRECATED_RETRO_CONFIG_KEYS if k in ds_config]
        scrubbed = {k: v for k, v in ds_config.items() if k not in _DEPRECATED_RETRO_CONFIG_KEYS}
        if removed:
            log.info(
                f"{_LOG_PREFIX} SCRUB  removed {len(removed)} deprecated retro_* key(s): "
                f"{removed}"
            )
        return scrubbed, removed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def memory_profile(self) -> RetroMemoryProfile:
        """The estimated GPU memory that RETRO would have consumed."""
        return self._profile

    @property
    def reclaimed_gb(self) -> float:
        """Reclaimed GPU memory in GB (convenience shorthand)."""
        return self._profile.total_gb

    @property
    def step(self) -> int:
        """Number of training steps observed since attachment."""
        return self._step

    @property
    def deprecated_keys_found(self) -> List[str]:
        """List of deprecated ``retro_*`` config keys found during audit."""
        return list(self._deprecated_keys_found)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event: str, message: str) -> None:
        """Emit a structured diagnostic log line.

        Format: ``[RBM] EVENT  message``
        """
        log.info(f"{_LOG_PREFIX} {event:<8}  {message}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default: int = 0) -> int:
    """Convert ``value`` to int, returning ``default`` on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _estimate_faiss_size_mb(index_str: str, ntrain: int) -> float:
    """Rough estimate of FAISS index size from index string and training count.

    FAISS index memory is dominated by:
    - PQ codebooks: ``M × ksub × dsub × 4`` bytes (FP32 centroids)
    - IVF centroids: ``nlist × d × 4`` bytes

    For a typical RETRO index ``OPQ32_64,IVF65536_HNSW8,PQ32``:
    - M=32, nlist=65536, d=64 → ~512 MB total with quantisation codes

    We use a simplified heuristic: 8 bytes per training vector + 10 MB fixed.

    Args:
        index_str: Faiss index factory string (may be empty).
        ntrain: Number of training vectors.

    Returns:
        Estimated index size in MB (float).
    """
    if not index_str or ntrain == 0:
        return 0.0

    # Parse PQ dimension M from string like "PQ32" or "OPQ32_64".
    import re
    pq_match = re.search(r'PQ(\d+)', index_str)
    M = int(pq_match.group(1)) if pq_match else 32

    # Parse IVF nlist.
    ivf_match = re.search(r'IVF(\d+)', index_str)
    nlist = int(ivf_match.group(1)) if ivf_match else 65536

    # Parse output dimension (after PQ, e.g. OPQ32_64 → 64).
    d_match = re.search(r'OPQ\d+_(\d+)', index_str)
    d = int(d_match.group(1)) if d_match else 64

    # IVF centroids: nlist × d × 4 bytes (FP32).
    ivf_bytes = nlist * d * 4

    # PQ codebooks: M × 256 × (d // M) × 4 bytes (FP32 sub-centroids).
    d_sub = max(1, d // M)
    pq_bytes = M * 256 * d_sub * 4

    # Index codes: ntrain × M bytes (PQ8 compressed).
    codes_bytes = ntrain * M

    total_mb = (ivf_bytes + pq_bytes + codes_bytes) / (1024 ** 2)
    return round(total_mb, 2)


def estimate_retro_vram_reclaimed(
    hidden_dim: int = 1024,
    n_layers: int = 24,
    seq_len: int = 2048,
    micro_batch_size: int = 4,
    n_neighbors: int = 2,
    chunk_len: int = 64,
    faiss_index_size_mb: float = 0.0,
    bert_enabled: bool = True,
) -> RetroMemoryProfile:
    """Standalone helper: estimate RETRO VRAM freed by upstream fc6969fbb.

    Convenience wrapper around ``RetrievalBudgetManager`` for callers that
    only need the memory profile and do not require the full manager lifecycle.

    Args:
        hidden_dim: Model hidden dimension.
        n_layers: Number of transformer layers.
        seq_len: Training sequence length.
        micro_batch_size: Micro-batch-size per GPU.
        n_neighbors: RETRO neighbor count.
        chunk_len: RETRO chunk length in tokens.
        faiss_index_size_mb: FAISS index total size in MB.
        bert_enabled: Whether BERT in-parallel embedder was active.

    Returns:
        ``RetroMemoryProfile`` with per-component and total estimates.

    Example::

        profile = estimate_retro_vram_reclaimed(
            hidden_dim=2048, n_layers=32, seq_len=4096,
            micro_batch_size=4, n_neighbors=2,
        )
        print(f"Freed: {profile.total_gb:.2f} GB per GPU")
    """
    cfg = RetrievalBudgetConfig(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        n_neighbors=n_neighbors,
        chunk_len=chunk_len,
        faiss_index_size_mb=faiss_index_size_mb,
        bert_enabled=bert_enabled,
    )
    mgr = RetrievalBudgetManager(cfg)
    return mgr.memory_profile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class _TestRetrievalBudgetManager:
    """Inline unit tests.  Run via: python -m deepspeed.runtime.retrieval_budget_manager"""

    # ------------------------------------------------------------------
    def test_memory_profile_zero_faiss(self):
        """Neighbor buffers + BERT only when no FAISS supplied."""
        cfg = RetrievalBudgetConfig(
            hidden_dim=1024, n_layers=24, seq_len=2048,
            micro_batch_size=4, n_neighbors=2, chunk_len=64,
            faiss_index_size_mb=0.0, bert_enabled=True,
        )
        mgr = RetrievalBudgetManager(cfg)
        p = mgr.memory_profile

        expected_neighbor_mb = (4 * 2 * 64 * 1024 * 2) / (1024 ** 2)
        assert abs(p.neighbor_buffer_mb - expected_neighbor_mb) < 0.5, (
            f"neighbor_mb mismatch: {p.neighbor_buffer_mb} vs {expected_neighbor_mb}"
        )
        assert p.bert_static_mb == float(_RETRO_BERT_STATIC_MB)
        assert p.faiss_gpu_mb == 0.0
        assert p.total_mb == p.neighbor_buffer_mb + p.bert_static_mb + p.faiss_gpu_mb
        print(f"  [PASS] memory_profile_zero_faiss: {p}")

    # ------------------------------------------------------------------
    def test_memory_profile_with_faiss(self):
        """FAISS mirror fraction is included in total."""
        cfg = RetrievalBudgetConfig(
            hidden_dim=1024, n_layers=24, seq_len=2048,
            micro_batch_size=4, n_neighbors=2, chunk_len=64,
            faiss_index_size_mb=600.0, bert_enabled=False,
        )
        mgr = RetrievalBudgetManager(cfg)
        p = mgr.memory_profile

        assert abs(p.faiss_gpu_mb - 600.0 * _RETRO_FAISS_GPU_FRACTION) < 0.1
        assert p.bert_static_mb == 0.0
        print(f"  [PASS] memory_profile_with_faiss: {p}")

    # ------------------------------------------------------------------
    def test_suggest_never_shrinks(self):
        """Suggested MBS is always >= original MBS."""
        cfg = RetrievalBudgetConfig(
            hidden_dim=512, n_layers=12, seq_len=1024,
            micro_batch_size=2, n_neighbors=2, chunk_len=64,
        )
        mgr = RetrievalBudgetManager(cfg)
        rec = mgr.suggest_micro_batch_size(current_micro_bs=2, grad_accum_steps=4)

        assert rec.suggested_micro_bs >= rec.original_micro_bs, (
            f"MBS shrank: {rec.original_micro_bs} → {rec.suggested_micro_bs}"
        )
        assert rec.scale_factor >= 1.0
        print(f"  [PASS] suggest_never_shrinks: {rec}")

    # ------------------------------------------------------------------
    def test_suggest_cap_respected(self):
        """Suggested MBS never exceeds cap multiplier × original."""
        cfg = RetrievalBudgetConfig(
            hidden_dim=128, n_layers=2, seq_len=128,
            micro_batch_size=1, n_neighbors=8, chunk_len=128,
            faiss_index_size_mb=10000.0, bert_enabled=True,
            batch_cap_multiplier=3.0,
        )
        mgr = RetrievalBudgetManager(cfg)
        rec = mgr.suggest_micro_batch_size(current_micro_bs=1)

        assert rec.suggested_micro_bs <= int(1 * 3.0), (
            f"cap violated: {rec.suggested_micro_bs} > {int(1 * 3.0)}"
        )
        print(f"  [PASS] suggest_cap_respected: {rec}")

    # ------------------------------------------------------------------
    def test_scrub_retro_keys(self):
        """scrub_retro_keys removes all deprecated retro_* keys."""
        raw = {
            "train_micro_batch_size_per_gpu": 4,
            "zero_optimization": {"stage": 2},
            "retro_project_dir": "/some/path",
            "retro_add_retriever": True,
            "retriever_seq_length": 256,
            "retro_gpt_chunk_length": 64,
        }
        scrubbed, removed = RetrievalBudgetManager.scrub_retro_keys(raw)

        retro_remaining = [k for k in scrubbed if "retro" in k or "retriev" in k.lower()]
        assert len(retro_remaining) == 0, f"retro keys remain: {retro_remaining}"
        assert "train_micro_batch_size_per_gpu" in scrubbed
        assert "zero_optimization" in scrubbed
        assert set(removed) == {
            "retro_project_dir", "retro_add_retriever",
            "retriever_seq_length", "retro_gpt_chunk_length",
        }
        print(f"  [PASS] scrub_retro_keys: removed={removed}")

    # ------------------------------------------------------------------
    def test_from_ds_config_audit(self):
        """from_ds_config emits DEPREC events for retro keys in config."""
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 4,
            "retro_project_dir": "/data/retro",
            "retro_gpt_chunk_length": 64,
            "retro_query_num_neighbors_query": 2,
            "retro_index_str": "OPQ32_64,IVF65536_HNSW8,PQ32",
            "retro_index_ntrain": 66625331,
            "retro_bert_tokenizer_type": "BertWordPieceLowerCase",
        }
        mc = {"hidden_dim": 1024, "n_layers": 24, "seq_len": 2048}
        mgr = RetrievalBudgetManager.from_ds_config(ds_cfg, mc)

        assert len(mgr.deprecated_keys_found) >= 4, (
            f"expected >=4 deprecated keys, got {len(mgr.deprecated_keys_found)}"
        )
        print(f"  [PASS] from_ds_config_audit: found {len(mgr.deprecated_keys_found)} deprecated keys")

    # ------------------------------------------------------------------
    def test_grad_accum_suggest(self):
        """suggest_grad_accum_steps reduces gas proportional to mbs increase."""
        cfg = RetrievalBudgetConfig(
            hidden_dim=512, n_layers=12, seq_len=512,
            micro_batch_size=4, n_neighbors=2, chunk_len=64,
            bert_enabled=True,
        )
        mgr = RetrievalBudgetManager(cfg)
        mgr.suggest_micro_batch_size(current_micro_bs=4, grad_accum_steps=8)
        new_gas = mgr.suggest_grad_accum_steps(
            current_gas=8, target_global_bs=64, world_size=4
        )
        # global_bs = mbs × gas × world_size, so gas = 64 / (new_mbs × 4)
        rec = mgr._last_recommendation
        expected_gas = max(1, int(math.ceil(64 / (rec.suggested_micro_bs * 4))))
        assert new_gas == expected_gas, f"gas mismatch: {new_gas} vs {expected_gas}"
        print(f"  [PASS] grad_accum_suggest: new_gas={new_gas} (mbs={rec.suggested_micro_bs})")

    # ------------------------------------------------------------------
    def test_faiss_size_estimator(self):
        """FAISS size estimator returns a plausible positive value."""
        mb = _estimate_faiss_size_mb("OPQ32_64,IVF65536_HNSW8,PQ32", ntrain=66625331)
        assert mb > 0, f"expected positive faiss size, got {mb}"
        print(f"  [PASS] faiss_size_estimator: {mb:.1f}MB for OPQ32_64,IVF65536_HNSW8,PQ32")

    # ------------------------------------------------------------------
    def test_estimate_helper(self):
        """Standalone estimate_retro_vram_reclaimed returns RetroMemoryProfile."""
        p = estimate_retro_vram_reclaimed(
            hidden_dim=2048, n_layers=32, seq_len=4096,
            micro_batch_size=4, n_neighbors=2,
        )
        assert isinstance(p, RetroMemoryProfile)
        assert p.total_mb > 0
        print(f"  [PASS] estimate_helper: {p}")

    # ------------------------------------------------------------------
    def run_all(self):
        import traceback
        tests = [m for m in dir(self) if m.startswith("test_")]
        passed, failed = 0, 0
        for name in tests:
            try:
                getattr(self, name)()
                passed += 1
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {name}: {exc}")
                traceback.print_exc()
        print(f"\n{'='*60}")
        print(f"RetrievalBudgetManager tests: {passed} passed, {failed} failed")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _TestRetrievalBudgetManager().run_all()
