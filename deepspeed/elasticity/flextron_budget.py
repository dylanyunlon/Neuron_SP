# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""flextron_budget.py — Subnet parameter and memory footprint estimator.

Mirrors Megatron 2d862fe0c flex_budget_utils.get_num_parameters and
get_memory_footprint, reinterpreted as a standalone DeepSpeed module
that does not depend on megatron.core or a fully-constructed TransformerConfig.

Design intent (upstream 2d862fe0c)
------------------------------------
``get_num_parameters`` counts how many trainable parameters a given
Flextron sub-network has, given the hybrid_pattern string ('M' = Mamba
layer, '*' = attention layer, 'E' = MoE/FFN layer) and per-budget integer
choice lists (mamba_int, mlp_int, emb_int, moe_expert_int).

``get_memory_footprint`` extends this with runtime cache sizes (KV cache for
attention layers, SSM state for Mamba) scaled by a ``MemoryConfig``
bytes-per-element profile.

DES-LOC reinterpretation
-------------------------
The upstream functions assume megatron.core is importable.  Here we
replicate their core arithmetic as pure-Python functions so DeepSpeed
can evaluate subnet sizes independently (e.g. during training on an
inference-only A6000 node that never imports Megatron).

Key algorithmic differences from the upstream:
1. ``count_subnet_params`` handles scalar *or* list inputs for mamba_nheads,
   ffn_hidden, num_experts — matching the upstream's ``flex_hetero_*``
   branching but without torch tensor operations (pure int arithmetic).
2. ``estimate_subnet_memory_gb`` accepts a ``DeslocMemoryProfile`` that
   mirrors ``MemoryConfig`` but is serialisable to plain dict without yaml.
3. Structured diagnostics at budget-list enumeration boundaries follow
   the M451 pattern (single ``[DS-FlextronBudget]`` event per evaluation,
   not per-layer noise).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[DS-FlextronBudget]"


# ---------------------------------------------------------------------------
# Memory profile (mirrors Megatron MemoryConfig without yaml dependency)
# ---------------------------------------------------------------------------

@dataclass
class DeslocMemoryProfile:
    """Bytes-per-element profile for Flextron memory estimation.

    Mirrors Megatron 2d862fe0c MemoryConfig, serialisable to plain dict.
    Resolution order (matches load_memory_config): preset < CLI override.

    Attributes
    ----------
    bpe_params : float
        Bytes per weight parameter element (2=BF16, 1=FP8/INT8, 0.5625=FP4).
    bpe_kv_cache : float
        Bytes per KV-cache element.
    bpe_ssm_cache : float
        Bytes per SSM-state element (conv_state + ssm_state, as in Mamba2).
    bpe_moe_buffer : float
        Bytes per MoE dispatch buffer element.
    param_budget_target : str
        'active'  → budget loss supervises on top-k active-expert params.
        'total'   → budget loss supervises on all params including inactive.
    """
    bpe_params:          float = 2.0
    bpe_kv_cache:        float = 2.0
    bpe_ssm_cache:       float = 2.0
    bpe_moe_buffer:      float = 2.0
    param_budget_target: str   = "active"

    # Named-preset registry (mirrors memory_profiles.yaml)
    _PRESETS: Dict[str, Dict] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.param_budget_target not in ("active", "total"):
            raise ValueError(
                f"param_budget_target must be 'active' or 'total', "
                f"got '{self.param_budget_target}'"
            )
        # Initialise _PRESETS if still empty (dataclass default_factory quirk)
        if not self._PRESETS:
            object.__setattr__(self, '_PRESETS', _BUILTIN_PRESETS)

    @classmethod
    def from_preset(cls, name: str) -> "DeslocMemoryProfile":
        """Construct from a named preset (mirrors load_memory_config preset lookup)."""
        if name not in _BUILTIN_PRESETS:
            raise ValueError(
                f"Unknown memory preset '{name}'. "
                f"Available: {list(_BUILTIN_PRESETS.keys())}"
            )
        p = _BUILTIN_PRESETS[name]
        return cls(
            bpe_params=float(p["params"]),
            bpe_kv_cache=float(p["kv_cache"]),
            bpe_ssm_cache=float(p["ssm_cache"]),
            bpe_moe_buffer=float(p.get("max_buffer", 2.0)),
            param_budget_target=p.get("param_budget_target", "active"),
        )

    def to_dict(self) -> dict:
        return {
            "bpe_params":          self.bpe_params,
            "bpe_kv_cache":        self.bpe_kv_cache,
            "bpe_ssm_cache":       self.bpe_ssm_cache,
            "bpe_moe_buffer":      self.bpe_moe_buffer,
            "param_budget_target": self.param_budget_target,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeslocMemoryProfile":
        return cls(
            bpe_params=d.get("bpe_params", 2.0),
            bpe_kv_cache=d.get("bpe_kv_cache", 2.0),
            bpe_ssm_cache=d.get("bpe_ssm_cache", 2.0),
            bpe_moe_buffer=d.get("bpe_moe_buffer", 2.0),
            param_budget_target=d.get("param_budget_target", "active"),
        )


# Mirrors memory_profiles.yaml — kept in sync manually; yaml is not required.
_BUILTIN_PRESETS: Dict[str, Dict] = {
    "bf16":       {"params": 2.0,    "kv_cache": 2.0,    "ssm_cache": 2.0, "max_buffer": 2.0, "param_budget_target": "active"},
    "fp8_kv":     {"params": 2.0,    "kv_cache": 1.0,    "ssm_cache": 2.0, "max_buffer": 2.0, "param_budget_target": "active"},
    "fp8_kv_ssm": {"params": 2.0,    "kv_cache": 1.0,    "ssm_cache": 1.0, "max_buffer": 2.0, "param_budget_target": "active"},
    "fp8_all":    {"params": 1.0,    "kv_cache": 1.0,    "ssm_cache": 1.0, "max_buffer": 1.0, "param_budget_target": "active"},
    "int8":       {"params": 1.0,    "kv_cache": 1.0,    "ssm_cache": 1.0, "max_buffer": 2.0, "param_budget_target": "active"},
    "fp4":        {"params": 0.5625, "kv_cache": 0.5625, "ssm_cache": 1.0, "max_buffer": 2.0, "param_budget_target": "active"},
}


# ---------------------------------------------------------------------------
# Parameter counter (mirrors flex_budget_utils.get_num_parameters)
# ---------------------------------------------------------------------------

def _mamba_param_count(
    nheads: int,
    headdim: int,
    d_state: int,
    hidden_size: int,
    ngroups: int = 8,
    conv_kernel: int = 4,
) -> int:
    """Count Mamba2 parameters for one layer given nheads.

    Mirrors the ``mamba_params(mamba_nheads)`` closure inside
    ``get_num_parameters`` in flex_budget_utils.py.

    Components
    ----------
    in_proj    : hidden * (d_inner*2 + 2*ngroups*d_state + nheads)
    conv1d     : (d_inner + 2*ngroups*d_state) * conv_kernel  + bias
    out_proj   : d_inner * hidden
    norms      : d_inner
    A_log, D   : nheads each
    dt_bias    : nheads
    """
    d_inner = nheads * headdim
    cdim = d_inner + 2 * ngroups * d_state
    conv = cdim * conv_kernel + cdim       # weight + bias

    in_proj  = hidden_size * (d_inner * 2 + 2 * ngroups * d_state + nheads)
    out_proj = d_inner * hidden_size
    norm     = d_inner
    A_log    = nheads
    D        = nheads
    dt_bias  = nheads
    ln_in    = hidden_size  # pre-mixer layer norm (1 element per hidden dim)

    return in_proj + conv + out_proj + norm + A_log + D + dt_bias + ln_in


def _attn_param_count(
    num_heads: int,
    num_query_groups: int,
    kv_channels: int,
    hidden_size: int,
) -> int:
    """Count self-attention parameters for one layer.

    Mirrors the ``att`` block in get_num_parameters:
      QKV proj : (num_heads + 2*num_query_groups) * kv_channels * hidden
      O proj   : num_heads * kv_channels * hidden
      input LN : hidden
    """
    qkv    = (num_heads + 2 * num_query_groups) * kv_channels * hidden_size
    o_proj = num_heads * kv_channels * hidden_size
    ln_in  = hidden_size
    return qkv + o_proj + ln_in


def _moe_param_count(
    ffn_hidden: int,
    hidden_size: int,
    num_experts: int,
    shared_expert_hidden: int = 0,
    topk: int = 1,
) -> Tuple[int, int]:
    """Count MoE/FFN parameters for one layer.

    Returns (total_params, active_params) mirroring ``moe_all`` / ``moe_active``
    in get_num_parameters.
    """
    ln_pre = hidden_size
    fc1 = ffn_hidden * (hidden_size * num_experts + shared_expert_hidden)
    fc2 = ffn_hidden * (hidden_size * num_experts + shared_expert_hidden)
    fc1_active = ffn_hidden * (hidden_size * topk + shared_expert_hidden)
    fc2_active = ffn_hidden * (hidden_size * topk + shared_expert_hidden)

    total  = ln_pre + fc1 + fc2
    active = ln_pre + fc1_active + fc2_active
    return total, active


def count_subnet_params(
    *,
    hybrid_pattern: str,
    hidden_size: int,
    # Mamba
    mamba_nheads: Union[int, Sequence[int]],
    mamba_headdim: int,
    mamba_d_state: int,
    # Attention
    num_attn_heads: int,
    num_query_groups: int,
    kv_channels: int,
    # FFN / MoE
    ffn_hidden: Union[int, Sequence[int]],
    num_experts: Union[int, Sequence[int]] = 1,
    shared_expert_hidden: int = 0,
    moe_topk: int = 1,
    # Embedding
    vocab_size: int,
    tied_vocab: bool = False,
) -> Tuple[int, int]:
    """Count (total_params, active_params) for a Flextron subnet.

    Mirrors get_num_parameters from flex_budget_utils.py.  Accepts both
    scalar and list inputs for mamba_nheads / ffn_hidden / num_experts
    (list = heterogeneous per-layer choices, matching flex_hetero_* flags).

    Parameters
    ----------
    hybrid_pattern : str
        One character per layer: 'M'=Mamba, '*'=Attention, 'E'=MoE/FFN.
    mamba_nheads : int or list[int]
        Number of Mamba heads (one value = shared; list = per-layer).
    ffn_hidden : int or list[int]
        FFN hidden size (one value = shared; list = per-MoE-layer).
    num_experts : int or list[int]
        Experts per MoE layer (1 = dense FFN).

    Returns
    -------
    (total_params, active_params) : (int, int)
    """
    # Normalise to iterators
    mamba_layers = [c for c in hybrid_pattern if c == 'M']
    attn_layers  = [c for c in hybrid_pattern if c == '*']
    moe_layers   = [c for c in hybrid_pattern if c == 'E']

    n_mamba = len(mamba_layers)
    n_moe   = len(moe_layers)

    mamba_heads_list = (
        [mamba_nheads] * n_mamba
        if isinstance(mamba_nheads, int)
        else list(mamba_nheads)
    )
    ffn_list = (
        [ffn_hidden] * n_moe
        if isinstance(ffn_hidden, int)
        else list(ffn_hidden)
    )
    expert_list = (
        [num_experts] * n_moe
        if isinstance(num_experts, int)
        else list(num_experts)
    )

    # Validate list lengths
    if len(mamba_heads_list) != n_mamba:
        raise ValueError(
            f"mamba_nheads list length {len(mamba_heads_list)} != "
            f"{n_mamba} Mamba layers in pattern '{hybrid_pattern}'"
        )
    if len(ffn_list) != n_moe:
        raise ValueError(
            f"ffn_hidden list length {len(ffn_list)} != "
            f"{n_moe} E-layers in pattern '{hybrid_pattern}'"
        )
    if len(expert_list) != n_moe:
        raise ValueError(
            f"num_experts list length {len(expert_list)} != "
            f"{n_moe} E-layers in pattern '{hybrid_pattern}'"
        )

    # Count
    total  = 0
    active = 0

    mamba_idx = 0
    moe_idx   = 0

    for c in hybrid_pattern:
        if c == 'M':
            p = _mamba_param_count(
                nheads=mamba_heads_list[mamba_idx],
                headdim=mamba_headdim,
                d_state=mamba_d_state,
                hidden_size=hidden_size,
            )
            total  += p
            active += p
            mamba_idx += 1
        elif c == '*':
            p = _attn_param_count(
                num_heads=num_attn_heads,
                num_query_groups=num_query_groups,
                kv_channels=kv_channels,
                hidden_size=hidden_size,
            )
            total  += p
            active += p
        elif c == 'E':
            t, a = _moe_param_count(
                ffn_hidden=ffn_list[moe_idx],
                hidden_size=hidden_size,
                num_experts=expert_list[moe_idx],
                shared_expert_hidden=shared_expert_hidden,
                topk=moe_topk,
            )
            total  += t
            active += a
            moe_idx += 1

    # Embedding and optional output layer
    embedding = vocab_size * hidden_size
    total  += embedding
    active += embedding
    if not tied_vocab:
        total  += vocab_size * hidden_size
        active += vocab_size * hidden_size

    # Final layer norm
    total  += hidden_size
    active += hidden_size

    return total, active


# ---------------------------------------------------------------------------
# KV-cache size counter (mirrors get_kv_cache_size)
# ---------------------------------------------------------------------------

def count_kv_cache_elements(
    *,
    hybrid_pattern: str,
    num_attn_heads: int,
    num_query_groups: int,
    kv_channels: int,
    mem_infer_seq_len: int = 131072,
    mem_batch_size: int = 1,
    prefill_chunk_size: int = 16384,
) -> int:
    """Count total KV-cache elements for one inference batch.

    Mirrors get_kv_cache_size from flex_budget_utils.py.
    Only attention ('*') layers contribute KV cache.
    """
    n_attn = hybrid_pattern.count('*')
    # kv cache per layer: 2 (K+V) * num_query_groups * kv_channels * seq_len * batch
    per_layer = 2 * num_query_groups * kv_channels * mem_infer_seq_len * mem_batch_size
    return n_attn * per_layer


def count_ssm_cache_elements(
    *,
    hybrid_pattern: str,
    mamba_nheads: Union[int, Sequence[int]],
    mamba_headdim: int,
    mamba_d_state: int,
    mem_batch_size: int = 1,
    ngroups: int = 8,
) -> int:
    """Count total SSM-state cache elements.

    Mirrors get_ssm_cache_size logic in flex_budget_utils.py.
    Each Mamba layer has a conv_state + ssm_state.
    """
    mamba_layers = [i for i, c in enumerate(hybrid_pattern) if c == 'M']
    n_mamba = len(mamba_layers)

    if isinstance(mamba_nheads, int):
        heads_list = [mamba_nheads] * n_mamba
    else:
        heads_list = list(mamba_nheads)

    total = 0
    for heads in heads_list:
        d_inner = heads * mamba_headdim
        conv_state = (d_inner + 2 * ngroups * mamba_d_state) * 1  # conv kernel size 1 stride
        ssm_state  = heads * mamba_d_state
        total += (conv_state + ssm_state) * mem_batch_size

    return total


# ---------------------------------------------------------------------------
# Full memory footprint (mirrors get_memory_footprint)
# ---------------------------------------------------------------------------

def estimate_subnet_memory_gb(
    *,
    hybrid_pattern: str,
    hidden_size: int,
    # Mamba
    mamba_nheads: Union[int, Sequence[int]],
    mamba_headdim: int,
    mamba_d_state: int,
    # Attention
    num_attn_heads: int,
    num_query_groups: int,
    kv_channels: int,
    # FFN / MoE
    ffn_hidden: Union[int, Sequence[int]],
    num_experts: Union[int, Sequence[int]] = 1,
    shared_expert_hidden: int = 0,
    moe_topk: int = 1,
    # Embedding
    vocab_size: int,
    tied_vocab: bool = False,
    # Inference sizing
    mem_infer_seq_len: int = 131072,
    mem_batch_size: int = 1,
    prefill_chunk_size: int = 16384,
    # Memory profile
    memory_profile: Optional[DeslocMemoryProfile] = None,
    # Diagnostics
    verbose: bool = False,
    rank: int = 0,
) -> float:
    """Estimate total inference memory in GB for one subnet configuration.

    Mirrors Megatron 2d862fe0c get_memory_footprint, reinterpreted to:
      1. Use DeslocMemoryProfile instead of MemoryConfig (no yaml dependency).
      2. Accept heterogeneous per-layer lists for mamba_nheads / ffn_hidden.
      3. Emit a structured diagnostic when verbose=True (M451 pattern).

    Returns
    -------
    float
        Estimated total memory in GB.
    """
    if memory_profile is None:
        memory_profile = DeslocMemoryProfile()   # BF16 defaults

    # ── Parameter memory ──────────────────────────────────────────────────
    total_params, active_params = count_subnet_params(
        hybrid_pattern=hybrid_pattern,
        hidden_size=hidden_size,
        mamba_nheads=mamba_nheads,
        mamba_headdim=mamba_headdim,
        mamba_d_state=mamba_d_state,
        num_attn_heads=num_attn_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        ffn_hidden=ffn_hidden,
        num_experts=num_experts,
        shared_expert_hidden=shared_expert_hidden,
        moe_topk=moe_topk,
        vocab_size=vocab_size,
        tied_vocab=tied_vocab,
    )
    # Select param count based on budget target (mirrors param_idx in upstream)
    param_count = (
        active_params
        if memory_profile.param_budget_target == "active"
        else total_params
    )
    mem_params_bytes = memory_profile.bpe_params * param_count

    # ── KV-cache memory ───────────────────────────────────────────────────
    kv_elems = count_kv_cache_elements(
        hybrid_pattern=hybrid_pattern,
        num_attn_heads=num_attn_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        mem_infer_seq_len=mem_infer_seq_len,
        mem_batch_size=mem_batch_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    mem_kv_bytes = memory_profile.bpe_kv_cache * kv_elems

    # ── SSM-state cache memory ────────────────────────────────────────────
    ssm_elems = count_ssm_cache_elements(
        hybrid_pattern=hybrid_pattern,
        mamba_nheads=mamba_nheads,
        mamba_headdim=mamba_headdim,
        mamba_d_state=mamba_d_state,
        mem_batch_size=mem_batch_size,
    )
    mem_ssm_bytes = memory_profile.bpe_ssm_cache * ssm_elems

    total_bytes = mem_params_bytes + mem_kv_bytes + mem_ssm_bytes
    total_gb = total_bytes / (1024 ** 3)

    # M451-style structured diagnostic at evaluation boundary
    if verbose and rank == 0:
        msg = (
            f"{_LOG_PREFIX} ESTIMATE "
            f"params={param_count:,}  "
            f"param_mem={mem_params_bytes / 1e9:.2f} GB  "
            f"kv_mem={mem_kv_bytes / 1e9:.2f} GB  "
            f"ssm_mem={mem_ssm_bytes / 1e9:.2f} GB  "
            f"total={total_gb:.2f} GB  "
            f"profile={memory_profile.param_budget_target}"
        )
        print(msg, flush=True)
        logger.info(msg)

    return total_gb


# ---------------------------------------------------------------------------
# Budget list scanner (mirrors FlextronModelManager._setup_param_loss_func)
# ---------------------------------------------------------------------------

def scan_budget_list_memory(
    budget_list: List[float],
    *,
    full_config: dict,
    memory_profile: Optional[DeslocMemoryProfile] = None,
    rank: int = 0,
) -> List[Tuple[float, float]]:
    """Estimate memory GB for each budget in budget_list.

    Calls ``estimate_subnet_memory_gb`` for each budget, scaling mamba_nheads
    and ffn_hidden by budget.  Mirrors FlextronModelManager iterating
    ``budget_list`` to compute ``self.all_param`` / ``self.active_param``.

    Parameters
    ----------
    budget_list : list of float
        Budgets in descending order.
    full_config : dict
        Full-model config with keys matching count_subnet_params signature
        plus 'mamba_nheads_full' (int) and 'ffn_hidden_full' (int) from
        which sub-network sizes are derived by multiplying by budget.
    memory_profile : DeslocMemoryProfile or None

    Returns
    -------
    list of (budget, estimated_gb)
    """
    if memory_profile is None:
        memory_profile = DeslocMemoryProfile()

    full_mamba = full_config.get("mamba_nheads_full", full_config.get("mamba_nheads", 0))
    full_ffn   = full_config.get("ffn_hidden_full",   full_config.get("ffn_hidden", 0))

    results = []
    for budget in budget_list:
        subnet_mamba = max(1, math.floor(full_mamba * budget))
        subnet_ffn   = max(1, math.floor(full_ffn   * budget))

        gb = estimate_subnet_memory_gb(
            hybrid_pattern=full_config["hybrid_pattern"],
            hidden_size=full_config["hidden_size"],
            mamba_nheads=subnet_mamba,
            mamba_headdim=full_config.get("mamba_headdim", 64),
            mamba_d_state=full_config.get("mamba_d_state", 128),
            num_attn_heads=full_config.get("num_attn_heads", 0),
            num_query_groups=full_config.get("num_query_groups", 1),
            kv_channels=full_config.get("kv_channels", 128),
            ffn_hidden=subnet_ffn,
            num_experts=full_config.get("num_experts", 1),
            shared_expert_hidden=full_config.get("shared_expert_hidden", 0),
            moe_topk=full_config.get("moe_topk", 1),
            vocab_size=full_config["vocab_size"],
            tied_vocab=full_config.get("tied_vocab", False),
            mem_infer_seq_len=full_config.get("mem_infer_seq_len", 131072),
            mem_batch_size=full_config.get("mem_batch_size", 1),
            memory_profile=memory_profile,
            verbose=False,
        )
        results.append((budget, gb))

    if rank == 0:
        scan_lines = "  ".join(f"b={b:.3f}->{gb:.2f}GB" for b, gb in results)
        msg = f"{_LOG_PREFIX} SCAN budget→mem: {scan_lines}"
        print(msg, flush=True)
        logger.info(msg)

    return results
