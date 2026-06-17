# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Heterogeneous multi-module parallelism for MIMO-style models in DES-LOC.

Mirrors Megatron 70a89aff3 — Add multi-module heterogeneous parallelism support
for MIMO model, reinterpreted as HeteroParallelismConfig + MIMORankAssigner for
DES-LOC clusters where different GPU tiers run different modules at different
tensor-parallel degrees.

Design intent (upstream 70a89aff3)
-------------------------------------
Megatron's commit introduces per-module TP/PP/DP grid configurations via
``HyperCommGrid`` so that, in a MIMO model (Multiple-Input Multiple-Output,
e.g. vision+audio encoders feeding a language backbone), each module can use
the parallelism strategy that fits its arithmetic intensity and memory footprint.

The commit adds three data structures:
  - ``ModuleStageInfo``: (is_first_stage, is_last_stage) per module per rank —
    tells each rank whether to build encoder projections, receive embeddings
    from a prior stage, or pass hidden states to a next stage.
  - ``RankRole``: a dict of {module_name → ModuleStageInfo} plus a ``ModuleLayout``
    enum (UNIFIED / NON_COLOCATED / COLOCATED).  UNIFIED means all modules
    share the same ranks and the classic single forward path runs.
    NON_COLOCATED means encoder ranks and LM ranks are disjoint; each rank
    runs only its assigned module(s).
  - ``MimoModelConfig.module_to_grid_map``: maps module names to grids so the
    model constructor can derive each rank's role without global topology queries.

The forward path is then split into three methods:
  ``_forward_all_modules`` (UNIFIED backward-compat path),
  ``_forward_encoders`` (NON_COLOCATED encoder ranks), and
  ``_forward_language_module`` (NON_COLOCATED LM ranks).

DES-LOC adaptation
-------------------
DES-LOC clusters pair A6000 GPUs (48 GB, ~310 TFLOP BF16) with H100s (80 GB,
~1979 TFLOP BF16).  Splitting tensor-parallel degree across GPU types is the
central DES-LOC optimisation:

  A6000 ranks → run vision/audio encoders at TP=1  (VRAM insufficient for TP≥2)
  H100 ranks  → run language backbone at TP=2      (3× the memory headroom)

The upstream ``module_to_grid_map`` + ``RankRole`` machinery is exactly what
DES-LOC needs, but translated away from Megatron's ``HyperCommGrid`` (which
depends on ``megatron.core``) into a DeepSpeed-native ``DeviceGrid`` that
queries ``torch.distributed`` process groups directly.

Key classes
-----------
``DeviceGrid``
    Lightweight replacement for Megatron's HyperCommGrid.  Stores a contiguous
    rank range, dimension names, and sizes; constructs named sub-groups on first
    access (lazy, because group creation is a collective that all ranks must
    call simultaneously).  No megatron.core dependency.

``ModuleStageInfo``
    Direct port of Megatron's ModuleStageInfo — (is_first_stage, is_last_stage).

``ModuleLayout``
    Enum mirroring Megatron's ModuleLayout: UNIFIED, NON_COLOCATED, COLOCATED.

``RankRole``
    Port of Megatron's RankRole.  Factory methods ``unified()`` and
    ``from_grid_map()`` let callers build a role for the current rank without
    hard-coding rank arithmetic.

``HeteroParallelismConfig``
    DES-LOC-specific wrapper that binds a ``module_to_grid_map`` together with
    per-module TP degree overrides.  ``build_role()`` drives ``RankRole``
    construction.  ``tensor_parallel_degree(module_name)`` returns the TP degree
    a DeepSpeed PipelineModule should use for the named module on this rank.

``MIMORankAssigner``
    Top-level helper: given a ``HeteroParallelismConfig`` and the current rank's
    physical GPU index, emits structured diagnostic events and returns the role.
    This is the single entry point DES-LOC training scripts call.

Diagnostic events (rank-0 only, logger.info + print, mirrors M451 pattern):
  [DS-HMIMO] ROLE_ASSIGN: per-rank module assignment and PP stage position.
  [DS-HMIMO] TP_SELECT:   tensor-parallel degree selected for each module.
  [DS-HMIMO] GRID_BUILD:  DeviceGrid sub-group construction timing.
  [DS-HMIMO] COLOCATED_WARN: when a rank hosts both encoder and LM modules
             (future COLOCATED mode), flags potential memory contention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HMIMO]"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language module key — mirrors Megatron's MIMO_LANGUAGE_MODULE_KEY
# ---------------------------------------------------------------------------

LANGUAGE_MODULE_KEY = "language"


# ---------------------------------------------------------------------------
# ModuleLayout — mirrors Megatron's ModuleLayout enum
# ---------------------------------------------------------------------------

class ModuleLayout(str, Enum):
    """Pipeline dispatch mode for MIMO multi-module parallelism.

    UNIFIED:
        No per-module grids.  All modules share the same ranks and the
        classic single-forward path is used.  Backward-compatible with
        pre-70a89aff3 behaviour.

    NON_COLOCATED:
        module_to_grid_map assigns non-overlapping rank ranges.  Each rank
        runs *either* encoder(s) *or* the language model, never both.
        The forward path dispatches to ``_forward_encoders`` or
        ``_forward_language_module`` based on RankRole.

    COLOCATED:
        (future) Overlapping rank ranges.  Encoder and LM share ranks but
        use different parallelism configs.  Placeholder — not yet implemented
        in Megatron upstream or DES-LOC.
    """

    UNIFIED = "unified"
    NON_COLOCATED = "non_colocated"
    COLOCATED = "colocated"


# ---------------------------------------------------------------------------
# DeviceGrid — pure-DeepSpeed replacement for Megatron's HyperCommGrid
# ---------------------------------------------------------------------------

@dataclass
class DeviceGrid:
    """Contiguous rank range with named parallelism dimensions.

    Attributes:
        rank_offset:  First global rank in this grid (inclusive).
        dim_names:    Ordered list of dimension names, e.g. ["tp", "pp", "dp"].
        dim_sizes:    Corresponding sizes.  Product must equal ``size``.

    Process groups are constructed lazily on the first call to ``get_group``
    because group creation is a collective — all ranks must participate.
    Scripts should call ``build_groups()`` once after all DeviceGrid objects
    are defined, before any training step.
    """

    rank_offset: int
    dim_names: List[str]
    dim_sizes: List[int]
    _groups: Dict[str, dist.ProcessGroup] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.dim_names) != len(self.dim_sizes):
            raise ValueError(
                f"DeviceGrid dim_names and dim_sizes must have equal length; "
                f"got {len(self.dim_names)} names and {len(self.dim_sizes)} sizes"
            )

    @property
    def size(self) -> int:
        """Total number of ranks in this grid."""
        n = 1
        for s in self.dim_sizes:
            n *= s
        return n

    def _stride(self, dim_idx: int) -> int:
        """Number of ranks between consecutive elements along dimension ``dim_idx``."""
        stride = 1
        for s in self.dim_sizes[dim_idx + 1:]:
            stride *= s
        return stride

    def build_groups(self) -> None:
        """Construct all named process groups.  Must be called as a collective."""
        if not dist.is_initialized():
            return
        for i, name in enumerate(self.dim_names):
            if name in self._groups:
                continue
            t0 = time.perf_counter()
            stride = self._stride(i)
            outer_stride = stride * self.dim_sizes[i]
            # Enumerate all groups along this dimension
            groups_for_dim: List[List[int]] = []
            # Outer loops over all dimensions except i
            outer = self.size // outer_stride
            inner = stride
            for o in range(outer):
                for inn in range(inner):
                    group_ranks = [
                        self.rank_offset + o * outer_stride + j * stride + inn
                        for j in range(self.dim_sizes[i])
                    ]
                    groups_for_dim.append(group_ranks)
            # Create all groups (collective), keep only the one this rank belongs to
            my_rank = dist.get_rank()
            my_group: Optional[dist.ProcessGroup] = None
            for group_ranks in groups_for_dim:
                pg = dist.new_group(ranks=group_ranks)
                if my_rank in group_ranks:
                    my_group = pg
            if my_group is not None:
                self._groups[name] = my_group
            elapsed = (time.perf_counter() - t0) * 1000
            _emit(
                "GRID_BUILD",
                f"dim={name} size={self.dim_sizes[i]} elapsed_ms={elapsed:.1f}",
                rank_zero_only=True,
            )

    def get_group(self, dim_name: str) -> Optional[dist.ProcessGroup]:
        """Return the process group for the named dimension, or None."""
        return self._groups.get(dim_name)

    def contains_rank(self, rank: int) -> bool:
        """True if ``rank`` falls within this grid's contiguous range."""
        return self.rank_offset <= rank < self.rank_offset + self.size

    def local_rank(self, rank: int) -> int:
        """Convert global rank to offset within this grid."""
        if not self.contains_rank(rank):
            raise ValueError(f"Rank {rank} is not in grid starting at {self.rank_offset}")
        return rank - self.rank_offset

    def pp_rank_and_size(self, global_rank: int) -> Tuple[int, int]:
        """Return (pp_rank, pp_size) for ``global_rank``, or (0, 1) if no PP dim."""
        if "pp" not in self.dim_names:
            return 0, 1
        pp_idx = self.dim_names.index("pp")
        pp_size = self.dim_sizes[pp_idx]
        local = self.local_rank(global_rank)
        stride = self._stride(pp_idx)
        pp_rank = (local // stride) % pp_size
        return pp_rank, pp_size


# ---------------------------------------------------------------------------
# ModuleStageInfo — direct port of Megatron's ModuleStageInfo
# ---------------------------------------------------------------------------

@dataclass
class ModuleStageInfo:
    """Pipeline stage position for a single module on this rank.

    Attributes:
        is_first_stage: This rank is the first PP stage for this module.
            Encoder ranks at the first stage receive raw modality inputs
            and produce initial embeddings.  LM ranks at the first stage
            combine encoder embeddings with text embeddings.
        is_last_stage: This rank is the last PP stage for this module.
            Encoder ranks at the last stage apply the input projection
            before handing embeddings to the LM.  LM ranks at the last
            stage compute the final logits / loss.
    """

    is_first_stage: bool
    is_last_stage: bool


# ---------------------------------------------------------------------------
# RankRole — port of Megatron's RankRole
# ---------------------------------------------------------------------------

@dataclass
class RankRole:
    """Describes which modules this rank participates in.

    Attributes:
        modules: Dict mapping module names to ModuleStageInfo for all modules
            this rank handles.  Encoder-only ranks omit LANGUAGE_MODULE_KEY;
            LM-only ranks omit all encoder keys.
        mode: Dispatch mode (UNIFIED / NON_COLOCATED / COLOCATED).
    """

    modules: Dict[str, ModuleStageInfo] = field(default_factory=dict)
    mode: ModuleLayout = ModuleLayout.UNIFIED

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def unified(cls, module_names: List[str]) -> "RankRole":
        """Create a unified role where every rank owns all modules, first+last stage.

        Mirrors Megatron's ``RankRole.unified()`` — backward-compatible path
        when no module_to_grid_map is provided.
        """
        return cls(
            modules={
                name: ModuleStageInfo(is_first_stage=True, is_last_stage=True)
                for name in module_names
            },
            mode=ModuleLayout.UNIFIED,
        )

    @classmethod
    def from_grid_map(
        cls,
        module_to_grid_map: Dict[str, DeviceGrid],
        modality_module_names: List[str],
        global_rank: Optional[int] = None,
    ) -> "RankRole":
        """Derive role for the current rank from a module-to-grid map.

        Mirrors Megatron's ``RankRole.from_grid_map()``, rewritten for
        DeepSpeed DeviceGrid instead of HyperCommGrid.

        Args:
            module_to_grid_map: Dict mapping module names to DeviceGrid objects.
                Must contain keys matching modality_module_names + LANGUAGE_MODULE_KEY.
            modality_module_names: Names of modality (non-language) modules.
            global_rank: Override for the current global rank.  Defaults to
                ``dist.get_rank()`` when omitted.

        Returns:
            RankRole for the current rank.

        Raises:
            ValueError: If grid map keys don't match expected module names.
            RuntimeError: If the current rank is not in any module grid.
        """
        expected_keys = set(modality_module_names) | {LANGUAGE_MODULE_KEY}
        grid_keys = set(module_to_grid_map.keys())
        if grid_keys != expected_keys:
            raise ValueError(
                f"module_to_grid_map keys must be exactly "
                f"{expected_keys}; got {grid_keys}. "
                f"Missing: {expected_keys - grid_keys}, Extra: {grid_keys - expected_keys}"
            )

        current_rank = global_rank if global_rank is not None else (
            dist.get_rank() if dist.is_initialized() else 0
        )
        modules: Dict[str, ModuleStageInfo] = {}

        for module_name, grid in module_to_grid_map.items():
            if not grid.contains_rank(current_rank):
                continue

            pp_rank, pp_size = grid.pp_rank_and_size(current_rank)
            is_first = pp_rank == 0
            is_last = pp_rank == pp_size - 1

            logger.info(
                f"{_LOG_PREFIX} ROLE_ASSIGN rank={current_rank} module={module_name} "
                f"pp_rank={pp_rank}/{pp_size} is_first={is_first} is_last={is_last}"
            )
            modules[module_name] = ModuleStageInfo(
                is_first_stage=is_first, is_last_stage=is_last
            )

        if not modules:
            raise RuntimeError(
                f"Rank {current_rank} is not present in any module grid. "
                f"Check HeteroParallelismConfig.module_to_grid_map."
            )

        # Detect colocated case (rank in both encoder and LM grids)
        has_enc = any(k != LANGUAGE_MODULE_KEY for k in modules)
        has_lm = LANGUAGE_MODULE_KEY in modules
        if has_enc and has_lm:
            _emit(
                "COLOCATED_WARN",
                f"rank={current_rank} participates in both encoder and LM modules. "
                f"COLOCATED mode is a future feature; memory contention is possible.",
                rank_zero_only=False,
            )
            return cls(modules=modules, mode=ModuleLayout.COLOCATED)

        return cls(modules=modules, mode=ModuleLayout.NON_COLOCATED)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_modality_modules(self) -> bool:
        """True if this rank runs any modality (non-language) encoder."""
        return any(k != LANGUAGE_MODULE_KEY for k in self.modules)

    @property
    def has_language_module(self) -> bool:
        """True if this rank runs the language backbone."""
        return LANGUAGE_MODULE_KEY in self.modules

    @property
    def modality_module_names(self) -> List[str]:
        """Names of modality modules this rank participates in."""
        return [k for k in self.modules if k != LANGUAGE_MODULE_KEY]

    def is_first_stage(self, module_name: str) -> bool:
        """True if this rank is the first PP stage for ``module_name``."""
        info = self.modules.get(module_name)
        return info.is_first_stage if info is not None else False

    def is_last_stage(self, module_name: str) -> bool:
        """True if this rank is the last PP stage for ``module_name``."""
        info = self.modules.get(module_name)
        return info.is_last_stage if info is not None else False

    def __repr__(self) -> str:
        parts = [f"mode={self.mode.value}"]
        for name, info in self.modules.items():
            parts.append(
                f"{name}(first={info.is_first_stage}, last={info.is_last_stage})"
            )
        return f"RankRole({', '.join(parts)})"


# ---------------------------------------------------------------------------
# HeteroParallelismConfig — DES-LOC extension
# ---------------------------------------------------------------------------

@dataclass
class HeteroParallelismConfig:
    """Per-module parallelism configuration for DES-LOC heterogeneous clusters.

    This is the DES-LOC counterpart to Megatron's ``MimoModelConfig.module_to_grid_map``.
    It combines the grid map with TP degree overrides so DeepSpeed's PipelineModule
    can select the right tensor-parallel group for each module at build time.

    Attributes:
        module_to_grid_map:
            Dict mapping module names to DeviceGrid objects.  When None, the
            UNIFIED layout is assumed and all modules share the same ranks and
            parallelism.  Must include LANGUAGE_MODULE_KEY plus all modality keys.

        module_tp_degrees:
            Optional override of TP degree per module.  When absent for a module,
            the TP degree is inferred from the "tp" dimension of its DeviceGrid.
            DES-LOC canonical mapping:
              "vision" → 1   (A6000: no headroom for TP split)
              "audio"  → 1   (A6000: same constraint)
              "language" → 2 (H100: 3× memory, benefits from TP=2 attention)

        modality_module_names:
            Names of non-language modality modules.  Must match the non-language
            keys in module_to_grid_map.

    Example (DES-LOC 2-node, 8-GPU cluster)::

        # Ranks 0-3: A6000, run vision + audio encoders at TP=1
        # Ranks 4-7: H100, run language backbone at TP=2 (PP=2)
        config = HeteroParallelismConfig(
            modality_module_names=["vision", "audio"],
            module_to_grid_map={
                "vision":   DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                "audio":    DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                "language": DeviceGrid(rank_offset=4, dim_names=["tp", "pp"],
                                       dim_sizes=[2, 2]),
            },
            module_tp_degrees={"vision": 1, "audio": 1, "language": 2},
        )
    """

    modality_module_names: List[str] = field(default_factory=list)
    module_to_grid_map: Optional[Dict[str, DeviceGrid]] = None
    module_tp_degrees: Optional[Dict[str, int]] = None

    def build_role(self, global_rank: Optional[int] = None) -> RankRole:
        """Construct the RankRole for the given rank.

        Args:
            global_rank: Override for current rank.  Defaults to dist.get_rank().

        Returns:
            RankRole describing which modules and pipeline stages this rank runs.
        """
        if self.module_to_grid_map is None:
            all_modules = self.modality_module_names + [LANGUAGE_MODULE_KEY]
            _emit(
                "ROLE_ASSIGN",
                f"UNIFIED mode — all ranks handle all modules: {all_modules}",
                rank_zero_only=True,
            )
            return RankRole.unified(all_modules)

        return RankRole.from_grid_map(
            module_to_grid_map=self.module_to_grid_map,
            modality_module_names=self.modality_module_names,
            global_rank=global_rank,
        )

    def tensor_parallel_degree(self, module_name: str) -> int:
        """Return the TP degree for ``module_name`` on this rank.

        Resolution order:
          1. Explicit override in module_tp_degrees.
          2. "tp" dimension size from the module's DeviceGrid.
          3. 1 (no tensor parallelism).

        Emits a TP_SELECT diagnostic on resolution.
        """
        if self.module_tp_degrees and module_name in self.module_tp_degrees:
            degree = self.module_tp_degrees[module_name]
            _emit("TP_SELECT", f"module={module_name} tp={degree} (explicit override)")
            return degree

        if self.module_to_grid_map and module_name in self.module_to_grid_map:
            grid = self.module_to_grid_map[module_name]
            if "tp" in grid.dim_names:
                idx = grid.dim_names.index("tp")
                degree = grid.dim_sizes[idx]
                _emit("TP_SELECT", f"module={module_name} tp={degree} (from DeviceGrid.tp dim)")
                return degree

        _emit("TP_SELECT", f"module={module_name} tp=1 (no tp dim, defaulting)")
        return 1

    def build_all_groups(self) -> None:
        """Call ``DeviceGrid.build_groups()`` for every grid in the map.

        Must be called once as a collective (all ranks must participate).
        """
        if self.module_to_grid_map is None:
            return
        for name, grid in self.module_to_grid_map.items():
            grid.build_groups()


# ---------------------------------------------------------------------------
# MIMORankAssigner — top-level entry point for DES-LOC training scripts
# ---------------------------------------------------------------------------

class MIMORankAssigner:
    """Assign RankRole for the current rank given a HeteroParallelismConfig.

    This is the single entry point DES-LOC training scripts and pipeline-module
    builders call to get the role for the current rank.  It handles:

      1. Group construction (``build_all_groups``).
      2. Role derivation via ``config.build_role()``.
      3. Structured diagnostic logging.
      4. Validation that COLOCATED mode (not yet implemented) is not silently used.

    Usage::

        config = HeteroParallelismConfig(
            modality_module_names=["vision", "audio"],
            module_to_grid_map={...},
            module_tp_degrees={"language": 2},
        )
        assigner = MIMORankAssigner(config)
        role = assigner.assign()  # call on all ranks simultaneously

        if role.has_modality_modules:
            # build encoder pipeline stages
            tp_degree = config.tensor_parallel_degree("vision")
        elif role.has_language_module:
            # build LM pipeline stages
            tp_degree = config.tensor_parallel_degree("language")
    """

    def __init__(self, config: HeteroParallelismConfig) -> None:
        self.config = config

    def assign(self, global_rank: Optional[int] = None) -> RankRole:
        """Build process groups and return the RankRole for this rank.

        Args:
            global_rank: Override for current rank.  Defaults to dist.get_rank().

        Returns:
            RankRole for the current rank.

        Raises:
            NotImplementedError: If the derived mode is COLOCATED (future work).
        """
        # 1. Build all DeviceGrid sub-groups (collective)
        self.config.build_all_groups()

        # 2. Derive role
        role = self.config.build_role(global_rank=global_rank)

        # 3. Validate
        if role.mode == ModuleLayout.COLOCATED:
            raise NotImplementedError(
                "COLOCATED multi-module parallelism is not yet implemented in DES-LOC. "
                "Use NON_COLOCATED (non-overlapping rank ranges per module) instead."
            )

        # 4. Emit per-rank summary at debug; rank-0 summary at info
        current_rank = global_rank if global_rank is not None else (
            dist.get_rank() if dist.is_initialized() else 0
        )
        logger.debug(f"{_LOG_PREFIX} rank={current_rank} → {role}")
        if current_rank == 0:
            _emit(
                "ROLE_ASSIGN",
                f"rank=0 role summary: {role}",
                rank_zero_only=False,  # already gated on current_rank==0
            )

        return role


# ---------------------------------------------------------------------------
# HeteroMIMOPipelineStageBuilder — convenience helper for stage construction
# ---------------------------------------------------------------------------

class HeteroMIMOPipelineStageBuilder:
    """Build DeepSpeed PipelineModule stage lists for a heterogeneous MIMO model.

    Given a RankRole and per-module layer factories, this class emits only the
    layer specs that belong to the current rank, enabling DeepSpeed's
    PipelineModule to handle NON_COLOCATED MIMO without knowing about modality
    routing.

    Each factory callable has the signature::

        def make_encoder_stages(
            module_name: str,
            is_first_stage: bool,
            is_last_stage: bool,
            tp_degree: int,
        ) -> List[nn.Module]:
            ...

        def make_lm_stages(
            is_first_stage: bool,
            is_last_stage: bool,
            tp_degree: int,
        ) -> List[nn.Module]:
            ...

    Usage::

        builder = HeteroMIMOPipelineStageBuilder(role, config)
        layers = builder.build(
            encoder_factory=make_encoder_stages,
            lm_factory=make_lm_stages,
        )
        engine = deepspeed.PipelineModule(layers=layers, ...)
    """

    def __init__(self, role: RankRole, config: HeteroParallelismConfig) -> None:
        self.role = role
        self.config = config

    def build(self, encoder_factory, lm_factory) -> list:
        """Return the layer list for this rank's pipeline stages.

        Args:
            encoder_factory: Callable(module_name, is_first_stage, is_last_stage,
                             tp_degree) → List[nn.Module].
            lm_factory:      Callable(is_first_stage, is_last_stage, tp_degree)
                             → List[nn.Module].

        Returns:
            Flat list of nn.Module objects for DeepSpeed PipelineModule.
        """
        layers: list = []

        if self.role.mode == ModuleLayout.UNIFIED:
            # All modules on all ranks — emit encoders then LM in order
            for mod_name in self.role.modality_module_names:
                info = self.role.modules[mod_name]
                tp = self.config.tensor_parallel_degree(mod_name)
                enc_layers = encoder_factory(mod_name, info.is_first_stage,
                                            info.is_last_stage, tp)
                layers.extend(enc_layers)
            if self.role.has_language_module:
                info = self.role.modules[LANGUAGE_MODULE_KEY]
                tp = self.config.tensor_parallel_degree(LANGUAGE_MODULE_KEY)
                lm_layers = lm_factory(info.is_first_stage, info.is_last_stage, tp)
                layers.extend(lm_layers)
            return layers

        if self.role.mode == ModuleLayout.NON_COLOCATED:
            if self.role.has_modality_modules:
                for mod_name in self.role.modality_module_names:
                    info = self.role.modules[mod_name]
                    tp = self.config.tensor_parallel_degree(mod_name)
                    enc_layers = encoder_factory(mod_name, info.is_first_stage,
                                                info.is_last_stage, tp)
                    layers.extend(enc_layers)
            elif self.role.has_language_module:
                info = self.role.modules[LANGUAGE_MODULE_KEY]
                tp = self.config.tensor_parallel_degree(LANGUAGE_MODULE_KEY)
                lm_layers = lm_factory(info.is_first_stage, info.is_last_stage, tp)
                layers.extend(lm_layers)
            return layers

        raise NotImplementedError(
            f"HeteroMIMOPipelineStageBuilder does not yet support {self.role.mode}"
        )


# ---------------------------------------------------------------------------
# ModalitySubmoduleBase — DeepSpeed analog of Megatron's ModalitySubmodules
# ---------------------------------------------------------------------------

import warnings
import torch.nn as nn


class ModalitySubmoduleBase(nn.Module):
    """Base class for encoder submodules in a DES-LOC heterogeneous MIMO pipeline.

    Mirrors Megatron's ModalitySubmodules base class changes in 70a89aff3:
      - ``from_spec`` now receives ``is_first_stage`` and ``is_last_stage`` so
        that projection layers are only built on the appropriate pipeline stage.
      - ``is_first_stage`` / ``is_last_stage`` are exposed as read-only properties.

    In DES-LOC the typical pattern is:
      - Encoder (feature extractor): built on first stage ranks.
      - Input projection (map encoder dim → LM dim): built on last stage ranks,
        since the projection output is what gets P2P-transferred to the LM ranks.
      - Output projection (decode embeddings → modality outputs): built on LM ranks
        that need to generate modality outputs (not common for encoders).

    Subclasses override ``encode()``, ``project()``, and optionally ``forward()``.
    """

    def __init__(
        self,
        is_first_stage: bool = True,
        is_last_stage: bool = True,
    ) -> None:
        super().__init__()
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        warnings.warn(
            "ModalitySubmoduleBase is experimental and part of DES-LOC heterogeneous "
            "MIMO support.  The API may change without notice.",
            UserWarning,
            stacklevel=2,
        )

    @property
    def is_first_stage(self) -> bool:
        """True if this rank is the first PP stage for this encoder module."""
        return self._is_first_stage

    @property
    def is_last_stage(self) -> bool:
        """True if this rank is the last PP stage for this encoder module."""
        return self._is_last_stage

    def encode(self, inputs: Dict) -> Optional[torch.Tensor]:
        """Run the encoder on raw modality inputs.

        Override in subclasses.  Should return a tensor of shape
        [total_tokens, hidden_dim] (flattened over batch × seq).
        """
        raise NotImplementedError

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project encoder output to language model dimension.

        Override in subclasses.  Only called on last-stage ranks.
        """
        raise NotImplementedError

    def forward(
        self,
        encoder_inputs: Optional[Dict] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Standard forward pass respecting pipeline stage position.

        - First stage: ``encode(encoder_inputs)`` → raw embeddings.
        - Last stage: ``project(embeddings)`` → LM-dim embeddings.
        - Middle stages (rare): pass ``hidden_states`` through any internal
          transformer layers if the encoder is itself pipelined.

        Mirrors Megatron's from_spec stage-aware construction: instead of
        building unused projection weights on every rank, DES-LOC builds
        them only on the stage that needs them.
        """
        if self.is_first_stage:
            if encoder_inputs is None:
                raise ValueError(
                    "encoder_inputs must be provided on first-stage encoder ranks"
                )
            embeddings = self.encode(encoder_inputs)
        else:
            # Intermediate or last stage: receive from prior stage
            embeddings = hidden_states

        if embeddings is None:
            return None

        if self.is_last_stage:
            return self.project(embeddings)

        return embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit(event: str, msg: str, rank_zero_only: bool = True) -> None:
    """Emit a structured diagnostic event, mirroring M451 pattern."""
    full = f"{_LOG_PREFIX} {event}: {msg}"
    if rank_zero_only:
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(full)
            print(full)
    else:
        logger.info(full)
        print(full)
