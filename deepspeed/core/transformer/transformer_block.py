# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerBlock — stack of TransformerLayers with DES-LOC tier-aware placement.

Ported from Megatron-LM megatron/core/transformer/transformer_block.py
(29 commits, M2260 → M3977) and extended for the Neuron_SP / DES-LOC project.

Key evolution tracked through the commit history:
  M2260 (5cc85f3a0) – Configurable double buffering for CPU offloading
  M2297 (a99f64742) – FP4 utils for nvfp4 recipe
  M2307 (5b75141b9) – Enable simplified checkpointing
  M2379 (f0d9fa97f) – Optimise attention preproc
  M2404 (e0efff9d0) – Inference-only full model CUDA graphs
  M2405/M2432        – Revert/replay CUDA graph changes
  M2856 (5f5741db9) – Replace global parallel state with explicit pg params
  M2879 (f19b59eed) – NVLS fused reduce-scatter + residual + RMSNorm + AG
  M2906 (2b343d739) – Refactor cuda_graph_scope (MoE)
  M2919 (1eed1d24f) – Typing pass
  M3009 (5247a1f46) – Support placing MTP layers into standalone stages
  M3038 (20d66d5c7) – Gated delta net for Qwen3-Next
  M3078 (10c6f010e) – Remove padding token from MoE routing loss
  M3086 (4cfaa7d59) – Revert above
  M3127 (71c49b56d) – Fix for PR-2142
  M3196 (1fdb29f76) – Sync request counts for EP inference
  M3231 (f68c7c10f) – Replace ModuleSpec with Protocols for LayerNorm
  M3301 (d4014908b) – Extract intermediate embeddings of transformer block
  M3344 (32efeffd2) – Re-enable full_iteration CUDA graphs for inference
  M3460 (bb451db2c) – Verbose error for model_parallel_size mismatch
  M3545 (fde4059a9) – Config param for retaining pinned CPU buffers
  M3563 (9ed8b0c4a) – Fix incorrect HAVE_TE detection
  M3591 (9054192b9) – Fix IndexError in uniform activation recompute
  M3717 (1daa19f89) – conditions_embeddings for DiT diffusion transformer
  M3723 (cc4cb0119) – Revert DiT conditions_embeddings
  M3954 (118933a85) – Support recomputing in HybridModel
  M3955 (925422cd8) – One single flag for inference mode
  M3977 (e41b37002) – Refactor CUDA graph API: full_iteration impl

DES-LOC tier-aware layer placement (key Neuron_SP extension):
-----------------------------------------------------------------
``TransformerBlock`` now:
  1. Assigns each layer to a tier (H100 / A6000 / unassigned) via
     ``config.get_layer_tier()``.
  2. Logs a summary at construction showing the per-tier breakdown per PP stage.
  3. Exposes ``get_desloc_tier_map()`` for the DES-LOC engine to route
     layers to the appropriate device pool.
  4. Uses ``config.desloc_tier_strategy`` to auto-assign layers when the
     explicit layer lists are not provided.

Pipeline-parallel support:
--------------------------
  * Even split (default): layers_per_rank = num_layers // pp_size
  * Explicit heterogeneous split: ``config.pipeline_layer_split``
  * Uneven first/last stage: ``config.num_layers_in_first_pipeline_stage`` /
    ``config.num_layers_in_last_pipeline_stage``
  * Virtual pipeline parallelism (VPP): ``config.virtual_pipeline_model_parallel_size``
  * Custom layout object: ``config.pipeline_model_parallel_layout``

Full-recompute and selective-recompute are controlled by
``config.recompute_granularity`` and ``config.recompute_modules``.

CPU offloading is controlled by ``config.cpu_offloading*`` flags.
FP8 / FP4 quantisation contexts wrap each layer individually
(non-delayed recipes) or the whole block (delayed FP8).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# DES-LOC requirement: use deepspeed.comm instead of torch.distributed
try:
    import deepspeed.comm as dist
except ImportError:
    import torch.distributed as dist  # type: ignore[no-redef]

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
    _build_norm,
    get_transformer_layer_offset,
)
from deepspeed.core.transformer.spec_utils import ModuleSpec, build_module
from deepspeed.core.transformer.identity_op import IdentityOp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_pp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_pipeline_model_parallel_rank
        return get_pipeline_model_parallel_rank()
    except Exception:
        return 0


def _get_pp_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_pipeline_model_parallel_world_size
        return get_pipeline_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# make_viewless_tensor helper (wraps deepspeed.core.utils if available)
# ---------------------------------------------------------------------------

def _make_viewless_tensor(t: Tensor, requires_grad: bool = True, keep_graph: bool = True) -> Tensor:
    """Return a viewless copy of *t*, guarding against schedule.py errors.

    Tries to use ``deepspeed.core.utils.make_viewless_tensor`` (which uses the
    JIT kernel for true viewlessness); falls back to ``.contiguous()`` so that
    the code path always works even without the CUDA extension.
    """
    try:
        from deepspeed.core.utils import make_viewless_tensor
        return make_viewless_tensor(inp=t, requires_grad=requires_grad, keep_graph=keep_graph)
    except Exception:
        return t.contiguous() if not t.is_contiguous() else t


# ---------------------------------------------------------------------------
# get_num_layers_to_build helper (M2260 era logic, aligned with Megatron)
# ---------------------------------------------------------------------------

def get_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
    """Determine the number of transformer layers to build for this PP stage.

    Supports:
      * Custom layout object (``config.pipeline_model_parallel_layout``).
      * Even split (default).
      * Explicit heterogeneous split (``config.pipeline_layer_split``).
      * Uneven first/last stage (``config.num_layers_in_first_pipeline_stage``
        / ``config.num_layers_in_last_pipeline_stage``).
      * Virtual pipeline parallelism (VPP).
      * Embedding / loss layer accounting.

    Args:
        config: TransformerConfig.
        vp_stage: Virtual pipeline stage (None if not using VPP).
        pp_rank: Pipeline rank override; uses live group if None.

    Returns:
        Number of layers to build on this PP (sub-)stage.
    """
    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1

    if pp_rank is None:
        pp_rank = _get_pp_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == pp_size - 1
    vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)

    # Custom layout object (takes priority)
    layout = getattr(config, "pipeline_model_parallel_layout", None)
    if layout is not None and hasattr(layout, "get_num_layers_to_build"):
        try:
            from deepspeed.core.transformer.enums import LayerType
            return layout.get_num_layers_to_build(layer_type=LayerType.decoder, vp_stage=vp_stage)
        except Exception:
            pass

    # DES-LOC heterogeneous split
    pipeline_layer_split: Optional[List[int]] = getattr(config, "pipeline_layer_split", None)
    if pipeline_layer_split is not None:
        if len(pipeline_layer_split) != pp_size:
            raise ValueError(
                f"pipeline_layer_split length {len(pipeline_layer_split)} "
                f"must equal pipeline_model_parallel_size {pp_size}"
            )
        if sum(pipeline_layer_split) != config.num_layers:
            raise ValueError(
                f"pipeline_layer_split sum {sum(pipeline_layer_split)} "
                f"must equal num_layers {config.num_layers}"
            )
        num_layers_per_pipeline_rank = pipeline_layer_split[pp_rank]
    elif (
        getattr(config, "num_layers_in_first_pipeline_stage", None) is not None
        or getattr(config, "num_layers_in_last_pipeline_stage", None) is not None
    ):
        # Uneven first/last stage — mirrors Megatron exactly
        layers_to_distribute = config.num_layers
        pipeline_stages_left = pp_size

        first_stage_layers = getattr(config, "num_layers_in_first_pipeline_stage", None)
        last_stage_layers = getattr(config, "num_layers_in_last_pipeline_stage", None)

        if first_stage_layers is not None:
            layers_to_distribute -= first_stage_layers
            pipeline_stages_left -= 1
        if last_stage_layers is not None:
            layers_to_distribute -= last_stage_layers
            pipeline_stages_left -= 1

        if pipeline_stages_left > 0:
            if layers_to_distribute % pipeline_stages_left != 0:
                raise ValueError(
                    f"With uneven pipelining the remaining {layers_to_distribute} layers "
                    f"must be divisible by {pipeline_stages_left} middle stages."
                )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        if is_first_pp_stage and first_stage_layers is not None:
            num_layers_per_pipeline_rank = first_stage_layers
        if is_last_pp_stage and last_stage_layers is not None:
            num_layers_per_pipeline_rank = last_stage_layers
    else:
        # Standard even split (with optional embedding/loss accounting)
        num_layers = config.num_layers
        account_embedding = getattr(config, "account_for_embedding_in_pipeline_split", False)
        account_loss = getattr(config, "account_for_loss_in_pipeline_split", False)
        if account_embedding:
            num_layers += 1
        if account_loss:
            num_layers += 1

        if num_layers % pp_size != 0:
            raise ValueError(
                f"num_layers {num_layers} must be divisible by "
                f"pipeline_model_parallel_size {pp_size}. "
                f"Hint: use pipeline_layer_split for heterogeneous PP splits."
            )
        num_layers_per_pipeline_rank = num_layers // pp_size

    # Virtual pipeline parallelism
    if vp_size is not None and pp_size > 1:
        if num_layers_per_pipeline_rank % vp_size != 0:
            raise ValueError(
                f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} "
                f"should be divisible by vp_size {vp_size}"
            )
        num_layers_to_build = num_layers_per_pipeline_rank // vp_size
    else:
        num_layers_to_build = num_layers_per_pipeline_rank

    # Subtract embedding/loss placeholder layers (mirrors Megatron is_vp_first/last_stage logic)
    account_embedding = getattr(config, "account_for_embedding_in_pipeline_split", False)
    account_loss = getattr(config, "account_for_loss_in_pipeline_split", False)

    def _is_vp_first_stage(vs, vps):
        return vs is None or vs == 0

    def _is_vp_last_stage(vs, vps):
        return vs is None or (vps is not None and vs == vps - 1) or vps is None

    if account_embedding:
        if _is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build = max(0, num_layers_to_build - 1)

    if account_loss:
        if _is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build = max(0, num_layers_to_build - 1)

    return num_layers_to_build


# ---------------------------------------------------------------------------
# TransformerBlockSubmodules (mirrors Megatron's dataclass)
# ---------------------------------------------------------------------------

@dataclass
class TransformerBlockSubmodules:
    """Dataclass for specifying the submodules of a transformer block.

    Args:
        layer_specs: List of layer spec builders (one per local layer).
        layer_norm: LayerNorm builder for the final norm.
    """

    layer_specs: Optional[List] = None
    layer_norm: Optional[object] = None


# ---------------------------------------------------------------------------
# _get_block_submodules — spec resolver (mirrors Megatron)
# ---------------------------------------------------------------------------

def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Retrieve or construct ``TransformerBlockSubmodules`` from *spec*.

    Handles three cases:
      1. ``spec`` is already a ``TransformerBlockSubmodules`` → return as-is.
      2. ``spec`` is a ``ModuleSpec`` for a ``TransformerBlock`` subclass
         → extract ``spec.submodules``.
      3. ``spec`` is a ``ModuleSpec`` for a ``BaseTransformerLayer`` subclass
         → fan out the spec for ``num_local_layers`` layers with the default
         norm implementation.

    Args:
        config: TransformerConfig.
        spec: Block spec or layer spec.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline rank override.

    Returns:
        Populated ``TransformerBlockSubmodules``.
    """
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            # Use the configured norm implementation (TE > Apex > Torch)
            norm_impl = _resolve_norm_impl(config)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers,
                layer_norm=norm_impl,
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


def _resolve_norm_impl(config: TransformerConfig):
    """Return the best available norm implementation class for *config*.

    Priority: TransformerEngine TENorm > Apex FusedLayerNorm > WrappedTorchNorm.
    Falls back to a factory that calls ``_build_norm`` if none are importable.
    """
    try:
        from deepspeed.core.extensions.transformer_engine import TENorm, HAVE_TE
        if HAVE_TE:
            return TENorm
    except ImportError:
        pass
    try:
        from deepspeed.core.fusions.fused_layer_norm import FusedLayerNorm
        return FusedLayerNorm
    except ImportError:
        pass

    # Fallback: return a callable class that wraps _build_norm
    class _WrappedTorchNorm:
        def __call__(self, *, hidden_size, eps, **kwargs):
            cfg = kwargs.get("config", config)
            return _build_norm(cfg, hidden_size=hidden_size)

    return _WrappedTorchNorm()


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(MegatronModule):
    """Stack of TransformerLayers with DES-LOC tier-aware layer placement.

    Main responsibilities:
      * Build the local layer stack for this PP stage from a spec or directly.
      * Assign DES-LOC tiers to each layer.
      * Manage full/selective gradient checkpointing (uniform / block).
      * Apply the final layer norm on the last PP stage (or last MTP layer).
      * Provide ``set_input_tensor()`` for PP receives.
      * Expose ``get_desloc_tier_map()`` for DES-LOC engine integration.
      * Wrap layers in inner FP8/FP4 quantisation contexts when needed.

    DES-LOC tier placement:
        Tier assignment is driven by ``TransformerConfig.get_layer_tier()``.
        Each ``TransformerLayer`` stores its tier in ``layer.desloc_tier``.
        ``TransformerBlock`` aggregates these at construction time and logs
        the per-stage breakdown:

            DES-LOC | PP stage 0/1 | layers [1..16] → H100: 16, A6000: 0, unassigned: 0

        The DES-LOC engine (``desloc_engine.py``) consults ``get_desloc_tier_map()``
        to route compute to the right device pool.

    Args:
        config: TransformerConfig.
        spec: TransformerBlockSubmodules, ModuleSpec, or None (builds directly).
        post_layer_norm: Whether to apply the final layer norm.
        pre_process: True if this is the first PP stage (receives embeddings).
        post_process: True if this is the last PP stage (runs final layernorm).
        pg_collection: ProcessGroupCollection for TP/PP/CP groups.
        vp_stage: Virtual pipeline stage.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Optional[Union[TransformerBlockSubmodules, ModuleSpec]] = None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[object] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config)

        self.config = config
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.pg_collection = pg_collection

        # PP receive buffer
        self.input_tensor: Optional[Tensor] = None

        # Resolve submodules from spec
        pp_rank = _get_pp_rank()
        if spec is not None:
            self.submodules = _get_block_submodules(config, spec, vp_stage, pp_rank)
        else:
            self.submodules = None

        # CPU offloading context (Megatron M2260: double-buffered offload)
        self._setup_cpu_offloading()

        # Build local layers
        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

        # Final layer norm
        self.final_layernorm: Optional[nn.Module] = None
        if self.has_final_layernorm_in_this_stage():
            if self.submodules is not None and self.submodules.layer_norm is not None:
                norm_cls = self.submodules.layer_norm
                try:
                    self.final_layernorm = norm_cls(
                        config=config,
                        hidden_size=config.hidden_size,
                        eps=config.layernorm_epsilon,
                    )
                except TypeError:
                    # Fallback for norm classes that don't accept config=
                    self.final_layernorm = _build_norm(config)
            else:
                self.final_layernorm = _build_norm(config)

        # Fused TP inference wiring (M3030 / M3063)
        if getattr(config, "inference_fuse_tp_communication", False):
            self._setup_fused_tp_communication()

        # DES-LOC: log stage tier summary
        self._log_desloc_summary()

    # ------------------------------------------------------------------
    # CPU offloading setup (M2260)
    # ------------------------------------------------------------------

    def _setup_cpu_offloading(self) -> None:
        """Initialise CPU offloading context managers (M2260 double-buffer).

        Tries to import TE's ``get_cpu_offload_context``; falls back to
        nullcontext if TE is not available.  Mirrors Megatron's block init.
        """
        get_cpu_offload_context = None
        try:
            from deepspeed.core.extensions.transformer_engine import (
                get_cpu_offload_context,
            )
        except ImportError:
            pass

        config = self.config
        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    config.cpu_offloading,
                    config.cpu_offloading_num_layers,
                    config.num_layers,
                    getattr(config, "cpu_offloading_activations", True),
                    getattr(config, "cpu_offloading_weights", False),
                    getattr(config, "cpu_offloading_double_buffering", True),
                    getattr(config, "cpu_offloading_retain_pinned_cpu_buffers", False),
                )
            )
            config._cpu_offloading_context = (
                self.offload_context if config.cpu_offloading else None
            )
        else:
            if getattr(config, "cpu_offloading", False):
                logger.warning(
                    "CPU offloading is enabled but TransformerEngine is not available; "
                    "falling back to no offloading."
                )
            self.offload_context = nullcontext()
            self.group_prefetch_offload_commit_async = None
            config._cpu_offloading_context = None

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Build the TransformerLayer stack for this PP stage.

        When a ``submodules`` spec is available, each layer is built via
        ``build_module`` using the corresponding ``layer_spec`` entry.
        Otherwise falls back to direct ``TransformerLayer`` construction.

        Global layer indices (1-based) are computed as:
            ``global_layer_number = local_index + layer_offset``
        where ``local_index`` is 1-based within this PP stage and
        ``layer_offset`` is the 0-based global index of the first layer here.

        Sets:
            self.layers           — nn.ModuleList of local layers.
            self._layer_offset    — 0-based global index of the first layer.
            self._num_local_layers — number of layers on this stage.
        """
        config = self.config
        pp_rank = _get_pp_rank()

        # Determine how many layers to build and the global offset
        if self.submodules is not None and self.submodules.layer_specs is not None:
            num_local = len(self.submodules.layer_specs)
        else:
            num_local = get_num_layers_to_build(config, self.vp_stage, pp_rank)

        offset = get_transformer_layer_offset(config, self.vp_stage, pp_rank)

        self._layer_offset = offset
        self._num_local_layers = num_local

        def _build_single_layer(layer_spec, local_idx: int) -> nn.Module:
            """Build one layer (with optional FP8/FP4 context wrapping)."""
            global_layer_number = local_idx + offset  # 1-based

            # Heterogeneous block: fetch per-layer config if supported
            if getattr(config, "heterogeneous_block_specs", False) and hasattr(
                config, "get_config_for_layer"
            ):
                layer_config = config.get_config_for_layer(global_layer_number)
            else:
                layer_config = config

            # Select quantisation context for this layer
            quantization_context = nullcontext()
            if getattr(layer_config, "fp8", False):
                try:
                    from deepspeed.core.fp8_utils import get_fp8_context
                    quantization_context = get_fp8_context(
                        layer_config, global_layer_number - 1, is_init=True
                    )
                except ImportError:
                    pass
            elif getattr(layer_config, "fp4", False):
                try:
                    from deepspeed.core.fp4_utils import get_fp4_context
                    quantization_context = get_fp4_context(
                        layer_config, global_layer_number - 1, is_init=True
                    )
                except ImportError:
                    pass

            with quantization_context:
                if layer_spec is not None:
                    return build_module(
                        layer_spec,
                        config=layer_config,
                        layer_number=local_idx,
                        pg_collection=self.pg_collection,
                        vp_stage=self.vp_stage,
                    )
                else:
                    return TransformerLayer(
                        layer_config,
                        layer_number=local_idx,
                        pg_collection=self.pg_collection,
                        vp_stage=self.vp_stage,
                    )

        if self.submodules is not None and self.submodules.layer_specs is not None:
            self.layers = nn.ModuleList([
                _build_single_layer(spec, i + 1)
                for i, spec in enumerate(self.submodules.layer_specs)
            ])
        else:
            self.layers = nn.ModuleList([
                _build_single_layer(None, i + 1)
                for i in range(num_local)
            ])

    # ------------------------------------------------------------------
    # Final layernorm placement (Megatron has_final_layernorm_in_this_stage)
    # ------------------------------------------------------------------

    def has_final_layernorm_in_this_stage(self) -> bool:
        """Check whether the final layernorm belongs on this PP stage.

        When MTP layers are present the layernorm is placed on the stage
        that holds the last decoder layer (``layer_number == config.num_layers``).
        Otherwise it goes on the last post-process stage.

        Mirrors Megatron's ``has_final_layernorm_in_this_stage`` (M3009).
        """
        # Check whether a norm implementation is configured
        has_norm_spec = (
            self.submodules is not None and self.submodules.layer_norm is not None
        ) or True  # direct-build always has a norm available

        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        if mtp_num_layers is None:
            return has_norm_spec and self.post_process and self.post_layer_norm
        else:
            # MTP: final layernorm lives on the stage hosting the last decoder layer
            for layer in self.layers:
                if layer.layer_number == self.config.num_layers:
                    return has_norm_spec and self.post_layer_norm
            return False

    # ------------------------------------------------------------------
    # Fused TP inference wiring (M3030 / M3063)
    # ------------------------------------------------------------------

    def _setup_fused_tp_communication(self) -> None:
        """Wire fused TP communication for all layers.

        Passes the next layer's QKV norm weights to the current layer's MLP FC2
        so the fused reduce-scatter + add + norm + all-gather kernel can be used
        at inference time.  Mirrors Megatron's ``_setup_fused_tp_communication``.
        """
        for i in range(len(self.layers)):
            current_layer = self.layers[i]
            if not hasattr(current_layer, "configure_fused_tp_inference"):
                continue

            next_qkv_norm_weights = None
            if i < len(self.layers) - 1:
                next_layer = self.layers[i + 1]
                if hasattr(next_layer, "get_qkv_layer_norm_weights"):
                    next_qkv_norm_weights = next_layer.get_qkv_layer_norm_weights()

            current_layer.configure_fused_tp_inference(
                skip_qkv_norm_and_all_gather=(i > 0),
                fc2_next_layer_norm_weights=next_qkv_norm_weights,
            )

    # ------------------------------------------------------------------
    # DES-LOC helpers
    # ------------------------------------------------------------------

    def _log_desloc_summary(self) -> None:
        """Log the DES-LOC tier breakdown for this PP stage."""
        tier_counts: Dict[str, int] = {"h100": 0, "a6000": 0, "unassigned": 0}
        for layer in self.layers:
            tier = getattr(layer, "desloc_tier", None)
            if tier == "h100":
                tier_counts["h100"] += 1
            elif tier == "a6000":
                tier_counts["a6000"] += 1
            else:
                tier_counts["unassigned"] += 1

        pp_rank = _get_pp_rank()
        pp_size = _get_pp_size()
        first = self._layer_offset + 1
        last = self._layer_offset + self._num_local_layers
        logger.info(
            "DES-LOC | PP stage %d/%d | layers [%d..%d] → "
            "H100: %d, A6000: %d, unassigned: %d",
            pp_rank, pp_size, first, last,
            tier_counts["h100"], tier_counts["a6000"], tier_counts["unassigned"],
        )

    def get_desloc_tier_map(self) -> Dict[int, str]:
        """Return a mapping of local-layer index (0-based) → tier string.

        Returns:
            Dict with keys in ``range(self._num_local_layers)`` and values
            ``"h100"``, ``"a6000"``, or ``"unassigned"``.

        Used by the DES-LOC engine to route layers to the correct device pool.
        """
        result: Dict[int, str] = {}
        for i, layer in enumerate(self.layers):
            tier = getattr(layer, "desloc_tier", None)
            result[i] = tier if tier is not None else "unassigned"
        return result

    def get_layer_by_global_index(self, global_layer_idx: int) -> Optional[TransformerLayer]:
        """Return the TransformerLayer for the given global (0-based) layer index.

        Args:
            global_layer_idx: 0-based global layer index
                (i.e. ``layer.layer_number - 1``).

        Returns:
            The ``TransformerLayer`` if it lives on this PP stage, else ``None``.
        """
        target_layer_number = global_layer_idx + 1  # convert to 1-based
        for layer in self.layers:
            if layer.layer_number == target_layer_number:
                return layer
        return None

    def compute_activation_memory(
        self,
        batch_size: int,
        seq_len: int,
        *,
        dtype_bytes: int = 2,
    ) -> Dict[str, int]:
        """Compute approximate activation memory for this PP stage by DES-LOC tier.

        Args:
            batch_size: Micro-batch size.
            seq_len: Sequence length.
            dtype_bytes: Bytes per element (2 for BF16/FP16, 4 for FP32).

        Returns:
            Dict mapping tier → approximate activation bytes::

                {
                    "h100": <bytes>,
                    "a6000": <bytes>,
                    "unassigned": <bytes>,
                }
        """
        h = self.config.hidden_size
        recompute = getattr(self.config, "recompute_granularity", None)

        if recompute == "full":
            reduction = 8
        elif recompute == "selective":
            reduction = 3
        else:
            reduction = 1

        act_per_layer = int(4 * batch_size * seq_len * h * dtype_bytes // reduction)

        result: Dict[str, int] = {"h100": 0, "a6000": 0, "unassigned": 0}
        for layer in self.layers:
            tier = getattr(layer, "desloc_tier", None) or "unassigned"
            result[tier] = result.get(tier, 0) + act_per_layer

        return result

    def get_desloc_layer_assignments(self) -> Dict[str, List[int]]:
        """Return per-tier lists of global layer indices (1-based).

        Returns:
            Dict mapping tier name → list of 1-based global layer indices.
        """
        assignments: Dict[str, List[int]] = {"h100": [], "a6000": [], "unassigned": []}
        for layer in self.layers:
            tier = getattr(layer, "desloc_tier", None) or "unassigned"
            assignments[tier].append(layer.layer_number)
        return assignments

    def clip_qk_all_layers(self) -> None:
        """Run QK logit clipping on all local layers that support it.

        Called by the DES-LOC engine at the end of each training step
        (after ``optimizer.step()``) when ``config.qk_clip`` is True.
        """
        if not getattr(self.config, "qk_clip", False):
            return
        for layer in self.layers:
            if getattr(layer, "has_clip_qk", False):
                try:
                    layer.clip_qk()
                except Exception as exc:
                    logger.debug(
                        "TransformerBlock: clip_qk skipped for layer %d: %s",
                        layer.layer_number, exc,
                    )

    def set_recompute_granularity_for_tier(
        self,
        tier: str,
        granularity: Optional[str],
    ) -> None:
        """Override activation recompute granularity for all layers of a given tier.

        Args:
            tier: ``"h100"``, ``"a6000"``, or ``"unassigned"``.
            granularity: ``"full"``, ``"selective"``, or ``None`` (no recompute).
        """
        assert granularity in (None, "full", "selective"), (
            f"granularity must be None, 'full', or 'selective', got {granularity!r}"
        )
        count = 0
        for layer in self.layers:
            layer_tier = getattr(layer, "desloc_tier", None) or "unassigned"
            if layer_tier == tier:
                layer.recompute_granularity = granularity
                recompute_modules = getattr(self.config, "recompute_modules", None) or []
                if hasattr(layer, "recompute_pre_mlp_layernorm"):
                    layer.recompute_pre_mlp_layernorm = (
                        granularity == "selective" and "layernorm" in recompute_modules
                    )
                count += 1
        logger.info(
            "TransformerBlock: set recompute_granularity=%r for %d %s-tier layers.",
            granularity, count, tier,
        )

    # ------------------------------------------------------------------
    # Layer access helpers (Megatron API compat — M3977)
    # ------------------------------------------------------------------

    def build_layer(self, layer_spec, layer_number: int, **kwargs):
        """Build a single transformer layer from a spec (Megatron API compat).

        This is the public-facing method that Megatron's ``TransformerBlock``
        uses internally.  In the deepspeed port, the actual construction is
        handled by ``_build_layers`` / ``_build_single_layer``; this method
        exists for callers that expect the Megatron interface.

        Args:
            layer_spec: The submodule spec for the layer.
            layer_number: 1-based local layer index.
            **kwargs: Extra kwargs forwarded to the layer constructor.

        Returns:
            A ``TransformerLayer`` (or ``MoETransformerLayer``) instance.
        """
        from .transformer_layer import TransformerLayer
        from .spec_utils import build_module

        if hasattr(layer_spec, 'module'):
            return build_module(layer_spec, config=self.config,
                                layer_number=layer_number, **kwargs)
        return TransformerLayer(
            config=self.config, layer_number=layer_number, **kwargs
        )

    def _get_layer(self, layer_number: int):
        """Get a layer by its 0-based local index (Megatron API compat).

        Megatron's ``TransformerBlock.forward`` uses ``self._get_layer(i)``
        to iterate over layers.  In the deepspeed port, ``self.layers`` is
        a standard ``nn.ModuleList`` so indexing works directly.

        Args:
            layer_number: 0-based local layer index.

        Returns:
            The ``TransformerLayer`` at the given local index.
        """
        return self.layers[layer_number]

    # ------------------------------------------------------------------
    # Pipeline-parallel helpers
    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Set input tensor for PP receive.

        Called by the pipeline schedule to inject the activation received
        from the previous PP stage via P2P communication.

        Args:
            input_tensor: ``[seq, batch, hidden]`` from the previous stage.
        """
        self.input_tensor = input_tensor

    # ------------------------------------------------------------------
    # Quantisation context helpers (M2297 / M2307 FP8 / FP4)
    # ------------------------------------------------------------------

    def _get_quantization_contexts(self):
        """Return ``(outer_ctx, use_inner)`` for FP8/FP4 quantisation wrapping.

        * Delayed FP8: wrap the entire forward with one outer context.
        * Non-delayed FP8 / FP4: use a per-layer inner context inside the loop.
        * No quantisation: both are nullcontext / False.
        """
        config = self.config
        fp8 = getattr(config, "fp8", False)
        fp4 = getattr(config, "fp4", False)

        if fp8:
            try:
                from deepspeed.core.fp8_utils import get_fp8_context
                from deepspeed.core.transformer.enums import Fp8Recipe
                fp8_recipe = getattr(config, "fp8_recipe", None)
                if fp8_recipe is not None and str(fp8_recipe).lower() == "delayed":
                    return get_fp8_context(config), False  # outer only
                else:
                    return nullcontext(), True  # inner only
            except ImportError:
                pass
        elif fp4:
            return nullcontext(), True  # inner only
        return nullcontext(), False

    def _get_inner_quantization_context(self, layer: nn.Module):
        """Return per-layer quantisation context (non-delayed FP8 or FP4)."""
        config = self.config
        try:
            if getattr(config, "fp8", False):
                from deepspeed.core.fp8_utils import get_fp8_context
                return get_fp8_context(config, layer.layer_number - 1)
            elif getattr(config, "fp4", False):
                from deepspeed.core.fp4_utils import get_fp4_context
                return get_fp4_context(config, layer.layer_number - 1)
        except ImportError:
            pass
        return nullcontext()

    # ------------------------------------------------------------------
    # Forward: non-checkpointed layer loop
    # ------------------------------------------------------------------

    def _forward_layers(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        attention_bias: Optional[Tensor],
        inference_context: Optional[object],
        packed_seq_params: Optional[object],
        sequence_len_offset: Optional[Tensor],
        padding_mask: Optional[Tensor],
        extract_layer_indices: Set[int],
        layer_offset: int,
        use_inner_quantization_context: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], List[Tensor]]:
        """Run all local layers sequentially (non-checkpointed path).

        Returns:
            (hidden_states, context, intermediate_hidden_states)
        """
        intermediate_hidden_states: List[Tensor] = []

        for l_no, layer in enumerate(self.layers):
            inner_ctx = (
                self._get_inner_quantization_context(layer)
                if use_inner_quantization_context
                else nullcontext()
            )
            with self.offload_context, inner_ctx:
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )

            # CPU offload commit (M2260 double-buffer)
            if (
                torch.is_grad_enabled()
                and getattr(self.config, "cpu_offloading", False)
                and self.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

            # Extract intermediate embeddings using global layer index (M3301)
            if (l_no + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

        return hidden_states, context, intermediate_hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        *,
        inference_params: Optional[object] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Forward pass through all local layers.

        If ``self.input_tensor`` is set (via :meth:`set_input_tensor`),
        it overrides *hidden_states*.

        Args:
            hidden_states: ``[seq, batch, hidden]``.
            attention_mask: Optional boolean mask ``[1, 1, seq, seq]``.
            context: Context tensor for cross-attention (encoder output).
            context_mask: Mask for cross-attention context.
            rotary_pos_emb: Rotary position embeddings.
            rotary_pos_cos: Rotary embedding cosine (for flash decode).
            rotary_pos_sin: Rotary embedding sine (for flash decode).
            rotary_pos_cos_sin: Combined cos/sin (dynamic batching flashinfer).
            attention_bias: Additive bias for attention logits.
            inference_context: Inference KV-cache context.
            packed_seq_params: THD packed sequence parameters.
            sequence_len_offset: Offset for inference CUDA graphs.
            padding_mask: Padding mask for heterogeneous batches.
            extract_layer_indices: Set of global layer indices (0-based) for
                which to collect intermediate hidden states (M3301).
            inference_params: Deprecated alias for inference_context.
            dynamic_inference_decode_only: CUDA graph runner selector.

        Returns:
            If ``extract_layer_indices`` is empty (or None): just hidden_states.
            If non-empty: ``(hidden_states, intermediate_hidden_states)`` tuple.
        """
        # Handle deprecated inference_params
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states: List[Tensor] = []

        # Compute global layer offset for intermediate embedding extraction
        pp_rank = _get_pp_rank()
        layer_offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        # PP receive: override hidden_states with tensor from previous stage
        if not self.pre_process:
            hidden_states = self.input_tensor

        # Make viewless tensor (Megatron: avoids deallocate_output_tensor issues)
        hidden_states = _make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)

        # Sequence-parallel RNG context
        if getattr(self.config, "sequence_parallel", False):
            try:
                from deepspeed.core import tensor_parallel as _tp
                rng_context = _tp.get_cuda_rng_tracker().fork()
            except Exception:
                rng_context = nullcontext()
        else:
            rng_context = nullcontext()

        # Quantisation contexts (FP8 delayed = outer; FP8 non-delayed / FP4 = inner)
        outer_quantization_context, use_inner_quantization_context = (
            self._get_quantization_contexts()
        )

        with rng_context, outer_quantization_context:
            # --- Activation checkpointing paths ---
            if getattr(self.config, "recompute_granularity", None) == "full" and self.training:
                recompute_method = getattr(self.config, "recompute_method", None)
                recompute_num_layers = getattr(self.config, "recompute_num_layers", None)
                num_layers_local = self.num_layers_per_pipeline_rank

                # Shared layer caller for checkpointed chunks
                def _call_layers(layer_slice, h, ctx):
                    for layer in self.layers[layer_slice]:
                        h, ctx = layer(
                            hidden_states=h,
                            attention_mask=attention_mask,
                            context=ctx,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                        )
                    return h, ctx

                if recompute_method == "uniform" and recompute_num_layers is not None:
                    # M3591 fix: clamp chunk_end BEFORE passing to checkpoint
                    layer_idx = 0
                    while layer_idx < num_layers_local:
                        chunk_end = min(layer_idx + recompute_num_layers, num_layers_local)
                        chunk_slice = slice(layer_idx, chunk_end)

                        def _make_fn(sl, ctx_ref):
                            def _fn(h):
                                out_h, _ = _call_layers(sl, h, ctx_ref)
                                return out_h
                            return _fn

                        hidden_states = torch.utils.checkpoint.checkpoint(
                            _make_fn(chunk_slice, context),
                            hidden_states,
                            use_reentrant=False,
                        )
                        # Collect intermediate embeddings at chunk boundary
                        for idx in range(layer_idx, chunk_end):
                            if (idx + layer_offset) in extract_layer_indices:
                                intermediate_hidden_states.append(hidden_states)
                        layer_idx = chunk_end

                elif recompute_method == "block" and recompute_num_layers is not None:
                    # Block: checkpoint first N layers, run rest normally
                    num_to_recompute = min(recompute_num_layers, num_layers_local)

                    def _make_block_fn(sl, ctx_ref):
                        def _fn(h):
                            out_h, _ = _call_layers(sl, h, ctx_ref)
                            return out_h
                        return _fn

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _make_block_fn(slice(0, num_to_recompute), context),
                        hidden_states,
                        use_reentrant=False,
                    )
                    for idx in range(0, num_to_recompute):
                        if (idx + layer_offset) in extract_layer_indices:
                            intermediate_hidden_states.append(hidden_states)

                    # Remaining layers without checkpointing
                    for l_no in range(num_to_recompute, num_layers_local):
                        inner_ctx = (
                            self._get_inner_quantization_context(self.layers[l_no])
                            if use_inner_quantization_context
                            else nullcontext()
                        )
                        with inner_ctx:
                            hidden_states, context = self.layers[l_no](
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                rotary_pos_cos_sin=rotary_pos_cos_sin,
                                attention_bias=attention_bias,
                                inference_context=inference_context,
                                packed_seq_params=packed_seq_params,
                                sequence_len_offset=sequence_len_offset,
                                padding_mask=padding_mask,
                            )
                        if (l_no + layer_offset) in extract_layer_indices:
                            intermediate_hidden_states.append(hidden_states)

                else:
                    # Fallback: checkpoint entire block as one unit
                    def _full_block(h):
                        out_h, _ = _call_layers(slice(None), h, context)
                        return out_h

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _full_block, hidden_states, use_reentrant=False
                    )

            else:
                # Standard (non-checkpointed) path
                hidden_states, context, intermediate_hidden_states = self._forward_layers(
                    hidden_states, attention_mask, context, context_mask,
                    rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin,
                    attention_bias, inference_context, packed_seq_params,
                    sequence_len_offset, padding_mask,
                    extract_layer_indices, layer_offset,
                    use_inner_quantization_context=use_inner_quantization_context,
                )

        # Final layer norm (last PP stage or last decoder layer with MTP)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm can produce a viewed tensor; make viewless to avoid
            # schedule.py's deallocate_output_tensor() errors.
            hidden_states = _make_viewless_tensor(
                hidden_states, requires_grad=True, keep_graph=True
            )

        # Edge case: empty block with no pre-process and no final norm →
        # clone to avoid in-place graph issues in pipeline schedules.
        if (
            not self.pre_process
            and len(self.layers) == 0
            and self.final_layernorm is None
        ):
            hidden_states = hidden_states.clone()

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        return hidden_states

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Generate a sharded state dictionary for the transformer block.

        Handles both homogeneous (uniform PP) and heterogeneous (non-uniform
        PP) layer layouts.  Non-homogeneous keys use the global layer index
        in the key path; homogeneous keys use a sharded offset.

        Mirrors Megatron's ``TransformerBlock.sharded_state_dict`` (M3231 era):
        applies ``replace_prefix_for_sharding`` on each layer's dict and then
        delegates non-layer modules to ``sharded_state_dict_default``.

        Args:
            prefix: Key prefix.
            sharded_offsets: PP/TP sharding offsets tuple.
            metadata: Optional dict; ``non_homogeneous_layers=True`` forces
                per-layer key paths.

        Returns:
            Dict mapping checkpoint key → tensor or ShardedTensor.
        """
        assert not sharded_offsets, "Unexpected sharded offsets passed to TransformerBlock"

        non_homogeneous_layers = (metadata or {}).get("non_homogeneous_layers", False)

        # Force non-homogeneous if layout is irregular
        if getattr(self.config, "hetereogenous_dist_checkpoint", False):
            non_homogeneous_layers = True

        moe_freq = getattr(self.config, "moe_layer_freq", 1)
        if isinstance(moe_freq, list) or (isinstance(moe_freq, int) and moe_freq > 1):
            non_homogeneous_layers = True

        lin_attn_freq = getattr(self.config, "linear_attention_freq", 0)
        if isinstance(lin_attn_freq, list) or (isinstance(lin_attn_freq, int) and lin_attn_freq > 1):
            non_homogeneous_layers = True

        if getattr(self.config, "heterogeneous_block_specs", False):
            non_homogeneous_layers = True

        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
        if singleton_local_shards:
            if (metadata or {}).get("non_homogeneous_layers") is False:
                logger.warning(
                    "non_homogeneous_layers=False is deprecated. "
                    "Setting non_homogeneous_layers=True."
                )
            non_homogeneous_layers = True

        sharded_state_dict: dict = {}
        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        pp_rank = _get_pp_rank()
        offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        for layer in self.layers:
            global_layer_offset = layer.layer_number - 1  # 0-based
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."

            if non_homogeneous_layers:
                sharded_prefix = f"{layer_prefix}{global_layer_offset}."
                sharded_pp_offset: tuple = ()
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = ((0, global_layer_offset, num_layers),)

            if hasattr(layer, "sharded_state_dict"):
                layer_sd = layer.sharded_state_dict(
                    state_dict_prefix, sharded_pp_offset, metadata
                )
            else:
                layer_sd = {
                    f"{state_dict_prefix}{k}": v
                    for k, v in layer.state_dict(prefix="").items()
                }

            # Remap keys: state_dict_prefix → sharded_prefix
            # Mirrors Megatron's replace_prefix_for_sharding
            if state_dict_prefix != sharded_prefix:
                for k in list(layer_sd.keys()):
                    new_k = k.replace(state_dict_prefix, sharded_prefix, 1)
                    if new_k != k:
                        layer_sd[new_k] = layer_sd.pop(k)

            sharded_state_dict.update(layer_sd)

        # Add non-layer modules (e.g. final_layernorm) via sharded_state_dict_default
        tp_group = None
        if self.pg_collection is not None and hasattr(self.pg_collection, "tp"):
            tp_group = self.pg_collection.tp

        for name, module in self.named_children():
            if module is self.layers:
                continue
            sub_prefix = f"{prefix}{name}."
            try:
                from deepspeed.core.transformer.utils import sharded_state_dict_default
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        sub_prefix,
                        sharded_offsets,
                        metadata,
                        tp_group=tp_group,
                    )
                )
            except Exception:
                # Fallback: plain state dict
                if hasattr(module, "sharded_state_dict"):
                    sharded_state_dict.update(
                        module.sharded_state_dict(sub_prefix, sharded_offsets, metadata)
                    )
                else:
                    for k, v in module.state_dict(prefix="").items():
                        sharded_state_dict[f"{sub_prefix}{k}"] = v

        return sharded_state_dict
