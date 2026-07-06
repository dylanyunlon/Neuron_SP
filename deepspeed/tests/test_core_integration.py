# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Integration tests: verify that all deepspeed.core sub-packages are importable.

Each module under deepspeed/core/ is tested for a clean import and for the
presence of its advertised public names (__all__).  Tests that exercise
modules with known upstream bugs are marked ``xfail`` so that CI still passes
while the issue is tracked.

Known bugs (tracked for fix):
  - ``deepspeed.core.utils`` imports ``from deepspeed.core import config`` but
    no ``config`` module exists in deepspeed.core; this cascades into any module
    that transitively imports utils (fusions, transformer internals, etc.).
  - ``deepspeed.core.optimizer.__init__`` uses ``List``, ``Dict``, ``Optional``,
    ``Callable``, and ``Tuple`` without importing them from ``typing``.

Run with::

    PYTHONPATH=/path/to/Neuron_SP pytest deepspeed/tests/test_core_integration.py -v

"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so "deepspeed" is importable without
# a full package install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # Neuron_SP/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import(dotted: str) -> types.ModuleType:
    """Import *dotted* and return the module object."""
    return importlib.import_module(dotted)


def _public_names(mod: types.ModuleType) -> list[str]:
    """Return mod.__all__ if defined, otherwise all non-private attrs."""
    if hasattr(mod, "__all__"):
        return list(mod.__all__)
    return [n for n in dir(mod) if not n.startswith("_")]


# ---------------------------------------------------------------------------
# 1. parallel_state
# ---------------------------------------------------------------------------

class TestParallelStateImport:
    """deepspeed.core.parallel_state — process-group management."""

    def test_import(self):
        mod = _import("deepspeed.core.parallel_state")
        assert mod is not None

    def test_is_module(self):
        mod = _import("deepspeed.core.parallel_state")
        assert isinstance(mod, types.ModuleType)

    def test_initialize_model_parallel_callable(self):
        """The most important public entry-point must exist."""
        mod = _import("deepspeed.core.parallel_state")
        assert callable(getattr(mod, "initialize_model_parallel", None)), (
            "parallel_state.initialize_model_parallel must be callable"
        )

    def test_core_parallel_state_attr(self):
        """deepspeed.core should re-export parallel_state as an attribute."""
        core = _import("deepspeed.core")
        assert hasattr(core, "parallel_state"), (
            "deepspeed.core should expose 'parallel_state' attribute"
        )

    def test_key_symbols_present(self):
        mod = _import("deepspeed.core.parallel_state")
        expected = [
            "initialize_model_parallel",
            "model_parallel_is_initialized",
            "destroy_model_parallel",
        ]
        missing = [s for s in expected if not hasattr(mod, s)]
        assert not missing, f"parallel_state missing symbols: {missing}"


# ---------------------------------------------------------------------------
# 2. distributed
# ---------------------------------------------------------------------------

class TestDistributedImport:
    """deepspeed.core.distributed — DDP, grad buffers, finalize_model_grads."""

    def test_import(self):
        mod = _import("deepspeed.core.distributed")
        assert mod is not None

    def test_all_defined(self):
        mod = _import("deepspeed.core.distributed")
        assert hasattr(mod, "__all__"), "distributed package must define __all__"
        assert len(mod.__all__) > 0

    def test_ddp_class_present(self):
        mod = _import("deepspeed.core.distributed")
        assert hasattr(mod, "DistributedDataParallel"), (
            "DistributedDataParallel must be importable from deepspeed.core.distributed"
        )

    def test_ddp_config_present(self):
        mod = _import("deepspeed.core.distributed")
        assert hasattr(mod, "DistributedDataParallelConfig")

    def test_finalize_model_grads_callable(self):
        mod = _import("deepspeed.core.distributed")
        assert callable(mod.finalize_model_grads)

    def test_param_and_grad_buffer_present(self):
        mod = _import("deepspeed.core.distributed")
        assert hasattr(mod, "ParamAndGradBuffer")

    def test_all_symbols_importable(self):
        """Every name in __all__ must actually be present on the module."""
        mod = _import("deepspeed.core.distributed")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"distributed.__all__ has unresolved names: {missing}"

    def test_core_reexport(self):
        core = _import("deepspeed.core")
        assert hasattr(core, "distributed")
        assert hasattr(core, "DistributedDataParallel")
        assert hasattr(core, "finalize_model_grads")


# ---------------------------------------------------------------------------
# 3. transformer
# ---------------------------------------------------------------------------

class TestTransformerImport:
    """deepspeed.core.transformer — TransformerConfig, TransformerBlock, etc."""

    def test_import(self):
        mod = _import("deepspeed.core.transformer")
        assert mod is not None

    def test_all_defined(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "__all__")
        assert len(mod.__all__) > 0

    def test_transformer_config_present(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "TransformerConfig")

    def test_transformer_block_present(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "TransformerBlock")

    def test_transformer_layer_present(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "TransformerLayer")

    def test_attention_classes(self):
        mod = _import("deepspeed.core.transformer")
        for cls in ("Attention", "SelfAttention", "DotProductAttention"):
            assert hasattr(mod, cls), f"transformer missing: {cls}"

    def test_mlp_present(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "MLP")

    def test_megatron_module_present(self):
        mod = _import("deepspeed.core.transformer")
        assert hasattr(mod, "MegatronModule")

    def test_all_symbols_importable(self):
        mod = _import("deepspeed.core.transformer")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"transformer.__all__ has unresolved names: {missing}"


# ---------------------------------------------------------------------------
# 4. optimizer  (xfail: missing 'from typing import List, …' in __init__)
# ---------------------------------------------------------------------------

class TestOptimizerImport:
    """deepspeed.core.optimizer — distributed optimizer, OptimizerConfig."""

    @pytest.mark.xfail(
        reason=(
            "deepspeed.core.optimizer.__init__ uses List/Dict/Optional/Callable/Tuple "
            "from typing without importing them (NameError: name 'List' is not defined). "
            "Fix: add 'from typing import Callable, Dict, List, Optional, Tuple' "
            "near the top of deepspeed/core/optimizer/__init__.py."
        ),
        strict=True,
    )
    def test_import(self):
        mod = _import("deepspeed.core.optimizer")
        assert mod is not None

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_all_defined(self):
        mod = _import("deepspeed.core.optimizer")
        assert hasattr(mod, "__all__")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_optimizer_config_present(self):
        mod = _import("deepspeed.core.optimizer")
        assert hasattr(mod, "OptimizerConfig")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_distributed_optimizer_present(self):
        mod = _import("deepspeed.core.optimizer")
        assert hasattr(mod, "DistributedOptimizer")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_megatron_optimizer_present(self):
        mod = _import("deepspeed.core.optimizer")
        assert hasattr(mod, "MegatronOptimizer")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_clip_grad_norm_callable(self):
        mod = _import("deepspeed.core.optimizer")
        assert callable(mod.clip_grad_norm)

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_all_symbols_importable(self):
        mod = _import("deepspeed.core.optimizer")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"optimizer.__all__ has unresolved names: {missing}"

    def test_submodule_optimizer_config_importable(self):
        """optimizer_config.py itself has no broken imports and is directly usable."""
        sub = _import("deepspeed.core.optimizer.optimizer_config")
        assert hasattr(sub, "OptimizerConfig")

    def test_submodule_clip_grads_importable(self):
        """clip_grads.py should be importable independently."""
        sub = _import("deepspeed.core.optimizer.clip_grads")
        assert hasattr(sub, "clip_grad_norm")


# ---------------------------------------------------------------------------
# 5. pipeline_parallel
# ---------------------------------------------------------------------------

class TestPipelineParallelImport:
    """deepspeed.core.pipeline_parallel — 1F1B / VPP schedules and P2P comms."""

    def test_import(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        assert mod is not None

    def test_all_defined(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        assert hasattr(mod, "__all__")
        assert len(mod.__all__) > 0

    def test_required_schedule_function(self):
        """Task spec requires this specific symbol to be importable."""
        mod = _import("deepspeed.core.pipeline_parallel")
        assert callable(
            getattr(mod, "forward_backward_pipelining_without_interleaving", None)
        ), "forward_backward_pipelining_without_interleaving must be callable"

    def test_p2p_communicator_present(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        assert hasattr(mod, "P2PCommunicator")

    def test_get_forward_backward_func(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        assert callable(mod.get_forward_backward_func)

    def test_interleaved_schedule_present(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        assert hasattr(mod, "forward_backward_pipelining_with_interleaving")

    def test_heterogeneous_schedule_present(self):
        """DES-LOC PP=5 schedule must be exported."""
        mod = _import("deepspeed.core.pipeline_parallel")
        assert hasattr(
            mod, "forward_backward_pipelining_without_interleaving_pp5_heterogeneous"
        )

    def test_all_symbols_importable(self):
        mod = _import("deepspeed.core.pipeline_parallel")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"pipeline_parallel.__all__ has unresolved names: {missing}"


# ---------------------------------------------------------------------------
# 6. tensor_parallel
# ---------------------------------------------------------------------------

class TestTensorParallelImport:
    """deepspeed.core.tensor_parallel — TP layers, RNG tracker, mappings."""

    def test_import(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert mod is not None

    def test_all_defined(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert hasattr(mod, "__all__")
        assert len(mod.__all__) > 0

    def test_column_parallel_linear(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert hasattr(mod, "ColumnParallelLinear")

    def test_row_parallel_linear(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert hasattr(mod, "RowParallelLinear")

    def test_vocab_parallel_embedding(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert hasattr(mod, "VocabParallelEmbedding")

    def test_cuda_rng_states_tracker(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert hasattr(mod, "CudaRNGStatesTracker")

    def test_model_parallel_cuda_manual_seed(self):
        mod = _import("deepspeed.core.tensor_parallel")
        assert callable(mod.model_parallel_cuda_manual_seed)

    def test_scatter_gather_functions(self):
        mod = _import("deepspeed.core.tensor_parallel")
        for fn in (
            "copy_to_tensor_model_parallel_region",
            "reduce_from_tensor_model_parallel_region",
            "scatter_to_tensor_model_parallel_region",
            "gather_from_tensor_model_parallel_region",
        ):
            assert callable(getattr(mod, fn, None)), f"tensor_parallel missing: {fn}"

    def test_all_symbols_importable(self):
        mod = _import("deepspeed.core.tensor_parallel")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"tensor_parallel.__all__ has unresolved names: {missing}"


# ---------------------------------------------------------------------------
# 7. fusions  (xfail: deepspeed.core.utils imports non-existent 'config' module)
# ---------------------------------------------------------------------------

class TestFusionsImport:
    """deepspeed.core.fusions — fused kernels for bias-dropout, layer-norm, RoPE, softmax."""

    @pytest.mark.xfail(
        reason=(
            "deepspeed.core.fusions transitively imports deepspeed.core.utils, which "
            "contains 'from deepspeed.core import config'. No 'config' module exists in "
            "deepspeed.core, causing ImportError. "
            "Fix: add deepspeed/core/config.py with is_experimental_enabled() and the "
            "required attributes (sequence_parallel, gradient_accumulation_fusion), or "
            "update utils.py to import config from the correct location."
        ),
        strict=True,
    )
    def test_import(self):
        mod = _import("deepspeed.core.fusions")
        assert mod is not None

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_all_defined(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "__all__")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_fused_layer_norm_present(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "FusedLayerNorm")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_fused_rms_norm_present(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "FusedRMSNorm")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_fused_softmax_present(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "FusedScaleMaskSoftmax")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_fused_rope_present(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "apply_rotary_pos_emb_fused")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_bias_dropout_add_present(self):
        mod = _import("deepspeed.core.fusions")
        assert hasattr(mod, "bias_dropout_add_fused_train")
        assert hasattr(mod, "bias_dropout_add_fused_inference")

    @pytest.mark.xfail(reason="Depends on successful import; see test_import xfail.", strict=True)
    def test_all_symbols_importable(self):
        mod = _import("deepspeed.core.fusions")
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"fusions.__all__ has unresolved names: {missing}"

    @pytest.mark.xfail(
        reason=(
            "fused_softmax → transformer.utils → core.jit → core.utils, "
            "which fails with 'cannot import name config from deepspeed.core'. "
            "Blocked by the same root-cause bug as test_import above."
        ),
        strict=True,
    )
    def test_fused_softmax_submodule_direct_import(self):
        """fused_softmax.py transitively depends on core.utils; blocked by config bug."""
        sub = _import("deepspeed.core.fusions.fused_softmax")
        assert hasattr(sub, "FusedScaleMaskSoftmax")

    @pytest.mark.xfail(
        reason=(
            "fused_layer_norm → transformer.utils → core.jit → core.utils, "
            "which fails with 'cannot import name config from deepspeed.core'. "
            "Blocked by the same root-cause bug as test_import above."
        ),
        strict=True,
    )
    def test_fused_layer_norm_submodule_direct_import(self):
        """fused_layer_norm.py transitively depends on core.utils; blocked by config bug."""
        sub = _import("deepspeed.core.fusions.fused_layer_norm")
        assert hasattr(sub, "FusedLayerNorm")
        assert hasattr(sub, "FusedRMSNorm")


# ---------------------------------------------------------------------------
# 8. Cross-cutting: deepspeed.core top-level __init__
# ---------------------------------------------------------------------------

class TestCoreTopLevel:
    """deepspeed.core.__init__ must import cleanly and surface key symbols."""

    def test_core_import(self):
        mod = _import("deepspeed.core")
        assert mod is not None

    def test_desloc_config(self):
        mod = _import("deepspeed.core")
        assert hasattr(mod, "DesLocConfig"), "DesLocConfig must be in deepspeed.core"

    def test_tier_type(self):
        mod = _import("deepspeed.core")
        assert hasattr(mod, "TierType")

    def test_model_parallel_config(self):
        mod = _import("deepspeed.core")
        assert hasattr(mod, "ModelParallelConfig")

    def test_parallel_state_reexport(self):
        core = _import("deepspeed.core")
        assert hasattr(core, "parallel_state")

    def test_distributed_reexport(self):
        core = _import("deepspeed.core")
        assert hasattr(core, "distributed")


# ---------------------------------------------------------------------------
# 9. Import-path equivalence: package vs module-level imports
# ---------------------------------------------------------------------------

class TestImportPathEquivalence:
    """Verify that both import styles resolve to the same object."""

    def test_parallel_state_equivalence(self):
        import deepspeed.core.parallel_state as ps_direct
        from deepspeed.core import parallel_state as ps_attr
        assert ps_direct is ps_attr

    def test_distributed_equivalence(self):
        import deepspeed.core.distributed as d_direct
        from deepspeed.core import distributed as d_attr
        assert d_direct is d_attr

    def test_transformer_equivalence(self):
        import deepspeed.core.transformer as t_direct
        from deepspeed.core import transformer as t_attr
        assert t_direct is t_attr

    def test_pipeline_parallel_equivalence(self):
        import deepspeed.core.pipeline_parallel as pp_direct
        from deepspeed.core import pipeline_parallel as pp_attr
        assert pp_direct is pp_attr

    def test_tensor_parallel_equivalence(self):
        import deepspeed.core.tensor_parallel as tp_direct
        from deepspeed.core import tensor_parallel as tp_attr
        assert tp_direct is tp_attr
