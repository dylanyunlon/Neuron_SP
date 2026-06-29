# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
End-to-end dry-run validation for desloc_engine — no GPU required.

Tests
-----
1. All public components of desloc_engine can be imported
   (DesLocEngine, TrainingConfig, TierSpec, PartitionPlan, PartitionStrategy,
   TierClass, HeteroRegistry, TierDiscovery, PartitionSolver, RMSNorm,
   CausalSelfAttention, MLP, TransformerBlock, MiniTransformer,
   build_warmup_cosine_scheduler, infinite_data_iter).
2. configs/7b_5gpu.yaml loads and contains expected top-level keys.
3. core_adapters.py — the four adapter functions return None / original value
   when their config switches are disabled (the common default).
4. core_adapters.BridgeToP2PWrapper can be instantiated with a mock bridge.
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers — minimal package stubs so we can load desloc_engine without GPU
# ---------------------------------------------------------------------------

def _stub_deepspeed_package() -> None:
    """
    Inject a minimal 'deepspeed' package stub and the three sub-modules that
    desloc_engine.py imports at module level, so the real deepspeed/__init__.py
    (which drags in apex, triton, cpuinfo …) is never executed.

    Idempotent: safe to call multiple times in the same process.
    """
    if "deepspeed" in sys.modules and hasattr(sys.modules["deepspeed"], "__path__"):
        # Already set up — nothing to do.
        return

    # ---- top-level deepspeed stub ----------------------------------------
    ds = types.ModuleType("deepspeed")
    ds.__path__ = [str(REPO_ROOT / "deepspeed")]
    ds.__package__ = "deepspeed"
    sys.modules["deepspeed"] = ds

    # ---- deepspeed.core stub ---------------------------------------------
    ds_core = types.ModuleType("deepspeed.core")
    ds_core.__path__ = [str(REPO_ROOT / "deepspeed" / "core")]
    ds_core.__package__ = "deepspeed.core"
    sys.modules["deepspeed.core"] = ds_core
    ds.core = ds_core  # type: ignore[attr-defined]

    # ---- deepspeed.core.parallel_state -----------------------------------
    ps = types.ModuleType("deepspeed.core.parallel_state")
    ps.__package__ = "deepspeed.core"
    ps.is_initialized = lambda: False
    ps.get_data_parallel_rank = lambda: 0
    ps.get_data_parallel_world_size = lambda: 1
    ps.get_data_parallel_group = lambda: None
    ps.get_pipeline_model_parallel_rank = lambda: 0
    ps.get_pipeline_model_parallel_world_size = lambda: 1
    ps.get_tensor_model_parallel_rank = lambda: 0
    ps.get_expert_model_parallel_world_size = lambda: 1
    ps.get_expert_model_parallel_rank = lambda: 0
    sys.modules["deepspeed.core.parallel_state"] = ps
    ds_core.parallel_state = ps  # type: ignore[attr-defined]

    # ---- deepspeed.core.stream_manager -----------------------------------
    sm = types.ModuleType("deepspeed.core.stream_manager")
    sm.__package__ = "deepspeed.core"

    class _StreamManager:
        @staticmethod
        def get_shard_sync_stream(_gpu_type: str):  # noqa: ANN001
            return None

    sm.StreamManager = _StreamManager
    sys.modules["deepspeed.core.stream_manager"] = sm
    ds_core.stream_manager = sm  # type: ignore[attr-defined]

    # ---- deepspeed.core.distributed --------------------------------------
    cdist = types.ModuleType("deepspeed.core.distributed")
    cdist.__package__ = "deepspeed.core"

    class _DDPConfig:
        def __init__(self, **kw):
            pass

    class _CoreDDP:
        def __init__(self, **kw):
            pass

    cdist.DistributedDataParallel = _CoreDDP
    cdist.DistributedDataParallelConfig = _DDPConfig
    cdist.finalize_model_grads = lambda **kw: None
    sys.modules["deepspeed.core.distributed"] = cdist
    ds_core.distributed = cdist  # type: ignore[attr-defined]

    # ---- deepspeed.runtime stub (parent package for desloc_engine) -------
    ds_rt = types.ModuleType("deepspeed.runtime")
    ds_rt.__path__ = [str(REPO_ROOT / "deepspeed" / "runtime")]
    ds_rt.__package__ = "deepspeed"
    sys.modules["deepspeed.runtime"] = ds_rt
    ds.runtime = ds_rt  # type: ignore[attr-defined]


def _load_desloc_engine() -> types.ModuleType:
    """
    Load deepspeed/runtime/desloc_engine.py as a proper module inside the
    'deepspeed.runtime' package, without running deepspeed/__init__.py.
    """
    _stub_deepspeed_package()

    mod_name = "deepspeed.runtime.desloc_engine"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = REPO_ROOT / "deepspeed" / "runtime" / "desloc_engine.py"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(path),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__package__ = "deepspeed.runtime"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_core_adapters() -> types.ModuleType:
    """
    Load deepspeed/runtime/core_adapters.py without the full package chain.
    """
    _stub_deepspeed_package()

    mod_name = "deepspeed.runtime.core_adapters"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = REPO_ROOT / "deepspeed" / "runtime" / "core_adapters.py"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(path),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__package__ = "deepspeed.runtime"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def desloc_module():
    """Lazily-loaded desloc_engine module (session-scoped for speed)."""
    return _load_desloc_engine()


@pytest.fixture(scope="session")
def adapters_module():
    """Lazily-loaded core_adapters module (session-scoped for speed)."""
    return _load_core_adapters()


@pytest.fixture(scope="session")
def yaml_config():
    """Parsed 7b_5gpu.yaml configuration."""
    cfg_path = REPO_ROOT / "configs" / "7b_5gpu.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Test 1 — import all components of desloc_engine
# ---------------------------------------------------------------------------

class TestDesLocEngineImports:
    """All public symbols from desloc_engine must be importable."""

    EXPECTED_CLASSES = [
        "DesLocEngine",
        "TrainingConfig",
        "TierSpec",
        "PartitionPlan",
        "PartitionStrategy",
        "TierClass",
        "HeteroRegistry",
        "TierDiscovery",
        "PartitionSolver",
        "RMSNorm",
        "CausalSelfAttention",
        "MLP",
        "TransformerBlock",
        "MiniTransformer",
    ]

    EXPECTED_FUNCTIONS = [
        "build_warmup_cosine_scheduler",
        "infinite_data_iter",
    ]

    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_class_importable(self, desloc_module, name):
        """Each class listed in EXPECTED_CLASSES must exist in desloc_engine."""
        assert hasattr(desloc_module, name), (
            f"desloc_engine is missing class: {name}"
        )
        obj = getattr(desloc_module, name)
        assert isinstance(obj, type), f"{name} should be a class, got {type(obj)}"

    @pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
    def test_function_importable(self, desloc_module, name):
        """Each function listed in EXPECTED_FUNCTIONS must exist in desloc_engine."""
        assert hasattr(desloc_module, name), (
            f"desloc_engine is missing function: {name}"
        )
        assert callable(getattr(desloc_module, name)), (
            f"{name} should be callable"
        )

    def test_training_config_instantiation(self, desloc_module):
        """TrainingConfig must instantiate with all defaults (no GPU needed)."""
        TrainingConfig = desloc_module.TrainingConfig
        cfg = TrainingConfig()
        # Check a few key defaults
        assert cfg.vocab_size == 32000
        assert cfg.hidden_size == 4096
        assert cfg.num_layers == 32
        assert cfg.num_heads == 32
        assert cfg.seq_len == 2048
        assert cfg.total_steps == 100_000
        assert cfg.grad_clip == 1.0
        assert cfg.activation_checkpointing is False

    def test_training_config_custom_values(self, desloc_module):
        """TrainingConfig accepts custom values and stores them correctly."""
        TrainingConfig = desloc_module.TrainingConfig
        cfg = TrainingConfig(
            vocab_size=1024,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            total_steps=100,
        )
        assert cfg.vocab_size == 1024
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 4
        assert cfg.total_steps == 100

    def test_training_config_cpu_offloading_derived_flag(self, desloc_module):
        """TrainingConfig.cpu_offloading is a derived property of cpu_offloading_num_layers."""
        TrainingConfig = desloc_module.TrainingConfig
        cfg_off = TrainingConfig(cpu_offloading_num_layers=0)
        assert cfg_off.cpu_offloading is False

        cfg_on = TrainingConfig(cpu_offloading_num_layers=4)
        assert cfg_on.cpu_offloading is True

    def test_training_config_core_adapter_switches_default_false(self, desloc_module):
        """All core module adapter switches must default to False (safe off-by-default)."""
        TrainingConfig = desloc_module.TrainingConfig
        cfg = TrainingConfig()
        assert cfg.use_core_scheduler is False
        assert cfg.use_pipeline_schedule is False
        assert cfg.use_dist_checkpointing is False
        assert cfg.use_bridge_communicator is False

    def test_partition_strategy_enum_members(self, desloc_module):
        """PartitionStrategy must have ZERO3_HETERO and PIPELINE_1F1B members."""
        PS = desloc_module.PartitionStrategy
        assert hasattr(PS, "ZERO3_HETERO")
        assert hasattr(PS, "PIPELINE_1F1B")

    def test_tier_class_enum_members(self, desloc_module):
        """TierClass must include H100, A6000, RTX_PRO_6000_BW, UNKNOWN."""
        TC = desloc_module.TierClass
        assert hasattr(TC, "H100")
        assert hasattr(TC, "A6000")
        assert hasattr(TC, "RTX_PRO_6000_BW")
        assert hasattr(TC, "UNKNOWN")

    def test_hetero_registry_instantiation(self, desloc_module):
        """HeteroRegistry must instantiate without arguments."""
        HR = desloc_module.HeteroRegistry
        registry = HR()
        assert hasattr(registry, "discover")
        assert hasattr(registry, "register_hooks")
        assert hasattr(registry, "get")
        assert len(registry) == 0

    def test_tier_discovery_has_discover_method(self, desloc_module):
        """TierDiscovery must expose a discover() method."""
        TD = desloc_module.TierDiscovery
        td = TD()
        assert callable(getattr(td, "discover", None))

    def test_partition_solver_has_solve_method(self, desloc_module):
        """PartitionSolver must expose a solve() method."""
        PS = desloc_module.PartitionSolver
        TrainingConfig = desloc_module.TrainingConfig
        cfg = TrainingConfig()
        solver = PS(tiers=[], config=cfg)
        assert callable(getattr(solver, "solve", None))

    def test_mini_transformer_instantiation_cpu(self, desloc_module):
        """MiniTransformer must instantiate on CPU with a minimal config."""
        import torch
        MiniTransformer = desloc_module.MiniTransformer
        TrainingConfig = desloc_module.TrainingConfig
        cfg = TrainingConfig(
            vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, seq_len=16
        )
        model = MiniTransformer(cfg)
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_build_warmup_cosine_scheduler(self, desloc_module):
        """build_warmup_cosine_scheduler must return a LambdaLR-compatible scheduler."""
        import torch
        from torch.optim import AdamW
        build_scheduler = desloc_module.build_warmup_cosine_scheduler
        TrainingConfig = desloc_module.TrainingConfig
        MiniTransformer = desloc_module.MiniTransformer

        cfg = TrainingConfig(
            vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
            total_steps=20, warmup_steps=5
        )
        model = MiniTransformer(cfg)
        opt = AdamW(model.parameters(), lr=1e-3)
        sched = build_scheduler(opt, warmup_steps=5, total_steps=20)
        assert hasattr(sched, "step")
        assert hasattr(sched, "get_last_lr")

    def test_infinite_data_iter(self, desloc_module):
        """infinite_data_iter must yield dicts with 'tokens' and 'labels' keys."""
        import torch
        infinite_data_iter = desloc_module.infinite_data_iter
        it = infinite_data_iter(
            vocab_size=64, batch_size=2, seq_len=8, device=torch.device("cpu")
        )
        batch = next(it)
        assert isinstance(batch, dict), "Expected dict from infinite_data_iter"
        assert "tokens" in batch
        assert "labels" in batch
        assert batch["tokens"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)


# ---------------------------------------------------------------------------
# Test 2 — load configs/7b_5gpu.yaml
# ---------------------------------------------------------------------------

class TestYamlConfig:
    """The 7b_5gpu.yaml config must be well-formed and contain expected keys."""

    TOP_LEVEL_KEYS = ["model", "training", "data", "parallelism", "desloc",
                      "nccl", "checkpoint", "eval", "logging"]

    def test_yaml_file_exists(self):
        """configs/7b_5gpu.yaml must exist on disk."""
        cfg_path = REPO_ROOT / "configs" / "7b_5gpu.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"

    @pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
    def test_top_level_key_present(self, yaml_config, key):
        """Each expected top-level section must be present in the config."""
        assert key in yaml_config, f"Missing top-level key: {key!r}"

    def test_model_section(self, yaml_config):
        """model section must contain architecture fields."""
        m = yaml_config["model"]
        assert m["vocab_size"] == 32064
        assert m["hidden_size"] == 4096
        assert m["num_layers"] == 32
        assert m["num_heads"] == 32
        assert m["seq_len"] == 2048

    def test_training_section(self, yaml_config):
        """training section must have LR and batch fields."""
        t = yaml_config["training"]
        assert t["max_lr"] == pytest.approx(3e-4)
        assert t["global_batch_size"] == 64
        assert t["grad_accum_steps"] == 4
        assert isinstance(t["micro_batch_size_per_gpu"], list)
        assert len(t["micro_batch_size_per_gpu"]) == 5

    def test_parallelism_section(self, yaml_config):
        """parallelism section must reflect a 5-GPU pipeline config."""
        p = yaml_config["parallelism"]
        assert p["pipeline_parallel"] == 5
        assert p["tensor_parallel"] == 1
        pp_split = p["pp_layer_split"]
        assert isinstance(pp_split, list)
        assert len(pp_split) == 5
        assert sum(pp_split) == yaml_config["model"]["num_layers"]

    def test_desloc_section(self, yaml_config):
        """desloc section must be enabled and have Kx/Ku/Kv parameters."""
        d = yaml_config["desloc"]
        assert d["enabled"] is True
        assert "Kx" in d
        assert "Ku" in d
        assert "Kv" in d
        assert d["zero_stage"] == 3

    def test_checkpoint_section(self, yaml_config):
        """checkpoint section must have save_every and dir."""
        c = yaml_config["checkpoint"]
        assert "save_every" in c
        assert "dir" in c

    def test_nccl_section_pcie_settings(self, yaml_config):
        """nccl section must reflect PCIe-only topology settings."""
        n = yaml_config["nccl"]
        # PCIe topology: P2P and IB disabled
        assert n["p2p_disable"] is True
        assert n["ib_disable"] is True

    def test_yaml_maps_to_training_config(self, desloc_module, yaml_config):
        """Key fields from yaml can be used to construct TrainingConfig."""
        TrainingConfig = desloc_module.TrainingConfig
        m = yaml_config["model"]
        t = yaml_config["training"]
        cfg = TrainingConfig(
            vocab_size=m["vocab_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            seq_len=m["seq_len"],
            total_steps=t["steps"],
            warmup_steps=t["warmup_steps"],
            max_lr=t["max_lr"],
            min_lr=t["min_lr"],
            global_batch_size=t["global_batch_size"],
            grad_accum_steps=t["grad_accum_steps"],
            micro_batch_size=t["micro_batch_size"],
            weight_decay=t["weight_decay"],
            grad_clip=t["grad_clip"],
            activation_checkpointing=t["activation_checkpointing"],
        )
        assert cfg.vocab_size == 32064
        assert cfg.num_layers == 32
        assert cfg.total_steps == 100_000
        assert cfg.activation_checkpointing is True


# ---------------------------------------------------------------------------
# Test 3 — core_adapters: adapter functions return None / original value
#           when config switches are disabled (the default)
# ---------------------------------------------------------------------------

class TestCoreAdaptersDisabled:
    """When config switches are OFF, all adapters fall through to their default."""

    @pytest.fixture()
    def disabled_config(self, desloc_module):
        """Return a TrainingConfig with all adapter switches disabled."""
        return desloc_module.TrainingConfig(
            use_core_scheduler=False,
            use_pipeline_schedule=False,
            use_dist_checkpointing=False,
            use_bridge_communicator=False,
        )

    def test_maybe_build_core_scheduler_returns_none(
        self, adapters_module, disabled_config
    ):
        """maybe_build_core_scheduler must return None when use_core_scheduler=False."""
        mock_optimizer = MagicMock()
        result = adapters_module.maybe_build_core_scheduler(
            mock_optimizer, disabled_config
        )
        assert result is None, (
            "Expected None from maybe_build_core_scheduler when switch is OFF, "
            f"got {result!r}"
        )

    def test_maybe_get_pipeline_forward_backward_returns_default(
        self, adapters_module, disabled_config
    ):
        """maybe_get_pipeline_forward_backward must return default_fn when switch is OFF."""
        sentinel_fn = MagicMock(name="default_fn")
        result = adapters_module.maybe_get_pipeline_forward_backward(
            disabled_config, default_fn=sentinel_fn
        )
        assert result is sentinel_fn, (
            "Expected default_fn back when use_pipeline_schedule=False, "
            f"got {result!r}"
        )

    def test_maybe_get_pipeline_forward_backward_returns_none_default(
        self, adapters_module, disabled_config
    ):
        """maybe_get_pipeline_forward_backward returns None when default_fn=None and switch is OFF."""
        result = adapters_module.maybe_get_pipeline_forward_backward(
            disabled_config, default_fn=None
        )
        assert result is None

    def test_maybe_build_dist_checkpoint_saver_returns_none(
        self, adapters_module, disabled_config
    ):
        """maybe_build_dist_checkpoint_saver must return None when switch is OFF."""
        result = adapters_module.maybe_build_dist_checkpoint_saver(disabled_config)
        assert result is None, (
            "Expected None from maybe_build_dist_checkpoint_saver when switch is OFF, "
            f"got {result!r}"
        )

    def test_maybe_build_bridge_communicator_returns_existing(
        self, adapters_module, disabled_config
        ):
        """maybe_build_bridge_communicator returns existing_p2p unchanged when switch is OFF."""
        mock_p2p = MagicMock(name="existing_p2p")
        result = adapters_module.maybe_build_bridge_communicator(
            disabled_config, mock_p2p
        )
        assert result is mock_p2p, (
            "Expected existing_p2p back when use_bridge_communicator=False, "
            f"got {result!r}"
        )

    def test_all_adapter_switches_checked_via_getattr(self, adapters_module):
        """All four adapters check their switch with getattr(..., False) so missing attr = OFF."""

        class _EmptyConfig:
            """Config with no attributes at all — all switches should behave as False."""

        cfg = _EmptyConfig()
        sentinel = MagicMock(name="sentinel")

        # 1. scheduler → None
        result = adapters_module.maybe_build_core_scheduler(MagicMock(), cfg)
        assert result is None

        # 2. pipeline → default_fn unchanged
        result = adapters_module.maybe_get_pipeline_forward_backward(cfg, sentinel)
        assert result is sentinel

        # 3. dist checkpointing → None
        result = adapters_module.maybe_build_dist_checkpoint_saver(cfg)
        assert result is None

        # 4. bridge communicator → existing unchanged
        result = adapters_module.maybe_build_bridge_communicator(cfg, sentinel)
        assert result is sentinel


# ---------------------------------------------------------------------------
# Test 4 — BridgeToP2PWrapper can be instantiated with a mock bridge
# ---------------------------------------------------------------------------

class TestBridgeToP2PWrapper:
    """BridgeToP2PWrapper must instantiate and expose the send_activation interface."""

    @pytest.fixture()
    def mock_bridge(self):
        """A mock object that mimics BridgeCommunicator's send/recv interface."""
        bridge = MagicMock(name="BridgeCommunicator")
        import torch
        bridge.send_forward.return_value = None
        bridge.recv_forward.return_value = torch.zeros(1, 4)
        bridge.send_backward.return_value = None
        bridge.recv_backward.return_value = torch.zeros(1, 4)
        return bridge

    def test_instantiation_with_mock_bridge(self, adapters_module, mock_bridge):
        """BridgeToP2PWrapper must instantiate with a mock bridge and no cache."""
        wrapper = adapters_module.BridgeToP2PWrapper(bridge=mock_bridge)
        assert wrapper is not None
        assert wrapper._bridge is mock_bridge
        assert wrapper._cache is None

    def test_instantiation_with_locality_cache(self, adapters_module, mock_bridge):
        """BridgeToP2PWrapper must accept an optional locality_cache argument."""
        mock_cache = MagicMock(name="SharedLocalityCache")
        wrapper = adapters_module.BridgeToP2PWrapper(
            bridge=mock_bridge, locality_cache=mock_cache
        )
        assert wrapper._cache is mock_cache

    def test_send_activation_interface_exists(self, adapters_module, mock_bridge):
        """BridgeToP2PWrapper must expose a send_activation callable."""
        wrapper = adapters_module.BridgeToP2PWrapper(bridge=mock_bridge)
        assert callable(getattr(wrapper, "send_activation", None)), (
            "BridgeToP2PWrapper must expose send_activation()"
        )

    def test_send_activation_local_copy(self, adapters_module, mock_bridge):
        """
        When src_device == dst_device, send_activation should make a local
        copy without calling the bridge's send/recv methods.

        The implementation calls tensor.to(f"cuda:{dst_device}") which
        requires CUDA. We patch torch.Tensor.to so the test runs CPU-only.
        """
        import torch
        from unittest.mock import patch

        wrapper = adapters_module.BridgeToP2PWrapper(bridge=mock_bridge)
        tensor = torch.zeros(2, 4)

        # Patch .to() to return the tensor unchanged (avoids actual CUDA call)
        with patch.object(torch.Tensor, "to", return_value=tensor):
            result = wrapper.send_activation(tensor, src_device=0, dst_device=0)

        # Bridge send/recv must NOT be called for a local copy
        mock_bridge.send_forward.assert_not_called()
        mock_bridge.recv_forward.assert_not_called()
        mock_bridge.send_backward.assert_not_called()
        mock_bridge.recv_backward.assert_not_called()

    def test_send_activation_forward_direction(self, adapters_module, mock_bridge):
        """
        When src_device < dst_device, send_activation routes through
        bridge.send_forward / bridge.recv_forward.
        """
        import torch
        fake_recv = torch.ones(2, 4)
        mock_bridge.recv_forward.return_value = fake_recv

        wrapper = adapters_module.BridgeToP2PWrapper(bridge=mock_bridge)
        tensor = torch.zeros(2, 4)

        result = wrapper.send_activation(tensor, src_device=0, dst_device=1)

        mock_bridge.send_forward.assert_called_once_with(tensor)
        mock_bridge.recv_forward.assert_called_once()
        assert result is fake_recv

    def test_send_activation_backward_direction(self, adapters_module, mock_bridge):
        """
        When src_device > dst_device, send_activation routes through
        bridge.send_backward / bridge.recv_backward.
        """
        import torch
        fake_recv = torch.ones(2, 4) * 2.0
        mock_bridge.recv_backward.return_value = fake_recv

        wrapper = adapters_module.BridgeToP2PWrapper(bridge=mock_bridge)
        tensor = torch.zeros(2, 4)

        result = wrapper.send_activation(tensor, src_device=2, dst_device=0)

        mock_bridge.send_backward.assert_called_once_with(tensor)
        mock_bridge.recv_backward.assert_called_once()
        assert result is fake_recv

    def test_send_activation_cache_hit_skips_transfer(
        self, adapters_module, mock_bridge
    ):
        """
        When a locality_cache returns a hit for cache_key, send_activation
        must return the cached tensor without calling the bridge.

        The implementation calls cached.to(f"cuda:{dst_device}"), so we
        patch torch.Tensor.to to avoid needing an actual GPU.
        """
        import torch
        from unittest.mock import patch

        cached_tensor = torch.full((2, 4), fill_value=42.0)

        mock_cache = MagicMock(name="SharedLocalityCache")
        mock_cache.get.return_value = cached_tensor

        wrapper = adapters_module.BridgeToP2PWrapper(
            bridge=mock_bridge, locality_cache=mock_cache
        )
        tensor = torch.zeros(2, 4)

        with patch.object(torch.Tensor, "to", return_value=cached_tensor):
            result = wrapper.send_activation(
                tensor, src_device=0, dst_device=1, cache_key="test_key"
            )

        # Cache must be queried
        mock_cache.get.assert_called_once_with("test_key")
        # Bridge must NOT be called on a cache hit
        mock_bridge.send_forward.assert_not_called()
        mock_bridge.recv_forward.assert_not_called()

    def test_send_activation_cache_miss_populates_cache(
        self, adapters_module, mock_bridge
    ):
        """
        On a cache miss, send_activation must call cache.put with the result.
        """
        import torch
        fake_recv = torch.ones(2, 4)
        mock_bridge.recv_forward.return_value = fake_recv

        mock_cache = MagicMock(name="SharedLocalityCache")
        mock_cache.get.return_value = None  # cache miss

        wrapper = adapters_module.BridgeToP2PWrapper(
            bridge=mock_bridge, locality_cache=mock_cache
        )
        tensor = torch.zeros(2, 4)

        result = wrapper.send_activation(
            tensor, src_device=0, dst_device=1, cache_key="miss_key"
        )

        # Cache must have been populated
        mock_cache.put.assert_called_once()
        call_args = mock_cache.put.call_args[0]
        assert call_args[0] == "miss_key"
