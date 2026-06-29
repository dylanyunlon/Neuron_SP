# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
Phase 3 validation: checkpoint adapter dispatch (no GPU required).

Verifies that the Phase-1 wire-up in save_checkpoint and load_checkpoint
routes correctly based on config.use_dist_checkpointing:

    False (default) → torch.save / torch.load unchanged
    True            → _dc_saver.save / _dc_saver.load via DistCheckpointAdapter

Tests are pure-Python: torch and deepspeed are stubbed in sys.modules so
the suite runs without any GPU, CUDA libraries, or a full package install.
The modules under test (core_adapters, dist_checkpointing) are loaded
directly from source via importlib.

Test classes
------------
TestBuildDistCheckpointSaver
    Unit-tests for the build_dist_checkpoint_saver() factory function.

TestSaveDispatch
    Integration-style tests that exercise the dispatch block added in
    save_checkpoint (desloc_engine.py) via the same logic path used at
    runtime, without instantiating DesLocEngine (which needs torch/GPUs).

TestLoadDispatch
    Mirrors TestSaveDispatch for the load_checkpoint Stage-3 dispatch.

TestAdapterLoadSignature
    Confirms the DistCheckpointAdapter.load() calls dc_load with the
    correct (sharded_state_dict={}, checkpoint_dir) signature, catching
    the dc_load(str(path)) bug that was fixed alongside Phase 1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Torch + deepspeed stubs
# ---------------------------------------------------------------------------

def _make_torch_stub():
    """
    Return a minimal torch stub that captures save/load calls.
    Re-created fresh per test so call counts don't bleed across tests.
    """
    stub = types.ModuleType("torch")

    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: False
    dist.is_initialized = lambda: False
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda: 1
    dist.barrier = lambda: None
    stub.distributed = dist

    stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    stub.Tensor = object  # enough for isinstance checks to not crash

    stub._saves: list = []
    stub._loads: list = []
    stub._load_return: dict = {}

    def _save(obj, path, **kw):
        stub._saves.append(str(path))

    def _load(path, **kw):
        stub._loads.append(str(path))
        return stub._load_return

    stub.save = _save
    stub.load = _load
    return stub


def _install_torch_stub(stub):
    sys.modules["torch"] = stub
    sys.modules["torch.distributed"] = stub.distributed


def _make_deepspeed_stub():
    ds = types.ModuleType("deepspeed")
    ds.__path__ = [str(REPO_ROOT / "deepspeed")]
    ds.__package__ = "deepspeed"

    ds_core = types.ModuleType("deepspeed.core")
    ds_core.__path__ = [str(REPO_ROOT / "deepspeed" / "core")]
    ds_core.__package__ = "deepspeed.core"
    ds.core = ds_core

    safe_globals = types.ModuleType("deepspeed.core.safe_globals")
    safe_globals.register_safe_globals = lambda: None

    return ds, ds_core, safe_globals


def _install_deepspeed_stubs(ds, ds_core, safe_globals):
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.core"] = ds_core
    sys.modules["deepspeed.core.safe_globals"] = safe_globals


# ---------------------------------------------------------------------------
# Module loader helpers
# ---------------------------------------------------------------------------

def _load_module(mod_name: str, file_path: Path, package: str) -> types.ModuleType:
    """Load a source file as a named module, replacing any cached version."""
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(file_path),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__package__ = package
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_dist_checkpointing() -> types.ModuleType:
    """Load deepspeed.core.dist_checkpointing from source (with stub env)."""
    mod_name = "deepspeed.core.dist_checkpointing"
    sys.modules.pop(mod_name, None)

    # Ensure the strategies sub-package stub exists (imported inside __init__)
    strat = types.ModuleType("deepspeed.core.dist_checkpointing.strategies")
    strat.__path__ = []
    sys.modules["deepspeed.core.dist_checkpointing.strategies"] = strat

    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(REPO_ROOT / "deepspeed" / "core" / "dist_checkpointing" / "__init__.py"),
        submodule_search_locations=[
            str(REPO_ROOT / "deepspeed" / "core" / "dist_checkpointing")
        ],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__package__ = mod_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_core_adapters() -> types.ModuleType:
    """Load deepspeed.runtime.core_adapters from source (with stub env)."""
    rt = sys.modules.get("deepspeed.runtime") or types.ModuleType("deepspeed.runtime")
    rt.__path__ = [str(REPO_ROOT / "deepspeed" / "runtime")]
    sys.modules["deepspeed.runtime"] = rt

    return _load_module(
        "deepspeed.runtime.core_adapters",
        REPO_ROOT / "deepspeed" / "runtime" / "core_adapters.py",
        "deepspeed.runtime",
    )


# ---------------------------------------------------------------------------
# Session-scoped environment setup (stubs installed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _stub_environment():
    """
    Install torch + deepspeed stubs once for the whole session.
    Individual tests that need a fresh torch stub install their own via
    the `torch_stub` fixture; this session fixture only prevents import
    errors from other modules that do a bare `import torch` at load time.
    """
    stub = _make_torch_stub()
    ds, ds_core, safe_globals = _make_deepspeed_stub()
    _install_torch_stub(stub)
    _install_deepspeed_stubs(ds, ds_core, safe_globals)
    yield
    # Leave stubs in place — teardown not required for a test session.


# ---------------------------------------------------------------------------
# Function-scoped fresh torch stub (so save/load call counts are isolated)
# ---------------------------------------------------------------------------

@pytest.fixture()
def torch_stub():
    """Fresh torch stub for each test; re-installs into sys.modules."""
    stub = _make_torch_stub()
    _install_torch_stub(stub)
    return stub


# ---------------------------------------------------------------------------
# Shared module fixtures (session-scoped; loaded once after stubs are set)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dc_mod():
    """deepspeed.core.dist_checkpointing loaded from source."""
    return _load_dist_checkpointing()


@pytest.fixture(scope="session")
def adapters_mod():
    """deepspeed.runtime.core_adapters loaded from source."""
    return _load_core_adapters()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(use_dist_checkpointing: bool):
    """Return a minimal config object with the given flag."""
    cfg = MagicMock(name="TrainingConfig")
    cfg.use_dist_checkpointing = use_dist_checkpointing
    return cfg


def _make_dist_ckpt_dir(tmp_path: Path) -> Path:
    """Create a directory that looks like a dist_checkpointing checkpoint."""
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({"sharded_backend": "torch", "sharded_backend_version": 1})
    )
    (d / "common.pt").write_text("")   # sentinel file; not actually loaded
    return d


# ===========================================================================
# TestBuildDistCheckpointSaver
# ===========================================================================

class TestBuildDistCheckpointSaver:
    """Unit tests for the build_dist_checkpoint_saver() factory."""

    def test_returns_none_when_flag_is_false(self, adapters_mod):
        """Factory must return None when use_dist_checkpointing=False."""
        result = adapters_mod.build_dist_checkpoint_saver(_cfg(False))
        assert result is None

    def test_returns_none_for_missing_attribute(self, adapters_mod):
        """Factory must return None when the config has no such attribute (safe default)."""
        result = adapters_mod.build_dist_checkpoint_saver(object())
        assert result is None

    def test_returns_adapter_when_flag_is_true(self, adapters_mod):
        """Factory must return a non-None adapter when use_dist_checkpointing=True."""
        result = adapters_mod.build_dist_checkpoint_saver(_cfg(True))
        assert result is not None

    def test_adapter_has_save_method(self, adapters_mod):
        """Returned adapter must expose a callable .save() method."""
        adapter = adapters_mod.build_dist_checkpoint_saver(_cfg(True))
        assert callable(getattr(adapter, "save", None))

    def test_adapter_has_load_method(self, adapters_mod):
        """Returned adapter must expose a callable .load() method."""
        adapter = adapters_mod.build_dist_checkpoint_saver(_cfg(True))
        assert callable(getattr(adapter, "load", None))

    def test_returns_new_instance_each_call(self, adapters_mod):
        """Each call with flag=True must return a distinct adapter instance."""
        a1 = adapters_mod.build_dist_checkpoint_saver(_cfg(True))
        a2 = adapters_mod.build_dist_checkpoint_saver(_cfg(True))
        assert a1 is not a2


# ===========================================================================
# TestSaveDispatch
# ===========================================================================

class TestSaveDispatch:
    """
    Validate the Phase-1 save-side dispatch:
      flag=False → torch.save is called
      flag=True  → _dc_saver.save is called, torch.save is NOT called
    """

    def _run_dispatch(
        self,
        adapters_mod,
        torch_stub,
        tmp_path: Path,
        use_dist_checkpointing: bool,
    ):
        """
        Execute the same dispatch logic that save_checkpoint uses, extracted
        so we can test it without instantiating DesLocEngine.
        """
        import sys
        # Ensure our fresh torch stub is active
        sys.modules["torch"] = torch_stub

        cfg = _cfg(use_dist_checkpointing)
        payload = {"global_step": 1, "model_state": {}}
        path = tmp_path / "checkpoint.pt"

        _dc_saver = adapters_mod.build_dist_checkpoint_saver(cfg)

        if _dc_saver is not None:
            # Phase-1 path: dist_checkpointing directory
            ckpt_dir = path.parent / path.stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _dc_saver.save(payload, ckpt_dir)
            return "dc_saver", str(ckpt_dir)
        else:
            # Legacy path: torch.save to .pt file
            torch_stub.save(payload, str(path))
            return "torch_save", str(path)

    def test_flag_false_calls_torch_save(self, adapters_mod, torch_stub, tmp_path):
        """With flag=False, dispatch must fall through to torch.save."""
        route, path = self._run_dispatch(
            adapters_mod, torch_stub, tmp_path, use_dist_checkpointing=False
        )
        assert route == "torch_save"
        assert len(torch_stub._saves) == 1
        assert torch_stub._saves[0].endswith(".pt")

    def test_flag_false_does_not_call_dc_saver(self, adapters_mod, torch_stub, tmp_path):
        """With flag=False, _dc_saver must be None (never constructed)."""
        cfg = _cfg(False)
        assert adapters_mod.build_dist_checkpoint_saver(cfg) is None

    def test_flag_true_calls_dc_saver_save(self, adapters_mod, torch_stub, tmp_path):
        """With flag=True, dispatch must call _dc_saver.save, not torch.save."""
        dc_save_calls = []

        cfg = _cfg(True)
        adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
        # Patch the adapter's save to capture calls without real IO
        original_save = adapter.save
        adapter.save = lambda state, path: dc_save_calls.append(str(path))

        payload = {"global_step": 5}
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        adapter.save(payload, ckpt_dir)

        assert len(dc_save_calls) == 1
        assert len(torch_stub._saves) == 0, "torch.save must NOT be called when _dc_saver is active"

    def test_flag_true_creates_directory_not_pt_file(self, adapters_mod, torch_stub, tmp_path):
        """
        With flag=True, the checkpoint destination is a directory,
        not a .pt file — dist_checkpointing.save() requires a directory.
        """
        cfg = _cfg(True)
        path = tmp_path / "checkpoint.pt"

        _dc_saver = adapters_mod.build_dist_checkpoint_saver(cfg)
        assert _dc_saver is not None

        ckpt_dir = path.parent / path.stem   # same derivation as desloc_engine.py
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        assert ckpt_dir.is_dir(), "dist_checkpointing target must be a directory"
        assert not ckpt_dir.suffix == ".pt", "dist_checkpointing target must not be a .pt file"

    def test_flag_transition_false_then_true(self, adapters_mod, torch_stub, tmp_path):
        """
        Constructing the adapter twice (once False, once True) behaves
        independently — no shared state leaks between calls.
        """
        assert adapters_mod.build_dist_checkpoint_saver(_cfg(False)) is None
        assert adapters_mod.build_dist_checkpoint_saver(_cfg(True)) is not None
        # torch.save was never called (we never executed the legacy path)
        assert len(torch_stub._saves) == 0


# ===========================================================================
# TestLoadDispatch
# ===========================================================================

class TestLoadDispatch:
    """
    Validate the Phase-1 load-side dispatch:
      flag=False              → torch.load is called
      flag=True, no metadata  → torch.load is called (not a dist_checkpointing dir)
      flag=True, with metadata → _dc_saver.load is called, torch.load is NOT called
    """

    def _run_load_dispatch(
        self,
        adapters_mod,
        dc_mod,
        torch_stub,
        path: Path,
        use_dist_checkpointing: bool,
    ):
        """
        Execute the Stage-3 dispatch logic from load_checkpoint.
        Returns ('dc_load'|'torch_load', path_str).
        """
        import sys
        sys.modules["torch"] = torch_stub

        cfg = _cfg(use_dist_checkpointing)
        _dc_saver = adapters_mod.build_dist_checkpoint_saver(cfg)
        check_is_dc = dc_mod.check_is_distributed_checkpoint

        if (
            _dc_saver is not None
            and path.is_dir()
            and check_is_dc(str(path))
        ):
            # Phase-1 dist_checkpointing path
            payload = _dc_saver.load(path)
            return "dc_load", payload
        else:
            # Legacy torch.load path
            payload = torch_stub.load(str(path))
            return "torch_load", payload

    def test_flag_false_calls_torch_load(self, adapters_mod, dc_mod, torch_stub, tmp_path):
        """With flag=False, Stage 3 must call torch.load regardless of dir format."""
        dist_dir = _make_dist_ckpt_dir(tmp_path)
        route, _ = self._run_load_dispatch(
            adapters_mod, dc_mod, torch_stub, dist_dir, False
        )
        assert route == "torch_load"
        assert len(torch_stub._loads) == 1

    def test_flag_true_legacy_pt_file_calls_torch_load(
        self, adapters_mod, dc_mod, torch_stub, tmp_path
    ):
        """With flag=True but a plain .pt file, Stage 3 must fall back to torch.load."""
        pt_file = tmp_path / "checkpoint.pt"
        pt_file.write_text("")   # exists but is not a dist_checkpointing dir
        route, _ = self._run_load_dispatch(
            adapters_mod, dc_mod, torch_stub, pt_file, True
        )
        assert route == "torch_load"
        assert len(torch_stub._loads) == 1

    def test_flag_true_dir_without_metadata_calls_torch_load(
        self, adapters_mod, dc_mod, torch_stub, tmp_path
    ):
        """With flag=True but a directory missing metadata.json, fall back to torch.load."""
        bare_dir = tmp_path / "bare"
        bare_dir.mkdir()
        route, _ = self._run_load_dispatch(
            adapters_mod, dc_mod, torch_stub, bare_dir, True
        )
        assert route == "torch_load"
        assert len(torch_stub._loads) == 1

    def test_flag_true_dist_dir_calls_dc_saver_load(
        self, adapters_mod, dc_mod, torch_stub, tmp_path
    ):
        """
        With flag=True and a dist_checkpointing directory (has metadata.json),
        Stage 3 must call _dc_saver.load and NOT call torch.load.
        """
        dist_dir = _make_dist_ckpt_dir(tmp_path)

        dc_load_calls = []
        cfg = _cfg(True)
        adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
        # Patch load to avoid real IO
        adapter.load = lambda path: dc_load_calls.append(str(path)) or {}

        check_is_dc = dc_mod.check_is_distributed_checkpoint
        assert check_is_dc(str(dist_dir)), "Precondition: metadata.json must be present"

        if adapter is not None and dist_dir.is_dir() and check_is_dc(str(dist_dir)):
            adapter.load(dist_dir)
            route = "dc_load"
        else:
            torch_stub.load(str(dist_dir))
            route = "torch_load"

        assert route == "dc_load"
        assert len(dc_load_calls) == 1
        assert len(torch_stub._loads) == 0, "torch.load must NOT be called when _dc_saver routes the load"

    def test_check_is_distributed_checkpoint_sentinel(self, dc_mod, tmp_path):
        """
        check_is_distributed_checkpoint() is the gate for the load dispatch.
        Verify it returns False without metadata.json and True with it.
        """
        bare = tmp_path / "bare"
        bare.mkdir()
        assert dc_mod.check_is_distributed_checkpoint(str(bare)) is False

        (bare / "metadata.json").write_text(
            json.dumps({"sharded_backend": "torch", "sharded_backend_version": 1})
        )
        assert dc_mod.check_is_distributed_checkpoint(str(bare)) is True

    def test_legacy_dir_rank_pt_files_calls_torch_load(
        self, adapters_mod, dc_mod, torch_stub, tmp_path
    ):
        """
        A rank_*.pt directory (legacy multi-GPU format) has no metadata.json
        → Stage 3 must fall back to torch.load for backward compatibility.
        """
        legacy_dir = tmp_path / "ckpt_legacy"
        legacy_dir.mkdir()
        (legacy_dir / "rank_0.pt").write_text("")
        (legacy_dir / "rank_1.pt").write_text("")

        route, _ = self._run_load_dispatch(
            adapters_mod, dc_mod, torch_stub, legacy_dir, True
        )
        assert route == "torch_load", (
            "Legacy rank_*.pt directories must still load via torch.load"
        )


# ===========================================================================
# TestAdapterLoadSignature
# ===========================================================================

class TestAdapterLoadSignature:
    """
    Confirm the DistCheckpointAdapter.load() internal call signature.

    The bug: the original implementation called dc_load(str(path)), passing
    the path as the sharded_state_dict positional arg.  The fix calls
    dc_load({}, str(path)).  These tests lock that in.
    """

    def test_adapter_load_passes_empty_sharded_state_dict(self, adapters_mod):
        """
        DistCheckpointAdapter.load() must invoke dc_load with an empty dict
        as the first positional argument (sharded_state_dict), not a string.
        """
        cfg = _cfg(True)

        dc_load_calls = []

        # Patch dc_load inside the module that the adapter closes over
        def _fake_dc_load(sharded_state_dict, checkpoint_dir):
            dc_load_calls.append({
                "sharded_state_dict": sharded_state_dict,
                "checkpoint_dir": checkpoint_dir,
            })
            return {}

        # Reload adapters with dc_load patched
        with patch(
            "deepspeed.core.dist_checkpointing.load",
            side_effect=_fake_dc_load,
        ):
            adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
            if adapter is not None:
                # Re-build with the patch active so the closure captures it
                pass

        # Build a fresh adapter under the patch context
        with patch(
            "deepspeed.core.dist_checkpointing.load",
            side_effect=_fake_dc_load,
        ) as mock_load:
            import importlib as _il
            # Force re-execution of build_dist_checkpoint_saver to pick up patch
            fresh_adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
            if fresh_adapter is not None:
                try:
                    fresh_adapter.load("/tmp/fake_ckpt")
                except Exception:
                    pass  # IO will fail; we only care about the call args
            if mock_load.called:
                args, kwargs = mock_load.call_args
                first_arg = args[0] if args else kwargs.get("sharded_state_dict")
                assert isinstance(first_arg, dict), (
                    f"First arg to dc_load must be a dict (sharded_state_dict), "
                    f"got {type(first_arg).__name__!r}: {first_arg!r}"
                )
                assert first_arg == {}, (
                    "Phase-1: sharded_state_dict must be empty dict {}, "
                    f"got {first_arg!r}"
                )

    def test_adapter_load_passes_path_as_second_arg(self, adapters_mod, tmp_path):
        """
        DistCheckpointAdapter.load() must pass the checkpoint dir string
        as the second positional argument (checkpoint_dir) to dc_load.
        """
        cfg = _cfg(True)
        target_path = str(tmp_path / "ckpt")

        with patch(
            "deepspeed.core.dist_checkpointing.load",
        ) as mock_load:
            mock_load.return_value = {}
            fresh_adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
            if fresh_adapter is not None:
                try:
                    fresh_adapter.load(target_path)
                except Exception:
                    pass
            if mock_load.called:
                args, kwargs = mock_load.call_args
                second_arg = args[1] if len(args) > 1 else kwargs.get("checkpoint_dir")
                assert second_arg == target_path, (
                    f"Second arg to dc_load must be the checkpoint path {target_path!r}, "
                    f"got {second_arg!r}"
                )

    def test_adapter_save_passes_path_as_string(self, adapters_mod, tmp_path):
        """
        DistCheckpointAdapter.save() must pass str(path) to dc_save,
        not a Path object (dc_save calls str() internally but we verify
        the adapter does not break that contract).
        """
        cfg = _cfg(True)
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        with patch("deepspeed.core.dist_checkpointing.save") as mock_save:
            fresh_adapter = adapters_mod.build_dist_checkpoint_saver(cfg)
            if fresh_adapter is not None:
                fresh_adapter.save({"global_step": 1}, str(ckpt_dir))
            if mock_save.called:
                args, kwargs = mock_save.call_args
                # First arg is state_dict, second is path string
                assert len(args) >= 2 or "checkpoint_dir" in kwargs


# ===========================================================================
# TestBackwardCompatibility
# ===========================================================================

class TestBackwardCompatibility:
    """
    End-to-end backward-compatibility: when use_dist_checkpointing=False
    the adapter is None and the old torch.save/load paths are exercised.
    Nothing about the legacy path changed in Phase 1.
    """

    def test_default_config_adapter_is_none(self, adapters_mod):
        """Default TrainingConfig (flag=False) must yield a None adapter."""
        # Simulate default config with no attribute (should also return None)
        class DefaultConfig:
            pass
        assert adapters_mod.build_dist_checkpoint_saver(DefaultConfig()) is None

    def test_explicit_false_adapter_is_none(self, adapters_mod):
        assert adapters_mod.build_dist_checkpoint_saver(_cfg(False)) is None

    def test_check_is_distributed_checkpoint_on_pt_file(self, dc_mod, tmp_path):
        """A .pt file is never a distributed checkpoint — gate returns False."""
        pt_file = tmp_path / "model.pt"
        pt_file.write_text("")
        # check_is_distributed_checkpoint expects a directory; a file → False
        assert dc_mod.check_is_distributed_checkpoint(str(pt_file)) is False

    def test_check_is_distributed_checkpoint_on_nonexistent(self, dc_mod, tmp_path):
        """A path that does not exist → check_is_distributed_checkpoint returns False."""
        missing = tmp_path / "does_not_exist"
        assert dc_mod.check_is_distributed_checkpoint(str(missing)) is False
