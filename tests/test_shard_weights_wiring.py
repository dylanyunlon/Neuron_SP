# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
Issue #590 ,  unit tests for the heterogeneous shard weight wiring.

Tests the full pipeline:

  run_pretrain.py -> query_runtime_config()
    -> apply_overrides(tc, overrides)
      -> tc.shard_weights = [...]
      -> tc.shard_weights_source = "runtime_query"
  desloc_engine.py -> resolve_shard_weights(config, tiers, ws)
    -> priority 1: config.shard_weights  (runtime_query)
    -> priority 2: free_vram_weights_from_tiers(tiers)
    -> priority 3: vram_weights_from_tiers(tiers)
    -> priority 4: None  (even_split)
  -> ShardState.build(model, rank, ws, device, vram_weights=weights)

AST call chain depth: 6 functions in 4 modules.

Run:
    python -m pytest tests/test_shard_weights_wiring.py -v
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub deepspeed top-level package so __init__.py (which drags in apex,
# cpuinfo, tqdm, regex …) is never executed.  Mirrors test_engine_dry_run.py.
# ---------------------------------------------------------------------------
def _stub_deepspeed() -> None:
    if "deepspeed" in sys.modules and hasattr(sys.modules["deepspeed"], "__path__"):
        return
    ds = types.ModuleType("deepspeed")
    ds.__path__ = [str(REPO_ROOT / "deepspeed")]
    ds.__package__ = "deepspeed"
    sys.modules["deepspeed"] = ds

    ds_rt = types.ModuleType("deepspeed.runtime")
    ds_rt.__path__ = [str(REPO_ROOT / "deepspeed" / "runtime")]
    ds_rt.__package__ = "deepspeed.runtime"
    sys.modules["deepspeed.runtime"] = ds_rt
    ds.runtime = ds_rt  # type: ignore[attr-defined]


_stub_deepspeed()

from deepspeed.runtime.zero3_hetero_shard import (
    vram_weights_from_tiers,
    free_vram_weights_from_tiers,
    resolve_shard_weights,
)
from deepspeed.runtime.runtime_config_query import apply_overrides, _validate
from deepspeed.runtime.desloc_config import TrainingConfig


# ── Fake TierSpec ─────────────────────────────────────────────────────────

@dataclass
class FakeTier:
    device_index: int
    total_mem_gb: float
    free_mem_gb: float = 0.0


# ── resolve_shard_weights priority tests ──────────────────────────────────

class TestResolveShardWeights:
    """Verify the 4-level priority in resolve_shard_weights (issue #590)."""

    def test_priority1_runtime_query_wins(self):
        cfg = TrainingConfig()
        cfg.shard_weights = [45.0, 45.0, 91.0]
        tiers = [FakeTier(0, 48, 46), FakeTier(1, 48, 46), FakeTier(2, 96, 93)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "runtime_query"
        assert w == [45.0, 45.0, 91.0]

    def test_priority2_free_vram_when_no_config(self):
        cfg = TrainingConfig()  # shard_weights is None
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44.5), FakeTier(2, 96, 91)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "vram_discovery"
        assert w == [45.0, 44.5, 91.0]

    def test_priority3_total_vram_fallback(self):
        cfg = TrainingConfig()
        tiers = [FakeTier(0, 48, 0), FakeTier(1, 48, 0), FakeTier(2, 96, 0)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "vram_discovery"
        # free=0 -> uses total-2.0
        assert w == [46.0, 46.0, 94.0]

    def test_priority4_even_split(self):
        cfg = TrainingConfig()
        w, src = resolve_shard_weights(cfg, None, 3)
        assert src == "even_split"
        assert w is None

    def test_wrong_length_falls_through(self):
        cfg = TrainingConfig()
        cfg.shard_weights = [45.0, 91.0]  # 2 != 3
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44), FakeTier(2, 96, 91)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "vram_discovery"

    def test_negative_weight_falls_through(self):
        cfg = TrainingConfig()
        cfg.shard_weights = [45.0, -1.0, 91.0]
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44), FakeTier(2, 96, 91)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "vram_discovery"

    def test_empty_tiers_gives_even_split(self):
        cfg = TrainingConfig()
        w, src = resolve_shard_weights(cfg, [], 3)
        assert src == "even_split"
        assert w is None


# ── apply_overrides tests ─────────────────────────────────────────────────

class TestApplyOverrides:
    """Verify special shard_weights handling in apply_overrides (issue #590)."""

    def test_sets_shard_weights_and_source(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"shard_weights": [45.0, 45.0, 91.0]})
        assert cfg.shard_weights == [45.0, 45.0, 91.0]
        assert cfg.shard_weights_source == "runtime_query"

    def test_int_normalised_to_float(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"shard_weights": [45, 45, 91]})
        assert cfg.shard_weights == [45.0, 45.0, 91.0]
        assert all(isinstance(w, float) for w in cfg.shard_weights)

    def test_rejects_non_positive(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"shard_weights": [45.0, 0.0, 91.0]})
        assert cfg.shard_weights is None  # unchanged

    def test_rejects_non_numeric(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"shard_weights": [45.0, "bad", 91.0]})
        assert cfg.shard_weights is None

    def test_cpu_offload_normalised_to_bool(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"cpu_offload_optimizer": [1, 1, 0]})
        assert cfg.cpu_offload_optimizer == [True, True, False]

    def test_other_overrides_still_work(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {"grad_accum_steps": 16, "max_lr": 1e-4})
        assert cfg.grad_accum_steps == 16
        assert cfg.max_lr == 1e-4


# ── free_vram_weights_from_tiers tests ────────────────────────────────────

class TestFreeVramWeights:

    def test_uses_free_vram(self):
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44.5), FakeTier(2, 96, 91)]
        assert free_vram_weights_from_tiers(tiers) == [45.0, 44.5, 91.0]

    def test_fallback_when_free_is_zero(self):
        tiers = [FakeTier(0, 48, 0), FakeTier(1, 48, 0), FakeTier(2, 96, 0)]
        assert free_vram_weights_from_tiers(tiers) == [46.0, 46.0, 94.0]

    def test_ordered_by_device_index(self):
        tiers = [FakeTier(2, 96, 91), FakeTier(0, 48, 45), FakeTier(1, 48, 44)]
        assert free_vram_weights_from_tiers(tiers) == [45.0, 44.0, 91.0]

    def test_empty(self):
        assert free_vram_weights_from_tiers([]) == []


# ── vram_weights_from_tiers regression ────────────────────────────────────

class TestVramWeightsFromTiers:

    def test_returns_total_mem(self):
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44), FakeTier(2, 96, 91)]
        assert vram_weights_from_tiers(tiers) == [48.0, 48.0, 96.0]

    def test_ordered_by_device_index(self):
        tiers = [FakeTier(2, 96), FakeTier(0, 48), FakeTier(1, 48)]
        assert vram_weights_from_tiers(tiers) == [48.0, 48.0, 96.0]


# ── TrainingConfig defaults ───────────────────────────────────────────────

class TestTrainingConfigDefaults:

    def test_shard_weights_default_none(self):
        assert TrainingConfig().shard_weights is None

    def test_cpu_offload_default_none(self):
        assert TrainingConfig().cpu_offload_optimizer is None

    def test_source_default_none(self):
        assert TrainingConfig().shard_weights_source is None


# ── _validate tests ───────────────────────────────────────────────────────

class TestValidate:

    def test_shard_weights_correct_length(self):
        r = _validate({"shard_weights": [45.0, 45.0, 91.0]}, 3)
        assert "shard_weights" in r

    def test_shard_weights_wrong_length(self):
        r = _validate({"shard_weights": [45.0, 91.0]}, 3)
        assert "shard_weights" not in r

    def test_shard_weights_wrong_type(self):
        r = _validate({"shard_weights": "not a list"}, 3)
        assert "shard_weights" not in r

    def test_cpu_offload_wrong_length(self):
        r = _validate({"cpu_offload_optimizer": [True]}, 3)
        assert "cpu_offload_optimizer" not in r

    def test_grad_accum_bounds(self):
        assert "grad_accum_steps" not in _validate({"grad_accum_steps": 0}, 3)
        assert "grad_accum_steps" not in _validate({"grad_accum_steps": 300}, 3)
        assert "grad_accum_steps" in _validate({"grad_accum_steps": 8}, 3)

    def test_unknown_fields_ignored(self):
        r = _validate({"unknown": 42, "grad_accum_steps": 4}, 3)
        assert "unknown" not in r
        assert r["grad_accum_steps"] == 4


# ── _parse_response tests ─────────────────────────────────────────────────

class TestParseResponse:

    def test_valid_json(self):
        from deepspeed.runtime.runtime_config_query import _parse_response
        raw = 'preamble\n{"shard_weights": [45, 45, 91]}\npostamble'
        assert _parse_response(raw)["shard_weights"] == [45, 45, 91]

    def test_error_envelope_rejected(self):
        from deepspeed.runtime.runtime_config_query import _parse_response
        raw = '{"error":{"message":"rate limit"},"type":"error"}'
        assert _parse_response(raw) is None

    def test_empty_returns_none(self):
        from deepspeed.runtime.runtime_config_query import _parse_response
        assert _parse_response("") is None
        assert _parse_response(None) is None

    def test_no_json_returns_none(self):
        from deepspeed.runtime.runtime_config_query import _parse_response
        assert _parse_response("no json here") is None


# ── end-to-end wiring simulation ──────────────────────────────────────────

class TestEndToEnd:
    """Simulate the full query -> apply -> resolve pipeline."""

    def test_runtime_query_path(self):
        cfg = TrainingConfig()
        apply_overrides(cfg, {
            "shard_weights": [45, 45, 91],
            "cpu_offload_optimizer": [True, True, False],
        })
        assert cfg.shard_weights == [45.0, 45.0, 91.0]
        assert cfg.shard_weights_source == "runtime_query"

        tiers = [FakeTier(0, 48, 46), FakeTier(1, 48, 46), FakeTier(2, 96, 93)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "runtime_query"
        assert w == [45.0, 45.0, 91.0]

    def test_fallback_when_query_fails(self):
        cfg = TrainingConfig()  # no overrides -> query failure
        tiers = [FakeTier(0, 48, 45), FakeTier(1, 48, 44.5), FakeTier(2, 96, 91)]
        w, src = resolve_shard_weights(cfg, tiers, 3)
        assert src == "vram_discovery"
        assert w == [45.0, 44.5, 91.0]

    def test_single_gpu_no_sharding(self):
        cfg = TrainingConfig()
        cfg.shard_weights = [45.0]
        w, src = resolve_shard_weights(cfg, None, 1)
        assert src == "runtime_query"
        assert w == [45.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
