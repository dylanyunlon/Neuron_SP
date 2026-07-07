# SPDX-License-Identifier: Apache-2.0
"""Unit tests for distrib_optimizer.py gap fixes — addresses #28.

Tests
-----
1.  Gap #1: step_with_ready_grads skips Adam when grad_norm > threshold (M4065).
2.  Gap #2: step_with_ready_grads calls get_grad_stats_parallel_group() not data_parallel_group.
3.  Gap #3a: _copy_model_grads_to_main_grads override populates _fp32_grad_shards from param.main_grad.
4.  Gap #3b: is_stub_optimizer property returns False on DistributedOptimizer.
5.  Gap #3c: reload_model_params copies BF16 model param data into _fp32_shards.
6.  Regression: step_with_ready_grads returns True when grad_norm <= threshold.
7.  Regression: step_with_ready_grads returns True when clip_grad disabled.

All tests run entirely in-process with mocked dist primitives — no GPU / NCCL required.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bootstrap the minimal deepspeed stub so distrib_optimizer imports cleanly
# ---------------------------------------------------------------------------

def _install_stubs():
    if "deepspeed" not in sys.modules:
        ds_stub = types.ModuleType("deepspeed")
        ds_stub.__path__ = []
        sys.modules["deepspeed"] = ds_stub

    core_stub = types.ModuleType("deepspeed.core")
    core_stub.__path__ = []
    sys.modules.setdefault("deepspeed.core", core_stub)

    ps_stub = types.ModuleType("deepspeed.core.parallel_state")
    ps_stub.is_initialized = lambda: False
    ps_stub.get_data_parallel_group = MagicMock(return_value=None)
    ps_stub.get_data_parallel_group_gloo = MagicMock(return_value=None)
    sys.modules["deepspeed.core.parallel_state"] = ps_stub
    core_stub.parallel_state = ps_stub

    mp_cfg_stub = types.ModuleType("deepspeed.core.model_parallel_config")
    class ModelParallelConfig:
        desloc = None
    mp_cfg_stub.ModelParallelConfig = ModelParallelConfig
    sys.modules["deepspeed.core.model_parallel_config"] = mp_cfg_stub

    dc_stub = types.ModuleType("deepspeed.core.desloc_config")
    class DesLocConfig:
        enabled = False
    class TierType:
        DATACENTER = "datacenter"
        PROFESSIONAL = "professional"
        BLACKWELL = "blackwell"
        CONSUMER = "consumer"
    dc_stub.DesLocConfig = DesLocConfig
    dc_stub.TierType = TierType
    sys.modules["deepspeed.core.desloc_config"] = dc_stub

    dist_stub = types.ModuleType("deepspeed.core.distributed")
    class _FakeParamAndGradBuffer:
        def __init__(self, params, buf_size=16):
            self.param_index_map = {}
            offset = 0
            for p in params:
                n = p.numel()
                self.param_index_map[p] = (offset, offset + n, 0)
                offset += n
        @property
        def dtype(self):
            return torch.bfloat16
    dist_stub.ParamAndGradBuffer = _FakeParamAndGradBuffer
    sys.modules["deepspeed.core.distributed"] = dist_stub

    oc_stub = types.ModuleType("deepspeed.core.optimizer.optimizer_config")
    from dataclasses import dataclass
    @dataclass
    class OptimizerConfig:
        lr: float = 1e-3
        adam_beta1: float = 0.9
        adam_beta2: float = 0.999
        adam_eps: float = 1e-8
        weight_decay: float = 0.01
        fp16: bool = False
        bf16: bool = True
        clip_grad: float = 1.0
        use_distributed_optimizer: bool = True
        decoupled_weight_decay: bool = False
        desloc_enabled: bool = False
        heterogeneous_shard_sizing: bool = False
        grad_norm_skip_threshold: float = float('inf')
        def is_ku_step(self, step): return False
        def is_kv_step(self, step): return False
    oc_stub.OptimizerConfig = OptimizerConfig
    sys.modules["deepspeed.core.optimizer.optimizer_config"] = oc_stub

    # Also register the optimizer subpackage stubs so the module can be registered
    opt_pkg_stub = types.ModuleType("deepspeed.core.optimizer")
    opt_pkg_stub.__path__ = []
    sys.modules.setdefault("deepspeed.core.optimizer", opt_pkg_stub)
    core_stub.optimizer = opt_pkg_stub


_install_stubs()

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "deepspeed.core.optimizer.distrib_optimizer",
    str(REPO_ROOT / "deepspeed/core/optimizer/distrib_optimizer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["deepspeed.core.optimizer.distrib_optimizer"] = _mod
try:
    _spec.loader.exec_module(_mod)
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = e


# ---------------------------------------------------------------------------
# Helper: patch clip_grad_norm in the loaded module directly
# ---------------------------------------------------------------------------

from contextlib import contextmanager

@contextmanager
def _patch_clip_grad_norm(return_value=None, side_effect=None):
    """Patch clip_grad_norm in the loaded distrib_optimizer module object."""
    original = getattr(_mod, "clip_grad_norm", None)
    if side_effect is not None:
        _mod.clip_grad_norm = side_effect
    else:
        def _fake(*args, **kwargs):
            return torch.tensor(float(return_value))
        _mod.clip_grad_norm = _fake
    try:
        yield _mod.clip_grad_norm
    finally:
        if original is not None:
            _mod.clip_grad_norm = original


# ---------------------------------------------------------------------------
# Helper to build a minimal DistributedOptimizer for testing
# ---------------------------------------------------------------------------

def _make_optimizer(clip_grad=1.0, skip_threshold=float('inf'), world_size=2, rank=0):
    if not _IMPORT_OK:
        return None

    DistributedOptimizer = _mod.DistributedOptimizer
    from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
    from deepspeed.core.model_parallel_config import ModelParallelConfig
    from deepspeed.core.distributed import ParamAndGradBuffer

    p1 = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    p2 = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    params = [p1, p2]

    config = OptimizerConfig(
        clip_grad=clip_grad,
        grad_norm_skip_threshold=skip_threshold,
        bf16=True,
        use_distributed_optimizer=True,
    )
    mp_config = ModelParallelConfig()
    buf = ParamAndGradBuffer(params)

    opt = object.__new__(DistributedOptimizer)

    opt.config = config
    _dummy_p = nn.Parameter(torch.zeros(6, dtype=torch.float32))
    _inner_adam = torch.optim.Adam(
        [_dummy_p],
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    opt.optimizer = _inner_adam
    opt.param_and_grad_buffers = [buf]
    opt.data_parallel_group = None
    opt.data_parallel_group_gloo = None
    opt.tier_assignments = None
    opt.data_parallel_world_size = world_size
    opt.data_parallel_rank = rank
    opt._step_count = 0
    opt._desloc = None
    opt._defer_param_sync = False
    opt._scale_one = torch.tensor(1.0)
    opt.grad_scaler = None

    buf_size = 12
    shard_size = buf_size // world_size
    shard_start = rank * shard_size
    shard_end = shard_start + shard_size

    fp32_shard = torch.zeros(shard_size, dtype=torch.float32)
    fp32_grad_shard = torch.zeros(shard_size, dtype=torch.float32)
    opt._fp32_shards = [fp32_shard]
    opt._fp32_grad_shards = [fp32_grad_shard]
    opt._buf_boundaries = [
        [(i * shard_size, (i + 1) * shard_size) for i in range(world_size)]
    ]

    shard_param = nn.Parameter(fp32_shard.clone(), requires_grad=True)
    opt._shard_params = [shard_param]

    # Update optimizer to track shard_param, keeping all Adam defaults
    _inner_adam.param_groups[0]["params"] = [shard_param]

    opt.grad_stats_parallel_group = None
    opt._apply_decoupled_weight_decay = MagicMock()
    opt.start_param_sync_for_bucket_group_subset = MagicMock()
    opt.sync_moments = MagicMock()

    return opt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestGap1GradNormSkipThreshold(unittest.TestCase):

    def test_skip_when_norm_exceeds_threshold(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=5.0)
        step_before = opt._step_count

        with _patch_clip_grad_norm(return_value=100.0):
            result = opt.step_with_ready_grads()

        self.assertFalse(result, "Expected False (step skipped) but got True")
        self.assertEqual(opt._step_count, step_before, "step_count should not increment on skip")
        opt._apply_decoupled_weight_decay.assert_not_called()
        opt.start_param_sync_for_bucket_group_subset.assert_not_called()

    def test_no_skip_when_norm_at_threshold(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=5.0)

        with _patch_clip_grad_norm(return_value=5.0):
            result = opt.step_with_ready_grads()

        self.assertTrue(result, "Expected True (step taken) at exact threshold")

    def test_no_skip_when_norm_below_threshold(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=50.0)

        with _patch_clip_grad_norm(return_value=1.5):
            result = opt.step_with_ready_grads()

        self.assertTrue(result)
        opt._apply_decoupled_weight_decay.assert_called_once()
        opt.start_param_sync_for_bucket_group_subset.assert_called_once()

    def test_default_threshold_never_skips(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=float('inf'))

        with _patch_clip_grad_norm(return_value=1e9):
            result = opt.step_with_ready_grads()

        self.assertTrue(result, "Default threshold=inf should never skip")

    def test_step_count_increments_on_successful_step(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=float('inf'))
        initial = opt._step_count

        with _patch_clip_grad_norm(return_value=0.5):
            opt.step_with_ready_grads()

        self.assertEqual(opt._step_count, initial + 1)

    def test_grads_zeroed_on_skip(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=2.0)
        opt._shard_params[0].grad = torch.full((6,), 99.0)

        with _patch_clip_grad_norm(return_value=100.0):
            opt.step_with_ready_grads()

        self.assertTrue(
            opt._shard_params[0].grad.eq(0.0).all(),
            "Grad tensor should be zeroed after skip",
        )


@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestGap2GradStatsParallelGroup(unittest.TestCase):

    def test_clip_uses_grad_stats_group_not_dp_group(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=float('inf'))

        fake_grad_stats_group = MagicMock(name="grad_stats_group")
        fake_dp_group = MagicMock(name="dp_group")
        opt.grad_stats_parallel_group = fake_grad_stats_group
        opt.data_parallel_group = fake_dp_group

        calls = []
        def recording_clip(parameters, max_norm, norm_type, model_parallel_group):
            calls.append(model_parallel_group)
            return torch.tensor(0.1)

        with _patch_clip_grad_norm(side_effect=recording_clip):
            opt.step_with_ready_grads()

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], fake_grad_stats_group,
            "clip_grad_norm must receive grad_stats_parallel_group, not data_parallel_group")
        self.assertIsNot(calls[0], fake_dp_group)

    def test_get_grad_stats_parallel_group_fallback(self):
        opt = _make_optimizer()
        opt.data_parallel_group = MagicMock(name="dp_fallback")
        if hasattr(opt, "grad_stats_parallel_group"):
            del opt.grad_stats_parallel_group

        group = opt.get_grad_stats_parallel_group()
        self.assertIs(group, opt.data_parallel_group)


@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestGap3CopyModelGrads(unittest.TestCase):

    def test_main_grad_copied_to_fp32_shard(self):
        opt = _make_optimizer(world_size=1, rank=0)
        buf = opt.param_and_grad_buffers[0]

        for i, (p, (ps, pe, _)) in enumerate(buf.param_index_map.items()):
            p.main_grad = torch.full((pe - ps,), float(i + 1), dtype=torch.bfloat16)

        opt._copy_model_grads_to_main_grads()

        fp32_grad = opt._fp32_grad_shards[0]
        self.assertFalse(fp32_grad.eq(0.0).all())

        params = list(buf.param_index_map.keys())
        p1, p2 = params[0], params[1]
        ps1, pe1, _ = buf.param_index_map[p1]
        ps2, pe2, _ = buf.param_index_map[p2]
        torch.testing.assert_close(fp32_grad[ps1:pe1], torch.ones(pe1 - ps1))
        torch.testing.assert_close(fp32_grad[ps2:pe2], torch.full((pe2 - ps2,), 2.0))

    def test_main_grad_cleared_after_copy(self):
        opt = _make_optimizer(world_size=1, rank=0)
        buf = opt.param_and_grad_buffers[0]
        for p, (ps, pe, _) in buf.param_index_map.items():
            p.main_grad = torch.ones(pe - ps, dtype=torch.bfloat16)

        opt._copy_model_grads_to_main_grads()

        for p in buf.param_index_map:
            self.assertIsNone(getattr(p, "main_grad", None))

    def test_shard_param_grad_attached(self):
        opt = _make_optimizer(world_size=1, rank=0)
        buf = opt.param_and_grad_buffers[0]
        for p, (ps, pe, _) in buf.param_index_map.items():
            p.main_grad = torch.zeros(pe - ps, dtype=torch.bfloat16)

        opt._copy_model_grads_to_main_grads()

        for sp, gs in zip(opt._shard_params, opt._fp32_grad_shards):
            self.assertIs(sp.grad, gs)

    def test_params_with_no_grad_contribute_zero(self):
        opt = _make_optimizer(world_size=1, rank=0)
        buf = opt.param_and_grad_buffers[0]
        for p in buf.param_index_map:
            p.main_grad = None
            p.grad = None

        try:
            opt._copy_model_grads_to_main_grads()
        except Exception as e:
            self.fail(f"raised: {e}")

        self.assertTrue(opt._fp32_grad_shards[0].eq(0.0).all())


@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestGap3IsStubOptimizer(unittest.TestCase):

    def test_is_stub_optimizer_false(self):
        opt = _make_optimizer()
        self.assertFalse(opt.is_stub_optimizer)

    def test_is_stub_optimizer_is_accessible(self):
        opt = _make_optimizer()
        val = opt.is_stub_optimizer
        self.assertIsInstance(val, bool)


@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestGap3ReloadModelParams(unittest.TestCase):

    def test_fp32_shards_populated_from_bf16_params(self):
        opt = _make_optimizer(world_size=1, rank=0)
        buf = opt.param_and_grad_buffers[0]

        for i, (p, (ps, pe, _)) in enumerate(buf.param_index_map.items()):
            p.data = torch.full((pe - ps,), float(i + 2), dtype=torch.bfloat16)

        for s in opt._fp32_shards:
            s.zero_()

        opt.reload_model_params()

        fp32_shard = opt._fp32_shards[0]
        params = list(buf.param_index_map.keys())
        p1, p2 = params[0], params[1]
        ps1, pe1, _ = buf.param_index_map[p1]
        ps2, pe2, _ = buf.param_index_map[p2]
        torch.testing.assert_close(fp32_shard[ps1:pe1], torch.full((pe1 - ps1,), 2.0))
        torch.testing.assert_close(fp32_shard[ps2:pe2], torch.full((pe2 - ps2,), 3.0))

    def test_reload_does_not_crash_with_empty_buffers(self):
        opt = _make_optimizer(world_size=1, rank=0)
        opt.param_and_grad_buffers = []
        opt._fp32_shards = []
        opt._buf_boundaries = []

        try:
            opt.reload_model_params()
        except Exception as e:
            self.fail(f"raised: {e}")


@unittest.skipUnless(_IMPORT_OK, f"distrib_optimizer import failed: {_IMPORT_ERR}")
class TestRegressionStepFlow(unittest.TestCase):

    def test_full_step_sequence(self):
        opt = _make_optimizer(clip_grad=1.0, skip_threshold=float('inf'))
        initial_step = opt._step_count

        with _patch_clip_grad_norm(return_value=0.5):
            result = opt.step_with_ready_grads()

        self.assertTrue(result)
        self.assertEqual(opt._step_count, initial_step + 1)
        opt._apply_decoupled_weight_decay.assert_called_once()
        opt.start_param_sync_for_bucket_group_subset.assert_called_once()

    def test_no_clip_grad_skips_norm_computation(self):
        opt = _make_optimizer(clip_grad=0.0, skip_threshold=float('inf'))

        original_clip = _mod.clip_grad_norm
        call_count = [0]
        def counting_clip(*a, **kw):
            call_count[0] += 1
            return original_clip(*a, **kw)
        _mod.clip_grad_norm = counting_clip
        try:
            result = opt.step_with_ready_grads()
        finally:
            _mod.clip_grad_norm = original_clip

        self.assertEqual(call_count[0], 0, "clip_grad_norm must not be called when clip_grad=0")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
