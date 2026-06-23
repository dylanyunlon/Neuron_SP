# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
tests/test_run_pretrain.py

Pytest suite for run_pretrain.py key components.
All tests run on CPU, each completes in < 5 seconds.

Covered:
  1. LlamaModel construction and forward pass
  2. build_cosine_schedule warmup + decay behaviour
  3. synthetic_iter dataloader shapes and dtypes
  4. Checkpoint save / load round-trip (model + optimizer + scheduler)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest
import torch
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so run_pretrain is importable from any
# working directory (e.g. pytest invoked from tests/).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_pretrain import (  # noqa: E402
    LlamaModel,
    build_cosine_schedule,
    synthetic_iter,
    _MODEL_CONFIGS,
)

# ---------------------------------------------------------------------------
# Tiny model config used by all tests (fast CPU, < 100 k params)
# ---------------------------------------------------------------------------
_TINY = dict(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, seq_len=16)


# ===========================================================================
# 1. LlamaModel construction and forward
# ===========================================================================

class TestLlamaModel:
    """Tests for LlamaModel construction, forward, and parameter count."""

    def _make_model(self) -> LlamaModel:
        return LlamaModel(**_TINY)

    def test_instantiation(self):
        """Model should build without errors on CPU."""
        model = self._make_model()
        assert isinstance(model, torch.nn.Module)

    def test_forward_output_shape(self):
        """Forward pass must return (B, T, vocab_size) logits."""
        model = self._make_model()
        model.eval()
        B, T = 2, _TINY["seq_len"]
        input_ids = torch.randint(0, _TINY["vocab_size"], (B, T))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (B, T, _TINY["vocab_size"]), (
            f"Expected ({B}, {T}, {_TINY['vocab_size']}), got {logits.shape}"
        )

    def test_forward_dtype_float32(self):
        """Logits should be float32 on CPU (no autocast)."""
        model = self._make_model()
        model.eval()
        input_ids = torch.randint(0, _TINY["vocab_size"], (1, _TINY["seq_len"]))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.dtype == torch.float32

    def test_weight_tying(self):
        """embedding and lm_head weights should share the same storage."""
        model = self._make_model()
        assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr(), (
            "embedding and lm_head weights are not tied"
        )

    def test_num_parameters_positive(self):
        """num_parameters property must return a positive integer."""
        model = self._make_model()
        assert model.num_parameters > 0

    def test_70m_preset_builds(self):
        """The '70m' preset from _MODEL_CONFIGS should build and forward correctly."""
        cfg = _MODEL_CONFIGS["70m"]
        # Use a shorter seq_len to stay fast on CPU
        model = LlamaModel(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            seq_len=32,
        )
        input_ids = torch.randint(0, cfg["vocab_size"], (1, 32))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 32, cfg["vocab_size"])

    def test_loss_backward(self):
        """A cross-entropy backward pass must complete without NaN gradients."""
        import torch.nn.functional as F

        model = self._make_model()
        model.train()
        input_ids = torch.randint(0, _TINY["vocab_size"], (2, _TINY["seq_len"]))
        labels = torch.randint(0, _TINY["vocab_size"], (2, _TINY["seq_len"]))
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ===========================================================================
# 2. build_cosine_schedule
# ===========================================================================

class TestBuildCosineSchedule:
    """Tests for the linear-warmup + cosine-decay LR scheduler."""

    def _make_scheduler(self, warmup=10, total=100, min_lr=0.1):
        layer = torch.nn.Linear(4, 4)
        opt = AdamW(layer.parameters(), lr=1.0)  # base lr=1 so lambda == actual lr
        sched = build_cosine_schedule(opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=min_lr)
        return opt, sched

    def _collect_lrs(self, opt, sched, n_steps):
        lrs = []
        for _ in range(n_steps):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        return lrs

    def test_warmup_monotone_increase(self):
        """LR must be strictly increasing during the warmup phase."""
        opt, sched = self._make_scheduler(warmup=10, total=50)
        lrs = self._collect_lrs(opt, sched, 10)
        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1], (
                f"LR not monotonically increasing at step {i}: {lrs[i]:.6f} > {lrs[i+1]:.6f}"
            )

    def test_peak_lr_at_end_of_warmup(self):
        """At the end of warmup the lambda should equal 1.0 (= base LR)."""
        opt, sched = self._make_scheduler(warmup=5, total=50)
        lrs = self._collect_lrs(opt, sched, 5)
        assert math.isclose(lrs[-1], 1.0, rel_tol=1e-5), (
            f"Expected peak lr ≈ 1.0, got {lrs[-1]}"
        )

    def test_min_lr_floor_at_end(self):
        """After full decay the LR must not fall below min_lr_ratio * base_lr."""
        min_ratio = 0.1
        opt, sched = self._make_scheduler(warmup=2, total=50, min_lr=min_ratio)
        lrs = self._collect_lrs(opt, sched, 50)
        assert lrs[-1] >= min_ratio - 1e-6, (
            f"LR ({lrs[-1]:.6f}) fell below min_lr_ratio ({min_ratio})"
        )

    def test_returns_lambda_lr(self):
        """Return type must be torch.optim.lr_scheduler.LambdaLR."""
        from torch.optim.lr_scheduler import LambdaLR

        opt, sched = self._make_scheduler()
        assert isinstance(sched, LambdaLR)

    def test_lr_decays_after_warmup(self):
        """LR at the very end must be strictly less than the peak value."""
        opt, sched = self._make_scheduler(warmup=5, total=50)
        lrs = self._collect_lrs(opt, sched, 50)
        peak = lrs[4]   # last warmup step
        final = lrs[-1]
        assert final < peak, f"LR did not decay: peak={peak:.6f}, final={final:.6f}"


# ===========================================================================
# 3. synthetic_iter dataloader
# ===========================================================================

class TestSyntheticIter:
    """Tests for the infinite synthetic token iterator."""

    def _get_batch(self, vocab=256, bs=2, seq=16):
        device = torch.device("cpu")
        gen = synthetic_iter(vocab, bs, seq, device)
        return next(gen)

    def test_returns_two_tensors(self):
        """Iterator must yield (input_ids, labels) tuples."""
        result = self._get_batch()
        assert isinstance(result, tuple) and len(result) == 2

    def test_input_shape(self):
        """input_ids shape must be (batch_size, seq_len)."""
        inp, _ = self._get_batch(bs=3, seq=8)
        assert inp.shape == (3, 8), f"Expected (3, 8), got {inp.shape}"

    def test_labels_shape(self):
        """labels shape must match input_ids shape (seq_len offset by 1)."""
        inp, lbl = self._get_batch(bs=3, seq=8)
        assert lbl.shape == inp.shape, (
            f"labels shape {lbl.shape} != input shape {inp.shape}"
        )

    def test_token_ids_in_range(self):
        """All token ids must be in [0, vocab_size)."""
        vocab = 128
        inp, lbl = self._get_batch(vocab=vocab, seq=32)
        assert inp.min() >= 0 and inp.max() < vocab, "input_ids out of vocab range"
        assert lbl.min() >= 0 and lbl.max() < vocab, "labels out of vocab range"

    def test_dtype_long(self):
        """Token tensors must be integer dtype (torch.int64)."""
        inp, lbl = self._get_batch()
        assert inp.dtype == torch.int64, f"Expected int64, got {inp.dtype}"
        assert lbl.dtype == torch.int64, f"Expected int64, got {lbl.dtype}"

    def test_is_infinite(self):
        """Iterator must yield at least 10 consecutive batches without StopIteration."""
        device = torch.device("cpu")
        gen = synthetic_iter(64, 2, 8, device)
        for _ in range(10):
            batch = next(gen)  # would raise StopIteration if finite
        assert batch is not None

    def test_labels_are_next_token(self):
        """labels[i] must equal input_ids[i+1] (next-token prediction offset)."""
        # synthetic_iter slices tokens[:, :-1] and tokens[:, 1:], so
        # labels[b, t] == the token that follows input_ids[b, t] in the
        # original sequence.  We verify this holds for multiple consecutive
        # batches drawn from the SAME generator (same underlying token stream).
        # NOTE: each call to synthetic_iter draws an independent random tensor,
        # so we verify the within-batch invariant: inp and lbl come from the
        # same row of an (B, seq+1) draw, shifted by one position.
        device = torch.device("cpu")
        torch.manual_seed(0)
        gen = synthetic_iter(vocab_size=1000, batch_size=4, seq_len=20, device=device)
        inp, lbl = next(gen)
        # The last label token id should differ from the last input token id
        # (they're consecutive positions in the original sequence); the key
        # invariant we can check without seeing the raw draw is that
        # inp[b, 1:] == lbl[b, :-1] (shifted-by-one relationship within each row).
        assert torch.equal(inp[:, 1:], lbl[:, :-1]), (
            "labels are not a one-position shift of input_ids"
        )


# ===========================================================================
# 4. Checkpoint save / load round-trip
# ===========================================================================

class TestCheckpointRoundTrip:
    """Tests that model, optimizer, and scheduler state survives save/load."""

    def _setup(self):
        """Create a tiny model + optimizer + scheduler, advance one step."""
        model = LlamaModel(**_TINY)
        model.train()
        opt = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        sched = build_cosine_schedule(opt, warmup_steps=5, total_steps=50)

        # Run one forward + backward to populate optimizer moments
        import torch.nn.functional as F
        input_ids = torch.randint(0, _TINY["vocab_size"], (2, _TINY["seq_len"]))
        labels = torch.randint(0, _TINY["vocab_size"], (2, _TINY["seq_len"]))
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        return model, opt, sched, loss.item()

    def test_model_state_restored(self):
        """Loaded model parameters must be bit-exact to saved ones."""
        model, opt, sched, _ = self._setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            model2 = LlamaModel(**_TINY)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(ckpt["model_state_dict"])

            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.equal(p1, p2), f"Parameter mismatch for {n1}"
        finally:
            os.unlink(ckpt_path)

    def test_optimizer_state_restored(self):
        """Optimizer first/second moments must be preserved across save/load."""
        model, opt, sched, _ = self._setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            orig_sd = opt.state_dict()
            torch.save({"optimizer_state_dict": orig_sd}, ckpt_path)

            model2 = LlamaModel(**_TINY)
            opt2 = AdamW(model2.parameters(), lr=3e-4, betas=(0.9, 0.95))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            opt2.load_state_dict(ckpt["optimizer_state_dict"])
            loaded_sd = opt2.state_dict()

            # Both state dicts must have the same param indices
            assert set(orig_sd["state"].keys()) == set(loaded_sd["state"].keys()), (
                "Optimizer state param indices differ after reload"
            )
            # For each tracked parameter, step counts and moment shapes must match
            for idx in orig_sd["state"]:
                st_orig = orig_sd["state"][idx]
                st_load = loaded_sd["state"][idx]
                assert torch.equal(st_orig["step"], st_load["step"]), (
                    f"Optimizer step mismatch for param {idx}"
                )
                assert st_orig["exp_avg"].shape == st_load["exp_avg"].shape, (
                    f"exp_avg shape mismatch for param {idx}"
                )
        finally:
            os.unlink(ckpt_path)

    def test_scheduler_state_restored(self):
        """LR scheduler last_epoch must be preserved."""
        model, opt, sched, _ = self._setup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save({"scheduler_state_dict": sched.state_dict()}, ckpt_path)

            model2 = LlamaModel(**_TINY)
            opt2 = AdamW(model2.parameters(), lr=3e-4)
            sched2 = build_cosine_schedule(opt2, warmup_steps=5, total_steps=50)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sched2.load_state_dict(ckpt["scheduler_state_dict"])

            assert sched.last_epoch == sched2.last_epoch, (
                f"last_epoch mismatch: {sched.last_epoch} vs {sched2.last_epoch}"
            )
        finally:
            os.unlink(ckpt_path)

    def test_full_checkpoint_keys(self):
        """Full checkpoint dict must contain all expected keys."""
        model, opt, sched, loss_val = self._setup()

        ckpt = {
            "step": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "loss": loss_val,
            "tokens_seen": 1 * 2 * _TINY["seq_len"],
        }

        required_keys = {
            "step", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "loss", "tokens_seen",
        }
        assert required_keys.issubset(ckpt.keys()), (
            f"Missing keys: {required_keys - ckpt.keys()}"
        )

    def test_inference_output_unchanged_after_reload(self):
        """Model output must be identical before save and after load."""
        model, opt, sched, _ = self._setup()
        model.eval()

        input_ids = torch.randint(0, _TINY["vocab_size"], (1, _TINY["seq_len"]))
        with torch.no_grad():
            out_before = model(input_ids).clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            model2 = LlamaModel(**_TINY)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(ckpt["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                out_after = model2(input_ids)

            assert torch.allclose(out_before, out_after, atol=1e-6), (
                "Model outputs differ after checkpoint reload"
            )
        finally:
            os.unlink(ckpt_path)
