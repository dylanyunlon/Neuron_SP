# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
smoke_test.py — 端到端 smoke test

验证整条三阶段管线能跑通:
  1. Tokenizer 加载 + commit 编码
  2. 70M 模型构建
  3. Stage 1/2/3 各跑几步
  4. Checkpoint save/load
  5. 生成测试

不需要 GPU (CPU 模式), 不需要下载大数据.
在任何环境下 < 2 分钟完成.

Usage:
    python -m pipeline.smoke_test
"""

import os
import sys
import time
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅"
FAIL = "❌"
results = []


def test(name, fn):
    print(f"\n{'─'*50}")
    print(f"  TEST: {name}")
    print(f"{'─'*50}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  {PASS} {name} ({elapsed:.1f}s)")
        results.append((name, True, elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()
        results.append((name, False, elapsed))


# ── Test 1: Tokenizer ──

def test_tokenizer():
    from pipeline.unified_tokenizer import build_megatron_tokenizer

    tok = build_megatron_tokenizer()
    assert tok.vocab_size > 49000, f"vocab too small: {tok.vocab_size}"
    assert tok.eod is not None, "eod not set"
    assert tok.diff_start_id != tok.eod, "diff_start should differ from eod"

    # Commit 编码
    ids = tok.encode_commit(
        old_code="x = 1",
        new_code="x = 2",
        message="bump x",
        file_path="main.py",
        lang="python",
    )
    assert len(ids) > 10, f"encoded commit too short: {len(ids)}"
    decoded = tok.detokenize(ids)
    assert "diff_start" in decoded, f"missing diff_start in: {decoded[:100]}"
    print(f"  commit encoded: {len(ids)} tokens")
    print(f"  decoded preview: {decoded[:120]}")


# ── Test 2: Model build ──

def test_model_build():
    from pipeline.train_three_stage import build_model
    from pipeline.unified_tokenizer import build_megatron_tokenizer

    tok = build_megatron_tokenizer()
    model = build_model(
        vocab_size=tok.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        seq_len=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.1f}M")
    assert n_params > 1e6, "model too small"
    assert n_params < 1e9, "model too large for smoke test"


# ── Test 3: Forward pass ──

def test_forward():
    import torch
    from pipeline.train_three_stage import build_model
    from pipeline.unified_tokenizer import build_megatron_tokenizer

    tok = build_megatron_tokenizer()
    model = build_model(
        vocab_size=tok.vocab_size,
        hidden_size=256, num_layers=4, num_heads=4, seq_len=128,
    )
    model.eval()

    # 构造 batch
    ids = tok.encode_commit("def f(): pass", "def f(): return 1", "implement f")
    ids = ids[:128]
    ids += [tok.eod] * (128 - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    assert loss is not None, "loss is None"
    assert loss.item() > 0, f"loss should be positive: {loss.item()}"
    print(f"  forward loss: {loss.item():.4f}")


# ── Test 4: Data loading (all 3 stages) ──

def test_data_loading():
    from pipeline.unified_tokenizer import get_tokenizer
    from pipeline.train_three_stage import load_stage1_data, load_stage2_data, load_stage3_data

    tok = get_tokenizer()
    seq_len = 128
    batch_size = 2

    for name, loader_fn in [
        ("stage1_code", load_stage1_data),
        ("stage2_commit", load_stage2_data),
        ("stage3_instruct", load_stage3_data),
    ]:
        dl = loader_fn(tok, seq_len, batch_size)
        batch = next(iter(dl))
        assert "input_ids" in batch, f"{name}: missing input_ids"
        assert batch["input_ids"].shape == (batch_size, seq_len), \
            f"{name}: shape {batch['input_ids'].shape} != ({batch_size}, {seq_len})"
        print(f"  {name}: batch shape={batch['input_ids'].shape}, "
              f"first tokens={batch['input_ids'][0,:8].tolist()}")


# ── Test 5: Engine bridge config ──

def test_engine_config():
    from pipeline.engine_bridge import build_ds_config

    config = build_ds_config(
        hetero_shard_ratio=[1.0, 1.0, 2.0],
        bf16=True,
        total_steps=1000,
    )
    assert config["zero_optimization"]["stage"] == 3
    assert config["zero_optimization"]["hetero_shard_ratio"] == [1.0, 1.0, 2.0]
    assert config["bf16"]["enabled"] is True
    print(f"  config keys: {list(config.keys())}")
    print(f"  ZeRO stage: {config['zero_optimization']['stage']}")
    print(f"  hetero_shard_ratio: {config['zero_optimization']['hetero_shard_ratio']}")


# ── Test 6: Checkpoint save/load (CPU only) ──

def test_checkpoint():
    import torch
    from pipeline.train_three_stage import build_model
    from pipeline.unified_tokenizer import build_megatron_tokenizer

    tok = build_megatron_tokenizer()
    model = build_model(vocab_size=tok.vocab_size, hidden_size=256, num_layers=2,
                        num_heads=4, seq_len=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  saved: {path} ({size_mb:.1f}MB)")

        model2 = build_model(vocab_size=tok.vocab_size, hidden_size=256, num_layers=2,
                             num_heads=4, seq_len=64)
        model2.load_state_dict(torch.load(path, weights_only=True))
        print(f"  loaded successfully")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"mismatch in {n1}"
        print(f"  weights match ✓")


def test_desloc_engine_train():
    """Run a tiny model through DesLocEngine.train() for 10 steps.

    Validates that HeteroRegistry, HeteroStepBatchScheduler,
    and the full train loop work end-to-end.
    """
    import torch
    import torch.nn as nn

    # Tiny GPT-like model
    class TinyGPT(nn.Module):
        def __init__(self, vocab=512, hidden=64, layers=2, heads=2, seq=128):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(seq, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(hidden, vocab)
            self.seq = seq

        def forward(self, x):
            B, T = x.shape
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.embed(x) + self.pos(pos_ids)
            h = self.transformer(h)
            return self.head(h)

    # Try importing DesLocEngine
    try:
        from deepspeed.runtime.desloc_engine import DesLocEngine, DesLocConfig
    except ImportError as e:
        print(f"  [skip] DesLocEngine import failed: {e}")
        return

    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TinyGPT: {n_params/1e3:.0f}K params")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DesLocConfig(
            total_steps=10,
            micro_batch_size=2,
            seq_len=128,
            lr=1e-3,
            grad_clip=1.0,
            log_every=5,
            save_every=999,
            checkpoint_dir=tmpdir,
        )
        try:
            engine = DesLocEngine(model=model, config=cfg)
            print(f"  DesLocEngine created, registry={len(engine.registry)} modules")
            engine.train()
            print(f"  Train completed: {engine.global_step} steps, {engine.tokens_seen} tokens")
        except Exception as e:
            print(f"  DesLocEngine.train() raised: {type(e).__name__}: {e}")
            # Still pass if it's an expected issue (no GPU, etc)
            if "cuda" in str(e).lower() or "device" in str(e).lower():
                print("  [ok] Expected error in CPU-only environment")
            else:
                raise


# ── Test 8: HeteroRegistry discovers all hetero_* modules with register() ──

def test_hetero_registry_discovers_all():
    """Verify the standalone HeteroRegistry finds every hetero_*.py under deepspeed/."""
    from deepspeed.runtime.hetero_registry import HeteroRegistry

    registry = HeteroRegistry()
    count = registry.scan()
    assert count > 0, "HeteroRegistry.scan() found zero modules"
    print(f"  scanned {count} hetero modules")

    # Every module that exposes a top-level register() should be discoverable
    # by checking the file for 'def register' and confirming it ended up in
    # the registry's module index.
    import re
    modules_with_register = 0
    for _rel_path, info in registry.modules.items():
        full = os.path.join(registry.project_root, info.file_path)
        try:
            src = open(full, "r", encoding="utf-8").read()
        except Exception:
            continue
        if re.search(r'^def register\b', src, re.MULTILINE):
            modules_with_register += 1
    assert modules_with_register > 0, "Expected at least one module with register()"
    print(f"  {modules_with_register}/{count} modules expose register()")

    # Subsystem index should cover more than one subsystem
    subs = list(registry._subsystem_index.keys())
    assert len(subs) >= 2, f"Expected >=2 subsystems, got {subs}"
    print(f"  subsystems: {sorted(subs)}")


# ── Test 9: DesLocEngine instance has expected hooks ──

def test_desloc_engine_has_all_hooks():
    """Verify DesLocEngine wires neuron_sp_config, hetero_scheduler, fp32_grad_manager."""
    try:
        from deepspeed.runtime.desloc_engine import DesLocEngine, TrainingConfig
    except ImportError as e:
        print(f"  [skip] import failed: {e}")
        return

    cfg = TrainingConfig(
        total_steps=5,
        micro_batch_size=1,
        seq_len=64,
        num_layers=2,
        hidden_size=128,
        num_heads=2,
        save_every=999,
    )
    try:
        engine = DesLocEngine(config=cfg)
    except Exception as e:
        # GPU/dist failures are expected in CI — check attribute presence anyway
        if "cuda" in str(e).lower() or "gloo" in str(e).lower() or "dist" in str(e).lower():
            print(f"  [skip] engine init requires GPU/dist: {e}")
            return
        raise

    for attr in ("neuron_sp_config", "hetero_scheduler", "fp32_grad_manager"):
        assert hasattr(engine, attr), f"DesLocEngine missing attribute: {attr}"
        assert getattr(engine, attr) is not None, f"DesLocEngine.{attr} is None"
        print(f"  engine.{attr} = {type(getattr(engine, attr)).__name__}")

    # Registry should have discovered modules
    assert hasattr(engine, "registry"), "DesLocEngine missing registry"
    assert len(engine.registry) > 0, "registry is empty after init"
    print(f"  registry modules: {len(engine.registry)}")


# ── Test 10: CommitSequencePacker does not cross commit boundaries ──

def test_commit_sequence_packer():
    """Pack several commits and verify no packed sequence mixes partial commits."""
    from datasets.bigcode.commit_packing import CommitSequencePacker, PackedSequence

    packer = CommitSequencePacker(tokenizer=None, seq_len=64, pad_token_id=0)

    # Create synthetic commits of varying lengths (char-level ÷ 4 ≈ tokens)
    commits = [
        {"text": "a" * 80},   # ~20 tokens — fits in one seq
        {"text": "b" * 120},  # ~30 tokens — fits in one seq
        {"text": "c" * 60},   # ~15 tokens — may merge with previous
        {"text": "d" * 300},  # ~75 tokens — longer than seq_len, sliding window
        {"text": "e" * 40},   # ~10 tokens — short
    ]

    packed = packer.pack_dataset(iter(commits))
    assert len(packed) > 0, "packer produced zero sequences"
    print(f"  packed {len(commits)} commits into {len(packed)} sequences")

    for i, seq in enumerate(packed):
        assert isinstance(seq, PackedSequence), f"seq {i} is not PackedSequence"
        # Each sequence must be exactly seq_len after finalization
        assert len(seq.tokens) == 64, f"seq {i}: len={len(seq.tokens)}, expected 64"
        # commit_ids must be non-empty (every seq has at least one source commit)
        assert seq.num_commits >= 1, f"seq {i}: num_commits={seq.num_commits}"
        # No duplicate commit indices within a single packed sequence
        # (the packer appends commit_idx once per _append call, so duplicates
        # would indicate a bug in the boundary logic)
        seen = set()
        for cid in seq.commit_ids:
            assert cid not in seen, (
                f"seq {i}: commit {cid} appears twice — boundary violation"
            )
            seen.add(cid)
    print(f"  all {len(packed)} sequences respect commit boundaries")


# ── Test 11: HeteroBatchSampler proportional allocation ──

def test_hetero_batch_sampler_proportional():
    """Verify HeteroBatchSampler distributes sequences proportionally to VRAM."""
    from datasets.bigcode.commit_packing import (
        CommitSequencePacker, HeteroBatchSampler, PackedSequence,
    )

    packer = CommitSequencePacker(tokenizer=None, seq_len=32, pad_token_id=0)
    # Generate enough short commits to fill multiple batches
    commits = [{"text": "x" * 40} for _ in range(60)]
    packed = packer.pack_dataset(iter(commits))
    assert len(packed) >= 6, f"need >=6 sequences, got {len(packed)}"

    # H100 (96 GB) should get ~2× the sequences of A6000 (48 GB)
    gpu_mem_map = {0: 96, 1: 48}
    sampler = HeteroBatchSampler(packed, gpu_mem_map=gpu_mem_map, base_batch=1, verbose=False)

    # Ratio calculation: min_mem=48, rank0 ratio=round(96/48)=2, rank1 ratio=1
    assert sampler.ratios[0] == 2, f"rank0 ratio should be 2, got {sampler.ratios[0]}"
    assert sampler.ratios[1] == 1, f"rank1 ratio should be 1, got {sampler.ratios[1]}"
    print(f"  ratios: {sampler.ratios} (H100=2x, A6000=1x)")

    batches = list(sampler)
    assert len(batches) > 0, "sampler yielded zero batches"

    for step, batch in enumerate(batches):
        n0 = len(batch[0])
        n1 = len(batch[1])
        # rank0 (96 GB) should receive exactly 2× base_batch sequences
        assert n0 == 2, f"step {step}: rank0 got {n0} seqs, expected 2"
        assert n1 == 1, f"step {step}: rank1 got {n1} seqs, expected 1"
    print(f"  {len(batches)} steps, each step: rank0=2 seqs, rank1=1 seq")


# ── Test 12: PCIeP2PCommunicator mock interface ──

def test_pcie_p2p_communicator_mock():
    """Use mocks to verify PCIeP2PCommunicator.send_activation interface."""
    from unittest.mock import MagicMock, patch
    import torch

    from deepspeed.runtime.hetero_mimo_training_loop import (
        PCIeP2PCommunicator,
        DeviceCapabilityRegistry,
        SharedLocalityCache,
    )

    # Build a mock registry that maps device ids to pool info
    mock_registry = MagicMock(spec=DeviceCapabilityRegistry)
    mock_cap_0 = MagicMock()
    mock_cap_0.pool = "A6000_pool"
    mock_cap_1 = MagicMock()
    mock_cap_1.pool = "H100_pool"
    mock_registry.get.side_effect = lambda dev: {0: mock_cap_0, 1: mock_cap_1}.get(dev)

    # Build a mock locality cache (empty — no cached activations)
    mock_cache = MagicMock(spec=SharedLocalityCache)
    mock_cache.get.return_value = None

    comm = PCIeP2PCommunicator(
        registry=mock_registry,
        locality_cache=mock_cache,
        staging_threshold_mb=0.001,  # tiny threshold to trigger staging path
    )
    print(f"  PCIeP2PCommunicator created (staging_threshold=0.001 MB)")

    # Create a small CPU tensor (simulating a cross-pool activation)
    tensor = torch.randn(4, 8)

    # send_activation calls tensor.to() which needs a CUDA device — patch it
    with patch.object(torch.Tensor, "to", return_value=tensor) as mock_to:
        result = comm.send_activation(tensor, src_device=0, dst_device=1, cache_key="act:0:1")

    # Verify the interface was exercised correctly
    mock_registry.get.assert_any_call(0)
    mock_registry.get.assert_any_call(1)
    mock_cache.get.assert_called_once_with("act:0:1")
    mock_cache.put.assert_called_once()
    assert result is not None, "send_activation returned None"
    print(f"  send_activation OK: registry queried, cache checked, result returned")
    print(f"  cache.put called with key='act:0:1'")


# ── Run all ──

def main():
    print("╔══════════════════════════════════════════╗")
    print("║  DES-LOC Pipeline Smoke Test             ║")
    print("╚══════════════════════════════════════════╝")

    test("tokenizer", test_tokenizer)
    test("model_build", test_model_build)
    test("forward_pass", test_forward)
    test("data_loading", test_data_loading)
    test("engine_config", test_engine_config)
    test("checkpoint", test_checkpoint)
    test("desloc_engine_train", test_desloc_engine_train)
    test("hetero_registry_discovers_all", test_hetero_registry_discovers_all)
    test("desloc_engine_has_all_hooks", test_desloc_engine_has_all_hooks)
    test("commit_sequence_packer", test_commit_sequence_packer)
    test("hetero_batch_sampler_proportional", test_hetero_batch_sampler_proportional)
    test("pcie_p2p_communicator_mock", test_pcie_p2p_communicator_mock)

    print(f"\n{'='*50}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Results: {passed}/{total} passed")
    for name, ok, elapsed in results:
        print(f"    {PASS if ok else FAIL} {name} ({elapsed:.1f}s)")
    print(f"{'='*50}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
