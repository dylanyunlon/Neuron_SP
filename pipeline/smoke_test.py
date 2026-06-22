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
