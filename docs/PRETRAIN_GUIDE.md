# Neuron_SP 7B 预训练指南 (ags1)

## 硬件拓扑

| GPU | 型号 | VRAM | BF16 TFLOPS | NUMA |
|-----|------|------|-------------|------|
| 0 | RTX A6000 | 49GB | 38.7 | NUMA0 |
| 1 | RTX PRO 6000 Blackwell | 98GB | ~300 | NUMA0 |
| 2 | H100 NVL | 96GB | 835 | NUMA0 |
| 3 | RTX A6000 | 49GB | 38.7 | NUMA1 |
| 4 | RTX PRO 6000 Blackwell | 98GB | ~300 | NUMA1 |

CPU: 2×AMD EPYC 9354 (128核), 1.5TB DDR5
互联: 全 PCIe, 无 NVLink

## 快速开始

```bash
# 1. 环境安装 (一次性)
bash setup_ags1.sh
conda activate neuron_sp

# 2. 验证 GPU
python benchmark_mfu.py --gpu-id 2    # 测 H100
python tools/profile_memory.py         # 估算显存

# 3. 准备数据 (先用少量测试)
python data/prepare_commits.py --num-samples 10000 --tokenizer-name bigcode/starcoder
python tools/count_tokens.py --path data/commits.bin

# 4. 启动训练
# 先用 synthetic 数据验证
bash launch_7b.sh --dry-run
python run_pretrain.py --model-size 70m --steps 100

# 用真实数据, 5卡 FSDP
torchrun --nproc_per_node=5 run_pretrain.py \
    --model-size 7b \
    --data-path data/commits.bin \
    --steps 50000 \
    --fsdp \
    --gradient-checkpointing \
    --save-every 1000 \
    --wandb-project neuron-sp-7b
```

## 配置说明

`configs/7b_commitpack.yaml` 包含完整的 7B 配置。用 `--config` 加载:

```bash
python run_pretrain.py --config configs/7b_commitpack.yaml
```

## 关键参数

| 参数 | 说明 | 7B 推荐值 |
|------|------|-----------|
| --fsdp | FSDP 分片 (替代 DDP) | 必须开 |
| --gradient-checkpointing | 激活重计算 | 必须开 (A6000 48GB) |
| --save-every | 保存间隔 | 1000 |
| --batch-size | 每 GPU micro batch | 1 (7B) 或 2 (1B) |
| --seq-len | 序列长度 | 2048 |

## 训练监控

```bash
# 实时 loss 曲线
python monitor/train_monitor.py --log-file logs/7b_pretrain_*.log --watch

# wandb 面板
# 浏览器打开 wandb.ai/your-project
```

## 断点续训

```bash
python run_pretrain.py --resume-from checkpoints/step_0005000.pt --steps 50000 ...
```

## 训完后转 HuggingFace 格式

```bash
python tools/convert_to_hf.py \
    --checkpoint-path checkpoints/step_0050000.pt \
    --output-dir models/neuron-sp-7b-hf
```

## 常见问题

**Q: OOM on A6000?**
确保 `--gradient-checkpointing` 和 `--fsdp` 都开了。batch_size=1。

**Q: NCCL timeout?**
跨 NUMA 通信慢。设 `NCCL_P2P_DISABLE=1` (launch_7b.sh 已设)。

**Q: 数据量够吗?**
CommitPack 4TB ≈ 数百B tokens, 远超 Chinchilla-optimal 的 140B。

**Q: 预计训多久?**
7B @ 20B tokens ≈ 2周 (5卡); 7B @ 140B tokens ≈ 3-4月 (5卡)。
