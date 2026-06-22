# Claude-11 任务: M596-M610 — 异构 Tensor Parallel 跨设备分片策略

Session: Claude-11 (M596-M610) | Base: latest main

## 目标
参考 Megatron 的 TP 实现，设计跨 A6000/H100 的非均匀张量分片。

## 步骤

### 1. 研读 Megatron TP
```bash
cat references/pretrain_frameworks/megatron_tensor_parallel/*.py | head -300
```

### 2. 设计异构 TP 分片
- 按显存和带宽比例分配不同大小的 tensor shard
- H100 (94GB, HBM3 3.9TB/s) → 更大的 shard
- A6000 (48GB, GDDR6 768GB/s) → 更小的 shard
- AllReduce 在 PCIe 上的分块优化

### 3. 实现并验证
- 修改 DES-LOC 的 TP wrapper
- 单步 forward/backward 正确性验证

## 交付物
- 修改的文件 + commit push
- TP 分片比例和通信量分析写入 desloc_results/phase5/
