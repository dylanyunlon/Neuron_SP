# Claude-14 任务: M641-M655 — Heterogeneous Memory Profiler

Session: Claude-14 (M641-M655) | Base: latest main

## 目标
实现异构 GPU 集群的实时内存 profiler，为 DES-LOC 的动态分区策略提供数据。

## 步骤

### 1. 查看现有分区代码
```bash
cat deepspeed/runtime/desloc_engine.py | grep -B5 -A30 "PartitionSolver\|TierDiscovery"
```

### 2. 实现 profiler
- 实时采集各 GPU 的 VRAM 使用、计算利用率、PCIe 带宽
- 通过 pynvml 获取 GPU 指标
- 输出 timeline JSON（可被 Chrome trace viewer 可视化）
- 检测 OOM 风险并预警

### 3. 集成
- 嵌入 DES-LOC 训练循环的 step callback
- 每 N steps 采样一次

## 交付物
- 修改的文件 + commit push
- 10 step 的 profiler 输出样例
