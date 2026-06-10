# Claude-2 任务: M461-M475 — Kx Sweep 消融实验

Session: Claude-2 (M461-M475) | Base: 待 Claude-1 结果后确定

## 目标
对 DES-LOC 的核心超参 Kx 做完整 sweep, 验证分解式同步的优势。

## 步骤

### 1. 同步代码
```bash
cd /data/jiacheng/system/cache/temp/nips2026/Neuron_SP
git pull origin main
cat desloc_results/phase4/  # 确认 Claude-1 的基线结果存在
```

### 2. 同步 Sweep (Kx=Ku=Kv)
```bash
for KX in 1 2 4 8 16 32; do
    echo "=== Sync sweep: Kx=Ku=Kv=$KX ==="
    DESLOC_KX=$KX DESLOC_KU=$KX DESLOC_KV=$KX bash run_13B_ags1.sh
done
```

### 3. 分解 Sweep (Ku=2Kx, Kv=4Kx)
```bash
for KX in 1 2 4 8; do
    KU=$((KX * 2))
    KV=$((KX * 4))
    echo "=== Decomposed: Kx=$KX Ku=$KU Kv=$KV ==="
    DESLOC_KX=$KX DESLOC_KU=$KU DESLOC_KV=$KV bash run_13B_ags1.sh
done
```

### 4. 结果汇总
每组实验记录:
- final_loss, best_loss, convergence_step
- total_comm_bytes, comm_reduction_ratio
- wall_clock_seconds, throughput (tokens/sec)
- 写入 `desloc_results/phase4/kx_sweep.json`

### 5. 提交
```bash
git add desloc_results/phase4/kx_sweep.json
git commit --signoff -m "Claude-2 M461: Kx sweep ablation results"
git push origin main
```

## 判据
- 分解式 (Ku≠Kv≠Kx) 应比同步式 (Ku=Kv=Kx) 在相同 comm budget 下 loss 更低
- Kx=8, Ku=16, Kv=32 预期为甜蜜点
- 如果大 Kx 发散, 记录发散点, 作为论文的稳定性讨论素材
