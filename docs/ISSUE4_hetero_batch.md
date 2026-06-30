# ISSUE 4 — 异构 Micro-Batch Sizing 优化方案

**作者**: HeteroBatchLead  
**日期**: 2025-06  
**状态**: 草案

---

## 1. 当前 Batch 分配比例

### 1.1 `configs/7b_5gpu.yaml`

```yaml
micro_batch_size_per_gpu: [2, 4, 4, 2, 4]   # [A6000, Blackwell, H100, A6000, Blackwell]
micro_batch_size: 2                           # 全局 fallback（最保守设备）
grad_accum_steps: 4
global_batch_size: 64   # = (2+4+4+2+4) × 4 = 64 ✓
```

| GPU | 型号 | VRAM | TFLOPS (BF16) | micro_batch_size | 相对比例 |
|-----|------|------|----------------|------------------|----------|
| GPU0 | A6000 | 49 GB | 38.7 | 2 | 1× |
| GPU1 | RTX PRO 6000 Blackwell | 98 GB | 未标注 | 4 | 2× |
| GPU2 | H100 NVL | 96 GB | 835 | 4 | 2× |
| GPU3 | A6000 | 49 GB | 38.7 | 2 | 1× |
| GPU4 | RTX PRO 6000 Blackwell | 98 GB | 未标注 | 4 | 2× |

**H100 vs A6000 当前比例: 4 : 2 = 2:1**

这个比例只反映了 VRAM 差距（96 GB vs 49 GB ≈ 2:1），完全没有考虑算力差距。

---

### 1.2 `_plan_zero3()` — `desloc_engine.py` L726

```python
if spec.tier == TierClass.H100:
    micro_bs[spec.device_index] = min(cfg.micro_batch_size * 4, 4)
else:
    micro_bs[spec.device_index] = cfg.micro_batch_size          # = 2
```

`cfg.micro_batch_size = 2`（yaml fallback 值），因此：

- **H100**: `min(2 × 4, 4)` = **4**（上限被硬编码为 4，无法突破）
- **A6000**: **2**

ZeRO-3 路径下实际比例同样是 **2:1**，且被 `min(..., 4)` 死锁在此值。

---

### 1.3 `HeteroStepBatchScheduler` / `HeteroMicrobatchAllocator`

**`DEFAULT_DES_LOC_DEVICE_PROFILES`（L199）:**

```python
DeviceProfile(device_id=0, sm_arch=86, vram_gb=48.0, capacity_weight=1.0, max_micro_batch_size=1)
DeviceProfile(device_id=1, sm_arch=86, vram_gb=48.0, capacity_weight=1.0, max_micro_batch_size=1)
DeviceProfile(device_id=2, sm_arch=90, vram_gb=96.0, capacity_weight=1.0, max_micro_batch_size=4)
```

三台设备 `capacity_weight` **全部是 1.0**，分配器会均等三分全局批量（H100 和 A6000 拿同样多的 microbatch）。H100 的 `max_micro_batch_size=4` 而 A6000 的 `max_micro_batch_size=1`，但权重相同意味着 H100 实际上被当成和 A6000 一样的弱设备来调度。

注释自相矛盾：L197 写"H100 跑 6 个微批，A6000 各跑 1 个 (1:1:6)"，但 `capacity_weight` 全为 1.0，实际产生的是 **1:1:1**。

**`allocate()` 核心逻辑（L453）:**

```python
share = int(num_microbatches * profile.capacity_weight / self._total_weight)
```

`_total_weight = 1.0 + 1.0 + 1.0 = 3.0`，每台设备拿 `1/3` 的微批，H100 的 21× 算力优势被完全忽视。

---

### 1.4 当前状态汇总

| 路径 | H100 比例 | A6000 比例 | 实际效果 |
|------|-----------|------------|----------|
| yaml `micro_batch_size_per_gpu` | 4 | 2 | **2:1**（仅反映 VRAM）|
| `_plan_zero3()` | `min(mbs×4, 4)`=4 | mbs=2 | **2:1**（被上限锁死）|
| `DEFAULT_DES_LOC_DEVICE_PROFILES` | weight=1.0 | weight=1.0 | **1:1**（完全均等）|
| Megatron M93504931 目标 | 6 | 1 | **6:1** |
| 算力真实比值 | 835 TFLOPS | 38.7 TFLOPS | **21:1** |

---

## 2. 最优比例推导

### 2.1 理论上限：纯 TFLOPS 比

```
H100 NVL  : 835 TFLOPS BF16
A6000     :  38.7 TFLOPS BF16
比值       : 835 / 38.7 ≈ 21.6:1
```

在理想情况（无 IO 瓶颈，无通信开销），H100 每步能处理的 token 数是 A6000 的 21.6 倍。

### 2.2 VRAM 约束

7B 模型在 BF16 精度下（ZeRO-3 分片，grad + optimizer state）：

- 参数约 14 GB（BF16），ZeRO-3 每设备持有约 14/N GB 分片（N=5 → ~2.8 GB/GPU）。
- 激活显存随 `micro_batch_size × seq_len` 线性增长，开启 `selective` checkpointing 后约 `0.3 GB/MBS`（seq=2048）。

| 设备 | VRAM | 安全可用 | 估算最大 MBS |
|------|------|----------|-------------|
| A6000 | 49 GB | ~44 GB | **8**（保守取 4）|
| H100 NVL | 96 GB | ~90 GB | **32**（保守取 16）|
| Blackwell | 98 GB | ~92 GB | **32**（保守取 16）|

A6000 并非限制 MBS 的瓶颈，当前取 2 是过于保守的。

### 2.3 PCIe 通信约束

ZeRO-3 每步需要两次 AllGather（前向 + 反向）和一次 ReduceScatter（梯度）。在 PCIe4/5 互联下（无 NVLink）：

- 跨 NUMA 带宽约 32–48 GB/s（CPU 桥接）。
- 增大 H100 MBS 不增加通信量（通信量由参数量决定，与 MBS 无关）；
- 但增大 A6000 MBS 会延长 A6000 的计算时间，加重 H100 等待。

**结论：应尽量提高 H100 的 MBS，同时降低 A6000 的相对份额（减少等待），而非两者均等增大。**

### 2.4 推荐目标比例

综合算力比（21:1）、VRAM 约束和 PCIe 同步开销，分两档：

**保守目标（近期可落地）：H100 : A6000 = 8:1**
- H100 MBS = 16，A6000 MBS = 2
- `grad_accum_steps` 维持 4
- `global_batch_size = (2+16+16+2+16) × ? → 需调整`（见 §3）

**激进目标（对齐 Megatron M93504931）：H100 : A6000 = 12:1**
- H100 MBS = 24（或 32），A6000 MBS = 2
- Blackwell 按 VRAM 对齐 H100，MBS = 24
- 需要实测 A6000 在 MBS=2 下的 step time 是否仍是全局瓶颈

> **注意**: Megatron 参考的 6:1 是针对 3-GPU 集群（2×A6000 + 1×H100）。5-GPU 集群中 Blackwell 存在，需要单独档位。Blackwell SM12.0 BF16 性能未标注，建议实测后按 TFLOPS 插值，暂估为 300–500 TFLOPS（介于 A6000 和 H100 之间）。

---

## 3. 修改方案

### 3.1 修改 `configs/7b_5gpu.yaml`

**当前:**
```yaml
training:
  micro_batch_size_per_gpu: [2, 4, 4, 2, 4]
  micro_batch_size: 2
  grad_accum_steps: 4
  global_batch_size: 64
```

**改为（保守目标）:**
```yaml
training:
  # 比例: A6000(2) : Blackwell(8) : H100(16) : A6000(2) : Blackwell(8)
  # 硬件对齐: H100 835 TFLOPS / A6000 38.7 TFLOPS ≈ 21:1 → 取保守 8:1
  # Blackwell 性能未知，暂按 VRAM 比例取 H100 一半 (8)
  micro_batch_size_per_gpu: [2, 8, 16, 2, 8]
  micro_batch_size: 2           # 保持 A6000 保守值作为 fallback
  grad_accum_steps: 2           # 每步总样本 = (2+8+16+2+8)×2 = 36×2 = 72
  global_batch_size: 72         # 或调整 grad_accum_steps=1 → gbs=36，再 DP 翻倍

  # 如需维持 global_batch_size=64 的整除性，取:
  # micro_batch_size_per_gpu: [2, 8, 16, 2, 8], grad_accum_steps=1 → sum=36 (不整除64)
  # 建议调整 global_batch_size=72 或 micro_batch_size_per_gpu: [1, 8, 16, 1, 6] sum=32 × 2 = 64
  # 最简方案（保持 gbs=64）:
  # micro_batch_size_per_gpu: [1, 8, 14, 1, 8]  sum=32, grad_accum=2 → 64 ✓
```

**最终推荐（保持 global_batch_size=64 不变）:**
```yaml
training:
  micro_batch_size_per_gpu: [2, 8, 16, 2, 8]
  micro_batch_size: 2
  grad_accum_steps: 1           # 单步消耗 36 样本；或 gbs=72 对应 grad_accum=2
  global_batch_size: 36         # 36 token/step × 4 grad_accum → 需重新对齐 warmup

  # 替代：如果必须保持 gbs=64，使用不对称分配
  # micro_batch_size_per_gpu: [2, 6, 12, 2, 6]   sum=28, ×2 grad_accum=56 (近似)
  # 或 [2, 8, 14, 2, 6]  sum=32, ×2=64 ✓（更推荐）
```

### 3.2 修改 `desloc_engine.py` — `_plan_zero3()` L726

**当前代码（L741–744）:**
```python
if spec.tier == TierClass.H100:
    micro_bs[spec.device_index] = min(cfg.micro_batch_size * 4, 4)
else:
    micro_bs[spec.device_index] = cfg.micro_batch_size
```

**问题：** `min(..., 4)` 硬上限完全锁死了 H100 的潜力，且没有 Blackwell 分支。

**修改为:**
```python
# 按设备 SM 架构和 VRAM 动态决定 MBS 上限
# tier_to_max_mbs 可来自 config，此处内联默认值
_TIER_MBS_MULTIPLIER = {
    TierClass.H100:     8,   # 835 TFLOPS; VRAM 96 GB → 8× base 为保守安全值
    TierClass.BLACKWELL: 4,  # 性能未标注，暂取 H100 一半
    TierClass.A6000:    1,   # 38.7 TFLOPS; VRAM 49 GB → 基准值
}

for spec in self.tiers:
    multiplier = _TIER_MBS_MULTIPLIER.get(spec.tier, 1)
    # 从 yaml 的 micro_batch_size_per_gpu 列表读取（如有），否则 fallback 到倍率
    if (cfg.micro_batch_size_per_gpu
            and spec.device_index < len(cfg.micro_batch_size_per_gpu)):
        micro_bs[spec.device_index] = cfg.micro_batch_size_per_gpu[spec.device_index]
    else:
        cap = cfg.vram_per_device_gb.get(spec.device_index, 48) / 48.0
        micro_bs[spec.device_index] = max(
            1,
            min(
                int(cfg.micro_batch_size * multiplier),
                int(cfg.micro_batch_size * cap * 2),   # VRAM 上限
            )
        )
    grad_accum[spec.device_index] = cfg.grad_accum_steps
```

**同时更新 notes 字符串** 以反映新的动态上限：
```python
return PartitionPlan(
    ...
    notes=(
        f"ZeRO-3 hetero: A6000={micro_bs.get(a6000_idx, '?')}, "
        f"H100={micro_bs.get(h100_idx, '?')}, "
        f"Blackwell={micro_bs.get(bw_idx, '?')} "
        f"(ratio H100:A6000={micro_bs.get(h100_idx,1)//max(micro_bs.get(a6000_idx,1),1)}:1)"
    ),
)
```

### 3.3 修改 `hetero_step_batch_scheduler.py` — `DEFAULT_DES_LOC_DEVICE_PROFILES`

**当前（L199–206，三设备旧集群假设）:**
```python
DEFAULT_DES_LOC_DEVICE_PROFILES: List[DeviceProfile] = [
    DeviceProfile(device_id=0, sm_arch=86, vram_gb=48.0, capacity_weight=1.0, max_micro_batch_size=1),
    DeviceProfile(device_id=1, sm_arch=86, vram_gb=48.0, capacity_weight=1.0, max_micro_batch_size=1),
    DeviceProfile(device_id=2, sm_arch=90, vram_gb=96.0, capacity_weight=1.0, max_micro_batch_size=4),
]
```

**问题：**
1. `capacity_weight` 全为 1.0，H100 与 A6000 被等价对待，分配器输出 1:1:1。
2. 不包含 Blackwell（SM12.0），5-GPU 集群的 GPU1/GPU4 没有对应 profile。
3. 注释声称"1:1:6"但代码根本不产生这个结果。

**修改为（5-GPU 集群，保守目标 8:1）:**
```python
# 5-GPU 异构集群配置 (ags1)
# capacity_weight 基于 TFLOPS 比，H100:Blackwell:A6000 = 21:8:1
# 保守取 8:4:1（留余量，防止 A6000 成极端瓶颈）
DEFAULT_DES_LOC_DEVICE_PROFILES: List[DeviceProfile] = [
    # GPU0: A6000 SM8.6  49GB  38.7 TFLOPS
    DeviceProfile(device_id=0, sm_arch=86, vram_gb=49.0,
                  capacity_weight=1.0, max_micro_batch_size=2,
                  loc_cache_size_mb=2048),
    # GPU1: Blackwell SM12.0  98GB  ~300 TFLOPS (估计值，待实测后调整)
    DeviceProfile(device_id=1, sm_arch=120, vram_gb=98.0,
                  capacity_weight=4.0, max_micro_batch_size=8,
                  loc_cache_size_mb=8192),
    # GPU2: H100 NVL SM9.0  96GB  835 TFLOPS
    DeviceProfile(device_id=2, sm_arch=90, vram_gb=96.0,
                  capacity_weight=8.0, max_micro_batch_size=16,
                  loc_cache_size_mb=8192),
    # GPU3: A6000 SM8.6  49GB  38.7 TFLOPS (mirror of GPU0)
    DeviceProfile(device_id=3, sm_arch=86, vram_gb=49.0,
                  capacity_weight=1.0, max_micro_batch_size=2,
                  loc_cache_size_mb=2048),
    # GPU4: Blackwell SM12.0  98GB  ~300 TFLOPS (mirror of GPU1)
    DeviceProfile(device_id=4, sm_arch=120, vram_gb=98.0,
                  capacity_weight=4.0, max_micro_batch_size=8,
                  loc_cache_size_mb=8192),
]
# 总权重 = 1+4+8+1+4 = 18
# 分配比例: A6000(1/18) : Blackwell(4/18) : H100(8/18) : A6000(1/18) : Blackwell(4/18)
# 即 H100:Blackwell:A6000 ≈ 8:4:1，与 yaml micro_batch_size_per_gpu=[2,8,16,2,8] 对齐
```

### 3.4 激进目标（可选，对标 Megatron 21:1）

若实测确认 A6000 不是同步瓶颈，可进一步激进：

```python
# 激进配置：H100:Blackwell:A6000 = 21:10:1（接近真实 TFLOPS 比）
DeviceProfile(device_id=0, sm_arch=86,  capacity_weight=1.0,  max_micro_batch_size=2)
DeviceProfile(device_id=1, sm_arch=120, capacity_weight=10.0, max_micro_batch_size=16)
DeviceProfile(device_id=2, sm_arch=90,  capacity_weight=21.0, max_micro_batch_size=32)
DeviceProfile(device_id=3, sm_arch=86,  capacity_weight=1.0,  max_micro_batch_size=2)
DeviceProfile(device_id=4, sm_arch=120, capacity_weight=10.0, max_micro_batch_size=16)
```

```yaml
# 对应 yaml
micro_batch_size_per_gpu: [2, 16, 32, 2, 16]
grad_accum_steps: 1
global_batch_size: 68   # 或取整到 64/72
```

---

## 4. 变更影响与风险

| 变更项 | 风险 | 缓解措施 |
|--------|------|----------|
| H100 MBS 2→16 | A6000 等待时间被进一步拉长 | 监控 `step_time` per-device；若 A6000 成为尾延迟，降低 H100 MBS |
| `min(..., 4)` 上限移除 | H100 OOM | 开启 `activation_checkpointing=full`，从 MBS=8 逐步递增 |
| `capacity_weight` 由 1→8 | `allocate()` 分配严重偏斜，小 GBS 时 A6000 拿不到微批 | 保持 `max(1, share)` 下限（代码已有），GBS 建议 ≥ 32 |
| `DEFAULT_DES_LOC_DEVICE_PROFILES` 增至 5 设备 | 旧 3-GPU checkpoint 不兼容 | 加版本字段 `profile_version: "5gpu_v1"` 到 checkpoint meta |
| Blackwell `sm_arch=120` | SM12.0 内核未在旧 PyTorch 编译（见 yaml 注释） | 确认 `torch>=2.7.0+cu126`，否则 capacity_weight 保持 1.0 回退 |

---

## 5. 推荐验证步骤

1. **基线测速**: 当前配置跑 100 步，记录 `tokens/sec` 和 per-GPU `step_time`。
2. **保守目标**: `micro_batch_size_per_gpu: [2, 8, 16, 2, 8]` + 对应 `capacity_weight`，再跑 100 步对比。
3. **梯度等价性检查**: 确认更改 MBS 后 `global_batch_size` 不变，loss 曲线不漂移（ZeRO-3 下需验证梯度归约正确性）。
4. **激进目标**: 如果步骤 2 中 H100 `step_time` 仍低于 A6000 的 2 倍，进一步调高到 `[2, 16, 32, 2, 16]`。
5. **Blackwell 实测**: 用 `nvitop` 或 `nvidia-smi dmon` 量化 GPU1/GPU4 实际算力，替换估计的 `capacity_weight=4.0`。
