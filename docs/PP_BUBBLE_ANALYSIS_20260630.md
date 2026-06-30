# Neuron_SP 异构 PP=5 1F1B 分析

## 0. 先把问题摆清楚:这个配置本身就有结构性矛盾

在分析之前要指出一个根本问题:你的配置里 **per-GPU micro-batch size 和 layer 数量没有按算力配比**,而是大致按"显存大小"或"卡的代次"配的。这会导致每个 stage 的计算时间(layers × mbs / TFLOPS)严重不均衡,而 1F1B 的 bubble 恰恰是由最慢 stage 决定的。下面定量算一下。

## 1. 估算各 stage 前向时间

1F1B 中单个 micro-batch 在 stage *i* 的前向时间近似正比于:

```
T_i ∝ (layers_i × mbs_i) / TFLOPS_i
```

(这是简化模型,假设每层 FLOPs 相同、忽略 attention 的 O(n²) 项和通信,只看相对比例,用于定位瓶颈足够)

| GPU | 型号 | TFLOPS(BF16) | layers | mbs | layers×mbs | 相对算力需求 (∝layers×mbs/TFLOPS) | 归一化延迟 |
|---|---|---|---|---|---|---|---|
| GPU0 | A6000 | 38.7 | 4 | 2 | 8 | 8/38.7 = **0.2067** | **1.00×** (基准) |
| GPU1 | RTX6000B | 300 | 8 | 8 | 64 | 64/300 = 0.2133 | 1.03× |
| GPU2 | H100 NVL | 835 | 8 | 16 | 128 | 128/835 = 0.1533 | 0.74× |
| GPU3 | A6000 | 38.7 | 4 | 2 | 8 | 8/38.7 = **0.2067** | **1.00×** |
| GPU4 | RTX6000B | 300 | 8 | 8 | 64 | 64/300 = 0.2133 | 1.03× |

**关键发现**:这五个 stage 的相对延迟其实**已经被你手工调得相当接近了**(0.74×~1.03×),GPU0/GPU3 这两个 A6000 是最慢的(并列瓶颈,差距不大),而不是直觉上"A6000 layers少所以一定快"——你用 mbs=2(很小)补偿了 A6000 的低算力。这点做得对,但还不够精确,而且 H100 明显**算力被浪费了**(0.74× vs 1.0×,有 26% 的余量没用上)。

## 2. 瓶颈识别:GPU0/GPU3 (A6000) 还是 GPU2 (H100)?

**瓶颈是 GPU0 和 GPU3(两个 A6000),不是 H100。**

定量对比:
- A6000 stage: 8 "layer-mbs units" / 38.7 TFLOPS = 0.2067(相对单位)
- H100 stage: 128 "layer-mbs units" / 835 TFLOPS = 0.1533(相对单位)
- 比值:A6000 比 H100 **慢约 35%**(0.2067/0.1533 ≈ 1.348)

H100 不是瓶颈,反而是 mbs=16 配的还不够大——它的算力是 A6000 的 21.6 倍,但只承担了 16/2=8 倍的 micro-batch,即便 layer 数相同(8 vs 4,2倍),它仍有大量空闲算力。

更麻烦的是 **PP=5 有两个并列瓶颈(GPU0 和 GPU3)**,这是异构 PP 里最差的情况之一——bubble 不取决于平均速度,而取决于 max(T_i),且两个慢 stage 会同时拖慢 forward 和 backward 两条链路,使 1F1B 的 in-flight micro-batch 数也跟着被瓶颈卡住。

## 3. Bubble Ratio 估算

标准 1F1B(同构)bubble 公式:
```
bubble_ratio = (PP - 1) / (m + PP - 1)
```
其中 m = micro-batch 数量(per stage,在 steady state 流过的 micro-batch 数)。

但这里是**异构 stage 时间**,标准公式不适用,需要用 max-stage-time 模型:

```
bubble_ratio_hetero ≈ (PP - 1) × T_max / (m × T_max + (PP - 1) × T_max)
            简化为: (PP - 1) / (m + PP - 1)   [当所有 stage 时间相等时退化为标准公式]
```

由于 T_max 由 GPU0/GPU3 决定(归一化值 1.00),而其他 stage 实际工作量更小却仍要"陪跑"等待,实际 bubble 比标准公式更差。用你的 grad_accum_steps=2 (即 m=2,每个 GPU 在一次 step 中只跑 2 个 micro-batch):

```
bubble_ratio = (PP - 1) / (m + PP - 1) = 4 / (2 + 4) = 4/6 ≈ 66.7%
```

**这是相当糟糕的 bubble ratio**,核心原因不是异构本身,而是 **grad_accum_steps=2 太小,完全撑不起 PP=5**。标准建议是 m ≥ 4×PP 才能把 bubble 压到 20% 以下;m=2 < PP=5 时,bubble 占比会超过 60-70%,且 pipeline 甚至可能没填满就要排空(m < PP-1 时是病态情况,这里 m=2 < PP-1=4)。

考虑到异构进一步放大效应(慢 stage 的等待被算力浪费的快 stage "放大"),**实际有效 bubble ratio 估计在 70-75% 区间**,即只有约 25-30% 的 GPU-时间在做有效计算。

## 4. 优化建议

### 4.1 紧急且影响最大:提高 grad_accum_steps
```
m=2  → bubble ≈ 67%
m=8  → bubble = 4/(8+4) = 33%
m=16 → bubble = 4/(16+4) = 20%
m=32 → bubble = 4/(32+4) = 11%
```
这是单项收益最大的改动,几乎不需要重新设计 layer/mbs 配比。把 grad_accum_steps 从 2 提到 16~32(在显存允许范围内),bubble 可以从 67% 降到 11~20%。

### 4.2 Layer Split 再平衡
当前 4/8/8/4/8=32 layers,A6000 算力占比仅为总算力的 38.7×2/(38.7×2+300×2+835) ≈ 5.9%,却分到了 4/32=12.5% 的层数,过载约 2倍。理论上按算力比例分配 layer 数(忽略 mbs 调节,纯按 FLOPS 占比):

```
总 TFLOPS = 38.7+300+835+38.7+300 = 1512.4
理论 layer 分配 (×32):
  GPU0(A6000):  38.7/1512.4 × 32 ≈ 0.82 layer  → 不可行(太少,且不能切分到这么细)
  GPU2(H100):  835/1512.4 × 32 ≈ 17.7 layer
```
这说明**纯按 FLOPS 比例分配 layer 不现实**——A6000 几乎分不到整数层。这正是为什么你选择"小 mbs + 固定 layer 数"的策略是合理方向,问题只在于校准精度不够。

更现实的两个调整方向:

**方向 A(推荐,小改动):微调 mbs 而不动 layer split**
保持 layer 4/8/8/4/8 不变,把 mbs 配比再精修到更接近的 T_i:
```
目标: layers_i × mbs_i / TFLOPS_i 相等
设基准 T = 0.18 (取 H100 当前值与 A6000 值之间)
  GPU0: mbs = T × 38.7 / 4 ≈ 1.75 → 实际只能整数,保持 mbs=2 或降到能整除的值
  GPU2: mbs = T × 835 / 8 ≈ 18.8 → 可以把 H100 的 mbs 从 16 提到 18~20,进一步压榨利用率
  GPU1/4: mbs = T × 300 / 8 ≈ 6.75 → 当前 mbs=8 略微偏高,可微降到 7
```
即:**H100 的 mbs 还可以再加大(16→18~20)**,因为它仍有约 26% 的算力余量未用满;A6000 已经在合理区间,不需要再降(再降 mbs=1 会导致通信/调度开销占比过高,得不偿失)。

**方向 B(较大改动):把 A6000 的 layer 数再砍 1~2 层,补给 H100**
例如 layer split 改为 3/8/10/3/8(共32层),把砍下来的 2 层加到 GPU2,需要重新切分模型(可能跨 transformer block 边界,工程复杂度高,且要重新做 checkpoint 转换),收益要看具体能否对齐到 attention head/层边界。**只有在方向 A 调完后仍有明显瓶颈差距(>15%)时才值得做。**

### 4.3 GPU0/GPU3 是双瓶颈,优先级一致
因为这两个 stage 几乎完全对称(都是 A6000, 4 layers, mbs=2),任何调整必须**同时应用到两者**,否则会产生新的不对称瓶颈,比单瓶颈更难调度(1F1B 的 schedule 复杂度会因为非对称 stage 时间分布而上升)。

## 5. Interleaved 1F1B 在异构场景下是否有优势?

**有优势,但收益和代价要分开看:**

**收益**:Interleaved (virtual pipeline, 如 Megatron 的 interleaved 1F1B)把每个 GPU 上的连续 layer 段拆成多个不连续 chunk,跨 stage 交替放置。这样的核心好处是:
```
bubble_ratio_interleaved ≈ (PP-1)/(m × v + PP - 1) / v   (v = virtual stages per GPU)
```
v=2 时,等效 bubble 可以再降低约一半(相对同样的 m)。对你这种 m 很小(=2)的场景,interleaved 的边际收益尤其大,因为它本质上是用"更细粒度的调度"去填充本来由小 m 造成的大量空泡。

**但对你的异构瓶颈(GPU0/GPU3 慢)无效**:interleaved 调度只优化"调度填充率",不改变每个 micro-batch 在 A6000 上实际需要的计算时间。只要 GPU0/GPU3 的单 chunk 计算时间仍然最长,它们依然是整个 pipeline 的硬下限,interleaved 不会让 A6000 变快。

**额外代价**:
- A6000 显存本来就紧张(只分到 4 layers 大概率就是因为显存受限),interleaved 需要在每个 GPU 上同时持有多个不连续 chunk 的 activation,显存压力会进一步上升,对 A6000 反而可能成为新瓶颈(被迫降 mbs,抵消调度增益)。
- 通信次数增加(更多 stage 间的 P2P send/recv),在异构网络拓扑(A6000 PCIe vs H100 NVL)下,A6000 的通信也可能成为新瓶颈,需要确认互联带宽。

**结论**:**优先做 4.1(提高 grad_accum_steps)和 4.2 方向A(微调 mbs)**,这两项零结构改动、收益最大。Interleaved 1F1B 可以作为第二阶段优化,但要先确认 A6000 显存余量,否则可能在显存压力下被迫缩小 mbs,抵消调度收益。

---

**一句话总结**:你的瓶颈主要不是"异构"本身(layer/mbs 配比已经做得接近合理),而是 **grad_accum_steps=2 远小于 PP=5**,导致 bubble ratio 高达 ~67%。先把 m 提到 16-32 把 bubble 压到 20% 以下,再微调 H100 的 mbs(16→18-20)榨干剩余算力,这两步做完后再评估是否需要 interleaved。