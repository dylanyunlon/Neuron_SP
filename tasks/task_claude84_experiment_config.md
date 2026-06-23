# Claude-84: 7B 预训练实验配置

## 任务
创建 ags1 集群专用的 DeepSpeed 和训练配置文件。

## 具体工作
1. 先 `cat deepspeed/runtime/desloc_engine.py | head -50` 看 config 格式
2. 创建 `experiments/configs/7b_ags1_desloc.yaml`:
   - model: 7B (32 layers, 4096 hidden, 32 heads)
   - GPU 分配: A6000×2 (NUMA0+NUMA1), H100 NVL×1, RTX PRO 6000 Blackwell×2
   - DES-LOC 参数: Kx=8, Ku=4, Kv=16
   - Batch: micro_batch per device 按显存比例
   - Precision: A6000=FP16, H100=BF16, Blackwell=BF16
   - Activation checkpointing: selective per tier
   - LOC cache: 200GB on NUMA0, 200GB on NUMA1
3. 创建 `experiments/configs/7b_ags1_baseline.yaml` (无 DES-LOC 对照)
4. 创建 `experiments/launch_7b.sh` 启动脚本

## 铁律
- 可以创建新文件在 experiments/ 下
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
