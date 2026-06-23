# Claude-115: DataLoader num_workers + pin_memory 配置

## Bug
ags1 有 128 核 CPU, 1.5TB RAM。DataLoader 默认 num_workers=0 严重浪费。
pin_memory=True 在 PCIe 传输时可以加速 H2D。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -rn "DataLoader\|num_workers\|pin_memory" run_pretrain.py pipeline/train_three_stage.py
3. 所有 DataLoader 调用加: num_workers=4, pin_memory=True, persistent_workers=True
4. 128核/5GPU = 25核/GPU, 4 workers per GPU 很合理
5. prefetch_factor=2 让下一个 batch 提前准备

## 铁律
- 只改 run_pretrain.py 和 pipeline/train_three_stage.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: DataLoader num_workers=4, pin_memory=True for ags1 128-core CPU"
- git push origin main
