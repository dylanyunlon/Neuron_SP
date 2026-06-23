# Claude-103: 训练 loss spike 检测器

## 任务
在 run_pretrain.py 的训练循环中添加 loss spike 检测和自动回滚。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A30 "def run_standalone" — 读训练循环
3. 添加 loss spike 检测:
   - 维护 loss EMA (alpha=0.01)
   - 如果当前 loss > 3x EMA, 标记为 spike
   - 连续 5 个 spike: 自动 reload 上一个 checkpoint, 降低 lr 50%
4. 用 logger.warning 报告 spike

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
