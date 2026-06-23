# Claude-96: dry-run 验证脚本 — 不跑训练只验证配置

## 任务
在 launch_7b.sh 中添加 --dry-run 模式。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat launch_7b.sh — 读现有启动脚本
3. 加 --dry-run: 只做 model init + 1 step forward + 1 step backward, 打印 memory 和 throughput
4. 确保 dry-run 后清理 GPU memory
5. 输出: per-GPU memory peak, estimated tokens/sec, estimated training days

## 铁律
- 只改 launch_7b.sh 和 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
