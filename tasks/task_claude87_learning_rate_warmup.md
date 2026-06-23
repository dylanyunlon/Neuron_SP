# Claude-87: Learning rate warmup + cosine decay 验证

## 任务
验证 run_pretrain.py 的 lr schedule 对7B模型的参数是否正确。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A 30 "build_cosine_schedule" — 读 lr schedule
3. cat configs/7b_commitpack.yaml — 读 lr 配置
4. 确保 warmup_steps=2000, peak_lr=3e-4, min_lr=3e-5, cosine decay 到 total_steps
5. 如果参数不对,修改 configs/7b_commitpack.yaml 和 run_pretrain.py

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
