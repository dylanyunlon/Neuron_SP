# Claude-121: learning rate 硬编码 3e-4

## Bug
run_pretrain.py:568: lr=3e-4 硬编码, 不读 yaml config。
7B 模型应该用 configs/7b_commitpack.yaml 里的 lr。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "lr.*=.*3e-4\|learning_rate" run_pretrain.py configs/7b_commitpack.yaml
3. 从 yaml config 读 lr: lr = yaml_cfg.get("optimizer", {}).get("lr", 3e-4)
4. 确保 configs/7b_commitpack.yaml 有 optimizer.lr 字段
5. 如果 yaml 没有, 添加: optimizer: { lr: 3e-4, weight_decay: 0.1, betas: [0.9, 0.95] }

## 铁律
- 只改 run_pretrain.py 和 configs/7b_commitpack.yaml
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: read lr from YAML config instead of hardcoded 3e-4"
- git push origin main
