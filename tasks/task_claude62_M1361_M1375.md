# Claude-62: M1361-M1375 — Wiring: DesLocEngine.train() 接入 setup_hetero_mimo_training

## 任务
在 desloc_engine.py 的 train() 方法中接入 MIMO pipeline 训练循环。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py — 通读 train() 方法
3. cat deepspeed/runtime/hetero_mimo_training_loop.py | head -150 — 读 setup_hetero_mimo_training 签名
4. 在 DesLocEngine.train() 开头调用 setup_hetero_mimo_training(),把返回的组件(PCIeP2PCommunicator, SharedLocalityCache等)存为实例变量
5. 在训练循环中用 PCIeP2PCommunicator 替代直接的 dist 调用

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
