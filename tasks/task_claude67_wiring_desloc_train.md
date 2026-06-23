# Claude-67: Wire hetero modules into DesLocEngine.train()

## 任务
把已有的 hetero 组件接入 DesLocEngine.train() 训练循环。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/runtime/desloc_engine.py 的 train() 方法 (line ~1115)
4. 读 deepspeed/runtime/hetero_gdn_selective_recompute.py 的 build_neuron_sp_config()
5. 在 train() 开头调用 build_neuron_sp_config() 得到 recompute_config，apply 到 model
6. 读 deepspeed/runtime/hetero_mimo_training_loop.py 的 setup_hetero_mimo_training()
7. 在 DesLocEngine.__init__() 里调 setup_hetero_mimo_training() 初始化 MIMO 循环
8. 确保 import 正确，不破坏现有逻辑
9. git add -A && git commit --signoff -m "wire build_neuron_sp_config + setup_hetero_mimo_training into DesLocEngine"
10. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 不重写任何 hetero 模块，只调用它们的 public API
- 作者: dylanyunlon <dogechat@163.com>
