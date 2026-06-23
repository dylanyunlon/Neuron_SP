# Claude-70: Wire hetero modules into train_three_stage.py

## 任务
让三阶段训练编排器调用已有的 hetero 组件。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 pipeline/train_three_stage.py
4. 读 pipeline/engine_bridge.py
5. 在每个 stage 的初始化中调用 build_neuron_sp_config() 和 setup_hetero_mimo_training()
6. 在 stage 切换时正确 teardown 和 re-init hetero 组件
7. git add -A && git commit --signoff -m "wire hetero modules into train_three_stage.py stages"
8. git push origin main

## 铁律
- 只改 pipeline/train_three_stage.py 和 pipeline/engine_bridge.py
- 作者: dylanyunlon <dogechat@163.com>
