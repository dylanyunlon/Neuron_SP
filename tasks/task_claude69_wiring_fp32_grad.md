# Claude-69: Wire HeteroFP32GradAccumManager into backward pass

## 任务
把 FP32 梯度累积管理器接入 DesLocEngine 的 backward 流程。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/runtime/hetero_fp32_grad_accum.py 的 HeteroFP32GradAccumManager
4. 在 DesLocEngine.__init__() 中实例化 HeteroFP32GradAccumManager
5. 在 train() 的 backward 后、optimizer.step() 前调用其同步方法
6. git add -A && git commit --signoff -m "wire HeteroFP32GradAccumManager into DesLocEngine backward"
7. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
