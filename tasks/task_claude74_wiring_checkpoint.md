# Claude-74: Wire hetero checkpoint into DesLocEngine

## 任务
接入异构 checkpoint 系统，支持 per-tier 异步保存。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/checkpoint/hetero_checkpoint_config.py
4. 读 deepspeed/checkpoint/hetero_async_checkpoint_save.py
5. 在 DesLocEngine 的 save_checkpoint() 中调用 hetero checkpoint API
6. 在 load_checkpoint() 中支持从 hetero checkpoint 恢复
7. git add -A && git commit --signoff -m "wire hetero async checkpoint into DesLocEngine"
8. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py 和 deepspeed/checkpoint/ 下相关文件
- 作者: dylanyunlon <dogechat@163.com>
