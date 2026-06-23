# Claude-68: Wire HeteroRegistry.register_hooks() into DesLocEngine

## 任务
让 HeteroRegistry 真正扫描并注册所有 hetero 模块的 hooks。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/runtime/hetero_registry.py 的 HeteroRegistry class
4. 读 deepspeed/runtime/desloc_engine.py 找 HeteroRegistry 的调用点
5. 在 DesLocEngine.__init__() 末尾调用 registry.register_hooks(self)
6. 确保 register_hooks 能找到并激活已有的 hetero_*.py 模块
7. git add -A && git commit --signoff -m "wire HeteroRegistry.register_hooks into DesLocEngine.__init__"
8. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py 和 deepspeed/runtime/hetero_registry.py
- 作者: dylanyunlon <dogechat@163.com>
