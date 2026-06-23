# Claude-56: M1271-M1285 — Wiring: SharedLocalityCache + PCIeP2PCommunicator 接入 MIMO

## 任务
在 train_three_stage.py 或 desloc_engine.py 中接入 SharedLocalityCache 和 PCIeP2PCommunicator。

## 具体工作
1. `cat deepspeed/runtime/hetero_mimo_training_loop.py | head -100` 看 SharedLocalityCache 和 PCIeP2PCommunicator 接口
2. 在 DesLocEngine.__init__ 中初始化 SharedLocalityCache(利用 1.5TB DRAM)
3. 在 train() 的跨 GPU 通信处接入 PCIeP2PCommunicator
4. 验证语法

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
