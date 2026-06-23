# Claude-52: M1211-M1225 — Wiring: HeteroFP32GradAccumManager 接入 backward

## 任务
在 `deepspeed/runtime/desloc_engine.py` 的 train() backward 阶段接入 HeteroFP32GradAccumManager。

## 具体工作
1. `cat deepspeed/runtime/desloc_engine.py` 先读
2. `head -60 deepspeed/runtime/hetero_fp32_grad_accum.py` 看接口
3. 在 train() 的 backward 后、grad clip 前，调用 FP32GradAccumManager 的 accumulate() 方法
4. 验证语法

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
