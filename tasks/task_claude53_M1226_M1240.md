# Claude-53: M1226-M1240 — Wiring: HeteroRecomputeConfig 接入 forward

## 任务
在 `deepspeed/runtime/desloc_engine.py` 的 forward 阶段接入 activation checkpointing。

## 具体工作
1. `cat deepspeed/runtime/desloc_engine.py` 先读
2. `grep -n "class HeteroRecomputeConfig\|def apply_" deepspeed/runtime/hetero_gdn_selective_recompute.py` 找接口
3. 在 __init__ 中初始化 HeteroRecomputeConfig，在 forward 中启用 selective recompute
4. 验证语法

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
