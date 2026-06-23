# Claude-55: M1256-M1270 — Wiring: CommitSequencePacker + HeteroBatchSampler 接入 data loading

## 任务
确保 CommitSequencePacker 和 HeteroBatchSampler 正确接入数据加载流程。

## 具体工作
1. `cat deepspeed/runtime/desloc_engine.py` 看数据加载部分
2. `grep -n "class CommitSequencePacker\|class HeteroBatchSampler" deepspeed/runtime/*.py datasets/bigcode/*.py` 找定义
3. 在 DesLocEngine 的数据加载初始化中，用 CommitSequencePacker 包装 dataset，用 HeteroBatchSampler 替换默认 sampler
4. 验证语法

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
