# Claude-65: M1406-M1420 — Wiring: HeteroStepBatchScheduler + CommitSequencePacker

## 任务
在 train_three_stage.py 中接入异构批次调度和 commit 感知序列打包。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat pipeline/train_three_stage.py — 通读
3. cat deepspeed/runtime/hetero_step_batch_scheduler.py | head -100 — 读接口
4. cat datasets/bigcode/commit_packing.py | head -80 — 读 CommitSequencePacker
5. 在 train_three_stage.py 的数据加载阶段接入 CommitSequencePacker
6. 用 HeteroStepBatchScheduler 替代固定 batch size 分配

## 铁律
- 只改 pipeline/train_three_stage.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
