# Claude-32: Wire CommitSequencePacker + HeteroBatchSampler into data loading

在 `pipeline/train_three_stage.py` 的 `load_stage2_data()` 和 `load_stage3_data()` 中:
1. `cat datasets/bigcode/commit_packing.py | grep -n "class CommitSequencePacker\|class HeteroBatchSampler"` 找到类
2. 用 `CommitSequencePacker` 替换当前的手动 tokenize+pad
3. 用 `HeteroBatchSampler` 替换默认 DataLoader sampler, 按 GPU 显存比例分配 batch
4. 验证: packing 后 padding ratio < 5%
