# Claude-136: CommitPack Data Preparation Pipeline

## Context
Currently using synthetic data. Need real CommitPack data for the paper.

## Task
1. Fix `data/prepare_commits.py` to:
   - Stream CommitPack from HuggingFace (`bigcode/commitpack`)
   - Tokenize with `transformers.AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")` (or sentencepiece fallback)
   - Pack sequences to `seq_len=2048` with padding
   - Save as memory-mapped numpy arrays: `data/commitpack_train.npy`, `data/commitpack_valid.npy`
   - Split: 99% train, 1% valid
2. Add `--data-path data/commitpack_train.npy` support in `run_pretrain.py` real_data_iter()
3. Target: at least 1B tokens for training

## Files
- `data/prepare_commits.py` — fix/rewrite
- `run_pretrain.py` — ensure real_data_iter works with .npy mmap files

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
