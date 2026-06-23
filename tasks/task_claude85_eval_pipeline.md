# Claude-85: 评估管线 — commit 补全评估

## 任务
完善 `pipeline/` 中的评估逻辑。

## 具体工作
1. `cat pipeline/smoke_test.py` 先读现有评估
2. 修改 `pipeline/train_three_stage.py` 的 Stage 3 eval 部分:
   - 加入 commit message → diff 的 BLEU/ROUGE-L 计算
   - 加入 perplexity 计算
   - 结果写入 `experiments/eval_results/`
3. 确保 `pipeline/unified_tokenizer.py` 支持 commit special tokens: <|commit_msg|>, <|diff_start|>, <|diff_end|>

## 铁律
- MODIFY EXISTING FILES ONLY (除了 experiments/eval_results/ 目录)
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
