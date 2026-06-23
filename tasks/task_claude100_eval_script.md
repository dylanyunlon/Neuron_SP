# Claude-100: eval/run_commit_eval.sh 一键评测脚本

## 任务
新建 eval/run_commit_eval.sh 一键评测。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat eval/run_eval.py — 读现有评测
3. cat eval/eval_config.yaml — 读配置
4. 新建 eval/run_commit_eval.sh:
   - 接受 --checkpoint 参数
   - 跑 HumanEval + MBPP (通过 lm-eval-harness 或自定义)
   - 跑 commit message 生成质量评测 (BLEU/ROUGE on CommitPackFT test set)
   - 输出 JSON summary
5. chmod +x eval/run_commit_eval.sh

## 铁律
- 可新建 eval/run_commit_eval.sh
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
