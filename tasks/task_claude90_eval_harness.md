# Claude-90: 集成 lm-evaluation-harness 评测脚本

## 任务
在 eval/ 目录下完善评测管线。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat eval/run_eval.py — 读现有评测脚本
3. cat eval/eval_config.yaml — 读评测配置
4. 在 eval/run_eval.py 中添加 HumanEval + MBPP 的 commit-aware 评测支持
5. 添加 eval/run_commit_eval.sh 一键评测脚本

## 铁律
- 只改已有文件,可新建 eval/run_commit_eval.sh
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
