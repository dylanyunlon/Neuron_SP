# Claude-73: Wire CommitPackFT (高质量子集) for Stage 3

## 任务
添加 CommitPackFT 2GB 高质量数据集支持，用于 Stage 3 指令微调。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 datasets/bigcode/DATASETS.md 和 samples/commitpackft_python_100.jsonl
4. 在 load_commits.py 中添加 CommitPackFT 格式解析
5. 添加 instruction template 格式化 (把 commit message 作为 instruction)
6. 确保 HeteroBatchSampler 能区分 Stage 3 的小数据集
7. git add -A && git commit --signoff -m "wire CommitPackFT for stage 3 instruction tuning"
8. git push origin main

## 铁律
- 只改 datasets/bigcode/ 下的文件
- 作者: dylanyunlon <dogechat@163.com>
