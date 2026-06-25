# Claude-140: Megatron commit dataset 导入验证 + 修复

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP && tree -L 1

## 任务
验证 Megatron-LM/pretrain_commit.py 的所有 import 链路能正确解析，修复断链。

1. cd Megatron-LM && python -c "import ast; ast.parse(open('pretrain_commit.py').read()); print('OK')"
2. 检查 megatron/core/datasets/commit_dataset.py 和 commitpack_streaming_dataset.py:
   - python -c "import ast; ast.parse(open('megatron/core/datasets/commit_dataset.py').read())"
   - python -c "import ast; ast.parse(open('megatron/core/datasets/commitpack_streaming_dataset.py').read())"
3. 检查 megatron/training/tokenizer/commit_tokenizer.py 语法
4. 检查 Megatron-LM/tools/extract_commit_corpus.py 语法
5. 修复所有 ast.parse 失败的文件
6. 验证跨文件 import 一致性:
   - commit_dataset.py 里 from ... import 的符号在目标文件里都存在
   - pretrain_commit.py 的 from megatron.core.datasets.commit_dataset import 的类名匹配
7. 同步检查根目录 data/commit_loader.py 和 datasets/bigcode/commit_packing.py 语法

## 铁律
- 不开新分支，直接 main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- GIT_TOKEN=$(cat .claude-hk-config/GIT_TOKEN 2>/dev/null || echo YOUR_TOKEN)
- 用 dispatch prompt 里提供的 GIT_TOKEN 设置 remote
- push 前: git pull --rebase origin main
- commit: git commit --signoff -m "fix: verify and repair Megatron commit dataset import chain"
