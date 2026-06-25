# Claude-142: eval pipeline 语法验证 + 修复

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

## 任务
验证 eval/ 目录下所有文件能 ast.parse, 修复断链。

1. python -c "import ast; ast.parse(open('eval/run_eval.py').read()); print('OK')"
2. python -c "import ast; ast.parse(open('eval/extract_commitpack_test.py').read()); print('OK')"
3. 检查 eval/eval_config.yaml 格式
4. 检查 run_eval.py 里的 import 链: 确保引用的模块都存在
5. 修复所有 ast.parse 失败和 import 不存在的问题
6. 验证 pipeline/smoke_test.py 语法
7. 验证 tools/generate_sample.py 和 tools/convert_to_hf.py 语法

## 铁律
- 不开新分支,直接 main。push前 git pull --rebase origin main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- commit --signoff -m "fix: eval pipeline syntax verification and import repair"
