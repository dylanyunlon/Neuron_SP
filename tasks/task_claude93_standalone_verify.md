# Claude-93: standalone 训练管线语法验证

## 任务
验证 standalone 训练管线所有文件语法正确且可 import。

## 具体工作
1. 对每个文件运行 `python3 -c "import ast; ast.parse(open('FILE').read())"`
2. 检查文件列表:
   - run_pretrain.py, models/llama_pretrain.py, data/prepare_commits.py
   - data/commit_loader.py, data/blend_datasets.py, data/__init__.py
   - tools/convert_to_hf.py, tools/count_tokens.py, tools/profile_memory.py
   - pipeline/smoke_test.py, pipeline/train_three_stage.py, pipeline/engine_bridge.py
3. 如果有语法错误，修复
4. 验证 `python3 -c "from models.llama_pretrain import LlamaConfig, Llama"` 是否能 import

## 铁律
- MODIFY EXISTING FILES ONLY (修bug用)
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
