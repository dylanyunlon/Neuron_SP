# C9: Wire commit tokenizer 的特殊 token 到 Megatron tokenizer

## 目标
改 `Megatron-LM/megatron/training/tokenizer/commit_tokenizer.py`，确保 commit 特殊 token (<COMMIT>, <MSG>, <FILE>, <ADD>, <DEL>, <CTX>) 正确注册到 Megatron 的 tokenizer 系统，并在 `pipeline/unified_tokenizer.py` 中可用。

## 具体改动
1. 在 `commit_tokenizer.py` 中检查特殊 token 的注册逻辑
2. 在 `pipeline/unified_tokenizer.py` 的 `build_megatron_tokenizer()` 中，确保调用 commit_tokenizer 的 `add_special_tokens()` 
3. 添加一个 `validate_commit_tokens(tokenizer)` 函数，验证所有 commit token 都正确注册

## 文件
改 `Megatron-LM/megatron/training/tokenizer/commit_tokenizer.py` 和 `pipeline/unified_tokenizer.py`
