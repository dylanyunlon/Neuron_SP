# Claude-102: commit special tokens 完善

## 任务
确保 unified_tokenizer.py 的 commit special tokens 完整。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat pipeline/unified_tokenizer.py — 通读
3. 确保这些 special tokens 存在且正确注册:
   <|commit_start|> <|commit_end|> <|diff_start|> <|diff_end|>
   <|file_start|> <|file_end|> <|msg_start|> <|msg_end|>
4. 确保 tokenizer.encode / decode 往返测试通过
5. 确保 CommitSequencePacker 用这些 token 做边界检测

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
