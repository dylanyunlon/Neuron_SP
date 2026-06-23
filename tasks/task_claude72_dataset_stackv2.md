# Claude-72: Wire Stack v2 PR/commit dataset adapter

## 任务
完善 The Stack v2 的 PR/commit 数据适配器。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 datasets/bigcode/the_stack_v2/ 目录
4. 读 datasets/bigcode/the_stack_v2/stackv2_commits.py
5. 完善 megatron_indexed.py: 确保能把 Stack v2 转为 Megatron indexed format
6. 在 load_commits.py 中添加 Stack v2 数据源
7. git add -A && git commit --signoff -m "wire Stack v2 PR/commit adapter into data pipeline"
8. git push origin main

## 铁律
- 只改 datasets/bigcode/ 下的文件
- 作者: dylanyunlon <dogechat@163.com>
