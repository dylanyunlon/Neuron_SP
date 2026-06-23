# Claude-101: run_pretrain.py --dry-run 模式

## 任务
在 run_pretrain.py 中添加 --dry-run flag。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -n "argparse\|add_argument" | head -30
3. 添加 --dry-run argument
4. 在 run_standalone() 中: 如果 dry_run, 只做 model init + 3 steps + 打印:
   - per-GPU peak memory
   - tokens/sec
   - estimated total training days (基于 total_tokens / tokens_per_sec)
5. dry-run 后 sys.exit(0)

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
