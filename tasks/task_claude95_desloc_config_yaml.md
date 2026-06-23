# Claude-95: DES-LOC 配置 YAML schema 完善

## 任务
完善 experiments/configs/ 下的训练配置。

## 具体工作
1. `cat experiments/configs/7b_ags1_desloc.yaml` 先读
2. 确保 YAML 包含:
   - model section: layers, hidden_size, heads, vocab_size, seq_len
   - desloc section: Kx, Ku, Kv, loc_cache_size_gb, sync_strategy
   - gpu_tiers section: 每个 GPU 的 tier, micro_batch_size, precision
   - data section: dataset paths, tokenizer, packing策略
   - training section: lr, warmup, total_steps, eval_every, checkpoint_every
   - optimizer section: AdamW params, weight_decay, gradient_clipping
3. 添加 `experiments/configs/1b_ags1_desloc.yaml` 用于快速验证（1B模型, 小数据）
4. 验证所有 YAML 格式正确: `python3 -c "import yaml; yaml.safe_load(open('FILE'))"`

## 铁律
- 可在 experiments/configs/ 创建新文件
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
