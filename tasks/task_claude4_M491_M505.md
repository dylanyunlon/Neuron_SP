# Claude-4 任务: M491-M505 — 多种子评估 + 统计显著性

Session: Claude-4 (M491-M505) | Base: 待 Claude-3 后确定

## 目标
用 5 个随机种子重复最佳配置, 计算 mean±std, 做 paired t-test。

## 步骤
```bash
for SEED in 42 123 456 789 1024; do
    SEED=$SEED DESLOC_KX=8 DESLOC_KU=16 DESLOC_KV=32 bash run_13B_ags1.sh
done
```

## 结果
- desloc_results/phase4/multi_seed.json
- 包含: mean, std, min, max, paired_ttest_pvalue (vs DDP)
