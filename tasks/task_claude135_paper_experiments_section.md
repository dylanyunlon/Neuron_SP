# Claude-135: NeurIPS Paper — Experiments Section Draft

## Context
Paper at `FAUST_nips2026/main.tex`. Need experiments section.

## Task
Write Section 5 (Experiments) in LaTeX:
1. **Setup**: ags1 cluster (H100 NVL 93GB + 2×A6000 48GB), PCIe Gen4, no NVLink, PyTorch 2.7.1
2. **Baselines**: Naive DDP (equal batch), FSDP, single-GPU H100
3. **DES-LOC configs**: ZERO3_HETERO with proportional grad_accum (H100:22, A6000:1)
4. **Metrics**: Training throughput (tok/s), MFU, time-to-convergence, peak memory
5. **Tables**: Table 1 (throughput comparison), Table 2 (memory breakdown per GPU tier)
6. **Figures**: Fig 3 (loss curve overlay), Fig 4 (GPU utilization timeline)
7. Use placeholder numbers with `\todo{}` markers — real numbers will come from Claude-134

## Files
- `FAUST_nips2026/main.tex` — Section 5

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
