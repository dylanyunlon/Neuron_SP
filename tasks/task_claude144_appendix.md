# Claude-144: 论文 Appendix 完善

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

## 任务
完善 FAUST_nips2026/main.tex 的 Appendix 部分。

1. 读 §C Hardware Topology Details (line ~1757)
2. 用 configs/7b_5gpu.yaml 的数据填充:
   - GPU 型号、VRAM、SM 版本、PCIe Gen、NUMA 归属
   - PP layer split: [4,8,8,4,8]
   - 拓扑矩阵 (NODE/SYS/PIX)
3. 读 §D Additional Ablation Results (line ~1810)
4. 用 experiments/convergence_report.json 的 summary_table 数据填充消融表:
   - Kx sweep 结果 (Kx=1,2,4,8,16 的 loss 和 throughput)
   - 通信缩减比例
5. 确保所有 table 的 \begin{table} 有 \label 和 \caption
6. 验证: grep -c "\\\\todo" FAUST_nips2026/main.tex 应该为0(除了\newcommand定义行)

## 铁律
- 不开新分支,直接 main。push前 git pull --rebase origin main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- commit --signoff -m "paper: complete Appendix C/D with hardware topology and ablations"
