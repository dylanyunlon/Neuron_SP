# Claude-143: 论文 Related Work 引用补充

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

## 任务
补充 FAUST_nips2026/main.tex §2 Related Work 的引用。

1. 读 §2 (line 169-274) 现有内容
2. 在已有 \cite{} 基础上, 补充以下方向的引用 (在 \bibliography 里加 bibtex):
   - DiLoCo (Douillard et al., 2024): distributed local SGD for LLM
   - FedAvg (McMahan et al., 2017): federated averaging baseline
   - Espresso (Wang et al., 2024): heterogeneous-aware parallel
   - SWARM (Ryabinin et al., 2023): decentralized heterogeneous training
   - Sequence parallelism: DeepSpeed-Ulysses (Jacobs et al., 2023), Ring Attention (Liu et al., 2023)
3. 每个新引用在正文加1-2句描述, 并在 references 加 bibtex entry
4. 验证: pdflatex 不会因缺 bib 报错 (检查 \bibliographystyle 和 .bib 文件)

## 铁律
- 不开新分支,直接 main。push前 git pull --rebase origin main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- commit --signoff -m "paper: update Related Work with DiLoCo, SWARM, Ulysses refs"
