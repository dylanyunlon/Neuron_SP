# Claude-96: 论文 Related Work + Introduction 填充

## 任务
填充 `FAUST_nips2026/main.tex` 的 Section 1 (Introduction) 和 Section 2 (Related Work)。

## 具体工作
1. `cat FAUST_nips2026/main.tex` 先读
2. Introduction:
   - 问题: 异构GPU集群训练效率低（同步等待最慢设备）
   - 方案: DES-LOC 分解同步 + LOC缓存 + AutoSP
   - 贡献: (1) 分解同步理论, (2) 异构缓存系统, (3) 5GPU 7B实验
3. Related Work:
   - Local SGD: McMahan et al. 2017, Lin et al. 2020, Ortiz et al. 2021
   - Heterogeneous training: PipeDream, HetPipe, FlexPipe
   - Sequence parallelism: Megatron-SP, USP, DeepSpeed-Ulysses
   - 说明DES-LOC如何区别于上述工作

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
