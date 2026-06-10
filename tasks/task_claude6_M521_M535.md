# Claude-6 任务: M521-M535 — 论文数据填充

Session: Claude-6 (M521-M535) | Base: 待 Claude-5 后确定

## 目标
将实验结果填入 FAUST_nips2026/ 论文。

## 步骤
1. 从 desloc_results/phase4/ 提取最佳数据
2. 填入论文主表: DDP vs LocalAdam vs DES-LOC
3. 填入消融表: Kx sweep 结果
4. 填入多种子表: mean±std
5. 更新 SOTA 对比 (如有)
6. 确保 pdflatex 编译通过

## 数据完整性闸门
- 每个数字必须指向 desloc_results/ 内的具体 JSON 文件
- source 字段格式: "desloc_results/phase4/xxx.json#key"
- 禁止 "attached by user" 手工转录
