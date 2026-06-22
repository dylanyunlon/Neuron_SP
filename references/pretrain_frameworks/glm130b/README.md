# GLM-130B 参考文件 (智谱AI / 清华大学)
上游: github.com/THUDM/GLM-130B (2022)
模型: GLM-130B → ChatGLM 系列

## 文件说明
- `initialize.py` — 模型初始化 + 分布式配置
- `benchmark.py` — 性能基准测试
- `evaluation_model.py` — 评估用模型封装
- `quantization_layers.py` — INT4/INT8 量化层
- `configs/` — 模型配置 (含 V100 版本)

## DES-LOC 参考点
- GLM 的 bidirectional + causal 混合预训练目标
- INT4 量化: 在 A6000 上可用量化推理腾出显存给训练
- SwissArmyTransformer 架构抽象
- configs/model_glm_130b_v100.sh: V100 级别 GPU 的适配经验
  (A6000 Ampere 算力类似 V100×2, 可参考)
