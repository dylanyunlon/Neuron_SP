# DES-LOC + AutoSP 实验对比报告
生成时间: 2026-04-22 08:00:55

## 配置
- 硬件: 2× RTX A6000 (48GB) + 1× H100 NVL (94GB)
- DES-LOC: Kx=32, Ku=96, Kv=192
- Seeds: 42 137 2024
- Steps: 500

## 实验矩阵
| Model | Method | Kx | Runs | OK |
|-------|--------|----|------|----|
| 125M | DDP | 1 | 3 | 3 |
| 125M | DDP | 32 | 1 | 1 |
| 125M | DESLOC | 1 | 2 | 2 |
| 125M | DESLOC | 128 | 2 | 2 |
| 125M | DESLOC | 16 | 2 | 2 |
| 125M | DESLOC | 2 | 2 | 2 |
| 125M | DESLOC | 32 | 20 | 20 |
| 125M | DESLOC | 4 | 2 | 2 |
| 125M | DESLOC | 64 | 2 | 2 |
| 125M | DESLOC | 8 | 2 | 2 |
| 125M | LocalAdam | 32 | 3 | 3 |
| 1.3B | DDP | 1 | 1 | 1 |
| 1.3B | DESLOC | 32 | 1 | 1 |
| 1.3B | LocalAdam | 32 | 1 | 1 |
| 1.7B | DDP | 1 | 1 | 1 |
| 1.7B | DESLOC | 32 | 1 | 1 |
| 350M | DDP | 1 | 3 | 3 |
| 350M | DESLOC | 32 | 3 | 3 |
| 350M | LocalAdam | 32 | 3 | 3 |
| 700M | DDP | 1 | 1 | 1 |
| 700M | DESLOC | 256 | 1 | 1 |
| 700M | DESLOC | 32 | 1 | 1 |
| 700M | DESLOC_nesterov | 32 | 1 | 1 |
| 700M | LocalAdam | 32 | 1 | 1 |

## 训练指标 (从NKI-FA日志提取)
| Model | Method | Kx | AvgLoss | Tok/s/GPU | MFU(%) | CommRed |
|-------|--------|----|---------|-----------|--------|---------|

## 通信量分析
| Method | x_syncs | u_syncs | v_syncs | Total | vs DDP |
|--------|---------|---------|---------|-------|--------|
| DDP | 500 | 500 | 500 | 1500 | 1.0× |
| LocalAdam | 15 | 15 | 15 | 45 | 33.3× |
| DES-LOC | 15 | 5 | 2 | 22 | 68.2× |

## Kx Sweep (RQ1)
| Kx | Runs | OK | CommRed |
|----|------|----|---------|
| 1 | 2 | 2 | 1.0x |
| 2 | 2 | 2 | 4.0x |
| 4 | 2 | 2 | 8.0x |
| 8 | 2 | 2 | 16.0x |
| 16 | 2 | 2 | 32.0x |
| 32 | 2 | 2 | 64.0x |
| 64 | 2 | 2 | 128.0x |
| 128 | 2 | 2 | 256.0x |

