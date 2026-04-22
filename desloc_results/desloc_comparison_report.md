# DES-LOC + AutoSP 实验对比报告
生成时间: 2026-04-22 08:05:17

## 配置
- 硬件: 2× RTX A6000 (48GB) + 1× H100 NVL (94GB)
- DES-LOC: Kx=32, Ku=96, Kv=192
- Seeds: 42 137 2024
- Steps: 500

## 实验矩阵
| Model | Method | Kx | Runs | OK |
|-------|--------|----|------|----|
| 125M | DDP | 1 | 6 | 6 |
| 125M | DDP | 32 | 2 | 2 |
| 125M | DESLOC | 128 | 4 | 4 |
| 125M | DESLOC | 1 | 4 | 4 |
| 125M | DESLOC | 16 | 4 | 4 |
| 125M | DESLOC | 2 | 4 | 4 |
| 125M | DESLOC | 32 | 40 | 40 |
| 125M | DESLOC | 4 | 4 | 4 |
| 125M | DESLOC | 64 | 4 | 4 |
| 125M | DESLOC | 8 | 4 | 4 |
| 125M | LocalAdam | 32 | 6 | 6 |
| 1.3B | DDP | 1 | 2 | 2 |
| 1.3B | DESLOC | 32 | 2 | 2 |
| 1.3B | LocalAdam | 32 | 2 | 2 |
| 1.7B | DDP | 1 | 2 | 2 |
| 1.7B | DESLOC | 32 | 2 | 2 |
| 350M | DDP | 1 | 6 | 6 |
| 350M | DESLOC | 32 | 6 | 6 |
| 350M | LocalAdam | 32 | 6 | 6 |
| 700M | DDP | 1 | 2 | 2 |
| 700M | DESLOC | 256 | 2 | 2 |
| 700M | DESLOC | 32 | 2 | 2 |
| 700M | DESLOC_nesterov | 32 | 2 | 2 |
| 700M | LocalAdam | 32 | 2 | 2 |

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
| 1 | 4 | 4 | 1.0x |
| 2 | 4 | 4 | 4.0x |
| 4 | 4 | 4 | 8.0x |
| 8 | 4 | 4 | 16.0x |
| 16 | 4 | 4 | 32.0x |
| 32 | 4 | 4 | 64.0x |
| 64 | 4 | 4 | 128.0x |
| 128 | 4 | 4 | 256.0x |

