# C10: Wire Blackwell RTX PRO 6000 tier 到 TierDiscovery

## 目标
改 `deepspeed/runtime/desloc_engine.py` 的 `TierDiscovery` 和 `TierClass`，添加 Blackwell (SM 12.0) GPU 的支持，让框架能正确识别和利用 RTX PRO 6000 Blackwell Max-Q。

## 具体改动
1. 在 `TierClass` enum 中添加 `BLACKWELL` tier
2. 在 `TierDiscovery._classify_gpu()` 中添加 SM 12.x 的识别逻辑:
   - SM 12.0, VRAM > 90GB → BLACKWELL tier
   - bf16_tflops ≈ 300 (Max-Q thermal)
3. 在 `PartitionSolver` 中考虑 Blackwell 卡的算力（300 TFLOPS BF16），更新层分配策略
4. 更新 docstring 的硬件规格，加入 GPU1/GPU4 的 Blackwell 信息

## 文件
只改 `deepspeed/runtime/desloc_engine.py`
