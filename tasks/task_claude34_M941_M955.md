# Claude-34: Wire SharedLocalityCache into activation offload

在 `deepspeed/runtime/hetero_mimo_training_loop.py` 中:
1. `grep -n "class SharedLocalityCache" deepspeed/runtime/hetero_mimo_training_loop.py`
2. 确认 `SharedLocalityCache` 在 `setup_hetero_mimo_training()` 中被正确初始化
3. 确认 cache 用于: (a) 跨 NUMA activation staging (b) checkpoint recompute 的中间结果缓存
4. 设置 cache 大小: NUMA0 用 400GB, NUMA1 用 400GB (总 1.5TB 中留 700GB 给 OS)
