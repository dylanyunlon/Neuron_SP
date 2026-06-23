# C5: Wire HeteroRegistry.register() into all hetero modules

## 目标
给 `deepspeed/runtime/hetero_*.py` 中缺少 `register()` 函数的模块添加标准化的注册接口，让 HeteroRegistry.register_hooks() 能正确发现和注册它们。

## 具体改动
1. 看 `desloc_engine.py` 中 `HeteroRegistry` 类的 `register_hooks()` 方法，理解它期望什么接口
2. 选以下 5 个关键模块添加 `register(engine)` 函数:
   - `hetero_elastic_batch.py`
   - `hetero_h2d_stream_sync.py`
   - `hetero_hybrid_stabilizer.py`
   - `hetero_offload_throttle.py`
   - `hetero_pinned_buffer_guard.py`
3. 每个 register() 函数接收 engine 实例，用 engine 的 hook 机制注册自己

## 文件
改上述 5 个 `deepspeed/runtime/hetero_*.py` 文件
