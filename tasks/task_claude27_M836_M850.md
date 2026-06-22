# Claude-27: Wire build_neuron_sp_config() into DesLocEngine.__init__()

在 `deepspeed/runtime/desloc_engine.py` 的 `DesLocEngine.__init__()` 中:
1. `cat deepspeed/runtime/hetero_gdn_selective_recompute.py | grep -n "def build_neuron_sp_config"` 找到签名
2. 在 `DesLocEngine.__init__()` 末尾调用 `build_neuron_sp_config()` 获取 `HeteroRecomputeConfig`
3. 用这个 config 对模型的每一层设置 `torch.utils.checkpoint` (A6000 层做 recompute, H100/Blackwell 层不做)
4. 加 print 诊断: 每个 device 的 recompute 策略
