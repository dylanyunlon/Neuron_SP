# Claude-86: 端到端集成测试

## 任务
扩展 `pipeline/smoke_test.py`，验证所有 hetero 模块的 wiring 是否正确。

## 具体工作
1. `cat pipeline/smoke_test.py` 先读
2. 添加测试用例:
   - test_hetero_registry_discovers_all: 检查 HeteroRegistry 能发现所有带 register() 的模块
   - test_desloc_engine_has_all_hooks: 验证 DesLocEngine 实例上有 neuron_sp_config, hetero_scheduler, fp32_grad_manager 等
   - test_commit_sequence_packer: 验证 CommitSequencePacker 不跨 commit 边界
   - test_hetero_batch_sampler_proportional: 验证 HeteroBatchSampler 按显存比例分配
   - test_pcie_p2p_communicator_mock: 用 mock 验证 PCIeP2PCommunicator 接口
3. `python3 -c "import ast; ast.parse(open('pipeline/smoke_test.py').read())"` 验证

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
