# Claude-97: PCIe 带宽诊断工具

## 任务
新建 tools/check_pcie_bandwidth.py，验证5张卡的实际PCIe带宽。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat REAL_GPU_BENCHMARK.py | head -50 — 读现有 benchmark 参考
3. 新建 tools/check_pcie_bandwidth.py:
   - 对每张GPU做 H2D / D2H bandwidth test (用 torch.cuda.Event timing, 256MB tensor)
   - 对每对GPU做 D2D P2P bandwidth test (如果canAccessPeer)
   - 输出表格: GPU, H2D GB/s, D2H GB/s, PCIe Gen (从nvidia-smi读), expected vs actual
4. 脚本可独立运行: python tools/check_pcie_bandwidth.py

## 铁律
- 可新建 tools/check_pcie_bandwidth.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
