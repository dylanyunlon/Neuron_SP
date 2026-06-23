# Claude-94: PCIe 带宽诊断脚本

## 任务
写一个 PCIe 带宽验证脚本,确认5张卡的实际 PCIe 带宽。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat REAL_GPU_BENCHMARK.py — 读现有 benchmark
3. 在 tools/ 下新建 check_pcie_bandwidth.py
4. 对每张 GPU 做 H2D/D2H bandwidth test (torch.cuda.Event timing)
5. 做 GPU-GPU P2P bandwidth test (如果 P2P 可用)
6. 输出表格: GPU pair, bandwidth, PCIe gen, expected vs actual

## 铁律
- 可新建 tools/check_pcie_bandwidth.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
