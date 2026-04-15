# DES-LOC Benchmark Suite - Quick Start Guide

## 📦 Package Contents

| File | Description |
|------|-------------|
| `FULL_PATCH.py` | Main benchmark code (6,247 lines) - All 7 figures |
| `run_desloc_benchmark.sh` | Launch script for CPU/simulation mode |
| `run_gpu_benchmark.sh` | Distributed GPU training script |
| `deploy_to_yotta_a100.sh` | Deploy to Yotta A100 server |
| `submit_slurm.sh` | SLURM cluster submission script |
| `desloc_benchmark_complete.zip` | Complete archive with results |

## 🚀 Quick Start

### Option 1: Local CPU Mode (Simulation)
```bash
# Run all 7 figures
./run_desloc_benchmark.sh --all

# Run specific figure
./run_desloc_benchmark.sh --figure 1
./run_desloc_benchmark.sh --figure 5
```

### Option 2: Yotta A100 GPU Server
```bash
# Deploy to server
./deploy_to_yotta_a100.sh

# SSH to server and run
ssh -i ~/.ssh/yotta_key.pem -p 30000 user@c3b02cv0zag2y-m.proxy.yottalabs.ai
cd /home/user/desloc_benchmark
./run_gpu_benchmark.sh
```

### Option 3: SLURM Cluster
```bash
# Submit job
sbatch submit_slurm.sh
```

### Option 4: Direct Python
```bash
# Run all figures
python FULL_PATCH.py --figures all --output ./results

# Run specific figures
python FULL_PATCH.py --figures figure1 figure5 figure7 --output ./results
```

## 📊 Figures Implemented

| Figure | Description | Key Result |
|--------|-------------|------------|
| Figure 1 | Rosenbrock Toy Problem | DES-LOC convergence visualization |
| Figure 2 | Momentum Change Rates | β1 ablation, half-life analysis |
| Figure 3 | Sync Period Ablation | Kx, Ku, Kv sensitivity |
| Figure 4 | Communication Reduction | 2× vs Local Adam |
| Figure 5 | Billion-Scale Training | 170× reduction vs DDP |
| Figure 6 | Outer Optimizer | Nesterov vs Averaging |
| Figure 7 | Muon Integration | 1.5× byte savings |

## 📁 Output Structure

```
desloc_benchmark_results/
├── SUMMARY.md           # Experiment summary
├── figure1/
│   ├── figure1_rosenbrock.png
│   ├── figure1_rosenbrock.pdf
│   └── figure1_data_*.json
├── figure2/
│   └── ...
├── figure3/
├── figure4/
├── figure5/
├── figure6/
└── figure7/
```

## 🔧 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- (Optional) PyTorch 2.0+ for GPU benchmarks
- (Optional) CUDA 12.0+ for distributed training

## 📖 Paper Reference

DES-LOC: Desynced Low Communication Adaptive Optimizers for Training Foundation Models
Authors: Alex Iacob et al.
Conference: ICLR 2026

## 🆘 Troubleshooting

**Import Error**: Install dependencies
```bash
pip install numpy matplotlib
```

**GPU Not Found**: Check CUDA installation
```bash
nvidia-smi
```

**Permission Denied**: Make scripts executable
```bash
chmod +x *.sh
```
