#!/bin/bash
#SBATCH --job-name=desloc_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/desloc_%j.out
#SBATCH --error=logs/desloc_%j.err

# =============================================================================
# DES-LOC Benchmark - SLURM Submission Script
# =============================================================================
# For HPC clusters with SLURM scheduler
# Usage: sbatch submit_slurm.sh
# =============================================================================

set -e

# Create log directory
mkdir -p logs

# Load modules (adjust for your cluster)
module load cuda/12.0 2>/dev/null || true
module load pytorch/2.0 2>/dev/null || true

# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "========================================="
echo "DES-LOC Benchmark on SLURM"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_NTASKS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "========================================="

# Show GPU info
nvidia-smi

# Run benchmark
echo ""
echo "Starting DES-LOC benchmark..."
srun python FULL_PATCH.py --output ./slurm_results_${SLURM_JOB_ID}

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo "Results: ./slurm_results_${SLURM_JOB_ID}"
