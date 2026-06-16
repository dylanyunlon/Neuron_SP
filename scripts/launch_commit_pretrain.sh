#!/bin/bash
# ===========================================================================
# Neuron_SP: Launch commit pretraining on heterogeneous GPUs
# ===========================================================================
#
# Hardware:
#   GPU0: RTX A6000 48GB (PCIe Gen1→fix to Gen4!)
#   GPU1: RTX A6000 48GB (PCIe Gen1→fix to Gen4!)
#   GPU2: H100 NVL 96GB  (PCIe Gen5)
#   CPU:  2× AMD EPYC 9354 (128 threads)
#   RAM:  ~1.5TB
#
# Model: 7B GPT (32 layers, hidden=4096, heads=32)
# Data:  GitHub commits formatted with commit_tokenizer.py
#
# Pipeline Parallel layout:
#   PP stage 0 (GPU0/A6000): layers 0-5   (6 layers)
#   PP stage 1 (GPU1/A6000): layers 6-11  (6 layers)
#   PP stage 2 (GPU2/H100):  layers 12-31 (20 layers)
#
# ===========================================================================

set -euo pipefail

# ===== Paths =====
NEURON_SP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MEGATRON_DIR="${NEURON_SP_DIR}/Megatron-LM"
DATA_DIR="${NEURON_SP_DIR}/data/commit_corpus"
CKPT_DIR="${NEURON_SP_DIR}/checkpoints/commit_7b"
LOG_DIR="${NEURON_SP_DIR}/logs"
TOKENIZER_MODEL="${NEURON_SP_DIR}/tokenizer/commit_sp.model"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# ===== Model config (7B) =====
NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
FFN_HIDDEN_SIZE=11008  # SwiGLU: 8/3 * hidden_size, rounded
SEQ_LENGTH=2048

# ===== Training config =====
MICRO_BATCH_SIZE=1          # per-GPU micro batch (conservative for 48GB A6000)
GLOBAL_BATCH_SIZE=8          # effective batch size across all GPUs
TRAIN_ITERS=100000           # ~14B tokens at 8*2048 tokens/step
LR=3e-4
MIN_LR=3e-5
WARMUP_ITERS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# ===== Parallelism =====
TP_SIZE=1                    # No tensor parallelism (no NVLink)
PP_SIZE=3                    # Pipeline parallel across 3 GPUs
DP_SIZE=1                    # Data parallel = world_size / (TP * PP)

# ===== Heterogeneous PP layer assignment =====
# H100 gets 60% of layers (20), each A6000 gets 20% (6)
# Megatron's --num-layers-per-virtual-pipeline-stage or custom split
PP_LAYER_SPLIT="6,6,20"

# ===== Commit-specific =====
FIM_RATE=0.15                # 15% Fill-in-the-Middle
CTX_LOSS_WEIGHT=0.5          # Loss weight for context lines

# ===== GPU visibility =====
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ===== NCCL tuning for PCIe (no NVLink) =====
export NCCL_P2P_DISABLE=1           # Disable P2P (PCIe Gen1 is broken)
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# ===== Launch =====
echo "============================================="
echo " Neuron_SP: Commit Pretraining (7B)"
echo "============================================="
echo " GPUs:  GPU0(A6000) + GPU1(A6000) + GPU2(H100)"
echo " Model: ${NUM_LAYERS}L / ${HIDDEN_SIZE}H / ${NUM_ATTN_HEADS}A"
echo " Data:  ${DATA_DIR}"
echo " PP:    ${PP_SIZE} stages (${PP_LAYER_SPLIT})"
echo " Batch: micro=${MICRO_BATCH_SIZE} global=${GLOBAL_BATCH_SIZE}"
echo " FIM:   ${FIM_RATE}"
echo "============================================="

DISTRIBUTED_ARGS=(
    --nproc_per_node 3
    --nnodes 1
    --master_addr localhost
    --master_port 29500
)

GPT_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}

    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}

    --pipeline-model-parallel-size ${PP_SIZE}
    --tensor-model-parallel-size ${TP_SIZE}

    --train-iters ${TRAIN_ITERS}
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-warmup-iters ${WARMUP_ITERS}
    --lr-decay-style cosine
    --weight-decay ${WEIGHT_DECAY}
    --clip-grad ${GRAD_CLIP}
    --adam-beta1 0.9
    --adam-beta2 0.95

    --bf16
    --initial-loss-scale 65536
    --use-distributed-optimizer

    --log-interval 10
    --eval-interval 1000
    --eval-iters 50
    --save-interval 5000
    --save ${CKPT_DIR}
    --load ${CKPT_DIR}

    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --swiglu
    --normalization RMSNorm
    --disable-bias-linear
)

COMMIT_ARGS=(
    --commit-data-path ${DATA_DIR}
    --fim-rate ${FIM_RATE}
    --ctx-loss-weight ${CTX_LOSS_WEIGHT}
)

# Vocab size: 32000 base + 43 special = 32043, pad to 32064 (multiple of 64)
DATA_ARGS=(
    --vocab-size 32064
    --tokenizer-type NullTokenizer
    --vocab-size 32064
)

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${MEGATRON_DIR}/pretrain_commit.py" \
    "${GPT_ARGS[@]}" \
    "${COMMIT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/pretrain_commit_$(date +%Y%m%d_%H%M%S).log"
