#!/bin/bash
# =============================================================================
# Deploy DES-LOC Benchmark to Yotta A100 Server
# =============================================================================
# Prerequisites:
#   1. SSH key at C:\Users\44797\0415_ssh\private_key.pem (Windows)
#      or ~/.ssh/yotta_key.pem (Linux/Mac)
#   2. Yotta pod running
#
# Usage: ./deploy_to_yotta_a100.sh
# =============================================================================

set -e

# Yotta server configuration
YOTTA_HOST="c3b02cv0zag2y-m.proxy.yottalabs.ai"
YOTTA_PORT=30000
YOTTA_USER="user"

# Detect OS and set key path
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]; then
    # Windows
    SSH_KEY="/c/Users/44797/0415_ssh/private_key.pem"
else
    # Linux/Mac
    SSH_KEY="${HOME}/.ssh/yotta_key.pem"
fi

# Files to deploy
FILES=(
    "FULL_PATCH.py"
    "run_desloc_benchmark.sh"
    "run_gpu_benchmark.sh"
)

# Remote directory
REMOTE_DIR="/home/user/desloc_benchmark"

echo "========================================="
echo "Deploying DES-LOC Benchmark to Yotta A100"
echo "========================================="
echo "Host: $YOTTA_HOST:$YOTTA_PORT"
echo "User: $YOTTA_USER"
echo "Key: $SSH_KEY"
echo "Remote: $REMOTE_DIR"
echo "========================================="

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found at $SSH_KEY"
    echo ""
    echo "Please ensure your SSH key is at the correct location:"
    echo "  Windows: C:\\Users\\44797\\0415_ssh\\private_key.pem"
    echo "  Linux/Mac: ~/.ssh/yotta_key.pem"
    exit 1
fi

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY -p $YOTTA_PORT"

# Create remote directory
echo ""
echo "Creating remote directory..."
ssh $SSH_OPTS ${YOTTA_USER}@${YOTTA_HOST} "mkdir -p $REMOTE_DIR"

# Upload files
echo ""
echo "Uploading files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Uploading $file..."
        scp $SSH_OPTS "$file" ${YOTTA_USER}@${YOTTA_HOST}:${REMOTE_DIR}/
    else
        echo "  Warning: $file not found, skipping"
    fi
done

# Set permissions
echo ""
echo "Setting permissions..."
ssh $SSH_OPTS ${YOTTA_USER}@${YOTTA_HOST} "chmod +x $REMOTE_DIR/*.sh"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
ssh $SSH_OPTS ${YOTTA_USER}@${YOTTA_HOST} << 'REMOTE_CMD'
cd /home/user/desloc_benchmark
pip install numpy matplotlib --quiet 2>/dev/null || pip install numpy matplotlib
REMOTE_CMD

# Verify GPU
echo ""
echo "Verifying GPU..."
ssh $SSH_OPTS ${YOTTA_USER}@${YOTTA_HOST} "nvidia-smi --query-gpu=name,memory.total --format=csv"

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "To run the benchmark, SSH to the server and execute:"
echo ""
echo "  ssh -i $SSH_KEY -p $YOTTA_PORT ${YOTTA_USER}@${YOTTA_HOST}"
echo "  cd $REMOTE_DIR"
echo "  ./run_desloc_benchmark.sh --all"
echo ""
echo "Or for GPU benchmark:"
echo "  ./run_gpu_benchmark.sh"
echo ""
