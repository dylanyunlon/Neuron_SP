#!/bin/bash
# =============================================================================
# DES-LOC Benchmark Suite - Launch Script
# =============================================================================
# Usage: ./run_desloc_benchmark.sh [OPTIONS]
#
# Options:
#   --all           Run all 7 figures (default)
#   --figure N      Run specific figure (1-7)
#   --gpu           Run GPU benchmarks (requires CUDA)
#   --output DIR    Output directory
#   --help          Show this help
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
OUTPUT_DIR="./desloc_benchmark_results"
RUN_ALL=true
RUN_GPU=false
FIGURES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --figure)
            RUN_ALL=false
            FIGURES+=("figure$2")
            shift 2
            ;;
        --gpu)
            RUN_GPU=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all           Run all 7 figures (default)"
            echo "  --figure N      Run specific figure (1-7)"
            echo "  --gpu           Run GPU benchmarks"
            echo "  --output DIR    Output directory"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ██████╗ ███████╗███████╗      ██╗      ██████╗  ██████╗   ║"
echo "║     ██╔══██╗██╔════╝██╔════╝      ██║     ██╔═══██╗██╔════╝   ║"
echo "║     ██║  ██║█████╗  ███████╗█████╗██║     ██║   ██║██║        ║"
echo "║     ██║  ██║██╔══╝  ╚════██║╚════╝██║     ██║   ██║██║        ║"
echo "║     ██████╔╝███████╗███████║      ███████╗╚██████╔╝╚██████╗   ║"
echo "║     ╚═════╝ ╚══════╝╚══════╝      ╚══════╝ ╚═════╝  ╚═════╝   ║"
echo "║                                                               ║"
echo "║     Desynced Low-Communication Optimizer Benchmark Suite      ║"
echo "║                         v1.0.0                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Check required packages
python3 -c "import numpy, matplotlib" 2>/dev/null || {
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install numpy matplotlib --quiet
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
echo -e "${GREEN}Starting DES-LOC Benchmark Suite...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ "$RUN_ALL" = true ]; then
    echo -e "${BLUE}Running all 7 figures...${NC}"
    python3 FULL_PATCH.py --output "$OUTPUT_DIR"
else
    echo -e "${BLUE}Running figures: ${FIGURES[*]}${NC}"
    python3 FULL_PATCH.py --figures "${FIGURES[@]}" --output "$OUTPUT_DIR"
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.pdf" | head -20
echo ""
echo -e "${YELLOW}View summary: cat $OUTPUT_DIR/SUMMARY.md${NC}"
