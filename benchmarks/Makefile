# Makefile for bench_swiglu_ln
#
# Usage:
#   make                    # default SM 8.6
#   make SM=90              # target H100
#   make SM=120             # target Blackwell
#   make clean

SM       ?= 86
NVCC     ?= nvcc
CXXFLAGS := -O3 -std=c++17 -arch=sm_$(SM) --use_fast_math
INCLUDES := -I../csrc/includes -I../csrc/hetero_reduce

FUSED_SRC := ../csrc/hetero_reduce/fused_swiglu_ln.cu
BENCH_SRC := bench_swiglu_ln.cu
TARGET    := bench_swiglu_ln

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(BENCH_SRC) $(FUSED_SRC)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f $(TARGET)
