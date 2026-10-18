# FraudGraph Intelligence — CUDA Baseline Makefile
# Targets standard modern GPUs (T4 sm_75, V100 sm_70, A100 sm_80)

NVCC ?= nvcc
CXX ?= g++

CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80

NVCC_FLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr -Xcompiler -Wall -I./include $(CUDA_ARCH)

BIN_DIR = bin
BUILD_DIR = build
TARGET = $(BIN_DIR)/fgi_baseline_runner

SRCS = src/fgi_baseline_kernels.cu src/main_baseline.cu
OBJS = $(patsubst src/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean
