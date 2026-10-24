# FraudGraph Intelligence — CUDA Makefile (Baseline & Phase 6 Optimized Kernels)
# Targets Google Colab NVIDIA Tesla T4 (Turing sm_75, CUDA 13.0 compatible)

NVCC ?= nvcc
CXX ?= g++

CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75

NVCC_FLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr -Xcompiler -Wall -I./include $(CUDA_ARCH)

BIN_DIR = bin
BUILD_DIR = build

TARGET_BASELINE = $(BIN_DIR)/fgi_baseline_runner
TARGET_COALESCED = $(BIN_DIR)/fgi_coalesced_runner
TARGET_STREAMS = $(BIN_DIR)/fgi_streams_runner
TARGET_DEGREE_AWARE = $(BIN_DIR)/fgi_degree_aware_runner

OBJS_BASELINE = $(BUILD_DIR)/fgi_baseline_kernels.o $(BUILD_DIR)/main_baseline.o
OBJS_COALESCED = $(BUILD_DIR)/fgi_coalesced_kernels.o $(BUILD_DIR)/main_coalesced.o
OBJS_STREAMS = $(BUILD_DIR)/fgi_coalesced_kernels.o $(BUILD_DIR)/main_streams.o
OBJS_DEGREE_AWARE = $(BUILD_DIR)/fgi_degree_aware_kernels.o $(BUILD_DIR)/main_degree_aware.o

all: $(TARGET_BASELINE) $(TARGET_COALESCED) $(TARGET_STREAMS) $(TARGET_DEGREE_AWARE)

$(TARGET_BASELINE): $(OBJS_BASELINE) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_COALESCED): $(OBJS_COALESCED) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_STREAMS): $(OBJS_STREAMS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_DEGREE_AWARE): $(OBJS_DEGREE_AWARE) | $(BIN_DIR)
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
