# Table 2: Computational Hardware & Runtime Environments

This table documents the dual-environment architecture used in FraudGraph Intelligence: the local Windows development/API runtime host and the remote Google Colab GPU benchmarking environment.

| Component / Specification | Local Host Environment (Development & API) | Remote GPU Environment (CUDA Benchmarks) |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Enterprise (10.0.26100 AMD64) | Ubuntu Linux 22.04 LTS (x86_64) |
| **Host CPU Architecture** | AMD64 x86_64 Multi-Core Processor | Intel Xeon Processor (2 vCPUs @ 2.20 GHz) |
| **Host System Memory** | 16.0 GB DDR4 | 12.7 GB Available System RAM |
| **Python Runtime** | Python 3.14.3 (64-bit MSC v.1944) | Python 3.10.12 (CPython x86_64) |
| **Key Python Libraries** | NumPy 2.5.4, SciPy 1.15.1, Pandas 3.0.6, Matplotlib 3.11.2, FastAPI 0.115.0, Uvicorn 0.32.0, Pytest 9.1.1 | NumPy 1.25.2, SciPy 1.11.4, Pandas 2.0.3, Matplotlib 3.7.1, Scikit-learn 1.2.2 |
| **Node.js & Tooling** | Node.js v22.14.0, npm 10.9.2, Vite 6.1.0/6.4.4, TypeScript 5.7.2 | Not Applicable (Headless C++/Python runner) |
| **Discrete GPU Accelerator** | None Detected / Unavailable (`local_gpu_detected: None`) | **NVIDIA Tesla T4** |
| **GPU Microarchitecture** | Not Applicable | Turing (TU104) |
| **Compute Capability** | Not Applicable | **SM 7.5** |
| **Streaming Multiprocessors** | 0 SMs | **40 SMs** (2,560 CUDA Cores, 320 Tensor Cores) |
| **GPU Dedicated VRAM** | 0 MB | **14,912.7 MB (14.6 GB GDDR6)** |
| **Peak Memory Bandwidth** | Host DRAM: ~38.4 GB/s | **320.0 GB/s** (256-bit bus @ 5000 MHz) |
| **NVIDIA Driver Version** | None Installed | **535.104.05** |
| **CUDA Toolchain (NVCC)** | Not Available (`NO_GPU_DETECTED_COLAB_READY`) | **CUDA 12.2** (`V12.2.140`) |
| **Timing Instrumentation** | Python `time.perf_counter_ns` (Host Wall-Clock) | Hardware `cudaEvent_t` (`cudaEventRecord`, `cudaEventSynchronize`) |
| **Role in Project** | Backend FastAPI server, Dashboard UI, Unit tests, CPU Reference, Synthetic Generator | Isolated GPU kernel benchmarks, warp coalescing experiments, degree-aware profiling |
