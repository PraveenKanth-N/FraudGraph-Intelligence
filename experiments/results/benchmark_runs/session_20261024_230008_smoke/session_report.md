# FraudGraph Intelligence — Phase 8 Benchmark Session Report

**Session ID:** `20261024_230008_smoke`
**Profile:** `smoke`
**Execution Status:** `COMPLETED`
**Timestamp:** `2026-10-24T23:15:52.305307`

---

## 1. System & Environment Metadata

- **Git Commit:** `e0bd1cbd94057cb2e558fa345324b386a3003f1c` (branch: `main`)
- **Host Platform:** Windows-11-10.0.26200-SP0
- **Python Version:** 3.14.3 | **NumPy:** 2.5.4
- **CPU Model:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 logical cores)
- **Host RAM:** 16,150 MB total (4,556 MB available at preflight)
- **GPU Model:** NVIDIA GeForce MX450 (Driver: 581.83, VRAM: 2,048 MB)

---

## 2. Benchmark Execution Summary

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Total Recorded Runs** | `36` | Total workload-engine instances recorded |
| **Measured Runs** | `12` | Completed executions meeting numerical tolerances |
| **Skipped Runs** | `24` | Bypassed due to toolchain or OOM guards |
| **Failed Runs** | `0` | Tolerance, non-finite, or execution errors |

---

## 3. Measured Performance Results

| Workload | Operator | Engine | Timing Boundary | Latency (ms) | Throughput (nodes/s) | Bandwidth (GB/s) | Verif Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `synthetic_tier_a_10k_max` | `max` | `cpu_reference` | `cpu_host_timer` | 86.788 ± 10.221 | 115,223.6 | 0.335 | `MEASURED` |
| `synthetic_tier_a_10k_max` | `max` | `cpu_vectorized` | `cpu_host_timer` | 87.416 ± 13.900 | 114,395.7 | 0.332 | `MEASURED` |
| `synthetic_tier_a_10k_mean` | `mean` | `cpu_reference` | `cpu_host_timer` | 114.860 ± 11.257 | 87,062.7 | 0.253 | `MEASURED` |
| `synthetic_tier_a_10k_mean` | `mean` | `cpu_vectorized` | `cpu_host_timer` | 2.576 ± 0.883 | 3,882,590.5 | 11.275 | `MEASURED` |
| `synthetic_tier_a_10k_sum` | `sum` | `cpu_reference` | `cpu_host_timer` | 100.346 ± 23.944 | 99,655.1 | 0.289 | `MEASURED` |
| `synthetic_tier_a_10k_sum` | `sum` | `cpu_vectorized` | `cpu_host_timer` | 3.871 ± 7.315 | 2,583,578.8 | 7.503 | `MEASURED` |
| `toy_graph_5nodes_max` | `max` | `cpu_reference` | `cpu_host_timer` | 0.060 ± 0.002 | 82,918.7 | 0.013 | `MEASURED` |
| `toy_graph_5nodes_max` | `max` | `cpu_vectorized` | `cpu_host_timer` | 0.040 ± 0.002 | 125,628.1 | 0.020 | `MEASURED` |
| `toy_graph_5nodes_mean` | `mean` | `cpu_reference` | `cpu_host_timer` | 0.035 ± 0.001 | 142,450.1 | 0.023 | `MEASURED` |
| `toy_graph_5nodes_mean` | `mean` | `cpu_vectorized` | `cpu_host_timer` | 0.138 ± 0.083 | 36,127.2 | 0.006 | `MEASURED` |
| `toy_graph_5nodes_sum` | `sum` | `cpu_reference` | `cpu_host_timer` | 0.035 ± 0.002 | 141,242.9 | 0.023 | `MEASURED` |
| `toy_graph_5nodes_sum` | `sum` | `cpu_vectorized` | `cpu_host_timer` | 0.073 ± 0.013 | 68,870.5 | 0.011 | `MEASURED` |

---

## 4. Skipped Implementation Runs

The following planned engine implementations were safely bypassed:

| Workload | Engine | Skip State | Reason |
| :--- | :--- | :--- | :--- |
| `synthetic_tier_a_10k_max` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_max` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_max` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_max` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_mean` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_mean` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_mean` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_mean` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_sum` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_sum` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_sum` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `synthetic_tier_a_10k_sum` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_max` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_max` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_max` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_max` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_mean` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_mean` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_mean` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_mean` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_sum` | `cuda_baseline` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_baseline_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_sum` | `cuda_coalesced` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_coalesced_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_sum` | `cuda_degree_aware` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_degree_aware_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |
| `toy_graph_5nodes_sum` | `cuda_streams` | `SKIPPED_CUDA_UNAVAILABLE` | CUDA execution unavailable on current host. Missing nvcc compiler or precompiled binary 'fgi_streams_runner.exe'. Dispatched benchmarks must be executed via the Google Colab workflow ('backend/gpu_engine/colab_runner/'). |

---

## 5. Timing Boundary & Scientific Integrity Disclosures

1. **Boundary Isolation:** CPU measurements use high-resolution host monotonic timers (`time.perf_counter_ns`), capturing CPU kernel execution and intermediate Python function dispatch. GPU measurements (when executed on CUDA devices) use device-side CUDA events (`cudaEventElapsedTime`) capturing pure kernel execution time.
2. **Zero Metric Fabrication:** If CUDA compilers (`nvcc`) or compiled binaries are unavailable on the local host, GPU runs are explicitly recorded as `SKIPPED_CUDA_UNAVAILABLE`. No synthetic or zero latency numbers are accepted as measurements.
3. **Tolerance Enforcement:** All measurements undergo operator-specific validation against the Phase 4 CPU reference. Any deviation exceeding operator tolerances causes immediate status rejection (`FAILED_TOLERANCE`).

Generated automatically by FraudGraph Intelligence (FGI-MOD-08).
