# Benchmark Experiment Protocol & Reproducibility Standard — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Document Status:** Phase 2 Authoritative Specification  

---

## 1. Benchmarking Integrity Standards

In strict accordance with the Master Engineering Directive:
- **Zero Fabrication:** Performance speedups, latency values, and throughput metrics must **NEVER** be hardcoded, faked, or estimated.
- **Explicit Measurement States:** Every metric reported in exported JSON/CSV files and UI dashboards must carry one of three explicit state labels:
  - `MEASURED`: Empirically recorded from an actual execution on physical GPU/CPU hardware.
  - `SAMPLE`: Illustrative reference data used strictly for mock frontend layout testing.
  - `NOT_RUN`: Benchmark configuration defined but not yet executed on hardware.

---

## 2. Experimental Factor Matrix

To thoroughly test Hypotheses $H_1$, $H_2$, and $H_3$, experiments will execute across the following orthogonal factors:

| Factor | Parameter Space | Values |
| :--- | :--- | :--- |
| **Compute Implementations** | Engine Architecture | 1. `cpu_reference` (NumPy vector-parallel)<br>2. `cuda_baseline_thread_per_node`<br>3. `cuda_optimized_warp_coalesced`<br>4. `cuda_optimized_shared_memory` |
| **Graph Scale ($N, M$)** | Node and Edge Counts | - **Tier A (Small / Debug):** $N = 10,000$, $M = 50,000$<br>- **Tier B (Medium / Workload):** $N = 50,000$, $M = 350,000$<br>- **Tier C (Large / Stress):** $N = 200,000$, $M = 1,800,000$ |
| **Feature Width ($D$)** | Node Feature Dimensionality | $D \in \{16, 32, 64, 128\}$ |
| **Degree Distributions** | Graph Topology Profile | 1. **Uniform Random:** Erdős–Rényi graph (low variance in degree)<br>2. **Power-Law Skew:** Barabási–Albert ($\gamma \approx 2.5$) |
| **Aggregation Operators** | Mathematical Operator | `mean` (primary), `sum`, `max` |

---

## 3. High-Precision Timing Methodology

### 3.1 Kernel Timing via CUDA Events
To eliminate host CPU scheduling noise and PCIe asynchronous queue latency, GPU kernel execution time must be measured exclusively with hardware CUDA events:
```cpp
cudaEvent_t start_event, stop_event;
cudaEventCreate(&start_event);
cudaEventCreate(&stop_event);

cudaEventRecord(start_event, stream);
fgi_aggregator_coalesced_kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop_event, stream);
cudaEventSynchronize(stop_event);

float kernel_time_ms = 0.0f;
cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
```

### 3.2 Decoupled PCIe Transfer Timing
Separate timers measure the data movement across the PCIe bus:
- $t_{\text{H2D}}$: Host-to-Device transfer of `row_ptr`, `col_idx`, and `features`.
- $t_{\text{kernel}}$: Pure GPU execution time.
- $t_{\text{D2H}}$: Device-to-Host transfer of the aggregated tensor $Y$.
- $t_{\text{e2e}}$: Total end-to-end elapsed time ($t_{\text{H2D}} + t_{\text{kernel}} + t_{\text{D2H}} + \text{synchronization}$).

### 3.3 Warm-Up & Repetition Protocol
1. **Warm-Up Runs:** Prior to taking measurements, execute **5 consecutive warm-up iterations** on the target GPU to stabilize GPU core clock frequencies (P-states) and warm up instruction caches.
2. **Repetition Count:** Execute each test configuration for $R = 20$ recorded iterations.
3. **Statistical Metrics:**
   - Sample Mean: $\bar{t} = \frac{1}{R} \sum_{i=1}^R t_i$
   - Standard Deviation: $\sigma = \sqrt{\frac{1}{R-1} \sum_{i=1}^R (t_i - \bar{t})^2}$
   - Median: $t_{50}$
   - 95th Percentile: $t_{95}$
   - Throughput (Edges/sec): $\text{Throughput} = \frac{M}{\bar{t}_{\text{kernel}} \times 10^{-3}}$
   - Effective Memory Bandwidth (GB/s):
     $$\text{BW}_{\text{eff}} = \frac{(N+1) \times 4 + M \times 4 + M \times D \times 4 + N \times D \times 4}{\bar{t}_{\text{kernel}} \times 10^{-3} \times 10^9}$$

---

## 4. Failure & Out-of-Memory (OOM) Handling

The GPU memory footprint is strictly bounded prior to launching kernels:
$$\text{Memory Required} = \Big( (N+1) \times 4 + M \times 4 + 2 \times N \times D \times 4 \Big) \text{ bytes}$$

If $\text{Memory Required} \ge 0.85 \times \text{VRAM}_{\text{free}}$:
1. Local execution halts immediately without attempting an unsafe allocation.
2. The experiment is logged with status `EXCEEDS_LOCAL_VRAM_COLAB_REQUIRED`.
3. The graph is marked for execution via the Google Colab cloud runner (T4 / A100 with 16 GB+ VRAM).

---

## 5. Machine-Readable JSON Result Schema

All benchmark results are automatically written to `experiments/results/{experiment_id}.json`:

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "EXP-20261018-001",
  "measurement_state": "NOT_RUN",
  "timestamp": null,
  "hardware": {
    "device_name": "NVIDIA GeForce MX450",
    "compute_capability": "7.5",
    "driver_version": "581.83",
    "total_vram_mb": 2048.0
  },
  "graph_configuration": {
    "tier": "Tier_A",
    "num_nodes": 10000,
    "num_edges": 50000,
    "feature_dim": 32,
    "degree_distribution": "power_law",
    "power_law_gamma": 2.5
  },
  "benchmark_parameters": {
    "implementation": "cuda_optimized_warp_coalesced",
    "operator": "mean",
    "warmup_iterations": 5,
    "recorded_repetitions": 20
  },
  "timing_results_ms": {
    "kernel_mean": null,
    "kernel_std": null,
    "kernel_p50": null,
    "kernel_p95": null,
    "h2d_transfer_mean": null,
    "d2h_transfer_mean": null,
    "end_to_end_mean": null
  },
  "performance_metrics": {
    "edges_per_second": null,
    "effective_bandwidth_gbps": null,
    "speedup_vs_cpu_kernel": null,
    "speedup_vs_cpu_e2e": null
  },
  "numerical_verification": {
    "status": "PENDING_EXECUTION",
    "max_absolute_error": null,
    "tolerance_threshold": 1.0e-5,
    "nan_detected": false
  }
}
```

---

## 6. Phase 4 CPU Benchmark Harness Implementation

### 6.1 Harness Module (`backend/cpu_engine/benchmark_cpu.py`)
- **Module Path:** [`backend/cpu_engine/benchmark_cpu.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/benchmark_cpu.py)
- **Invocation:**
  ```powershell
  .\.venv\Scripts\python.exe backend/cpu_engine/benchmark_cpu.py --warmup 3 --repeat 10
  ```
- **Measurement Protocol:**
  - Timer: `time.perf_counter_ns` (monotonic nanosecond clock).
  - Graph loading isolated from aggregation timing.
  - Metrics collected: `mean`, `std`, `median_p50`, `p95`, `min`, `max`, `nodes_per_second`, `edges_per_second`.
  - Automated output export to `experiments/results/{experiment_id}.json` with status `MEASURED`.
- **Baseline Measurements Recorded (Host: Intel Core i5-1135G7 @ 2.40GHz, 8 Cores):**
  - `toy_graph_5nodes.npz`:
    - Mean Reference: $0.033 \text{ ms}$ ($245,927 \text{ edges/s}$) | Status: `PASSED`
    - Mean Vectorized: $0.036 \text{ ms}$ ($219,118 \text{ edges/s}$) | Status: `PASSED`
  - `synthetic_tier_a_10k.npz` ($N=10,000, M=94,945, D=32$):
    - Mean Reference: $133.177 \text{ ms}$ ($712,923 \text{ edges/s}$) | Status: `PASSED`
    - Mean Vectorized (SpMM): $6.325 \text{ ms}$ ($15,011,945 \text{ edges/s}$) | Status: `PASSED`
    - Sum Reference: $166.877 \text{ ms}$ ($568,953 \text{ edges/s}$) | Status: `PASSED`
    - Sum Vectorized (SpMM): $3.624 \text{ ms}$ ($26,196,494 \text{ edges/s}$) | Status: `PASSED`
    - Max Reference: $167.173 \text{ ms}$ ($567,945 \text{ edges/s}$) | Status: `PASSED`

