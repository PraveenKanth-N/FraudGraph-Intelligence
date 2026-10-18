# Experiment Protocol & Benchmark Standard — FraudGraph Intelligence (FGI)

## 1. Benchmarking Integrity Standard

In strict accordance with Section 3, Section 10, and Section 12 of the Master Engineering Directive:
- **Zero Fabrication:** Benchmark timings, kernel speedups, and hardware throughput metrics must **NEVER** be synthesized, fabricated, or hardcoded.
- **Explicit Unexecuted State:** Any experiment configuration that has not yet been executed on physical hardware must display the explicit status:
  `STATUS: AWAITING_GPU_RUN` or `STATUS: NOT_YET_EXECUTED`.
- **Timing Decoupling:** Every benchmark run must strictly separate:
  1. **Kernel Execution Time:** High-resolution GPU device runtime measured using `cudaEventRecord` / `cudaEventElapsedTime`.
  2. **Memory Transfer Time:** PCIe Host-to-Device (`cudaMemcpyHostToDevice`) and Device-to-Host (`cudaMemcpyDeviceToHost`) transfer latencies.
  3. **End-to-End Latency:** Complete wall-clock duration from graph input ingestion to output tensor return.

---

## 2. Standard Benchmark Execution Protocol

### Step 1: Warm-up Phase
- To eliminate GPU clock ramp-up latency and CUDA context creation overhead, execute **5 warm-up iterations** prior to recording timing events.
- Discard all warm-up measurements from statistics.

### Step 2: Repetition & Statistical Aggregation
- Execute each experiment configuration across **$R \ge 20$ independent repetitions**.
- Record raw observations vector: $T = [t_1, t_2, \dots, t_R]$.
- Compute summary statistics:
  - Arithmetic Mean: $\mu = \frac{1}{R} \sum t_i$
  - Standard Deviation: $\sigma = \sqrt{\frac{1}{R-1} \sum (t_i - \mu)^2}$
  - Median: $t_{50}$
  - 95th Percentile: $t_{95}$
  - 99th Percentile: $t_{99}$

### Step 3: Numerical Verification Gate
Prior to recording speedup, the output tensor $Y_{\text{GPU}}$ must be compared against the CPU reference tensor $Y_{\text{CPU}}$:
$$\max_{i, k} |Y_{\text{GPU}}[i, k] - Y_{\text{CPU}}[i, k]| \le \text{atol} = 10^{-5}$$
If verification fails, the benchmark run is marked as `VERIFICATION_FAILED` and invalid.

---

## 3. Experiment Result JSON Schema

All results are saved under `experiments/results/{experiment_id}.json` adhering to this schema:

```json
{
  "experiment_id": "EXP-20261018-001",
  "timestamp": "2026-10-18T19:30:00Z",
  "graph_tier": "Tier_A",
  "num_nodes": 10000,
  "num_edges": 50000,
  "feature_dim": 32,
  "hardware": {
    "device_name": "NVIDIA GeForce MX450",
    "compute_capability": "7.5",
    "driver_version": "581.83"
  },
  "implementation": "cuda_coalesced_warp_per_node",
  "repetitions": 20,
  "warmup_runs": 5,
  "numerical_verification": {
    "status": "PASSED",
    "max_absolute_error": 3.2e-6,
    "tolerance_threshold": 1.0e-5
  },
  "timing_ms": {
    "h2d_transfer_mean": 1.42,
    "kernel_exec_mean": 0.88,
    "d2h_transfer_mean": 1.35,
    "end_to_end_mean": 3.65,
    "kernel_exec_std": 0.04
  },
  "speedup_vs_cpu_reference": {
    "kernel_only_speedup": 18.4,
    "end_to_end_speedup": 4.2
  }
}
```
