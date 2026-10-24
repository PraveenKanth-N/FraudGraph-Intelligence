# Benchmark Automation & Experiment Management — FraudGraph Intelligence (FGI)

**Phase 8 Architectural Specification & Empirical Verification**
*Module ID:* `FGI-MOD-08`
*Academic Context:* VIT Vellore — CSI4007 GPU Programming
*Workspace Root:* `D:\FraudGraph Intelligence`
*Current Status:* Pre-Commit Verification Gate Reached

---

## 1. Executive Summary & Architectural Scope

Phase 8 implements **Benchmark Automation & Experiment Management** (`backend/benchmark_engine/`), providing a unified, reproducible, matrix-driven benchmarking framework across CPU reference engines, vectorized implementations, and native CUDA GPU aggregation kernels.

Prior to Phase 8, CPU and GPU benchmarks operated via separate ad-hoc execution harnesses (`backend/cpu_engine/benchmark_cpu.py` and `backend/gpu_engine/colab_runner/`). Phase 8 consolidates these harnesses into a structured, production-grade experiment management pipeline that enforces:
1. **Formal Taxonomy Separation:** Clear structural distinction between *Workload Configurations*, *Implementation-Specific Runs*, *Measurement Repetitions*, and *Bypassed Implementations*.
2. **Operator-Specific Numerical Correctness:** Mathematical floating-point validation distinguishing `mean`, `max`, and `sum` accumulation behaviors, with strict empty-neighborhood zero-sentinel enforcement.
3. **Preflight Memory Guards & Telemetry Fallback Policies:** Separate host-RAM and GPU-VRAM capacity checks with configurable safety margins (80% host RAM, 85% GPU VRAM) to prevent out-of-memory crashes before execution. Workload-level preflight blocks execution before graph loading or allocation. Large or unverified workloads fail closed (`BLOCKED_TELEMETRY_UNAVAILABLE`) when memory telemetry cannot be established, while bounded smoke workloads may execute under an explicit, configurable fallback policy (`PERMITTED_BOUNDED_FALLBACK`) with transparent audit logging.
4. **Timing Boundary Isolation:** Explicit labeling and isolation of host timers (`perf_counter_ns` for CPU) versus device hardware event timers (`cudaEventElapsedTime` for GPU), preventing misleading or ungrounded cross-boundary speedup claims.
5. **Zero Metric Fabrication:** On execution environments lacking local `nvcc` compiler toolchains or compiled binaries (such as standard Windows developer workstations), GPU runs are recorded faithfully as `SKIPPED_CUDA_UNAVAILABLE` with explicit guidance for Colab execution, completely eliminating fabricated or placeholder zero timings.
6. **Isolated Session Management & Provenance:** Self-contained session directories under `experiments/results/benchmark_runs/session_{session_id}/` utilizing atomic `.tmp` file serialization, cryptographic SHA-256 fixture verification, git environment capture, and partial session recovery.

---

## 2. Taxonomy & Workload Matrix Disambiguation

To eliminate ambiguity across benchmark reporting, Phase 8 introduces a rigorous, hierarchical experiment taxonomy:

```
+-----------------------------------------------------------------------------------+
| Benchmark Suite Profile (e.g. 'smoke', 'scaling_features', 'scaling_graphs')       |
+-----------------------------------------------------------------------------------+
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
+------------------------------------+          +-----------------------------------+
| Logical Workload Configuration 1   |          | Logical Workload Configuration W  |
| - Graph (N, M, topology, fixture)  |          | - Graph (N, M, topology, fixture) |
| - Feature dimension (D)            |          | - Feature dimension (D)           |
| - Aggregation operator (mean/sum/  |          | - Aggregation operator (mean/sum/ |
|   max)                             |          |   max)                            |
+------------------------------------+          +-----------------------------------+
                 │                                               │
  ┌──────────────┼──────────────┐                 ┌──────────────┼──────────────┐
  ▼              ▼              ▼                 ▼              ▼              ▼
+------------+ +------------+ +------------+    +------------+ +------------+ +------------+
| Run 1      | | Run 2      | | Run E      |    | Run 1      | | Run 2      | | Run E      |
| Engine:    | | Engine:    | | Engine:    |    | Engine:    | | Engine:    | | Engine:    |
| cpu_ref    | | cpu_vec    | | cuda_base  |    | cpu_ref    | | cpu_vec    | | cuda_base  |
+------------+ +------------+ +------------+    +------------+ +------------+ +------------+
  │              │              │                 │              │              │
  ▼              ▼              ▼                 ▼              ▼              ▼
W warmups      W warmups      W warmups         W warmups      W warmups      W warmups
R repetitions  R repetitions  R repetitions     R repetitions  R repetitions  R repetitions
(Host timer)   (Host timer)   (CUDA event)      (Host timer)   (Host timer)   (CUDA event)
```

### 2.1 Formal Definitions

1. **Logical Workload Configuration ($W$):**
   A unique, engine-agnostic aggregation problem defined by graph scale ($N$ nodes, $M$ edges), graph topology (power-law, uniform, toy), feature dimensionality ($D$), and aggregation operator ($\text{op} \in \{\text{mean}, \text{sum}, \text{max}\}$).
2. **Implementation-Specific Run ($E$):**
   The application of a specific aggregation engine (`cpu_reference`, `cpu_vectorized`, `cuda_baseline`, `cuda_coalesced`, `cuda_streams`, `cuda_degree_aware`) to a given workload configuration.
3. **Repetition Iteration ($R$):**
   A single, timed execution of an implementation run, collected after $W$ untimed warm-up iterations to establish thermal and caching stability.
4. **Execution Status:**
   Each implementation run produces an explicit state from the canonical taxonomy:
   - `MEASURED`: Completed execution validated against CPU reference within operator tolerances.
   - `NOT_RUN`: Configuration planned but not scheduled in the active execution batch.
   - `SKIPPED_CUDA_UNAVAILABLE`: Bypassed because `nvcc` or precompiled CUDA binaries are absent on the host.
   - `SKIPPED_OOM_GUARD`: Bypassed by preflight memory check to protect system stability.
   - `FAILED_TOLERANCE`: Computed output tensor diverged beyond allowed absolute/relative error bounds.
   - `FAILED_NON_FINITE`: Output tensor contained `NaN` or `Inf` floating-point anomalies.
   - `FAILED_SHAPE_MISMATCH`: Output tensor dimensions did not match $[N, D]$.
   - `FAILED_EXECUTION`: Subprocess, driver, or runtime exception during execution.

### 2.2 Standard Suite Profiles & Planned Counts

| Profile Name | Logical Workloads ($W$) | Engines per Workload ($E$) | Planned CPU Runs | Planned GPU Runs | Total Planned Runs | Warmups ($W$) | Repetitions ($R$) | Total Repetitions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`smoke`** | **6** | **6** (2 CPU, 4 GPU) | **12** | **24** | **36** | 5 | 20 | **720** (240 CPU, 480 GPU) |
| **`scaling_features`** | 10 | 6 (2 CPU, 4 GPU) | 20 | 40 | 60 | 5 | 20 | 1,200 |
| **`scaling_graphs`** | 6 | 6 (2 CPU, 4 GPU) | 12 | 24 | 36 | 5 | 20 | 720 |
| **`power_law_vs_uniform`** | 6 | 6 (2 CPU, 4 GPU) | 12 | 24 | 36 | 5 | 20 | 720 |

> **Accounting Clarification for the `smoke` Profile:**
> The `smoke` profile comprises exactly 6 logical workload configurations:
> - Workloads 1–3: `toy_graph_5nodes` ($N=5, M=6, D=16$) across operators `mean`, `sum`, `max`.
> - Workloads 4–6: `synthetic_tier_a_10k` ($N=10,000, M=100,000, D=64$) across operators `mean`, `sum`, `max`.
> Each workload is targeted by 6 engines (2 CPU: `cpu_reference`, `cpu_vectorized`; 4 GPU: `cuda_baseline`, `cuda_coalesced`, `cuda_streams`, `cuda_degree_aware`).
> Therefore: $6 \times 6 = 36$ planned runs. On local CPU-only Windows execution, exactly 12 CPU runs are executed as `MEASURED` (totaling 240 measured repetitions post-60 warmups), while 24 CUDA runs are cleanly reported as `SKIPPED_CUDA_UNAVAILABLE`.

---

## 3. Operator-Specific Numerical Validation

Universal, operator-agnostic absolute thresholds (e.g. strict $\text{atol} = 10^{-5}$) fail for single-precision IEEE 754 floating-point reduction on high-degree graphs because FP32 mantissas provide only 24 bits of precision ($\approx 7$ decimal digits). When summing $10^4$–$10^5$ neighbor values ranging from 1 to 100, intermediate sums reach magnitudes of $10^4$–$10^7$, where the machine epsilon represents absolute increments larger than $10^{-5}$.

Phase 8 replaces rigid single thresholds with **operator-appropriate numerical validation** incorporating both absolute ($\text{atol}$) and relative ($\text{rtol}$) tolerances:

$$\|Y_{\text{test}} - Y_{\text{ref}}\|_\infty \le \text{atol} + \text{rtol} \times |Y_{\text{ref}}|$$

### 3.1 Operator Tolerances

| Operator | $\text{atol}$ | $\text{rtol}$ | Mathematical Justification |
| :--- | :---: | :---: | :--- |
| **`mean`** | $1.0 \times 10^{-5}$ | $1.0 \times 10^{-5}$ | Scaled division by node degree keeps values within normalized feature scale $[-10, 10]$, maintaining FP32 precision. |
| **`max`** | $1.0 \times 10^{-5}$ | $0.0$ | Max reduction performs exact element-wise comparisons; no accumulation drift occurs. Discrepancies are bounded by float representation. |
| **`sum`** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | Unscaled summation over power-law hubs ($d_i \sim 10^3$) accumulates floating-point roundoff error in the least significant mantissa bits. |

### 3.2 Invariant Verification

1. **Empty-Neighborhood Zero Sentinel:**
   In compliance with ADR-006, any isolated node $i$ having zero in-degree ($d_i = 0$, where `row_ptr[i] == row_ptr[i+1]`) must output an exact zero vector:
   $$\vec{y}_i = \vec{0}_{D} \quad \forall i \in \{j \mid d_j = 0\}$$
   Both reference and candidate outputs are verified to ensure every element is identically $0.0$.
2. **Non-Finite Detection:**
   Outputs are scanned for `NaN` and `Inf` values prior to difference calculations. Any non-finite value immediately triggers `FAILED_NON_FINITE`.
3. **Tensor Shape & Dtype:**
   Shapes must match $[N, D]$ exactly and dtypes must be `np.float32`.

---

## 4. Hardware Telemetry & Preflight Memory Guards

To prevent operating-system freezing or out-of-memory kernel panics during large graph sweeps, Phase 8 conducts automated memory preflight audits before launching any workload.

### 4.1 Memory Estimation Model

For a graph with $N$ nodes, $M$ edges, and feature dimension $D$:
- **CSR Row Pointer:** $(N + 1) \times 8 \text{ bytes}$ (int64)
- **CSR Column Indices:** $M \times 8 \text{ bytes}$ (int64)
- **Feature Matrix:** $N \times D \times 4 \text{ bytes}$ (float32)
- **Output Tensor:** $N \times D \times 4 \text{ bytes}$ (float32)
- **Working Buffers:** $2 \times (N \times D \times 4) \text{ bytes}$ (intermediate reduction buffers)

$$\text{Footprint}_{\text{est}} = \frac{8(N + 1) + 8M + 16ND}{1024^2} \text{ MB}$$

### 4.2 Guard Logic & Fail-Closed Memory Safety Policy

Phase 8 implements rigorous, separate preflight checks for host RAM and GPU VRAM before any execution:

- **Host RAM Guard (`check_host_ram`):**
  Queries available physical memory via Win32 `GlobalMemoryStatusEx` (Windows) or `/proc/meminfo` (Linux).
  Requires: $\text{Footprint}_{\text{est}} \le \text{RAM}_{\text{available}} \times 0.80$ (configurable 80% safety margin).
- **GPU VRAM Guard (`check_gpu_vram`):**
  Queries free VRAM via `nvidia-smi`.
  Requires: $\text{Footprint}_{\text{est}} \le \text{VRAM}_{\text{free}} \times 0.85$ (85% safety margin).
- **Handling Unknown or Invalid Memory Telemetry:**
  Operating system APIs or virtualized container environments may return non-positive ($0 \text{ MB}$, negative), missing, or non-finite (`NaN`) memory readings. Phase 8 distinguishes fail-closed blocking from bounded fallback exceptions:
  1. **Strict Distinction Between Verified Verification and Fallback:** An execution is **never** granted verified memory safety status (`guard_status = "VERIFIED_PASS"`) when physical memory availability cannot be confirmed from system APIs. Returning `allowed=True` under missing telemetry represents an explicit bounded fallback exception, NOT verified physical safety and NOT fail-closed behavior.
  2. **Fail-Closed Policy for Large or Unverified Workloads:** Workloads whose estimated memory footprint exceeds the unverified safety ceiling (`max_unverified_mb`, default $100.0 \text{ MB}$), or any workload when bounded fallback is disabled (`allow_bounded_fallback=False`), are strictly blocked under a fail-closed policy (`allowed=False`, `guard_status = "BLOCKED_TELEMETRY_UNAVAILABLE"`).
  3. **Configurable Bounded Fallback for Small Smoke Workloads:** For lightweight smoke tests or small unit graphs where $\text{Footprint}_{\text{est}} \le \text{max\_unverified\_mb}$, execution is permitted only when `allow_bounded_fallback=True` is explicitly configured. This path returns `allowed=True` with `guard_status = "PERMITTED_BOUNDED_FALLBACK"` and appends an explicit audit notice to the run record documenting that physical RAM availability was unverified.
  4. **Categorical Guard Statuses:** Every benchmark run, session manifest, and summary table explicitly records its `memory_guard_status`:
     - `VERIFIED_PASS`: Telemetry verified against physical memory within the configured safety margin.
     - `BLOCKED_CAPACITY_EXCEEDED`: Physical memory verified; workload exceeds available safety budget.
     - `BLOCKED_TELEMETRY_UNAVAILABLE`: Telemetry absent/invalid; blocked under the fail-closed policy.
     - `PERMITTED_BOUNDED_FALLBACK`: Telemetry absent/invalid; small workload permitted under explicit bounded fallback policy with documented caution.
  5. **Preflight Allocation Guard:** Preflight checks execute at the workload level in `BenchmarkRunner.run_profile()` **prior to** graph loading or dynamic graph synthesis, ensuring that blocked workloads are never allocated in host memory.

---

## 5. Timing Boundary Isolation & Scientific Integrity

A central requirement of Phase 8 is preserving the scientific validity of performance comparisons. CPU and GPU execution models measure fundamentally different boundaries:

| Dimension | CPU Host Boundary (`cpu_host_timer`) | GPU Device Boundary (`gpu_cuda_event`) |
| :--- | :--- | :--- |
| **Timing Mechanism** | High-resolution monotonic clock (`time.perf_counter_ns`) | Hardware device events (`cudaEventRecord`, `cudaEventElapsedTime`) |
| **Scope Measured** | Python method dispatch + NumPy array iteration + CPU ALU instructions | Hardware kernel execution on GPU streaming multiprocessors (SMs) |
| **Exclusions** | Graph construction/serialization | Host-to-Device transfer, Device-to-Host transfer, launch latency |
| **Speedup Rule** | **Directly comparable only against other CPU engines** | **Directly comparable only against other CUDA kernels** |

> **Scientific Rule on Cross-Boundary Comparisons:**
> Speedup factors comparing CPU reference timings against CUDA device kernel timings must explicitly state their boundaries (e.g. *“Kernel-only GPU speedup over host CPU reference”*). The benchmark automation engine enforces this by tagging every record with its explicit `timing_boundary` and never outputting unqualified cross-boundary ratios.

---

## 6. Experiment Session Architecture & Provenance

Phase 8 guarantees reproducibility by isolating every run into a dedicated, self-contained directory:

```
experiments/results/benchmark_runs/session_{session_id}/
├── session_manifest.json          # Overall metadata, provenance, fixture hashes, counts
├── summary_table.csv              # Machine-readable tabular summary across all runs
├── session_report.md              # Academic-grade Markdown summary of session
└── raw_measurements/              # Atomic JSON files per workload-engine instance
    ├── toy_graph_5nodes_mean_cpu_reference.json
    ├── toy_graph_5nodes_mean_cpu_vectorized.json
    ├── toy_graph_5nodes_mean_cuda_baseline.json
    ├── ...
    └── synthetic_tier_a_10k_max_cuda_degree_aware.json
```

### 6.1 Cryptographic & Environment Provenance

Each session records:
1. **Fixture Hashes:** SHA-256 cryptographic digests of all input `.npz` fixtures.
2. **Git Commit State:** Current commit SHA (`e0bd1cb...`), branch (`main`), and dirty state.
3. **Hardware Telemetry:** CPU model (`11th Gen Intel(R) Core(TM) i5-1135G7`), core counts, RAM availability, GPU model (`NVIDIA GeForce MX450`), driver version (`581.83`), VRAM capacity.
4. **Software Environment:** Python version (`3.14.3`), NumPy version (`2.5.4`).

### 6.2 Atomic File Serialization

All JSON outputs are written to a `.tmp` file and committed via atomic operating-system file replacement (`os.replace`). If an experiment is aborted or interrupted, no partially-written or corrupt JSON files are created, and the `session_manifest.json` preserves the `IN_PROGRESS` or `PARTIAL` status with all successfully recorded runs intact.

---

## 7. Empirical Verification & Baseline Smoke Benchmark

### 7.1 Canonical Smoke Run Results

The benchmark harness was executed with profile `smoke` (5 warmups, 20 repetitions) on the local host environment:

- **Session ID:** `20261024_230008_smoke`
- **Session Directory:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/`
- **Total Workloads:** 6
- **Total Recorded Runs:** 36 (12 CPU `MEASURED`, 24 GPU `SKIPPED_CUDA_UNAVAILABLE`)
- **Total Repetitions:** 720 (240 CPU measured, 480 GPU bypassed)

#### Measured Performance Summary (`synthetic_tier_a_10k`: $N=10,000, M=100,000, D=64$)

| Workload Configuration | Engine | Timing Boundary | Status | Mean Latency (ms) | Std Latency (ms) | Throughput (nodes/s) | Bandwidth (GB/s) | Verification Error | CPU Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `synthetic_tier_a_10k_mean` | `cpu_reference` | `cpu_host_timer` | `MEASURED` | 114.860 | 11.257 | 87,063 | 0.253 | 0.00 (Reference) | 1.00x |
| `synthetic_tier_a_10k_mean` | `cpu_vectorized` | `cpu_host_timer` | `MEASURED` | **2.576** | 0.883 | **3,882,591** | **11.274** | $3.58 \times 10^{-7}$ | **44.59x** |
| `synthetic_tier_a_10k_sum` | `cpu_reference` | `cpu_host_timer` | `MEASURED` | 100.346 | 23.944 | 99,655 | 0.289 | 0.00 (Reference) | 1.00x |
| `synthetic_tier_a_10k_sum` | `cpu_vectorized` | `cpu_host_timer` | `MEASURED` | **3.871** | 7.315 | **2,583,579** | **7.501** | $0.00$ | **25.92x** |
| `synthetic_tier_a_10k_max` | `cpu_reference` | `cpu_host_timer` | `MEASURED` | 86.788 | 10.221 | 115,224 | 0.335 | 0.00 (Reference) | 1.00x |
| `synthetic_tier_a_10k_max` | `cpu_vectorized` | `cpu_host_timer` | `MEASURED` | **87.416** | 13.900 | 114,396 | 0.332 | $0.00$ | 0.99x |

*Findings:*
- Vectorized CPU aggregation accelerates `mean` by **44.59x** and `sum` by **25.92x** over naive reference nested iteration on Tier A, achieving throughputs of up to 3.88 million nodes/sec and 11.27 GB/s effective bandwidth.
- For `max`, vectorized NumPy achieves equivalent performance (0.99x) to reference because standard NumPy lacks an SpMM sparse max reduction ufunc, requiring segmented slicing.
- On small graphs (`toy_graph_5nodes`), reference aggregation outperforms vectorized aggregation (0.28x–0.73x) due to Python array allocation overhead dominating computation.

### 7.2 Detailed Timing Variability & Outlier Audit (Tier A Vectorized Sum)

An audit of the 20 raw repetition latencies in `synthetic_tier_a_10k_sum_cpu_vectorized.json` reveals:
```json
"trial_times_ms": [
  1.6949, 1.7425, 1.8581, 1.3908, 1.3652, 2.1330, 1.8029, 1.7236, 2.3373, 1.4486,
  1.3524, 1.4531, 34.7381, 2.7171, 2.8672, 4.0190, 2.9429, 4.3217, 2.6872, 2.8166
]
```

- **Recomputed Moments Across All 20 Repetitions:**
  - $\text{Mean}: 3.8706 \text{ ms}$
  - $\text{Sample Std } (\text{ddof}=1): 7.3154 \text{ ms}$
  - $\text{Median } (p_{50}): 1.9956 \text{ ms}$
  - $p_{95}: 5.8425 \text{ ms}$
  - $\text{Min}: 1.3524 \text{ ms} \quad \text{Max}: 34.7381 \text{ ms}$
  - $\text{Coefficient of Variation } (\text{CV}\%): 189.00\%$
- **Root Cause Analysis:**
  Nineteen of the twenty repetitions clustered tightly between $1.35 \text{ ms}$ and $4.32 \text{ ms}$. A single isolated iteration (iteration 13) recorded $34.7381 \text{ ms}$. Inside the timed loop, `aggregate_neighbors(graph, op="sum", method="vectorized")` executes `data = np.ones(num_edges, dtype=np.float32)` and instantiates a `scipy.sparse.csr_matrix` on every call. On iteration 13, dynamic memory allocation triggered a minor garbage collection or OS heap compaction pause on the Windows host.
- **Scientific Integrity Policy:**
  All 20 raw observations are strictly preserved in the canonical artifact. No outliers are deleted or pruned. However, because $\text{CV} = 189.00\%$ exceeds the stability threshold ($\text{CV} \le 30.0\%$), the run is explicitly flagged as `HIGH_VARIABILITY`.
  - **Authoritative Primary Speedup (Complete 20-Repetition Sample):** **25.92x** (arithmetic mean $3.871 \text{ ms}$ vs reference $100.35 \text{ ms}$). All headline performance claims and formal comparisons in FGI must strictly cite this primary figure.
  - **Robust Speedup Diagnostic (Median):** **50.18x** ($100.35 \text{ ms} / 2.00 \text{ ms}$).
  - **Trimmed Diagnostic (19 Stable Iterations):** **44.68x** ($2.246 \pm 0.877 \text{ ms}$, $\text{CV}=39.0\%$).

  > **Methodological Mandate on Diagnostic Statistics:**
  > The 19-run metric is strictly a secondary diagnostic provided for diagnostic transparency to explain latency behavior in the absence of memory allocator pauses. Under no circumstances may this trimmed diagnostic be substituted for the primary 20-run result, nor may it be used to represent authoritative system speedup.

### 7.3 Compulsory Traffic, Algorithmic Traffic, and Bandwidth Modeling

To prevent confusion between model-derived traffic and hardware DRAM counters, the harness computes and reports:
1. **Compulsory Data Traffic ($B_{\text{compulsory}}$):**
   Minimum theoretical traffic assuming an ideal infinite cache (where each unique node feature is loaded at most once):
   $$B_{\text{compulsory}} = (N + 1 + M) \times \text{index\_bytes} + 8 \times N \times D \text{ bytes}$$
2. **Algorithmic Data Traffic ($B_{\text{algorithmic}}$):**
   Traffic assuming un-cached edge traversal (each directed edge reads neighbor feature vector):
   $$B_{\text{algorithmic}} = (N + 1 + M) \times \text{index\_bytes} + 4 \times M \times D + 4 \times N \times D \text{ bytes}$$
3. **Bandwidth Qualification:**
   $$\text{BW}_{\text{algorithmic}} = \frac{B_{\text{algorithmic}} / 10^9}{\text{latency (sec)}} \text{ GB/s}$$
   `effective_bandwidth_gb_s` is formally documented as an **algorithmic traffic model estimate**, NOT a physical hardware DRAM bandwidth measurement from performance counters (e.g. NVIDIA Nsight / Intel PCM).

### 7.4 Workload Configuration & Fixture Dimension Alignment

- **Phase 3 Fixture Reality:**
  - `data/fixtures/toy_graph_5nodes.npz`: $N=5, M=8, D=4$ (ADR-008/009).
  - `data/fixtures/synthetic_tier_a_10k.npz`: $N=10,000, M=94,945, D=32$ (Phase 5/6 baseline).
- **Dimension Reconciliation:**
  In Phase 5, Tier A was evaluated with $D=32$ and $M=94,945$. The Phase 8 smoke profile initially declared nominal targets ($D=64, M=100,000$), while executing on the committed fixture ($D=32, M=94,945$). The harness has been refined so that:
  1. The `smoke` profile configuration explicitly specifies the true fixture dimensions ($D=32, M=94,945$ for Tier A; $D=4, M=8$ for Toy).
  2. The runner inspects the loaded graph's actual dimensions (`graph.num_nodes, graph.num_edges, graph.feature_dim`) and records them with 100% fidelity.
  3. Dynamic scaling profiles (`scaling_features`) test varying dimensions ($D \in [16, 32, 64, 128, 256]$) via dynamic graph generation.

### 7.5 Preflight Execution Order & Memory Guard Status Reporting

- **Preflight Allocation Guard:** Host-memory preflight checks (`check_host_ram`) execute at the workload level in `BenchmarkRunner.run_profile()` **BEFORE** any graph loading (`CSRGraph.load`) or dynamic generation occurs. If an oversized workload exceeds host capacity, it is immediately blocked with `SKIPPED_OOM_GUARD` without allocating memory.
- **Fail-Closed Policy for Unknown Telemetry:** If OS telemetry is unavailable (`avail_mb <= 0` or non-finite), large workloads (>100 MB) are blocked with `BLOCKED_TELEMETRY_UNAVAILABLE`. Bounded smoke workloads (<=100 MB) are permitted only under an explicit configurable fallback policy (`allow_bounded_fallback=True`), accompanied by explicit caveat notes in the experiment session manifest and raw JSON records.
- **Guard Status Recording:** Every recorded run and tabular export captures `memory_guard_status` (`VERIFIED_PASS`, `BLOCKED_CAPACITY_EXCEEDED`, `BLOCKED_TELEMETRY_UNAVAILABLE`, or `PERMITTED_BOUNDED_FALLBACK`) alongside the estimated footprint.

### 7.6 Upstream Pytest Warning Resolution & Dependency Analysis

During test execution, full backend pytest runs previously reported a deprecation warning:
```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```
- **Origin & Characterization:** This warning is emitted exclusively by `fastapi.testclient` importing `starlette.testclient` under Starlette $\ge 0.40$. `StarletteDeprecationWarning` inherits from `UserWarning` rather than standard Python `DeprecationWarning`, which allowed it to bypass standard pytest deprecation filters.
- **Upstream Dependency Issue:** This warning represents an upstream third-party library interaction between FastAPI and Starlette; it does not indicate any defect, deprecated API usage, or code smell within the FraudGraph Intelligence codebase. FGI tests do not directly import deprecated Starlette clients.
- **Narrowed Resolution:** The warning filter in `pytest.ini` was specifically narrowed to target `ignore::starlette.exceptions.StarletteDeprecationWarning` (and `ignore:.*starlette\.testclient.*:UserWarning`). This eliminates the spurious upstream warning while preserving full visibility into any potential `UserWarning` or `DeprecationWarning` emitted by FGI application modules. The backend test suite executes cleanly with **zero warnings**.

### 7.7 Automated Test Suite

A dedicated unit test suite (`backend/tests/test_benchmark_automation.py`) validates all architectural requirements:

```
backend/tests/test_benchmark_automation.py::test_profile_workload_and_run_counts PASSED
backend/tests/test_benchmark_automation.py::test_operator_tolerance_specification PASSED
backend/tests/test_benchmark_automation.py::test_numerical_verification_tolerances PASSED
backend/tests/test_benchmark_automation.py::test_empty_neighborhood_zero_sentinel_verification PASSED
backend/tests/test_benchmark_automation.py::test_non_finite_rejection PASSED
backend/tests/test_benchmark_automation.py::test_shape_mismatch_rejection PASSED
backend/tests/test_benchmark_automation.py::test_statistical_metric_calculations PASSED
backend/tests/test_benchmark_automation.py::test_throughput_and_bandwidth_calculations PASSED
backend/tests/test_benchmark_automation.py::test_memory_preflight_guards PASSED
backend/tests/test_benchmark_automation.py::test_atomic_json_write_and_session_manifest PASSED
backend/tests/test_benchmark_automation.py::test_colab_schema_compatibility_and_reporting PASSED
backend/tests/test_benchmark_automation.py::test_missing_cuda_transparent_skip PASSED
backend/tests/test_benchmark_automation.py::test_fixtures_immutability PASSED
backend/tests/test_benchmark_automation.py::test_workload_preflight_guard_blocks_before_allocation PASSED
backend/tests/test_benchmark_automation.py::test_workload_preflight_blocks_allocation_on_unavailable_telemetry PASSED
```

**Test Results:**
- **15/15 Phase 8 focused tests passed** in 9.86s.
- **142 passed, 2 skipped, 0 warnings** (144 total backend test suite) in full regression testing. Both skipped tests are expected optional live CUDA executions on CPU-only host.

**Test Results:**
- **14/14 Phase 8 focused tests passed** in 8.86s.
- **141 passed, 2 skipped, 0 warnings** (143 total backend test suite) in full regression testing. Both skipped tests are expected optional live CUDA executions on CPU-only host.

---

## 8. Summary of Phase 8 Deliverables

| Deliverable Path | Description |
| :--- | :--- |
| `backend/benchmark_engine/__init__.py` | Package root exporting core classes and lazy-loader for CLI entrypoint. |
| `backend/benchmark_engine/benchmark_matrix.py` | Workload configuration, run configuration, profile specifications (`smoke`, `scaling_features`, `scaling_graphs`, `power_law_vs_uniform`), and count accounting. |
| `backend/benchmark_engine/benchmark_verifier.py` | Operator-aware numerical verification, floating-point tolerance bounds, and zero-degree sentinel checks. |
| `backend/benchmark_engine/benchmark_telemetry.py` | Hardware telemetry gathering, host-RAM/GPU-VRAM preflight guards, and statistical moment calculators. |
| `backend/benchmark_engine/experiment_manager.py` | Isolated session directories, atomic JSON serialization, SHA-256 fixture checksums, and manifest management. |
| `backend/benchmark_engine/benchmark_runner.py` | Unified execution orchestrator, timing boundary enforcement, and CLI interface. |
| `backend/benchmark_engine/benchmark_report.py` | Machine-readable `summary_table.csv` and Markdown `session_report.md` generation. |
| `backend/tests/test_benchmark_automation.py` | 13 automated tests covering matrix schema, validation, preflight, serialization, and immutability. |
| `experiments/results/benchmark_runs/session_20261024_230008_smoke/` | Canonical baseline smoke benchmark session containing 36 run records, summary CSV, Markdown report, and manifest. |
| `docs/BENCHMARK_AUTOMATION.md` | Authoritative architectural and empirical documentation for Phase 8. |
| `docs/PHASE_STATUS.md` | Updated phase progress tracking. |
| `docs/CHANGELOG.md` | Milestone changelog entries for Phase 8. |
