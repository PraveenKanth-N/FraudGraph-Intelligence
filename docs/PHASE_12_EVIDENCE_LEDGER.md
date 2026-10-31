# Phase 12 Evidence Ledger — FraudGraph Intelligence (FGI)

**Project:** FraudGraph Intelligence (FGI)  
**Research Title:** *Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation*  
**Course:** CSI4007 — GPU Programming Capstone  
**Institution:** Vellore Institute of Technology (VIT), Vellore Campus  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline HEAD:** `8afa28cf1172c006f0242f530267e8cc4466f9d6` (Branch `main`)  
**Audit Timestamp:** 2026-10-25T17:55:00+05:30  
**Audit Authority:** Phase 12 Research Finalization & Scientific Evidence Verification  

---

## 1. Executive Summary & Audit Mandate

In strict accordance with the Phase 12 Master Execution Protocol, this Evidence Ledger establishes the authoritative, verified mapping between every quantitative empirical claim in FraudGraph Intelligence and its immutable source artifact on disk.

Every metric, latency, speedup, confusion matrix element, and memory footprint in the final submission report is classified under one of four rigorous evidentiary tiers:
1. **MEASURED:** Directly captured via hardware timers (`cudaEvent_t`, Python `perf_counter`) in reproducible runs.
2. **DERIVED:** Mathematically computed from measured metrics (e.g., speedup ratio, PR-AUC trapezoidal integration, percentage improvement).
3. **HISTORICAL:** Recorded in previous authoritative project phases (e.g., Colab Tesla T4 CUDA benchmarks) and preserved byte-for-byte in repository archives.
4. **QUALITATIVE:** Engineering trade-off analysis, micro-architectural explanations, and boundary qualifications.

No metrics are synthesized, fabricated, or extrapolated.

---

## 2. Canonical Dataset & Topological Evidence Ledger

**Canonical Fixture Path:** `data/fixtures/synthetic_fraud_10k.npz`  
**Canonical Fixture SHA-256:** `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430` (Verified immutable)  
**Manifest Path:** `data/fixtures/synthetic_fraud_10k_manifest.json`  

| Metric / Claim | Measured Value | Evidentiary Tier | Source Artifact & Key / Array Location | Scientific Caveat & Interpretation Boundary |
| :--- | :--- | :--- | :--- | :--- |
| **Node Count ($N$)** | 10,000 | MEASURED | `synthetic_fraud_10k.npz` -> `features.shape[0]` | Represents 10,000 synthetic transaction records. |
| **Input Feature Dimension ($D$)** | 32 | MEASURED | `synthetic_fraud_10k.npz` -> `features.shape[1]` | 32 continuous normalized tabular features per node. |
| **Directed Edge Count ($M$)** | 231,984 | MEASURED | `synthetic_fraud_10k.npz` -> `col_idx.shape[0]` | Directed causal edges connecting historical transactions sharing card or device IDs. |
| **CSR Row Offsets ($N+1$)** | 10,001 | MEASURED | `synthetic_fraud_10k.npz` -> `row_ptr.shape[0]` | Standard Compressed Sparse Row format indices. |
| **Isolated Node Count** | 1,155 (11.55%) | MEASURED | `synthetic_fraud_10k_manifest.json` -> `dataset_summary.num_isolated_nodes` | Nodes with 0 inbound neighbors within the 7-day lookback window. |
| **Graph Density** | 0.00232 | DERIVED | `manifest.json` -> `stats.density` | Computed as $M / (N \times (N-1))$. Highly sparse graph. |
| **Node Degree (Min / Mean / Median / Max)** | 0 / 23.20 / 6.0 / 252 | MEASURED | `synthetic_fraud_10k_manifest.json` -> `stats.degree_*` | Highly skewed in-degree distribution; max hub has 252 inbound edges. |
| **Total Fraud Labels** | 350 | MEASURED | `synthetic_fraud_10k.npz` -> `np.sum(labels)` | Ground truth binary labels (0 = licit, 1 = fraud). |
| **Total Licit Labels** | 9,650 | MEASURED | `synthetic_fraud_10k.npz` -> `10000 - np.sum(labels)` | Licit background transactions. |
| **Class Imbalance (Prevalence)** | 3.50% | DERIVED | `manifest.json` -> `fraud_stats.fraud_percentage` | Extreme class imbalance reflecting realistic banking fraud regimes. |
| **Fraud Typology: Ring Fraud** | 245 cases (70.0%) | HISTORICAL | `synthetic_fraud_10k_manifest.json` -> `label_breakdown.ring_fraud` | Collusive rings sharing cyclical cards/devices across time. |
| **Fraud Typology: Opportunistic** | 105 cases (30.0%) | HISTORICAL | `synthetic_fraud_10k_manifest.json` -> `label_breakdown.opportunistic_fraud` | High-velocity, anomalous amount single-node spikes. |
| **Chronological Train Split** | 7,000 nodes (70.0%), 239 frauds (3.41%) | MEASURED | `fraud_model_evaluation_20261024_212032.json` -> `dataset_split.train_*` | First 70% chronological transactions ($t \le 21.0$ days). |
| **Chronological Val Split** | 1,500 nodes (15.0%), 58 frauds (3.87%) | MEASURED | `fraud_model_evaluation_20261024_212032.json` -> `dataset_split.val_*` | Intermediate 15% transactions ($21.0 < t \le 25.5$ days). |
| **Chronological Test Split** | 1,500 nodes (15.0%), 53 frauds (3.53%) | MEASURED | `fraud_model_evaluation_20261024_212032.json` -> `dataset_split.test_*` | Final unseen 15% transactions ($25.5 < t \le 30.0$ days). |
| **Temporal Lookback Window** | 7.0 days ($604,800$ s) | MEASURED | `manifest.json` -> `metadata.temporal_window_days` | Hard causal boundary; no edges formed across $> 7$ days span. |
| **Graph Causal Invariant** | Strictly acyclic forward | MEASURED | `synthetic_fraud_10k.npz` -> `timestamps[u] < timestamps[v]` | Zero future-to-past edges; zero self-loops ($u \ne v$). |

### 2.1 Graph Fixture Disambiguation Matrix

To eliminate cross-workload ambiguity, the four repository fixtures and their topological parameters are formally disambiguated below:

| Fixture Path | Nodes ($N$) | Edges ($M$) | Feat Dim ($D$) | Max Degree ($d_{\max}$) | Hub Profile | Primary Experimental Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/fixtures/synthetic_fraud_10k.npz` | 10,000 | 231,984 | 32 | **252** | Power-law, ring/burst structures | Downstream ML evaluation, transaction ledger, BFS explorer |
| `data/fixtures/synthetic_skewed_10k.npz` | 10,000 | 94,970 | 32 | **6,340** | Extreme super-hub (56 hubs $d>32$, 19 hubs $d>128$) | CUDA skewed stress benchmarks (Experiment 6.3) |
| `data/fixtures/synthetic_tier_a_10k.npz` | 10,000 | 94,945 | 32 | **10** | Uniform bounded degree ($d \in [8, 10]$) | CUDA uniform benchmarks (Experiment 6.1, 6.2) |
| `data/fixtures/toy_graph_5nodes.npz` | 5 | 8 | 4 | **3** | Hand-crafted deterministic graph | Unit test verification gate (`atol <= 1e-5`) |

---

## 3. CPU Aggregation & Vectorized Acceleration Ledger

**Primary Source Artifact:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`  
**Benchmark Workload:** Synthetic Tier A 10k Workload ($N=10,000, M=100,000, D=64$, Power-Law Topology) and Toy Workload ($N=5, M=6, D=16$)  
**Host Platform:** AMD64 64-bit, Python 3.14.3, NumPy 2.5.4, SciPy 1.15.1  

| Workload & Operator | Engine | Mean Latency (ms) | Median (ms) | p95 (ms) | CV% | Measured Speedup vs Ref | Compulsory Bandwidth | Evidentiary Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier A 10k: MEAN** | CPU Reference (Sequential) | 114.8598 ms | 115.1114 ms | 129.4814 ms | 9.80% | 1.00x (Baseline) | 0.0522 GB/s | MEASURED |
| **Tier A 10k: MEAN** | CPU Vectorized (SpMM SciPy) | 2.5756 ms | 2.3322 ms | 4.0347 ms | 34.29% | **44.60x** | 2.3296 GB/s | MEASURED |
| **Tier A 10k: MAX** | CPU Reference (Sequential) | 86.7878 ms | 83.9624 ms | 101.8732 ms | 11.78% | 1.00x (Baseline) | 0.0691 GB/s | MEASURED |
| **Tier A 10k: MAX** | CPU Vectorized (NumPy/SciPy) | 87.4159 ms | 86.5556 ms | 105.7071 ms | 15.90% | **0.99x** (No speedup) | 0.0686 GB/s | MEASURED |
| **Toy Graph: SUM** | CPU Reference (Sequential) | 0.0354 ms | 0.0342 ms | 0.0406 ms | 6.48% | 1.00x (Baseline) | 0.0208 GB/s | MEASURED |
| **Toy Graph: SUM** | CPU Vectorized (SpMM) | 0.0726 ms | 0.0683 ms | 0.1034 ms | 18.60% | **0.49x** (Slowdown) | 0.0101 GB/s | MEASURED |
| **Historical Phase 3 SUM Ref** | CPU Reference (Loop) | 166.8770 ms | 166.2100 ms | 175.4000 ms | 3.12% | 1.00x | — | HISTORICAL |
| **Historical Phase 3 MEAN Ref** | CPU Reference (Loop) | 133.1770 ms | 132.8500 ms | 141.2000 ms | 2.85% | 1.00x | — | HISTORICAL |
| **Historical Phase 3 MAX Ref** | CPU Reference (Loop) | 167.1730 ms | 166.9000 ms | 176.1000 ms | 2.94% | 1.00x | — | HISTORICAL |
| **Historical Phase 3 SUM SpMM** | CPU Vectorized (SpMM) | 3.6240 ms | 3.5100 ms | 4.1200 ms | 8.45% | **46.05x** | — | HISTORICAL |
| **Historical Phase 3 MEAN SpMM** | CPU Vectorized (SpMM) | 6.3250 ms | 6.1800 ms | 7.0500 ms | 7.90% | **21.06x** | — | HISTORICAL |

> [!CRITICAL]
> **Scientific Integrity Attribution:**
> The peak **44.60x speedup** (and historical **46.05x speedup**) is strictly a **CPU Vectorized SpMM vs. CPU Sequential Python Reference Loop** comparison. It is **NOT** a GPU speedup. Under no circumstances may this be attributed to CUDA hardware execution.

---

## 4. CUDA GPU Acceleration & Optimization Ledger

**Source Artifacts:** `experiments/results/exp_gpu_*_20261024_142037.json` (Colab Tesla T4 Authoritative Benchmark Run)  
**Hardware Specifications:** NVIDIA Tesla T4, 14,912.7 MB VRAM, 40 Streaming Multiprocessors (SMs), Compute Capability 7.5 (Turing), Driver 535.104.05, CUDA 12.2  
**Dataset Scale:** $N=10,000$ transactions, $D=32$ features; Tier A ($M=94,945$), Skewed ($M=94,970$)  

### 4.1 Tier A Uniform Graph Results ($N=10,000, M=94,945, D=32$)

| Implementation Kernel | Operator | Kernel Latency (ms) | H2D Transfer (ms) | D2H Transfer (ms) | End-to-End Latency (ms) | Effective Bandwidth (GB/s) | Edges / Sec | Evidentiary Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA Baseline (Thread/Node)** | SUM | 0.068595 ms | 0.600544 ms | 0.322272 ms | 0.991411 ms | 43.44 GB/s | $1.384 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Baseline (Thread/Node)** | MEAN | 0.067947 ms | 0.580416 ms | 0.336032 ms | 0.984395 ms | 43.85 GB/s | $1.397 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Baseline (Thread/Node)** | MAX | 0.067662 ms | 0.561408 ms | 0.334368 ms | 0.963438 ms | 44.04 GB/s | $1.403 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Coalesced (1 Warp/Node)** | SUM | **0.017192 ms** | 0.548672 ms | 0.316704 ms | 0.882568 ms | 173.32 GB/s | $5.523 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Coalesced (1 Warp/Node)** | MEAN | **0.016995 ms** | 0.573792 ms | 0.391744 ms | 0.982531 ms | 175.33 GB/s | $5.587 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Coalesced (1 Warp/Node)** | MAX | **0.016878 ms** | 0.607936 ms | 0.387296 ms | 1.012110 ms | 176.54 GB/s | $5.625 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Pinned Streams** | SUM | 0.016698 ms | 0.159072 ms | 0.106880 ms | **0.282650 ms** | 178.46 GB/s | $5.686 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Pinned Streams** | MEAN | 0.017197 ms | 0.157248 ms | 0.106560 ms | **0.281005 ms** | 173.28 GB/s | $5.521 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Pinned Streams** | MAX | **0.016360 ms** | 0.156768 ms | 0.105344 ms | **0.278472 ms** | 182.14 GB/s | $5.803 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Degree-Aware Hybrid** | SUM | 0.020469 ms | 0.161024 ms | 0.105856 ms | 0.396069 ms | 145.58 GB/s | $4.639 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Degree-Aware Hybrid** | MEAN | 0.020126 ms | 0.163776 ms | 0.106144 ms | 0.343334 ms | 148.05 GB/s | $4.717 \times 10^9$ | HISTORICAL / MEASURED |
| **CUDA Degree-Aware Hybrid** | MAX | 0.021536 ms | 0.164448 ms | 0.107744 ms | 0.382614 ms | 138.36 GB/s | $4.409 \times 10^9$ | HISTORICAL / MEASURED |

### 4.2 Skewed Power-Law Graph Results (`synthetic_skewed_10k.npz`: $N=10,000, M=94,970, D=32, d_{\max}=6,340$)

| Implementation Kernel | Operator | Kernel Latency (ms) | End-to-End Latency (ms) | Speedup vs Baseline Kernel | Speedup vs Coalesced Kernel | Engineering Finding & Nuance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA Baseline (Thread/Node)** | SUM | 6.518180 ms | 7.489540 ms | 1.00x | — | Catastrophic warp divergence on power-law hubs. |
| **CUDA Baseline (Thread/Node)** | MEAN | 6.985980 ms | 7.898550 ms | 1.00x | — | Warp serialization across irregular degree lists. |
| **CUDA Baseline (Thread/Node)** | MAX | 7.214300 ms | 8.140290 ms | 1.00x | — | Max reduction divergence within single SIMT warps. |
| **CUDA Coalesced (1 Warp/Node)** | SUM | 0.212122 ms | 1.096890 ms | **30.73x** | 1.00x | Coalesced 128-byte DRAM reads eliminate memory stalls. |
| **CUDA Coalesced (1 Warp/Node)** | MEAN | 0.219774 ms | 1.091170 ms | **31.79x** | 1.00x | Coalescing yields $>31\times$ acceleration under skew. |
| **CUDA Coalesced (1 Warp/Node)** | MAX | 0.233390 ms | 1.133040 ms | **30.91x** | 1.00x | Feature-dimension warp cooperation. |
| **CUDA Degree-Aware Hybrid** | MEAN | **0.064858 ms** | **0.396447 ms** | **107.71x** | **3.39x** | Partitioning low/medium/high nodes maximizes occupancy. |
| **CUDA Degree-Aware Hybrid** | MAX | **0.110626 ms** | **0.443224 ms** | **65.21x** | **2.11x** | Block-level cooperative reduction for hub nodes. |
| **CUDA Degree-Aware Hybrid** | SUM | **0.464106 ms** | **0.802457 ms** | **14.04x** | **0.46x (REGRESSION)** | **SUM regressed by $2.19\times$ vs Coalesced ($0.212 \to 0.464$ ms) because hub_degree_block_kernel serializes accumulation onto Warp 0 to guarantee float32 non-associative numerical stability (atol <= 1e-5), leaving 7 of 8 warps idle.** |

> [!NOTE]
> **Micro-Architectural Root Cause of Degree-Aware SUM Regression:**
> Source code inspection of `fgi_degree_aware_kernels.cu` (lines 160–173) confirms that no atomic instructions are present in `hub_degree_block_kernel`. For `OP_SUM`, the kernel executes `if (warp_id == 0)` serial accumulation. In multi-warp tree reductions, out-of-order IEEE 754 single-precision floating-point addition causes numerical drift up to $3.97 \times 10^{-4}$ on the 6,340-edge super-hub, violating the project's $\text{atol} \le 10^{-5}$ tolerance gate. To guarantee exact numerical equivalence with the CPU reference, accumulation was restricted to Warp 0. This intentional numerical safeguard forces 7 out of 8 warps (87.5% of threads per block) to remain idle while Warp 0 loops through all 6,340 neighbors, explaining the measured $2.19\times$ slowdown relative to Coalesced (which distributes nodes across warps).

> [!IMPORTANT]
> **Memory Transfer Bottleneck Finding:**
> On Tier A baseline, host-to-device ($0.58$ ms) and device-to-host ($0.33$ ms) PCIe transfer latency totals $0.91$ ms, which is **$13.4\times$ greater than the isolated kernel execution time** ($0.068$ ms).
> Utilizing asynchronous pinned memory streams reduced end-to-end latency from $0.98$ ms to $0.28$ ms (**$3.5\times$ speedup**).

---

## 5. Fraud Classification & Feature Enrichment Ledger

**Primary Source Artifact:** `experiments/results/fraud_model_evaluation_20261024_212032.json`  
**Summary CSV Artifact:** `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`  
**Dataset Evaluation Split:** 7,000 Train / 1,500 Val / 1,500 Test (Chronological, strictly causal)  
**Positive Test Ground Truth:** 53 Frauds out of 1,500 test transactions (3.533% prevalence)  

| Model Family | Representation | Input Features ($D$) | Test PR-AUC | Test ROC-AUC | Optimal Test F1 | Test Precision | Test Recall | TP | FP | FN | TN | Enrichment Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Raw Tabular | 32 | 0.308794 | 0.785503 | 0.344828 | 0.317460 | 0.377358 | 20 | 43 | 33 | 1404 | 0.748 ms |
| **Logistic Regression** | **Graph MAX Enriched** | 64 | **0.331471** | **0.786481** | **0.326531** | 0.255319 | **0.452830** | 24 | 70 | 29 | 1377 | 60.047 ms |
| **Logistic Regression** | Graph MEAN Enriched | 64 | 0.300959 | 0.776284 | 0.320000 | 0.247423 | 0.452830 | 24 | 73 | 29 | 1374 | 6.415 ms |
| **Logistic Regression** | Multi-Operator (Mean+Max) | 96 | 0.247094 | 0.780000 | 0.338710 | 0.295775 | 0.396226 | 21 | 50 | 32 | 1397 | 69.780 ms |
| **Logistic Regression** | Graph SUM Enriched | 64 | 0.148773 | 0.749932 | 0.275862 | 0.217391 | 0.377358 | 20 | 72 | 33 | 1375 | 6.424 ms |
| **Random Forest** | Raw Tabular | 32 | 0.244511 | 0.778605 | 0.255708 | 0.168675 | **0.528302** | 28 | 138 | 25 | 1309 | 0.707 ms |
| **Random Forest** | **Graph MEAN Enriched** | 64 | **0.258714** | **0.779185** | **0.306122** | **0.333333** | 0.283019 | 15 | **30** | 38 | 1417 | 6.031 ms |
| **Random Forest** | Graph MAX Enriched | 64 | 0.186520 | 0.734636 | 0.218579 | 0.153846 | 0.377358 | 20 | 110 | 33 | 1337 | 67.245 ms |
| **Random Forest** | Multi-Operator (Mean+Max) | 96 | 0.217212 | 0.772353 | 0.297872 | 0.341463 | 0.264151 | 14 | 27 | 39 | 1420 | 66.972 ms |
| **Random Forest** | Graph SUM Enriched | 64 | 0.123620 | 0.742375 | 0.177419 | 0.112821 | 0.415094 | 22 | 173 | 31 | 1274 | 4.640 ms |

### Key Machine Learning Findings & Evidentiary Rationales:
1. **MAX Aggregation for Linear Detectors & Precision Trade-off:** Logistic Regression achieves its highest test PR-AUC (**0.3315**, +7.3% relative gain over raw 0.3088) and improves Recall ($37.7\% \to 45.3\%$, capturing 4 additional fraud cases) with MAX neighbor representation. However, this reflects an explicit operational trade-off: false positives rise from 43 to 70, reducing Precision ($31.7\% \to 25.5\%$) and lowering optimal F1 ($0.3448 \to 0.3265$). This profile is well-suited for high-severity fraud contexts where undetected fraud costs exceed manual triage costs.
2. **False Positive Suppression vs. Severe Recall Collapse in Non-Linear Ensembles:** In Random Forest, MEAN representation dramatically cuts false alarms from **138 down to 30** (a **78.3% false-positive reduction**), nearly doubling precision from $16.9\%$ to $33.3\%$ and lifting PR-AUC from 0.2445 to 0.2587 and F1 from 0.2557 to 0.3061. However, this introduces a severe operational trade-off: Recall collapses from $52.8\%$ to $28.3\%$ (-24.5 percentage points), with false negatives rising from 25 to 38 and true positive detections falling from 28 to 15. In risk-averse environments, this loss of sensitivity may require compensatory alert thresholds.
3. **Catastrophic Failure of Unnormalized SUM:** Both models suffer severe degradation under SUM aggregation (LR drops to 0.1488, RF drops to 0.1236) because unnormalized summation correlates with high node in-degree rather than true relational fraudulence, flooding the decision boundary with high-degree benign accounts.
4. **Validation Threshold Freezing:** All test classification decisions (F1, precision, recall, confusion matrix) were evaluated using decision thresholds selected strictly on the validation set, ensuring zero test data leakage.

---

## 6. Software Verification & Test Suite Ledger

**Verification Timestamp:** 2026-10-25T17:51:15+05:30  
**Git HEAD:** `8afa28cf1172c006f0242f530267e8cc4466f9d6`  

| Test Suite / Inspection Target | Command / Execution Method | Total Tests | Passed | Skipped | Failed | Execution Time | Evidentiary Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Backend Unit & Integration Suite** | `python -m pytest -o filterwarnings="ignore::DeprecationWarning"` | 190 | **188** | **2** | 0 | 89.94s | **VERIFIED PASS** |
| **Frontend Polling Lifecycle Tests** | `node frontend/tests/polling_lifecycle.test.mjs` | 21 | **21** | 0 | 0 | 142.96ms | **VERIFIED PASS** |
| **Frontend API Client Client Tests** | `node frontend/tests/subgraph_transactions_client.test.mjs` | 3 | **3** | 0 | 0 | 7.49ms | **VERIFIED PASS** |
| **TypeScript Typecheck** | `npm run typecheck` (`tsc -b`) | — | Clean | 0 | 0 | 1.84s | **VERIFIED PASS (0 Errors)** |
| **Frontend Production Build** | `npm run build` (`tsc -b && vite build`) | — | Clean | 0 | 0 | 5.81s | **VERIFIED PASS (Dist generated)** |
| **Canonical Fixture Hash Audit** | SHA-256 on `synthetic_fraud_10k.npz` | 1 | 1 | 0 | 0 | 0.05s | **VERIFIED MATCH (`aecafe3b...`)** |
| **Git Working Tree Check** | `git status --short` | — | Clean | — | — | 0.02s | **VERIFIED (0 staged, 0 modified)** |

### Explicit Limitations & Disclosures:
- **Two Skipped Backend Tests:** `test_cuda_baseline.py` and `test_feature_enrichment.py` skip gracefully when local CUDA hardware is not detected (`NO_GPU_DETECTED_COLAB_READY`).
- **Live Socket Integration Test:** `frontend/tests/api_integration.test.mjs` was omitted during automated test sweeps because port 8000 was bound by an external background process and port 5173 lacked an active Vite development server. It remains untracked and preserved.
- **Local NVCC Toolchain:** No local NVIDIA GPU or NVCC compiler is present on the Windows host. CUDA results are strictly derived from the verified Colab Tesla T4 benchmark logs.

---
*Ledger Compiled and Certified for Phase 12 Submission by FraudGraph Intelligence Verification Engine.*
