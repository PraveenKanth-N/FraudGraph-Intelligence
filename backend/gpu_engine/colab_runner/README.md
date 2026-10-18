# Google Colab GPU Workflow Guide — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Course Code:** CSI4007  
**Project:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Module:** `backend/gpu_engine/colab_runner/`  

---

## 1. Architectural Rationale

As established in the Phase 0 audit and Architectural Decision Record [`ADR-003`](file:///D:/FraudGraph%20Intelligence/docs/DECISIONS.md), the local development host features an NVIDIA GeForce MX450 GPU but lacks the local NVIDIA CUDA Compiler (`nvcc`) and MSVC build tools (`cl.exe`).

In strict adherence to the **Master Engineering Directive**:
- FGI never fabricates benchmark numbers, throughput metrics, or GPU speedups.
- Rather than forcing unverified multi-gigabyte local installations, FGI implements a **Two-Tier Reproducible Execution Workflow**:
  - **Tier 1 (Host Development):** Python 3.14 + FastAPI + CPU Reference Engine (`aggregators.py`) + React/TypeScript Frontend.
  - **Tier 2 (Cloud GPU Acceleration):** Google Colab GPU runtimes (NVIDIA T4 / V100 / A100) executing native CUDA C++ kernels compiled with official NVIDIA CUDA Toolkits.

---

## 2. Step-by-Step Execution Protocol

```
+--------------------------------------------------------------------------------------------------+
|                                    COLAB WORKFLOW PIPELINE                                       |
+--------------------------------------------------------------------------------------------------+
|   [Local Host]              [Google Colab Runtime]               [Local Host]                    |
|                                                                                                  |
| 1. Package Bundle      -> 2. Upload Bundle / Open Notebook   -> 4. Download Results Archive      |
|    (CUDA source +           (Verify GPU with nvidia-smi)           (fgi_phase5_gpu_results.zip)  |
|     raw graph arrays +    3. Execute run_colab_benchmarks.py    5. Ingest into experiments/      |
|     CPU references)          - Compile with nvcc                   (Strict schema validation     |
|                              - Correctness gate (toy)               and speedup calculations)    |
|                              - Benchmark matrix (Tier A)                                         |
+--------------------------------------------------------------------------------------------------+
```

### Step 1: Generate the Self-Contained Bundle
On the local host, run:
```powershell
.\.venv\Scripts\python.exe backend/gpu_engine/colab_runner/package_colab_bundle.py
```
This utility:
- Unpacks `data/fixtures/toy_graph_5nodes.npz` and `data/fixtures/synthetic_tier_a_10k.npz` into raw C++-loadable binary arrays (`row_ptr.bin`, `col_idx.bin`, `features.bin`, `meta.txt`).
- Precomputes and serializes exact CPU reference outputs (`ref_mean.bin`, `ref_sum.bin`, `ref_max.bin`).
- Bundles all native CUDA sources (`fgi_baseline_kernels.cu`, `main_baseline.cu`), headers (`fgi_cuda_common.cuh`, `fgi_baseline_kernels.cuh`), and `Makefile`.
- Creates `experiments/colab_bundle_fgi_phase5.zip`.

### Step 2: Open Google Colab & Enable GPU
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Open or upload [`backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb`](file:///D:/FraudGraph%20Intelligence/backend/gpu_engine/colab_runner/fgi_colab_workflow.ipynb).
3. Ensure GPU hardware acceleration is active:
   `Runtime` -> `Change runtime type` -> Select **T4 GPU** (or A100 if Colab Pro is available).

### Step 3: Upload & Execute
1. In the Colab file browser (left panel), upload `colab_bundle_fgi_phase5.zip`.
2. Run the notebook cells sequentially:
   - Cell 1: `!nvidia-smi` and `!nvcc --version` (detects real physical GPU).
   - Cell 2: Extracts `colab_bundle_fgi_phase5.zip`.
   - Cell 3: Executes `!python3 run_colab_benchmarks.py`.
   - Cell 4: Automatically downloads `fgi_phase5_gpu_results.zip`.

### Step 4: Ingest Verified Results Back into Local Workspace
When the results archive is downloaded to your machine, run:
```powershell
.\.venv\Scripts\python.exe backend/gpu_engine/colab_runner/ingest_colab_results.py <path_to_downloaded_fgi_phase5_gpu_results.zip>
```
The ingestion utility will:
- Check that all JSON files conform to the FGI benchmark schema.
- Enforce that `measurement_state` is `MEASURED`.
- Verify that `numerical_verification.status` is `PASSED` with max absolute error $\le 10^{-5}$.
- Archive results into `experiments/results/`.

---

## 3. Correctness Gate & Tolerance Standards

The Colab runner enforces a strict two-stage gate:
1. **Stage 1 (Correctness Gate):** Executes on `toy_graph_5nodes` across Sum, Mean, and Max. Every output float is compared against the precomputed CPU reference. If $\max |y_{\text{gpu}} - y_{\text{cpu}}| > 1.0 \times 10^{-5}$ or if any NaN/Inf is detected, the pipeline halts immediately.
2. **Stage 2 (Scale Benchmarks):** Only after Stage 1 passes does the runner execute the multi-repetition benchmark on `synthetic_tier_a_10k` ($N=10,000, M=94,945, D=32$).

---

## 4. Troubleshooting

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `nvidia-smi: command not found` | Standard CPU runtime selected. | In Colab, go to `Runtime -> Change runtime type` and select **T4 GPU**. |
| `nvcc fatal: Value 'sm_XX' is not defined` | Old CUDA compiler version. | The provided Makefile targets `sm_70`, `sm_75`, `sm_80`. Colab uses CUDA 12.x which natively supports these. |
| `CUDA error: out of memory` | Graph feature tensor exceeds GPU VRAM. | The runner checks `cudaMemGetInfo` before allocating. Reduce graph size if testing beyond Tier B. |
