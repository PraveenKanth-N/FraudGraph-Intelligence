# Research Specification & Experimental Design — FraudGraph Intelligence (FGI)

**Course:** VIT Vellore — CSI4007 GPU Programming  
**Document Status:** Phase 2 Authoritative Specification  

---

## 1. Refined Research Questions

### **RQ1: Comparative Performance Baseline**
*How does a native CUDA parallel neighbor aggregation kernel compare in latency and throughput against an optimized, vector-parallel CPU reference implementation across synthetic and real-world graph topologies with varying degree distributions?*

### **RQ2: Memory-Aware CUDA Optimizations**
*To what extent can memory-aware optimizations (specifically warp-per-node thread mapping with memory coalescing and threadblock shared-memory caching) reduce execution time and increase memory bandwidth utilization compared to a naive thread-per-node CUDA baseline?*

### **RQ3: Dimensional & Topological Sensitivity**
*How do graph scale ($N \in [10^4, 2 \times 10^5]$), degree distribution (uniform vs. power-law skew), and feature width ($D \in \{16, 32, 64, 128\}$) affect the relative balance between PCIe transfer latency and GPU kernel compute time, and what are the numerical precision boundaries across single-precision floating-point aggregation operations?*

---

## 2. Formal Testable Hypotheses & Variables

### **Hypothesis 1 ($H_1$): Warp-Coalesced Speedup over Baseline CUDA**
- **Statement:** For graphs with power-law degree distributions ($\gamma \in [2.0, 3.0]$) and feature dimensionality $D \ge 32$, a warp-per-node kernel that coalesces feature-dimension global memory reads will achieve at least a $2.0\times$ speedup in kernel execution time compared to a naive thread-per-node CUDA baseline.
- **Independent Variables:**
  - Thread mapping strategy: Thread-per-node vs. Warp-per-node (32 threads per node).
  - Feature dimensionality: $D \in \{16, 32, 64, 128\}$.
  - Degree distribution: Uniform random vs. Power-law (Barabási–Albert / R-MAT).
- **Dependent Variables:**
  - Kernel execution time $t_{\text{kernel}}$ (milliseconds, measured via CUDA events).
  - Global memory load efficiency ($\%$, measured via memory access transactions).
- **Supporting Evidence:** Measured mean kernel time $t_{\text{coalesced}} \le 0.50 \times t_{\text{baseline}}$ across $\ge 20$ repeated benchmark runs with $D \ge 32$.
- **Rejecting Evidence:** $t_{\text{coalesced}} > 0.50 \times t_{\text{baseline}}$ or no statistically significant difference ($\alpha = 0.01$).

### **Hypothesis 2 ($H_2$): Transfer-Bound to Compute-Bound Transition**
- **Statement:** In an end-to-end GPU acceleration pipeline, as the feature dimensionality $D$ increases from 16 to 128 at constant graph topology, the ratio of kernel computation time to total end-to-end time ($\frac{t_{\text{kernel}}}{t_{\text{e2e}}}$) will increase monotonically, resulting in higher end-to-end speedups over CPU reference execution.
- **Independent Variables:** Feature dimension $D \in \{16, 32, 64, 128\}$.
- **Dependent Variables:**
  - Host-to-Device transfer time $t_{\text{H2D}}$ (ms).
  - Kernel computation time $t_{\text{kernel}}$ (ms).
  - Device-to-Host transfer time $t_{\text{D2H}}$ (ms).
  - Ratio $R_{\text{compute}} = \frac{t_{\text{kernel}}}{t_{\text{H2D}} + t_{\text{kernel}} + t_{\text{D2H}}}$.
  - End-to-end speedup factor $S_{\text{e2e}} = \frac{t_{\text{CPU}}}{t_{\text{e2e}}}$.
- **Supporting Evidence:** Monotonic increase in $R_{\text{compute}}$ and $S_{\text{e2e}}$ as $D$ scales from 16 to 128.
- **Rejecting Evidence:** $R_{\text{compute}}$ remains static or decreases due to memory allocation overheads or bus saturation.

### **Hypothesis 3 ($H_3$): Numerical Equivalence Preservation**
- **Statement:** The single-precision floating-point output matrix produced by optimized CUDA kernels ($Y_{\text{GPU}}$) will maintain an absolute numerical tolerance of $\le 10^{-5}$ compared to the double-precision rounded CPU reference output ($Y_{\text{CPU}}$) across all valid non-isolated nodes.
- **Independent Variables:** Kernel implementation (Baseline vs. Coalesced vs. Shared).
- **Dependent Variables:** Maximum absolute error: $\epsilon_{\max} = \max_{i, k} |Y_{\text{GPU}}[i, k] - Y_{\text{CPU}}[i, k]|$.
- **Supporting Evidence:** $\epsilon_{\max} \le 1.0 \times 10^{-5}$ across all benchmark test graphs.
- **Rejecting Evidence:** $\epsilon_{\max} > 1.0 \times 10^{-5}$, or presence of NaN / Inf values for nodes with degree $d_i \ge 0$.

---

## 3. Separation of Project Contributions

To maintain absolute academic and research integrity, the contributions of FraudGraph Intelligence are explicitly divided into three distinct categories:

### 3.1 Engineering Contribution (Practical & Systemic)
1. **Production-Grade CSR Engine:** Robust, deterministic implementation of Compressed Sparse Row graph representations tailored for financial relational structures with zero-degree protection.
2. **Modular Multi-Tier Software Stack:** Decoupled React 18 frontend (Obsidian Cybernetics design system), FastAPI REST backend, and dual-tier execution adapter enabling verified cloud GPU execution when local hardware toolchains are absent.
3. **Decoupled Telemetry Pipeline:** Comprehensive profiling infrastructure that transparently isolates PCIe memory transfer overheads from pure GPU kernel runtimes.

### 3.2 Experimental Contribution (Empirical & Evaluative)
1. **Systematic Micro-Benchmarking:** Controlled experimental comparison of three distinct CUDA parallel reduction designs (thread-per-node, warp-per-node with coalescing, and shared-memory tiling) across controlled graph scales ($10^4$ to $2 \times 10^5$ nodes) and feature widths (16 to 128).
2. **Topological Skew Analysis:** Empirical measurement of how degree variance and power-law distribution degrade naive SIMT execution efficiency on modern NVIDIA architectures (Turing / Ampere).
3. **Downstream Utility Assessment:** Controlled empirical evaluation quantifying the classification gain (PR-AUC, F1-score) of appending 1-hop aggregated neighborhood features to baseline tabular transaction classifiers on imbalanced financial fraud data.

### 3.3 Modest, Defensible Research Contribution
1. **Memory-Aware Parallel Neighbor Aggregation Specialization:** Demonstrating that co-optimizing warp thread scheduling along the feature dimension—rather than the sparse topological dimension alone—provides a simple, portable, and highly effective acceleration strategy for GNN message-passing workloads without requiring heavy third-party deep learning framework overheads.

---

## 4. Defensible Scope Boundaries (What is NOT Claimed)
- **NO New Theoretical Algorithm:** We do not claim to have invented a fundamentally new mathematical sparse matrix operation.
- **NO State-of-the-Art Claim:** We do not claim superiority over fully fused industrial frameworks (e.g. cuSPARSE with specialized hardware tensor-core acceleration).
- **NO Guaranteed Speedup:** Speedup targets are formal hypotheses to be tested, never claimed as pre-ordained outcomes.
- **NO Production Fraud Prevention:** Transaction classifications are experimental forensic demonstrations, not live operational banking solutions.
