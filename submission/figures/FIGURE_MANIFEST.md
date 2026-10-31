# Figure Manifest — FraudGraph Intelligence (FGI)

**Directory:** `submission/figures/`  
**Generated Date:** 2026-10-25T17:56:00+05:30  
**Resolution:** 300 DPI (PNG) + Scalable Vector Graphics (SVG) + PDF (where indicated)  
**Scientific Attribution:** All figures are strictly derived from verified disk artifacts in `data/fixtures/` and `experiments/results/`. Zero synthetic values or unverified mock vectors were generated.

---

## Catalog of Publication Figures

### Figure 1: Test PR-AUC Comparison Across Representations
- **Files:** `fig1_test_pr_auc_comparison.png`, `fig1_test_pr_auc_comparison.svg`, `fig1_test_pr_auc_comparison.pdf`
- **Source Artifact:** `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`, `fraud_model_evaluation_20261024_212032.json`
- **Fields / Variables:** `primary_results_causal_history[*].test_metrics.pr_auc`, `dataset_split.test_prevalence` (3.53%)
- **Target Report Section:** Section 15 & 17 (Fraud Feature Enrichment & Experimental Results)
- **Scientific Interpretation:** Demonstrates that MAX aggregation provides the highest PR-AUC for Logistic Regression (0.3315 vs raw 0.3088, +7.3% relative gain) and lifts fraud recall from 37.7% to 45.3%, though with lower precision (25.5% vs 31.7%) and slightly lower F1 (0.3265 vs 0.3448). Unnormalized SUM causes catastrophic degradation due to hub degree bias.

### Figure 2: Test ROC-AUC Comparison Across Representations
- **Files:** `fig2_test_roc_auc_comparison.png`, `fig2_test_roc_auc_comparison.svg`, `fig2_test_roc_auc_comparison.pdf`
- **Source Artifact:** `summary_evaluation_metrics.csv`
- **Fields / Variables:** `primary_results_causal_history[*].test_metrics.roc_auc`
- **Target Report Section:** Section 15 & 17 (Model Evaluation)
- **Scientific Interpretation:** Shows compression of ROC-AUC scores (0.73 - 0.79) under extreme class imbalance (3.5%), highlighting why PR-AUC is the necessary primary metric for fraud evaluation.

### Figure 3: Precision-Recall Curve Availability & Protocol Notice
- **Files:** `fig3_pr_curve_omission_notice.png`, `fig3_pr_curve_omission_notice.svg`, `fig3_pr_curve_omission_notice.pdf`
- **Source Artifact:** `evaluation_protocol.threshold_selection_protocol` in `fraud_model_evaluation_20261024_212032.json`
- **Target Report Section:** Section 16 (Benchmarking Methodology & Scientific Integrity)
- **Scientific Interpretation:** Documents adherence to scientific integrity rules: continuous PR curve plotting is explicitly omitted because per-example continuous score vectors were not stored in the canonical summary artifact; only discrete trapezoidal PR-AUC is reported.

### Figure 4: Test Confusion Matrices at Validation-Selected Thresholds
- **Files:** `fig4_confusion_matrices.png`, `fig4_confusion_matrices.svg`, `fig4_confusion_matrices.pdf`
- **Source Artifact:** `summary_evaluation_metrics.csv`
- **Fields / Variables:** `confusion_matrix` (TP, FP, FN, TN) for LR Raw, LR Max, RF Raw, RF Mean
- **Target Report Section:** Section 15 & 17 (Fraud Detection Performance)
- **Scientific Interpretation:** Visualizes decision boundaries under frozen validation thresholds. Highlights the Random Forest MEAN trade-off: false alerts drop from 138 down to 30 (78.3% reduction) and precision lifts from 16.9% to 33.3%, but recall drops from 52.8% to 28.3% (FN increases from 25 to 38).

### Figure 5: Fraud Detection Operating Trade-off Comparison
- **Files:** `fig5_operating_tradeoffs.png`, `fig5_operating_tradeoffs.svg`, `fig5_operating_tradeoffs.pdf`
- **Source Artifact:** `summary_evaluation_metrics.csv`
- **Fields / Variables:** Precision vs Recall vs False Positives across model architectures
- **Target Report Section:** Section 15 & 17 (Operational Banking Trade-offs)
- **Scientific Interpretation:** Illustrates the practical tradeoff between operational investigation costs (governed by false alerts) and fraud loss mitigation (governed by recall).

### Figure 6: Computational Timing Profile & Hardware Boundary Qualification
- **Files:** `fig6_runtime_breakdown.png`, `fig6_runtime_breakdown.svg`, `fig6_runtime_breakdown.pdf`
- **Source Artifact:** `fraud_model_evaluation_20261024_212032.json`
- **Fields / Variables:** `timing_ms.feature_enrichment_ms`, `timing_ms.model_training_ms`, `timing_ms.test_inference_ms`
- **Target Report Section:** Section 16 & 18 (System Runtime & Micro-Architectural Separation)
- **Scientific Interpretation:** Explicitly separates host Python process timings (scikit-learn training and inference) from GPU aggregation kernel execution boundaries.

### Figure 7: Context-Policy Sensitivity Analysis
- **Files:** `fig7_context_policy_sensitivity.png`, `fig7_context_policy_sensitivity.svg`, `fig7_context_policy_sensitivity.pdf`
- **Source Artifact:** `fraud_model_evaluation_20261024_212032.json`
- **Fields / Variables:** `primary_results_causal_history[*].test_metrics.pr_auc` vs `sensitivity_results_train_only[*].test_metrics.pr_auc`
- **Target Report Section:** Section 15 & 20 (Causal Graph Leakage & Sensitivity)
- **Scientific Interpretation:** Proves stability of neighbor representations when restricting context to historical training nodes versus full causal lookback windows.

### Figure 8: CUDA Kernel Execution Latency Across Architectural Optimizations
- **Files:** `fig8_cuda_kernel_latency_comparison.png`, `fig8_cuda_kernel_latency_comparison.svg`
- **Source Artifact:** `experiments/results/exp_gpu_*_20261024_142037.json`
- **Fields / Variables:** `timing_results_ms.kernel_mean` across Baseline, Warp-Coalesced, Degree-Aware, and Pinned Streams for Tier A Uniform ($M=94,945$) and Skewed Stress graphs ($M=94,970, d_{\max}=6,340$ super-hub) on Tesla T4
- **Target Report Section:** Section 12, 13, 14, 17 (CUDA Acceleration & Kernel Architecture)
- **Scientific Interpretation:** Highlights the 4.0x kernel speedup achieved by warp coalescing on uniform graphs, and the dramatic >30x speedup under skewed topologies where baseline thread-per-node suffers catastrophic SIMT divergence on the 6,340-edge super-hub.

### Figure 9: PCIe Data Transfer Bottleneck & Pinned Stream Overlap
- **Files:** `fig9_pcie_transfer_vs_kernel_breakdown.png`, `fig9_pcie_transfer_vs_kernel_breakdown.svg`
- **Source Artifact:** `exp_gpu_baseline_tier_a_sum_20261024_142037.json`, `exp_gpu_coalesced_tier_a_sum_20261024_142037.json`, `exp_gpu_streams_tier_a_sum_20261024_142037.json`
- **Fields / Variables:** `h2d_transfer_mean`, `kernel_mean`, `d2h_transfer_mean`, `end_to_end_mean`
- **Target Report Section:** Section 13 & 17 (Memory-Aware Optimization & PCIe Transfers)
- **Scientific Interpretation:** Formulates the central micro-architectural finding of FGI: PCIe data transfers (0.91 ms) dominate raw kernel compute (0.068 ms) by 13.5x. Shows how asynchronous double-buffering with pinned memory cuts end-to-end latency by 3.5x (0.99 ms down to 0.28 ms).

### Figure 10: Degree-Aware Dispatch on Power-Law Skewed Graph
- **Files:** `fig10_degree_aware_skew_tradeoff.png`, `fig10_degree_aware_skew_tradeoff.svg`
- **Source Artifact:** `exp_gpu_degree_aware_skewed_*_20261024_142037.json`, `exp_gpu_coalesced_skewed_*_20261024_142037.json`
- **Fields / Variables:** `kernel_mean` for MEAN, MAX, SUM under Coalesced vs Degree-Aware on `synthetic_skewed_10k.npz` ($d_{\max}=6,340$)
- **Target Report Section:** Section 14 & 17 (Degree-Aware Dispatch & Nuanced Trade-offs)
- **Scientific Interpretation:** Demonstrates nuanced engineering reality: degree-aware binning accelerates MEAN by 3.4x (0.065 ms) and MAX by 2.1x (0.111 ms), but REGRESSES SUM by 2.19x (0.464 ms). Code inspection of `fgi_degree_aware_kernels.cu` shows Warp 0 executes single-warp serial accumulation on hub nodes to enforce strict float32 non-associative numerical equivalence (atol <= 1e-5), leaving 7 of 8 warps idle in hub blocks. Zero atomic operations were present.

### Figure 11: CPU Vectorized SpMM vs Reference Sequential Loop
- **Files:** `fig11_cpu_vectorization_speedup.png`, `fig11_cpu_vectorization_speedup.svg`
- **Source Artifact:** `experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv`
- **Fields / Variables:** `mean_latency_ms` for `cpu_reference` vs `cpu_vectorized` on Tier A MEAN and MAX
- **Target Report Section:** Section 11 & 17 (CPU Reference & Vectorization Engine)
- **Scientific Interpretation:** Demonstrates that SciPy sparse matrix multiplication achieves 44.60x speedup over a sequential Python loop for MEAN, but fails to accelerate MAX (0.99x) due to the absence of native SpMM max-reduction kernels in classical BLAS/sparse libraries.

### Figure 12: Graph Topology In-Degree Characteristics and Temporal Dataset Split
- **Files:** `fig12_graph_topology_and_split.png`, `fig12_graph_topology_and_split.svg`
- **Source Artifact:** `data/fixtures/synthetic_fraud_10k.npz`, `synthetic_fraud_10k_manifest.json`
- **Fields / Variables:** `row_ptr` degree diffs (histogram), `dataset_split` temporal boundaries and fraud counts
- **Target Report Section:** Section 9 & 10 (Transaction Graph Representation & Synthetic Dataset)
- **Scientific Interpretation:** Illustrates the power-law in-degree tail (max in-degree 252, 1,155 isolated nodes) and the strictly causal chronological 70/15/15 split preventing temporal data leakage.
