# Phase 7.5 Evaluation Report — Research-Grade Reporting & Visualizations

**Project ID:** FraudGraph Intelligence (FGI)
**Academic Course:** VIT Vellore — CSI4007 GPU Programming
**Workspace:** `D:\FraudGraph Intelligence`
**Phase:** 7.5 — Research-Grade Evaluation Reporting & Visualizations
**Evaluation Target:** Controlled Downstream Fraud Classification Benchmark
**Date:** 2026-10-24
**Status:** Pre-Commit Review Complete

---

## 1. Executive Summary & Provenance

Phase 7.5 constructs the publication-grade reporting, visualization, and tabular analysis layer for the FraudGraph Intelligence evaluation program. It converts the canonical experiment evaluation artifact generated in Phase 7.4 into high-resolution multi-format figures (PNG at 300 DPI, vector SVG, vector PDF), an exhaustive machine-readable CSV summary table, and a cryptographic JSON manifest recording SHA-256 provenance across all inputs and outputs.

### 1.1 Cryptographic Provenance & Artifact Integrity

All reported metrics, figures, and tables are derived directly and deterministically from the canonical Phase 7.4 evaluation artifact and Phase 7.2 dataset fixture. No metric values, curve coordinates, or confusion matrix counts are hardcoded or manually transcribed.

| Artifact Role | File Path | Format | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Input: Synthetic Dataset Fixture** | `data/fixtures/synthetic_fraud_10k.npz` | Compressed NPZ | `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430` |
| **Input: Canonical Evaluation JSON** | `experiments/results/fraud_model_evaluation_20261024_212032.json` | JSON Schema | `155304b7453b15dda1f103817e32471a2ffb2357de9cb298cc7b470d8d0efc90` |
| **Output: Evaluation Metrics CSV** | `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv` | Full-Precision CSV | `86523fd6d6f998602124ba101972eba15dc5b27bc97c852b3dee475dd6f77ba4` |
| **Output: Verification Manifest** | `experiments/results/evaluation_reporting_7_5/manifest.json` | JSON Manifest | Dynamic (records all figure and CSV hashes) |

### 1.2 Execution Environment

- **Python Runtime:** Python 3.14.3 (`MSC v.1944 64-bit AMD64`)
- **Core Libraries:** `matplotlib 3.11.2`, `pandas 3.0.6`, `numpy 2.5.4`, `scikit-learn 1.9.1`, `scipy 1.18.1`
- **Host CPU Platform:** Windows 11 Enterprise (Build 10.0.26200-SP0), Intel64 Family 6 Model 140 Stepping 1
- **GPU Architecture Reference:** NVIDIA Tesla T4 (40 SMs, 2,560 CUDA cores, 14.9 GB VRAM, Driver 580.82.07, CUDA 13.0)

---

## 2. Comprehensive Evaluation Metric Tables

### 2.1 Primary Streaming Benchmark (`causal_history`)

Under the `causal_history` streaming transaction policy, each transaction $i$ aggregates feature embeddings from any preceding transaction $j < i$ with $t_j < t_i$. Test evaluation is conducted on $N=1,500$ transactions ($53$ frauds, class prevalence $\pi \approx 0.0353$) using decision thresholds frozen strictly on the validation partition ($N=1,500$, $58$ frauds).

| Model | Representation | Dim ($D$) | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Val $\tau^*$ | Test Precision | Test Recall | Test F1 | TP | FP | TN | FN | $t_{\text{enrich}}$ (ms) | $t_{\text{train}}$ (ms) | $t_{\text{test}}$ (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `raw` | 32 | 0.2590 | 0.8213 | **0.3088** | 0.7855 | 0.1684 | 0.3175 | 0.3774 | **0.3448** | 20 | 43 | 1404 | 33 | 0.75 | 7.18 | 0.19 |
| **Logistic Regression** | `mean` | 64 | 0.2016 | 0.8239 | 0.3010 | 0.7763 | 0.1090 | 0.2474 | 0.4528 | 0.3200 | 24 | 73 | 1374 | 29 | 6.42 | 7.21 | 0.21 |
| **Logistic Regression** | `max` | 64 | 0.2405 | 0.8209 | **0.3315** | **0.7865** | 0.1090 | 0.2553 | 0.4528 | 0.3265 | 24 | 70 | 1377 | 29 | 60.05 | 10.55 | 0.18 |
| **Logistic Regression** | `multi_operator` | 96 | 0.2070 | 0.8084 | 0.2471 | 0.7800 | 0.1387 | 0.2958 | 0.3962 | 0.3387 | 21 | 50 | 1397 | 32 | 69.78 | 24.53 | 0.27 |
| **Logistic Regression** | `sum` | 64 | 0.3054 | 0.8100 | 0.1488 | 0.7499 | 0.2278 | 0.2174 | 0.3774 | 0.2759 | 20 | 72 | 1375 | 33 | 6.42 | 9.23 | 0.29 |
| **Random Forest** | `raw` | 32 | 0.1967 | 0.7495 | 0.2445 | 0.7786 | 0.0991 | 0.1687 | **0.5283** | 0.2557 | 28 | 138 | 1309 | 25 | 0.71 | 852.12 | 26.34 |
| **Random Forest** | `mean` | 64 | 0.2259 | 0.7484 | **0.2587** | **0.7792** | 0.1882 | **0.3333** | 0.2830 | **0.3061** | 15 | 30 | 1417 | 38 | 6.03 | 1507.65 | 26.61 |
| **Random Forest** | `max` | 64 | 0.1941 | 0.7648 | 0.1865 | 0.7346 | 0.1189 | 0.1538 | 0.3774 | 0.2186 | 20 | 110 | 1337 | 33 | 67.25 | 1343.34 | 26.17 |
| **Random Forest** | `multi_operator` | 96 | 0.1751 | 0.7586 | 0.2172 | 0.7724 | 0.1684 | 0.3415 | 0.2642 | 0.2979 | 14 | 27 | 1420 | 39 | 66.97 | 1760.42 | 25.27 |
| **Random Forest** | `sum` | 64 | 0.1519 | 0.7648 | 0.1236 | 0.7424 | 0.1387 | 0.1128 | 0.4151 | 0.1774 | 22 | 173 | 1274 | 31 | 4.64 | 1392.83 | 25.63 |

### 2.2 Context Policy Sensitivity Benchmark (`train_only_context`)

The `train_only_context` sensitivity policy strictly prunes any graph edge pointing to validation or test nodes ($j \ge 7,000$). In this regime, validation and test nodes aggregate solely from historical training nodes. This experiment measures how test classification behaves when test transactions are separated from the training set by a 15-day validation buffer ($[7000, 8500)$).

| Model | Representation | `causal_history` Test PR-AUC | `train_only_context` Test PR-AUC | $\Delta$ PR-AUC | `causal_history` Test F1 | `train_only_context` Test F1 | $\Delta$ F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `raw` | 0.3088 | 0.3088 | 0.0000 | 0.3448 | 0.3448 | 0.0000 |
| **Logistic Regression** | `mean` | 0.3010 | 0.2793 | -0.0216 | 0.3200 | 0.3026 | -0.0174 |
| **Logistic Regression** | `max` | 0.3315 | 0.2817 | -0.0497 | 0.3265 | 0.3269 | +0.0004 |
| **Logistic Regression** | `multi_operator` | 0.2471 | 0.2644 | +0.0173 | 0.3387 | 0.3043 | -0.0344 |
| **Logistic Regression** | `sum` | 0.1488 | 0.2845 | +0.1358 | 0.2759 | 0.3212 | +0.0453 |
| **Random Forest** | `raw` | 0.2445 | 0.2445 | 0.0000 | 0.2557 | 0.2557 | 0.0000 |
| **Random Forest** | `mean` | 0.2587 | 0.2021 | -0.0566 | 0.3061 | 0.2857 | -0.0204 |
| **Random Forest** | `max` | 0.1865 | 0.2028 | +0.0163 | 0.2186 | 0.1948 | -0.0238 |
| **Random Forest** | `multi_operator` | 0.2172 | 0.1393 | -0.0779 | 0.2979 | 0.1266 | -0.1713 |
| **Random Forest** | `sum` | 0.1236 | 0.2288 | +0.1052 | 0.1774 | 0.2835 | +0.1060 |

---

## 3. Publication Visualizations & Analysis

The Phase 7.5 reporting pipeline generated 7 dedicated figures in three formats (`.png` at 300 DPI, `.svg` vector, and `.pdf` vector). All files reside in `experiments/results/evaluation_reporting_7_5/`.

### Figure 1: Test PR-AUC Comparison Across Representations

- **Files:** `fig1_test_pr_auc_comparison.png`, `.svg`, `.pdf`
- **Source Metrics:** `primary_results_causal_history[*].test_metrics.pr_auc`, `dataset_split.test_prevalence`
- **Visual Encoding:** Grouped vertical bar chart comparing Logistic Regression (blue) and Random Forest (green) across the 5 representations (`raw`, `mean`, `max`, `multi_operator`, `sum`). Includes the horizontal dashed red reference line for minority class test prevalence ($\pi \approx 0.0353$).
- **Key Empirical Observations:**
  1. *Substantial Separation Above Random:* All models significantly exceed the baseline prevalence ($\pi = 0.0353$), confirming that the synthetic classification task is non-trivial.
  2. *Operator-Specific Gains:* Logistic Regression achieves its highest PR-AUC under `max` aggregation ($0.3315$ vs $0.3088$ raw, a $+7.35\%$ relative gain), while Random Forest achieves its highest PR-AUC under `mean` aggregation ($0.2587$ vs $0.2445$ raw, a $+5.81\%$ relative gain).
  3. *Unnormalized Sum Degradation:* Both classifiers experience sharp performance drops under unnormalized `sum` aggregation ($0.1488$ for LR, $0.1236$ for RF). In skewed power-law degree distributions, raw sums scale with vertex in-degree rather than local behavioral intensity, corrupting the feature manifold.

### Figure 2: Test ROC-AUC Comparison Across Representations

- **Files:** `fig2_test_roc_auc_comparison.png`, `.svg`, `.pdf`
- **Source Metrics:** `primary_results_causal_history[*].test_metrics.roc_auc`
- **Visual Encoding:** Grouped vertical bar chart with y-axis bounded on $[0.50, 0.90]$ and an explicit note on metric compression under class imbalance.
- **Key Empirical Observations:**
  1. *Metric Compression:* While PR-AUC exhibits large variance ($0.1236$ to $0.3315$), ROC-AUC remains compressed between $0.7346$ and $0.7865$.
  2. *Class Imbalance Artifact:* Because non-fraud transactions constitute $96.5\%$ of the test set ($1,447 / 1,500$), the False Positive Rate denominator ($FP + TN = 1,447$) is large. Consequently, substantial numbers of false alerts produce small changes in the False Positive Rate, compressing ROC-AUC. PR-AUC is therefore the more sensitive and informative metric for fraud detection.

### Figure 3: Precision–Recall Curve Availability & Protocol Notice

- **Files:** `fig3_pr_curve_omission_notice.png`, `.svg`, `.pdf`
- **Source Fields:** `evaluation_protocol.threshold_selection_protocol`
- **Visual Encoding:** Methodological card explaining why continuous PR curves are omitted in accordance with scientific integrity safeguards.
- **Scientific Integrity Safeguard:**
  - The canonical Phase 7.4 experiment JSON stores aggregate summary scalars, precision/recall at operating points, and confusion matrices. It does not store the $1,500$-element continuous test prediction score vectors.
  - Generating continuous PR curves from summary scalars would require fabricating continuous scores or assuming artificial distributions. In accordance with strict empirical standards, continuous curves are omitted, directing the reader to Figure 1 (PR-AUC) and Figure 5 (discrete operating trade-offs).

### Figure 4: Test Confusion Matrices at Validation-Selected Thresholds

- **Files:** `fig4_confusion_matrices.png`, `.svg`, `.pdf`
- **Source Fields:** `primary_results_causal_history[*].test_metrics.optimal_f1_evaluation.confusion_matrix`
- **Visual Encoding:** $2 \times 2$ grid of confusion matrix heatmaps for the 4 primary configurations:
  1. Logistic Regression: Raw Features ($D=32, \tau^* = 0.1684$)
  2. Logistic Regression: Enriched Max ($2D=64, \tau^* = 0.1090$)
  3. Random Forest: Raw Features ($D=32, \tau^* = 0.0991$)
  4. Random Forest: Enriched Mean ($2D=64, \tau^* = 0.1882$)
- **Key Empirical Observations:**
  - *Random Forest False Alert Reduction:* Adding `mean` graph features to Random Forest decreases false positives from $138$ to $30$ (a $-78.3\%$ false alarm reduction), increasing precision from $16.87\%$ to $33.33\%$, with recall adjusting from $52.83\%$ ($28/53$) to $28.30\%$ ($15/53$) under validation threshold selection.
  - *Logistic Regression Max Recall Boost:* For Logistic Regression, `max` aggregation increases true positives from $20$ to $24$ (recall from $37.74\%$ to $45.28\%$), while false positives increase from $43$ to $70$.

### Figure 5: Fraud Detection Operating Trade-off Comparison

- **Files:** `fig5_operating_tradeoffs.png`, `.svg`, `.pdf`
- **Source Fields:** Precision, Recall, F1, False Positives, False Negatives from optimal F1 evaluations.
- **Visual Encoding:** Dual-panel display:
  - *Panel A:* Grouped bar comparison of Precision, Recall, and F1 across raw and graph-enriched representations.
  - *Panel B:* Error volume breakdown showing False Positives (red) and False Negatives (orange).
- **Key Empirical Observations:**
  - Clearly demonstrates that no single representation dominates on all operating metrics simultaneously. `max` enables higher recall at the expense of additional investigation workload (higher FP), while `mean` enables higher precision and compact alert volumes (lower FP) at the expense of lower recall.

### Figure 6: Computational Timing Profile & Hardware Boundary Qualification

- **Files:** `fig6_runtime_breakdown.png`, `.svg`, `.pdf`
- **Source Fields:** `timing_ms.feature_enrichment_ms`, `timing_ms.model_training_ms`, `timing_ms.test_inference_ms`
- **Visual Encoding:** Three distinct panels:
  - *Panel A:* CPU Feature Enrichment Latency across representations (`raw`: $0.75\text{ ms}$, `sum`: $6.42\text{ ms}$, `mean`: $6.42\text{ ms}$, `max`: $60.05\text{ ms}$, `multi-op`: $69.78\text{ ms}$).
  - *Panel B:* Classifier Training and Test Scoring Latency.
  - *Panel C:* Hardware Execution Profile & Timing Boundary Qualification card.
- **Boundary-Qualified Hardware Qualification:**
  - *Explicit Non-Equivalence:* CPU enrichment timing ($60.05\text{ ms}$ for `max`) represents a host Python iterative loop. In contrast, the Phase 6.3 CUDA Degree-Aware kernel execution time ($0.022\text{ ms}$ on Tesla T4) represents isolated GPU hardware execution measured via `cudaEventElapsedTime`.
  - *No Unqualified Speedup Claim:* These timing boundaries are fundamentally different (host process pipeline vs isolated device kernel). Claiming an end-to-end acceleration ratio between them would be scientifically invalid without measuring end-to-end host-device memory transfers and framework overhead.

### Figure 7: Context-Policy Sensitivity Analysis

- **Files:** `fig7_context_policy_sensitivity.png`, `.svg`, `.pdf`
- **Source Fields:** `primary_results_causal_history[*].test_metrics.pr_auc`, `sensitivity_results_train_only[*].test_metrics.pr_auc`
- **Visual Encoding:** Paired comparison bars across representations for both classifiers, showing `causal_history` (blue) alongside `train_only_context` (purple).
- **Key Empirical Observations:**
  - Pruning all graph context across the 15-day validation buffer reduces test PR-AUC for high-performing representations (RF `mean` drops from $0.2587$ to $0.2021$; LR `max` drops from $0.3315$ to $0.2817$).
  - Conversely, `sum` aggregation exhibits the opposite behavior ($0.1488 \to 0.2845$ for LR), because removing recent neighbors lowers the variance of extreme in-degrees in power-law subgraphs.

---

## 4. Scientific Integrity Disclosures & Methodological Invariants

1. **Synthetic Benchmark Boundary:**
   - All evaluations reported herein are conducted strictly on a deterministic synthetic benchmark fixture (`data/fixtures/synthetic_fraud_10k.npz`). The 3.5% fraud rate reflects realistic class imbalance, but the feature distributions and transaction graph topology do not represent live banking data or production financial records.
2. **Frozen Validation Threshold Protocol:**
   - In financial fraud detection, selecting decision thresholds on the test set is a common form of metric snooping. In this study, optimal F1 thresholds ($\tau^*$) and target recall thresholds ($\tau_{0.50}$) were identified exclusively on validation set predictions and frozen prior to test scoring.
3. **Train-Only Preprocessing Isolation:**
   - The feature scaler (`StandardScaler`) was fitted strictly on $X_{\text{train}}$. Validation and test features were transformed using training statistics ($(\vec{x} - \vec{\mu}_{\text{train}}) / \vec{\sigma}_{\text{train}}$), guaranteeing zero transductive feature leakage.
4. **Label Quarantine:**
   - Ground-truth binary labels $y \in \{0, 1\}$ were strictly quarantined from graph construction, edge selection, neighbor aggregation, and feature transformation.
5. **No Metric Fabrication:**
   - Where continuous prediction score vectors were not available in the summary artifact, continuous curves were omitted rather than fabricated, preserving total auditability.

---

## 5. Verification Checklist

- [x] All 7 figures generated in PNG (300 DPI), SVG, and PDF formats.
- [x] Machine-readable CSV (`summary_evaluation_metrics.csv`) matches source JSON across all 20 rows.
- [x] Verification manifest (`manifest.json`) generated with dynamic SHA-256 hashes.
- [x] All 11 unit tests in `backend/tests/test_reporting.py` passing.
- [x] Full backend test suite passing (127 passed, 2 skipped).
- [x] Immutable input files (`synthetic_fraud_10k.npz`, `fraud_model_evaluation_20261024_212032.json`) verified identical before and after reporting runs.
- [x] Zero metric fabrication and complete boundary disclosures maintained.
