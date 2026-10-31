# Table 6: Downstream Fraud Classification Performance & Enriched Representation Comparison

**Primary Source Artifact:** `experiments/results/fraud_model_evaluation_20261024_212032.json`  
**Summary Metrics CSV:** `experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv`  
**Dataset Split:** Chronological 70% Train (7,000 nodes, 239 frauds) / 15% Validation (1,500 nodes, 58 frauds) / 15% Test (1,500 nodes, 53 frauds)  
**Positive Test Ground Truth:** 53 Frauds (3.533% prevalence) | Random Guessing PR-AUC Baseline: 0.0353  
**Threshold Selection Protocol:** Optimal F1 decision threshold selected strictly on Validation Set and frozen for Test Evaluation  

| Model Architecture | Graph Feature Representation | Feature Dim ($D$) | Test PR-AUC | Test ROC-AUC | Optimal Test F1 | Test Precision | Test Recall | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Feature Enrichment Time (ms) | Training Time (ms) | Test Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline Raw Tabular | 32 | 0.308794 | 0.785503 | 0.344828 | 0.317460 | 0.377358 | 20 | 43 | 33 | 1404 | 0.748 ms | 7.177 ms | 0.188 ms |
| **Logistic Regression** | **Graph MAX Enriched** | **64** | **0.331471** | **0.786481** | **0.326531** | **0.255319** | **0.452830** | **24** | **70** | **29** | **1377** | **60.047 ms** | **10.549 ms** | **0.182 ms** |
| **Logistic Regression** | Graph MEAN Enriched | 64 | 0.300959 | 0.776284 | 0.320000 | 0.247423 | 0.452830 | 24 | 73 | 29 | 1374 | 6.415 ms | 7.206 ms | 0.215 ms |
| **Logistic Regression** | Multi-Operator (Mean+Max) | 96 | 0.247094 | 0.780000 | 0.338710 | 0.295775 | 0.396226 | 21 | 50 | 32 | 1397 | 69.780 ms | 24.533 ms | 0.274 ms |
| **Logistic Regression** | Graph SUM Enriched | 64 | 0.148773 | 0.749932 | 0.275862 | 0.217391 | 0.377358 | 20 | 72 | 33 | 1375 | 6.424 ms | 9.228 ms | 0.286 ms |
| **Random Forest** | Baseline Raw Tabular | 32 | 0.244511 | 0.778605 | 0.255708 | 0.168675 | **0.528302** | **28** | **138** | **25** | **1309** | 0.707 ms | 852.122 ms | 26.337 ms |
| **Random Forest** | **Graph MEAN Enriched** | **64** | **0.258714** | **0.779185** | **0.306122** | **0.333333** | **0.283019** | **15** | **30** | **38** | **1417** | **6.031 ms** | **1507.652 ms** | **26.612 ms** |
| **Random Forest** | Graph MAX Enriched | 64 | 0.186520 | 0.734636 | 0.218579 | 0.153846 | 0.377358 | 20 | 110 | 33 | 1337 | 67.245 ms | 1343.342 ms | 26.173 ms |
| **Random Forest** | Multi-Operator (Mean+Max) | 96 | 0.217212 | 0.772353 | 0.297872 | 0.341463 | 0.264151 | 14 | 27 | 39 | 1420 | 66.972 ms | 1760.416 ms | 25.266 ms |
| **Random Forest** | Graph SUM Enriched | 64 | 0.123620 | 0.742375 | 0.177419 | 0.112821 | 0.415094 | 22 | 173 | 31 | 1274 | 4.640 ms | 1392.833 ms | 25.632 ms |

> [!NOTE]
> **Key Analytical Findings & Operational Metric Trade-offs:**
> 1. **Logistic Regression MAX Metric Trade-off:** Logistic Regression enriched with MAX neighbor vectors achieved the highest overall PR-AUC of **0.3315** (+7.3% relative gain over raw tabular 0.3088) and lifted fraud recall from 37.7% to **45.3%** (catching 24 out of 53 frauds vs 20). However, this represents an operational metric trade-off: precision declined from **31.7% to 25.5%** (false positives increased from 43 to 70), and optimal F1 slightly dropped from **0.3448 to 0.3265**. MAX aggregation acts as a high-sensitivity alert filter that captures more true fraud at the expense of higher analyst review volume.
> 2. **Random Forest MEAN Metric Trade-off:** In Random Forest, MEAN neighbor enrichment reduced false alerts from 138 down to 30 (**78.3% false-positive reduction**), lifting precision from **16.9% to 33.3%** and lifting F1 from **0.2557 to 0.3061** (PR-AUC slightly increased from 0.2445 to 0.2587). However, this must NOT be evaluated as an unqualified improvement: **fraud recall dropped drastically from 52.8% to 28.3%** (-24.5 percentage points), with false negatives surging from **25 to 38** (missing 13 additional fraudulent transactions; true positives fell from 28 to 15). MEAN enrichment shifts the operational operating point toward extreme conservatism, drastically cutting investigator fatigue but letting more fraud slip through.
> 3. **Catastrophic Failure of SUM Aggregation:** Unnormalized SUM aggregation degrades PR-AUC by more than 50% across both models (LR: 0.3088 $\to$ 0.1488; RF: 0.2445 $\to$ 0.1236). This confirms that raw sums conflate high node in-degree with fraudulence, flooding classifiers with high-degree benign accounts.
