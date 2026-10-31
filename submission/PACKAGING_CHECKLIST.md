# FraudGraph Intelligence (FGI): Final Submission Packaging Checklist

**Project:** FraudGraph Intelligence (FGI)  
**Academic Context:** CSI4007 — GPU Programming Capstone  
**Institution:** Vellore Institute of Technology (VIT), Vellore Campus  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline Commit:** `8afa28cf1172c006f0242f530267e8cc4466f9d6` (Branch `main`)  
**Document Purpose:** Quick-reference pre-submission checklist and viva defense protocol for the student.

---

## 1. Pre-Flight Repository Verification Checklist

Execute these fast-path commands in PowerShell to verify complete repository integrity before demonstration or submission:

- [ ] **1. Git Baseline Verification:**
  ```powershell
  cd "D:\FraudGraph Intelligence"
  git status --short
  git rev-parse HEAD
  ```
  *Expected Output:* HEAD equals `8afa28cf1172c006f0242f530267e8cc4466f9d6`. 0 staged files, 0 modified tracked files.
- [ ] **2. Canonical Dataset Cryptographic Integrity:**
  ```powershell
  Get-FileHash data\fixtures\synthetic_fraud_10k.npz -Algorithm SHA256
  ```
  *Expected Output:* `AECAFE3B1FD608C18D62AA1586D0AB6F88B867BD92B20E0F6BD133606C7C4430`
- [ ] **3. Backend Automated Test Suite:**
  ```powershell
  python -m pytest -o filterwarnings="ignore::DeprecationWarning"
  ```
  *Expected Output:* `188 passed, 2 skipped in ~90s` (Exit code 0).
- [ ] **4. Frontend Node Test Suites & TypeScript Compilation:**
  ```powershell
  node frontend/tests/polling_lifecycle.test.mjs
  node frontend/tests/subgraph_transactions_client.test.mjs
  cd frontend
  npm run typecheck
  npm run build
  cd ..
  ```
  *Expected Output:* 24 Node tests pass; 0 TypeScript errors; clean `dist/` asset bundle generated.

---

## 2. Live Demonstration Service Startup Checklist

Ensure both services start cleanly in separate terminal sessions before presenting to the evaluation panel:

### Terminal 1: FastAPI Backend Server
```powershell
cd "D:\FraudGraph Intelligence"
.venv\Scripts\activate
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- [ ] Verify startup log: `Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`
- [ ] Verify API Health check in browser: `http://localhost:8000/api/v1/health`
- [ ] Verify Swagger UI: `http://localhost:8000/docs`

### Terminal 2: React / Vite Analytical Frontend
```powershell
cd "D:\FraudGraph Intelligence\frontend"
npm run dev
```
- [ ] Verify startup log: `VITE v6.4.4 ready in ~300 ms ... Local: http://127.0.0.1:5173/`
- [ ] Open Dashboard: `http://localhost:5173/`

---

## 3. 8-Minute Demonstration Flow (Viva Presentation Guide)

Follow this minute-by-minute outline detailed in [`submission/DEMO_AND_VIVA_GUIDE.md`](DEMO_AND_VIVA_GUIDE.md):

| Time Slot | Presentation Section | Screen Target | Key Talking Points & Live Actions |
| :--- | :--- | :--- | :--- |
| **0:00 – 1:00** | **Motivation & Problem Statement** | Overview / Slides | Explain relational financial fraud (money mule rings, synthetic identities). Contrast tabular models (blind to links) with graph neighbor aggregation. Introduce GPU challenges: power-law skew and PCIe bottlenecks. |
| **1:00 – 2:00** | **System Architecture & Telemetry** | `/` (Dashboard) | Point to live KPI telemetry (10k nodes, 231k edges, 3.5% fraud). Open System Health Drawer (top right): explain local CPU host vs remote Tesla T4 Colab benchmarking. |
| **2:00 – 3:00** | **Dataset Registry & CSR Format** | `/datasets` | Highlight SHA-256 verification pill (`aecafe3b...`). Explain CSR layout (2.24 MB footprint, $>178\times$ compression vs dense matrix, 1,155 isolated nodes, strict causal forward-in-time invariant). |
| **3:00 – 4:15** | **Topological Subgraph Explorer** | `/graph` | Extract Center Node `0` at Depth `2` using BFS. Show fraud rings clustered around shared cards/devices. Show defensive edge truncation pill (`is_truncated: true`) preventing DOM freeze. |
| **4:15 – 5:15** | **Transaction Ledger & Filtering** | `/transactions` | Demonstrate server-side pagination across 10,000 transactions. Toggle `Fraud Only` to filter to the 350 ground-truth cases. |
| **5:15 – 6:30** | **CUDA Acceleration Lab** | `/cuda-lab` | **The Core HPC Defense:**<br>1. *Baseline (Thread/Node):* Catastrophic SIMT divergence ($6.98$ ms on skewed graph).<br>2. *Warp-Coalesced (1 Warp/Node):* Aligned 128-byte transactions $\to$ **$31.8\times$ speedup** ($0.22$ ms).<br>3. *PCIe Bottleneck:* PCIe transfer ($0.91$ ms) exceeds compute ($0.068$ ms) by $13.5\times$.<br>4. *Pinned Streams:* Overlapped double-buffering cuts latency by **$3.5\times$** ($0.99 \to 0.28$ ms).<br>5. *Degree-Aware Skew:* **$3.4\times$ gain** on MEAN, but **$2.19\times$ regression** on SUM due to Warp 0 serial accumulation to prevent float32 drift ($\text{atol} \le 10^{-5}$). |
| **6:30 – 7:30** | **Downstream ML Evaluation** | `/evaluation` | Explain PR-AUC primacy over ROC-AUC under 3.5% imbalance. Highlight operational trade-offs: LR MAX boosts PR-AUC to 0.3315 and Recall to 45.3% (Precision drops to 25.5%); RF MEAN cuts false alerts by 78.3% (Recall drops from 52.8% to 28.3%). Show catastrophic collapse of SUM ($>50\%$ drop). |
| **7:30 – 8:00** | **Summary & Defense** | Reports / Conclusion | Summarize verified software tests (188 pytest, 24 Node). Disclose local GPU absence and deferred WebGPU shaders. Invite faculty questions. |

---

## 4. High-Stakes Viva Defense Cheat Sheet

Be prepared to answer these top technical questions from faculty:

1. **Why does Warp Coalescing yield >31x speedup on the skewed graph?**  
   *Answer:* Consecutive threads in a warp load feature dimensions $d \in [0, 31]$ contiguously in memory for the same neighbor, triggering a single 128-byte DRAM transaction instead of 32 scattered transactions. Bandwidth increases from 43.8 GB/s to 175.3 GB/s.
2. **What caused the Degree-Aware SUM regression on the skewed graph?**  
   *Answer:* Out-of-order multi-warp tree reductions across the 6,340-edge super-hub introduce IEEE 754 float32 non-associative rounding drift ($3.97 \times 10^{-4}$), violating the project's $\text{atol} \le 10^{-5}$ tolerance. To guarantee mathematical parity, `hub_degree_block_kernel` restricts SUM accumulation to Warp 0 (`if (warp_id == 0)`), leaving 7 of 8 warps idle in hub blocks. Zero atomic instructions were used.
3. **Why is PR-AUC superior to ROC-AUC for this project?**  
   *Answer:* With 3.5% fraud prevalence, the 1,447 licit transactions flood the false positive rate calculation in ROC-AUC ($FPR = FP / (FP + TN)$), compressing ROC-AUC scores into an artificially high 0.73–0.79 band. PR-AUC evaluates Precision ($TP / (TP + FP)$) directly against Recall, making it sensitive to false alarm volume.
4. **Why did Random Forest MEAN reduce false positives but hurt recall?**  
   *Answer:* MEAN aggregation smooths extreme individual features by averaging across neighbors. In non-linear decision trees, this suppresses spurious high-value anomalies, slashing false alarms from 138 to 30 (78.3% reduction), but it also attenuates isolated fraud signals, collapsing recall from 52.8% to 28.3% (-24.5 pp).

---

## 5. Manual PDF Export Instructions

Since headless document compilation tools (`pandoc`, `typst`) are not installed on the Windows host:

1. **Open the Manuscript:** Open `submission/report/FraudGraph_Intelligence_Final_Report.md` in VS Code or Chrome/Edge.
2. **Export to PDF:**
   - In VS Code: Use the *Markdown PDF: Export (pdf)* extension.
   - In Chrome / Edge: Open rendered Markdown preview, press `Ctrl + P`, set Destination to **Save as PDF**, check **Background graphics**, and set Margins to **Default / 0.75 in**.
3. **Save Target:** Save the output as `submission/report/FraudGraph_Intelligence_Final_Report.pdf`.
4. **Verify Export:** Confirm that Mermaid diagrams, mathematical formulas, and Table 1B render legibly.

---

## 6. Personal Tasks to Complete Before Submission

- [ ] Export `FraudGraph_Intelligence_Final_Report.md` to PDF.
- [ ] Obtain the official VIT CSI4007 course rubric and verify that title, team registration numbers, and guide details match requirements.
- [ ] Practice the 8-minute demonstration walkthrough using the live dashboard.
- [ ] If submitting via VTOP as a ZIP file, compress the `submission/` folder along with `docs/PHASE_12_EVIDENCE_LEDGER.md`.

---
*Checklist Compiled and Certified for FraudGraph Intelligence Capstone Submission.*
