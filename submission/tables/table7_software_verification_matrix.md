# Table 7: Software Suite Verification Matrix

**Verification Audit Timestamp:** 2026-10-25T17:51:15+05:30  
**Target Repository Workspace:** `D:\FraudGraph Intelligence`  
**Git Branch & Commit Baseline:** `main` @ `8afa28cf1172c006f0242f530267e8cc4466f9d6`  

| Quality Gate / Test Scope | Exact Verification Command | Target Components Tested | Tests Run | Pass | Skip | Fail | Execution Duration | Exit Code / Audit Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Backend Unit & API Suite** | `python -m pytest -o filterwarnings="ignore::DeprecationWarning"` | CPU Aggregators, CSR Graph, Synthetic Generator, Feature Enrichment, Health API, Dashboard API, Graph Subgraph API, Transactions Pagination API, Benchmark Engine | 190 | **188** | **2** | 0 | 89.94s | **EXIT 0 (PASS)** |
| **Frontend Polling Lifecycle** | `node frontend/tests/polling_lifecycle.test.mjs` | Monotonic request sequencing, race conditions, start/stop transitions, page visibility pauses, error classifications (F-01 to F-07) | 21 | **21** | 0 | 0 | 142.96ms | **EXIT 0 (PASS)** |
| **Frontend API Client Suite** | `node frontend/tests/subgraph_transactions_client.test.mjs` | Typed fetch callers (`getSubgraph`, `getTransactions`), query param serialization, response parsing | 3 | **3** | 0 | 0 | 7.49ms | **EXIT 0 (PASS)** |
| **Frontend TypeScript Types** | `npm run typecheck` (`tsc -b`) | Complete React TypeScript AST, component props, API response interfaces, route param types | — | **0 Errors** | 0 | 0 | 1.84s | **EXIT 0 (PASS)** |
| **Frontend Production Build** | `npm run build` (`tsc -b && vite build`) | Production Vite bundler, CSS minification, chunk splitting, asset pipeline (`dist/`) | 1,621 modules | **Clean Build** | 0 | 0 | 5.81s | **EXIT 0 (PASS)** |
| **Canonical Fixture Hash** | `Get-FileHash data/fixtures/synthetic_fraud_10k.npz` | Cryptographic integrity of benchmark dataset | 1 | **Match** | 0 | 0 | 0.05s | **SHA256: `aecafe3b...`** |
| **Git Working Tree Audit** | `git status --short` | Tracked changes, index staging, unexpected modifications | — | **Clean** | — | — | 0.02s | **0 Staged, 0 Modified** |

### Explicit Disclosures on Test Scope & Environment:
1. **Skipped Backend Tests (2):** `test_cuda_baseline.py` and `test_feature_enrichment.py` skip gracefully when `torch.cuda.is_available()` is False and local NVCC is absent.
2. **Headless Audit Multi-Process Socket Test:** `frontend/tests/api_integration.test.mjs` requires a live backend on port 8000 and Vite server on port 5173. During headless verification sweeps, port 8000 was bound by an external background process (`uvicorn` returning 404), so live multi-process socket tests were omitted from automated batch sweeps. The script remains preserved and untracked.
3. **Browser E2E Automation:** Headless browser automation (Playwright/Cypress) is not configured in `frontend/package.json`. All frontend verification is conducted via Node test runner, TypeScript compiler, and Vite build pipeline.
