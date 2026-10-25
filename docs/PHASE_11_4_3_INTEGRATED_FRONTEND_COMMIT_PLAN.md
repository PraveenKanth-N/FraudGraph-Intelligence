# Phase 11.4.3 — Integrated Frontend Commit Plan

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Current Git HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Plan Date:** October 25, 2026  
**Document Status:** PENDING USER REVIEW & EXPLICIT STAGING AUTHORIZATION  
**Deliverable File:** `docs/PHASE_11_4_3_INTEGRATED_FRONTEND_COMMIT_PLAN.md`  

---

## 1. Executive Summary & Verification Statement

This document defines the exact file allowlist, functional architecture, dependency matrix, and verification proof for consolidating all frontend improvements across Phases 9C through 11.3.1 into **one single integrated frontend commit**.

### Explicit Execution Statement
> [!IMPORTANT]
> **This plan has NOT been staged or committed.**  
> In strict accordance with the mandatory approval gate, zero files have been staged (`git diff --cached` is 100% clean), no commits have been created, no files have been discarded, and Git HEAD remains strictly at `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`. All canonical fixtures and benchmark outputs remain pristine.

---

## 2. Integrated Frontend Architecture & Rationale

In the original Phase 11.4.2 plan, frontend modifications were distributed across Commits 1, 2, 3, and 4. However, because TypeScript interfaces, shared API client singletons (`apiClient.ts`), layout status props (`SystemHealthDrawer.tsx`), and data visualizers share tight intra-frontend coupling, splitting individual files across commits led to preflight conflicts where pure files depended on uncommitted types or client methods.

Consolidating all frontend changes into **one integrated frontend commit** achieves:
1. **Self-Contained Coherence:** Every frontend route, hook, visualizer, and type definition is completely satisfied without broken imports or missing props.
2. **Strict Backend Decoupling:** Zero backend application code or backend tests are bundled into this commit. The frontend gracefully communicates with backend contracts via resilient fallbacks and typed models.
3. **Pristine Build Verification:** Verified to build cleanly via Vite and typecheck with zero errors via TypeScript compiler.
4. **Isolated Maintenance:** Completely excludes compiler build artifacts (`frontend/tsconfig.tsbuildinfo`) and historical documentation.

---

## 3. Exact Proposed File Allowlist

The proposed commit contains exactly **22 files** (12 tracked modified files, 10 untracked files).

| # | Exact File Path | Status | Functional Role & Reason for Inclusion | Backend / API Dependencies | Required Validating Tests | Multi-Phase Origin? |
|---|---|:---:|---|---|---|:---:|
| 1 | `frontend/src/types/telemetry.ts` | Untracked | Defines `TelemetryState` (`IDLE`, `POLLING`, `AVAILABLE`, `DEGRADED`, `HISTORICAL`), intervals, and configuration interfaces. | None (pure types) | `polling_lifecycle.test.mjs` | Phase 9C.1.2 |
| 2 | `frontend/src/types/api.ts` | Tracked M | Consolidated API models: error details (`ApiErrorKind`), dataset availability, artifact states (`artifact_state`), subgraph visualizer models (`SubgraphResponse`), and ledger models (`TransactionListResponse`). | Backend JSON schemas | `subgraph_transactions_client.test.mjs`, `tsc --noEmit` | **YES** (Phases 9C, 10, 11.2B) |
| 3 | `frontend/src/services/apiClient.ts` | Untracked | Resilient fetch client with abort signal propagation, timeout management, error classification, and typed endpoint callers (`getHealth`, `getDashboardSummary`, `getDatasets`, `getSubgraph`, `getTransactions`). | `/api/v1/*` endpoints | `polling_lifecycle.test.mjs`, `subgraph_transactions_client.test.mjs` | **YES** (Phases 9C.1.2, 11.2B) |
| 4 | `frontend/src/services/api.ts` | Tracked M | Re-exports `apiClient` singleton and provides backwards-compatible async delegators for legacy page callers. | `apiClient.ts` | `tsc --noEmit`, `npm run build` | **YES** (Phases 9C, 11.2B) |
| 5 | `frontend/src/hooks/usePollingResource.ts` | Untracked | Base telemetry lifecycle hook managing interval timers, document visibility pause/resume, monotonic sequence counters, and state transitions. | `telemetry.ts` | `polling_lifecycle.test.mjs` | Phase 9C.1.2 |
| 6 | `frontend/src/hooks/useHealthPolling.ts` | Untracked | Specialized poller for `/api/v1/health` with 15s interval and immediate manual trigger. | `apiClient.getHealth` | `polling_lifecycle.test.mjs` | Phase 9C.1.2 |
| 7 | `frontend/src/hooks/useDashboardPolling.ts` | Untracked | Specialized poller for `/api/v1/dashboard/summary` with 30s interval. | `apiClient.getDashboardSummary` | `polling_lifecycle.test.mjs` | Phase 9C.1.2 |
| 8 | `frontend/src/hooks/useDatasetsPolling.ts` | Untracked | Specialized poller for `/api/v1/datasets` with 30s interval. | `apiClient.getDatasets` | `polling_lifecycle.test.mjs` | Phase 9C.1.2 |
| 9 | `frontend/src/hooks/index.ts` | Untracked | Barrel export for telemetry polling hooks. | `hooks/*` | `tsc --noEmit` | Phase 9C.1.2 |
| 10 | `frontend/src/components/layout/AppLayout.tsx` | Tracked M | Universal application layout; orchestrates health poller, provides stabilized context object to children via `useMemo`, and passes telemetry props to header and drawer. | `useHealthPolling` | `tsc --noEmit`, `npm run build` | Phase 9C.1.2 |
| 11 | `frontend/src/components/layout/TopNav.tsx` | Tracked M | Header navigation; renders live telemetry pulse indicator and visual state badges (`POLLING`, `DEGRADED`, `HEALTHY`). | `telemetry.ts` | `tsc --noEmit`, `npm run build` | Phase 9C.1.2 |
| 12 | `frontend/src/components/layout/SystemHealthDrawer.tsx` | Tracked M | Slide-out diagnostic drawer; displays runtime telemetry, last probe timestamp, probe failure warnings, and truthful hardware accelerator status without fabricated CC 7.5 claims. | `telemetry.ts`, `api.ts` | `tsc --noEmit`, `npm run build` | **YES** (Phases 9C, 11.3.1) |
| 13 | `frontend/src/pages/OverviewPage.tsx` | Tracked M | System overview dashboard; displays live telemetry polling state, artifact failure state cards (`AVAILABLE`, `MISSING`, `INVALID`), dynamic variability badges, and truthful null GPU indicators. | `useDashboardPolling` | `tsc --noEmit`, `npm run build` | **YES** (Phases 9C, 10, 10.2.1, 11.3.1) |
| 14 | `frontend/src/pages/DatasetsPage.tsx` | Tracked M | Dataset inventory; displays verified fixture count, SHA-256 provenance badges, manual fixture refresh, and degraded telemetry banners. | `useDatasetsPolling` | `tsc --noEmit`, `npm run build` | Phase 9C.2 |
| 15 | `frontend/src/pages/GraphExplorerPage.tsx` | Tracked M | Concentric radial SVG subgraph visualizer; implements node search, 1–3 hop selection, zoom/pan navigation, depth coloring, and forensic node inspector. | `fetchSubgraph` (`/graph/subgraph`) | `subgraph_transactions_client.test.mjs`, `tsc --noEmit` | Phase 11.2B |
| 16 | `frontend/src/pages/TransactionsPage.tsx` | Tracked M | Forensic transaction ledger table; implements real backend pagination (offset/limit), fraud/licit filtering, and live query summaries. | `fetchTransactions` (`/transactions`) | `subgraph_transactions_client.test.mjs`, `tsc --noEmit` | Phase 11.2B |
| 17 | `frontend/src/pages/CudaLabPage.tsx` | Tracked M | CUDA diagnostic laboratory; includes live host probe button and honest `'NO GPU'` badge with `'No physical GPU detected on host'` subtext. | `useHealthPolling` | `tsc --noEmit`, `npm run build` | **YES** (Phases 9C, 11.3.1) |
| 18 | `frontend/src/pages/EvaluationCenterPage.tsx` | Tracked M | Model evaluation center; displays dynamic PR-AUC/ROC-AUC metrics from backend with artifact state handling and manual refresh. | `useDashboardPolling` | `tsc --noEmit`, `npm run build` | Phase 10 |
| 19 | `frontend/src/pages/ExperimentManagerPage.tsx` | Tracked M | Benchmark experiment manager; renders live session provenance and execution repetitions with artifact failure banners. | `useDashboardPolling` | `tsc --noEmit`, `npm run build` | Phase 10 |
| 20 | `frontend/tests/polling_lifecycle.test.mjs` | Untracked | 21 unit tests covering timer races, monotonic sequencing, abort vs timeout classification, document visibility, and state transitions. | Node.js test runner | 21 passed | Phase 9C.1.2 |
| 21 | `frontend/tests/subgraph_transactions_client.test.mjs` | Untracked | 3 unit tests validating `getSubgraph` and `getTransactions` query parameter formatting and response parsing. | Node.js test runner | 3 passed | Phase 11.2B |
| 22 | `frontend/tests/api_integration.test.mjs` | Untracked | 14 integration test specifications cataloging live HTTP contracts for FastAPI backend (:8000) and Vite proxy (:5173). | Node.js test runner | Authored Phase 10 | Phase 10 |

---

## 4. Explicit Exclusions From This Commit

The following files are explicitly **excluded** from this frontend commit:

1. **`frontend/tsconfig.tsbuildinfo`:**
   - *Status:* Tracked M.
   - *Reason for Exclusion:* Compiler cache artifact committed in Phase 1 (`5063a56`). Will be untracked in a dedicated maintenance commit (Commit 7) alongside `.gitignore` configuration.
2. **Backend Application Files (`backend/app/...`, `backend/cpu_engine/...`):**
   - *Files:* `dashboard.py`, `health.py`, `main.py`, `graph.py`, `transactions.py`, `synthetic_fraud_generator.py`.
   - *Reason for Exclusion:* Belongs to dedicated backend feature commits (Dashboard Artifact Handling, Graph/Transaction APIs, Telemetry/Provenance Fixes) to preserve clean architectural separation between frontend and backend.
3. **Backend Test Files (`backend/tests/...`):**
   - *Files:* `test_dashboard.py`, `test_health.py`, `test_graph_api.py`, `test_transactions_api.py`, `test_synthetic_fraud_generator.py`.
   - *Reason for Exclusion:* Packaged alongside their respective backend implementation commits.
4. **Historical Documentation & Audit Reports (`docs/PHASE_*.md`):**
   - *Reason for Exclusion:* Preserved as untracked historical records to be committed in the final documentation archive commit.

---

## 5. Verification Evidence & Quality Gates

All automated verification gates were executed and confirmed before generating this plan:

### 5.1 Unit & Client Test Suites
```bash
cd "D:\FraudGraph Intelligence\frontend"
node --test tests/polling_lifecycle.test.mjs tests/subgraph_transactions_client.test.mjs
```
- **Total Tests:** 24
- **Passed:** 24 (21 polling lifecycle + 3 subgraph/transaction client)
- **Failures:** 0
- **Duration:** 318 ms

### 5.2 TypeScript Static Typecheck
```bash
cd "D:\FraudGraph Intelligence\frontend"
npx tsc --noEmit
```
- **Exit Code:** 0
- **Type Errors:** 0

### 5.3 Production Vite Bundle Build
```bash
cd "D:\FraudGraph Intelligence\frontend"
npm run build
```
- **Exit Code:** 0
- **Modules Transformed:** 1,621 modules
- **Output Bundles:**
  - `dist/index.html` (0.87 kB)
  - `dist/assets/index-ChKqFaYm.css` (1.89 kB)
  - `dist/assets/index-D8SLlGAb.js` (295.30 kB)
- **Build Duration:** 2.62 s

### 5.4 Repository Whitespace & Conflict Check
```bash
git diff --check
```
- **Exit Code:** 0
- **Result:** No whitespace errors, trailing whitespace, or Git merge conflict markers.

### 5.5 Canonical Datasets & Experiments Protection
- `data/` status: Clean (unmodified)
- `experiments/` status: Clean (unmodified)
- `data/fixtures/synthetic_fraud_10k.npz` SHA-256: `aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430` (Verified matching)

---

## 6. Proposed Staging Commands & Commit Specification

Upon user authorization, the following explicit file list will be staged without using `git add .` or `git add -A`:

```bash
git add frontend/src/types/telemetry.ts \
        frontend/src/types/api.ts \
        frontend/src/services/apiClient.ts \
        frontend/src/services/api.ts \
        frontend/src/hooks/index.ts \
        frontend/src/hooks/usePollingResource.ts \
        frontend/src/hooks/useHealthPolling.ts \
        frontend/src/hooks/useDashboardPolling.ts \
        frontend/src/hooks/useDatasetsPolling.ts \
        frontend/src/components/layout/AppLayout.tsx \
        frontend/src/components/layout/TopNav.tsx \
        frontend/src/components/layout/SystemHealthDrawer.tsx \
        frontend/src/pages/OverviewPage.tsx \
        frontend/src/pages/DatasetsPage.tsx \
        frontend/src/pages/GraphExplorerPage.tsx \
        frontend/src/pages/TransactionsPage.tsx \
        frontend/src/pages/CudaLabPage.tsx \
        frontend/src/pages/EvaluationCenterPage.tsx \
        frontend/src/pages/ExperimentManagerPage.tsx \
        frontend/tests/polling_lifecycle.test.mjs \
        frontend/tests/subgraph_transactions_client.test.mjs \
        frontend/tests/api_integration.test.mjs
```

### Proposed Commit Metadata
- **Commit Message:**  
  `feat(frontend): implement integrated telemetry, subgraph explorer, transaction ledger, and artifact handling`
- **Estimated Commit Scope:** 22 files changed (+1,850 lines, -240 lines).
- **Post-Commit Verification:** Run `git status`, verify HEAD update, verify index is clean of frontend files (except `tsconfig.tsbuildinfo`), and confirm no network push occurred.
