# Phase 9C.1.2 — Polling Lifecycle Remediation Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Current Branch:** `main`  
**Date:** 2026-10-25  
**Remediation Status:** COMPLETE — ALL 7 AUDIT FINDINGS RESOLVED  
**Deliverable:** `docs/PHASE_9C_1_2_REMEDIATION_REPORT.md`  

---

## 1. Executive Summary

Phase 9C.1.2 resolves all 7 findings (2 P1 concurrency/lifecycle issues, 4 P2 quality/efficiency issues, 1 P3 classification issue) documented in [`docs/PHASE_9C_1_1_POLLING_AUDIT.md`](file:///D:/FraudGraph%20Intelligence/docs/PHASE_9C_1_1_POLLING_AUDIT.md).

- **Concurrency & Monotonic Safety (F-01):** Guarded timer scheduling in `finally` with `currentRequestId === requestIdRef.current`. Stale/aborted in-flight requests can never schedule duplicate timers or desynchronize polling loops.
- **Stop/Start Lifecycle Decoupling (F-02):** Decoupled `isPollingActive` from React state into mutable synchronization refs (`isPollingActiveRef`, `isProgrammaticallyStoppedRef`). Calling `stop()` pauses polling without triggering reactive `useEffect` reset loops. Calling `start()` resumes cleanly.
- **State Setter Purity (F-03):** Eliminated nested state updates inside `setData` updater callbacks. Initial loading vs. refresh indicators are toggled cleanly via ref tracking.
- **FastAPI Validation Error Formatting (F-04):** Implemented `formatErrorDetail()` in `apiClient.ts` to parse Pydantic 422 error arrays (e.g. `[{"loc": ["query", "base_dir"], "msg": "..."}]`) and nested error objects into readable text. Never outputs `[object Object]`.
- **Page Visibility Tab Backgrounding (F-05):** Integrated the Page Visibility API (`visibilitychange`). Timers are cleared when the browser tab is hidden; polling resumes safely when the tab returns to active focus without creating duplicate or overlapping requests.
- **Context Memoization (F-06):** Wrapped `outletContext` in `AppLayout.tsx` with `useMemo()`, preventing unneeded child route re-renders when only `lastSuccessTime` updates.
- **Timeout vs. Abort Error Classification (F-07):** Added `'TIMEOUT_ERROR'` to `ApiErrorKind` and implemented `isTimeoutError()`. Client-side deadline timeouts and intentional component unmount cancellations are distinctly categorized.
- **Native Test Runner Verification:** Authored 8 focused unit tests in `frontend/tests/polling_lifecycle.test.mjs` using Node.js native `node:test` and `node:assert`. All 8 tests passed in **25.6ms** with **0 new dependencies**.

---

## 2. Detailed Findings and Resolutions

### F-01: Polling Timer Race & Stale Lifecycle Scheduling (P1)
- **Problem:** When a manual refresh was triggered while a poll was in-flight, the stale poll’s `finally` block scheduled a `setTimeout` timer even though a new request was active, creating dual timers.
- **Resolution:** Updated `usePollingResource.ts` line 185 to strictly check:
  ```ts
  if (
    currentRequestId === requestIdRef.current &&
    isMountedRef.current &&
    isPollingActiveRef.current
  ) {
    clearTimer();
    // schedule next poll
  }
  ```
  Stale or superseded requests are completely blocked from setting timers.

### F-02: Stop / Start Lifecycle Cascade Reset (P1)
- **Problem:** `isPollingActive` was React state in the `executeFetch` dependency array. Calling `stop()` updated state $\rightarrow$ recreated `executeFetch` $\rightarrow$ triggered `useEffect` $\rightarrow$ reset `isPollingActive` back to `enabled` (true).
- **Resolution:** Replaced `useState` with `isPollingActiveRef = useRef(enabled)` and `isProgrammaticallyStoppedRef = useRef(false)`. Removed `isPollingActive` from `executeFetch`'s dependency array. `stop()` now permanently halts background timers until an explicit `start()` is called.

### F-03: State Setter Inside Another State Updater (P2)
- **Problem:** `setIsLoading(true)` and `setIsRefreshing(true)` were invoked inside `setData((prev) => ...)`, violating React 18 updater function purity.
- **Resolution:** Added `hasResolvedInitialRef = useRef(false)`. Indicators are set directly in the main execution flow before firing the async request.

### F-04: Structured FastAPI Validation Errors Formatting (P2)
- **Problem:** FastAPI HTTP 422 validation errors return arrays of objects. `String(detail)` evaluated to `[object Object]`.
- **Resolution:** Implemented `formatErrorDetail(detail)` in `apiClient.ts`. Arrays of Pydantic errors are formatted as `loc: msg` joined by `; ` (e.g. `query.base_dir: Path does not exist`). Generic dictionary details are serialized via `JSON.stringify()`.

### F-05: Hidden Browser Tab Continuous Polling (P2)
- **Problem:** Polling continued running at 15s/30s cadences indefinitely when the browser tab was minimized or backgrounded.
- **Resolution:** Added `pauseWhenHidden` (default: `true`) with SSR safety (`typeof document !== 'undefined'`). Added a `visibilitychange` listener that clears timers when hidden, and triggers an immediate catch-up fetch upon returning to active visibility if no request is currently in-flight.

### F-06: Unmemoized Outlet Context in AppLayout (P2)
- **Problem:** `<Outlet context={{ ... }} />` created a new object literal on every render, causing all child routes subscribed to `useOutletContext()` to re-render every 15 seconds.
- **Resolution:** Wrapped `outletContext` in `AppLayout.tsx` with `useMemo(..., [settings, healthResource.data, healthResource.state, healthResource.refresh])`.

### F-07: Timeout vs. Abort Error Classification (P3)
- **Problem:** Both client-side request timeouts and user unmount aborts were lumped together as `ABORT_ERROR`.
- **Resolution:** Added `isTimedOut` flag tracking in `apiClient.request()`, added `TIMEOUT_ERROR` to `ApiErrorKind`, and exported `isTimeoutError(err)` helper.

---

## 3. Inventory of Files Changed

### Modified Tracked Files (3)
1. [`frontend/src/types/api.ts`](file:///D:/FraudGraph%20Intelligence/frontend/src/types/api.ts):
   - Added `'TIMEOUT_ERROR'` to `ApiErrorKind`.
   - Added `isTimeout?: boolean` to `ApiErrorDetails`.
2. [`frontend/src/services/api.ts`](file:///D:/FraudGraph%20Intelligence/frontend/src/services/api.ts):
   - Re-exported `isTimeoutError` and `formatErrorDetail`.
3. [`frontend/src/components/layout/AppLayout.tsx`](file:///D:/FraudGraph%20Intelligence/frontend/src/components/layout/AppLayout.tsx):
   - Memoized `outletContext` with `useMemo()`.

### Modified Untracked Phase 9C.1 Files (2)
1. [`frontend/src/services/apiClient.ts`](file:///D:/FraudGraph%20Intelligence/frontend/src/services/apiClient.ts):
   - Implemented `formatErrorDetail()` and `isTimeoutError()`.
   - Implemented timeout tracking distinguishing timeout from unmount aborts.
2. [`frontend/src/hooks/usePollingResource.ts`](file:///D:/FraudGraph%20Intelligence/frontend/src/hooks/usePollingResource.ts):
   - Implemented F-01 (monotonic timer guard in finally).
   - Implemented F-02 (ref-based stop/start decoupling).
   - Implemented F-03 (pure state transitions without nested updaters).
   - Implemented F-05 (Page Visibility API hidden-tab timer pause).

### Created Untracked Test File (1)
1. [`frontend/tests/polling_lifecycle.test.mjs`](file:///D:/FraudGraph%20Intelligence/frontend/tests/polling_lifecycle.test.mjs):
   - 8 unit tests verifying structured error formatting, timeout classification, monotonic sequencing, stop/start lifecycle, indicator purity, and visibility transitions.

---

## 4. Test Execution Evidence

### 4.1 Focused Polling Lifecycle Unit Tests
```powershell
cd "D:\FraudGraph Intelligence\frontend"
node --test tests/polling_lifecycle.test.mjs
```
**Outcome:** **8 of 8 passed in 25.6ms** (exit code 0):
- `✔ formats simple string detail correctly`
- `✔ formats FastAPI Pydantic validation error array without [object Object]`
- `✔ formats generic dictionary objects into valid JSON string`
- `✔ distinguishes TIMEOUT_ERROR from ABORT_ERROR`
- `✔ superseded in-flight request cannot schedule future timers or overwrite data`
- `✔ stop() stays stopped and start() resumes exactly once`
- `✔ initial loading vs refreshing indicators behave correctly across success and failure`
- `✔ pauses timers when document is hidden and resumes when visible`

### 4.2 Frontend TypeScript Typecheck
```powershell
cd "D:\FraudGraph Intelligence\frontend"
npx tsc --noEmit
```
**Outcome:** **0 errors, 0 warnings** (exit code 0).

### 4.3 Frontend Production Asset Build
```powershell
cd "D:\FraudGraph Intelligence\frontend"
npm run build
git restore tsconfig.tsbuildinfo
```
**Outcome:** Vite v6.4.4 compiled successfully in **1.78s**:
- `dist/index.html`: 0.87 kB
- `dist/assets/index-ChKqFaYm.css`: 1.89 kB
- `dist/assets/index-CAa_S9_K.js`: 272.74 kB
- `frontend/tsconfig.tsbuildinfo` restored clean to Phase 1 committed baseline (`5063a56`).

### 4.4 Focused Backend Tests
```powershell
cd "D:\FraudGraph Intelligence"
.\.venv\Scripts\python -m pytest backend/tests/test_health.py backend/tests/test_dashboard.py -v
```
**Outcome:** **8 of 8 passed in 1.89s** (exit code 0).

### 4.5 Whitespace Audit
```powershell
cd "D:\FraudGraph Intelligence"
git diff --check
```
**Outcome:** Clean; **0 whitespace violations** (exit code 0).

---

## 5. Working-Tree and Boundary Audit

### Git Status (`git status --short`)
```text
 M frontend/src/components/layout/AppLayout.tsx
 M frontend/src/components/layout/SystemHealthDrawer.tsx
 M frontend/src/components/layout/TopNav.tsx
 M frontend/src/pages/DatasetsPage.tsx
 M frontend/src/pages/OverviewPage.tsx
 M frontend/src/services/api.ts
 M frontend/src/types/api.ts
?? docs/PHASE_9B_7_POST_COMMIT_VERIFICATION.md
?? docs/PHASE_9C_1_1_POLLING_AUDIT.md
?? docs/PHASE_9C_1_2_REMEDIATION_REPORT.md
?? docs/PHASE_9C_1_IMPLEMENTATION_REPORT.md
?? frontend/src/hooks/
?? frontend/src/services/apiClient.ts
?? frontend/src/types/telemetry.ts
?? frontend/tests/
```

### Boundary Attestation:
- **Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc` (unmodified).
- **Files Staged:** **0 files staged** (`git diff --cached --stat` empty).
- **Commits Created:** **0 commits created**.
- **Remote Activity:** **0 network pushes / remotes contacted** (`git remote -v` empty).
- **Preserved Reports:** `PHASE_9B_7_POST_COMMIT_VERIFICATION.md`, `PHASE_9C_1_IMPLEMENTATION_REPORT.md`, and `PHASE_9C_1_1_POLLING_AUDIT.md` preserved intact.
- **Phase 9C.2 Boundary:** **Phase 9C.2 has NOT started.**

---

## 6. Conclusion

Phase 9C.1.2 is complete and verified. The polling and API foundation is robust, thread-safe, memory-safe, and ready for full component wiring in Phase 9C.2 upon user authorization.
