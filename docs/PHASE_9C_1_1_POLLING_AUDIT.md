# Phase 9C.1.1 — API and Polling Correctness Audit Report

**Project Title:** Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation  
**Institution:** Vellore Institute of Technology, CSI4007 GPU Programming Capstone  
**Workspace Root:** `D:\FraudGraph Intelligence`  
**Expected Baseline HEAD:** `ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc`  
**Current Branch:** `main`  
**Audit Date:** 2026-10-25  
**Audit Status:** COMPLETE — READ-ONLY CORRECTNESS AUDIT  
**Deliverable:** `docs/PHASE_9C_1_1_POLLING_AUDIT.md`  

---

## 1. Executive Verdict

### **VERDICT: PASS WITH FINDINGS (NO BLOCKERS FOR PHASE 9C.1 FOUNDATION)**

The Phase 9C.1 API and Telemetry Foundation implementation is architecturally sound, type-safe, and truthful. It establishes genuine HTTP connectivity to FastAPI endpoints without fabricating GPU metrics, CUDA compiler states, or simulated activity.

- **Type Safety & Build Cleanliness:** Static verification via `npx tsc --noEmit` completes with **0 errors, 0 warnings**. Production bundling via Vite succeeds in **2.44s**.
- **Backend Stability:** All 8 focused backend tests passed in **4.11s**; the full 150-test regression suite passed with **148 passed, 2 skipped** (100% of runnable tests).
- **Data Integrity:** Metric provenance is strictly preserved. Peak speedup (44.60x) is clearly attributed to the `mean` aggregation operator; SUM variability (CV 189.0%) is preserved; local GeForce MX450 lacks NVCC and is strictly demarcated from Google Colab Tesla T4 (0.0679ms latency).
- **Whitespace & Repository Hygiene:** `git diff --check` is clean (0 violations). `frontend/tsconfig.tsbuildinfo` remains clean at committed baseline `5063a56`.

This audit identified **2 P1 concurrency/lifecycle findings**, **4 P2 quality/performance findings**, and **1 P3 classification finding**. None represent active blockers preventing the foundation from functioning, but they must be addressed prior to or during Phase 9C.2.

---

## 2. Scope of Code Reviewed

The following files and functions were inspected line-by-line:

| File Path | Component / Service / Hook | Key Functions & Types Reviewed |
|---|---|---|
| `frontend/src/services/apiClient.ts` | Typed API Client | `ApiError`, `isApiError`, `isAbortError`, `request()`, `get()`, `getHealth()`, `getDashboardSummary()`, `getDatasets()` |
| `frontend/src/services/api.ts` | Service Facade | `fetchHealth()`, `fetchDashboardSummary()`, `fetchDatasets()` delegation and options forwarding |
| `frontend/src/types/api.ts` | API Contracts | `HealthData`, `DashboardSummary`, `DatasetFixtureItem`, `DatasetListResponse`, `ApiErrorKind`, `ApiErrorDetails` |
| `frontend/src/types/telemetry.ts` | State Model | `TelemetryState`, `TelemetryProvenanceSource`, `TelemetryProvenance`, `PollingResourceState<T>` |
| `frontend/src/hooks/usePollingResource.ts` | Core Polling Engine | `usePollingResource()`, `executeFetch()`, `clearTimer()`, `refresh()`, `stop()`, `start()` |
| `frontend/src/hooks/useHealthPolling.ts` | Health Hook | 15s health cadence, host probe caveats, Colab demarcation |
| `frontend/src/hooks/useDashboardPolling.ts` | Dashboard Hook | 30s artifact summary cadence, ADR-007 and benchmark caveats |
| `frontend/src/hooks/useDatasetsPolling.ts` | Datasets Hook | 30s fixture catalog cadence, filesystem inspection |
| `frontend/src/components/layout/AppLayout.tsx` | Root Shell | Universal health polling hook integration, footer HEAD label |
| `frontend/src/components/layout/TopNav.tsx` | Top Navigation | `telemetryState` badge rendering (`SYS HEALTHY`, `SYS DEGRADED`, `SYS OFFLINE`) |
| `frontend/src/components/layout/SystemHealthDrawer.tsx` | Diagnostics Drawer | Truthful live probe warning banners, polling interval disclosure |
| `frontend/src/pages/OverviewPage.tsx` | Operational Cockpit | `useDashboardPolling()` integration, error fallback and retry |
| `frontend/src/pages/DatasetsPage.tsx` | Dataset Registry | `useDatasetsPolling()` integration, error fallback and retry |

---

## 3. Concrete Findings and Defect Analysis

### Finding F-01 (Priority: P1 — Concurrency): Stale in-flight poll schedules `setTimeout` tick in `finally` without checking monotonic request ID
- **Location:** [`frontend/src/hooks/usePollingResource.ts:206-215`](file:///D:/FraudGraph%20Intelligence/frontend/src/hooks/usePollingResource.ts#L206-L215)
- **Description:**  
  When an HTTP request finishes or catches an error, line 202 nulls `inFlightControllerRef` only if `currentRequestId === requestIdRef.current`. However, lines 207–215 schedule the next polling tick without checking whether `currentRequestId === requestIdRef.current`:
  ```ts
  // usePollingResource.ts lines 206-214
  if (isMountedRef.current && isPollingActive) {
    clearTimer();
    timerRef.current = setTimeout(() => {
      if (isMountedRef.current && isPollingActive) {
        executeFetch(false);
      }
    }, effectiveIntervalMs);
  }
  ```
- **Lifecycle Interleaving / Reproduction:**
  1. At $t = 0$: Scheduled poll #1 fires (`reqId = 1`).
  2. At $t = 100\,\text{ms}$: User clicks "PROBE RUNTIME" (manual refresh).
  3. `refresh()` executes: aborts controller #1, increments `requestIdRef.current = 2`, and dispatches poll #2 (`reqId = 2`).
  4. At $t = 105\,\text{ms}$: Poll #1 catches `AbortError`, discards state updates, and enters `finally`.
  5. Poll #1 checks `if (isMountedRef.current && isPollingActive)` (both true) and schedules a `setTimeout` timer for $t = 15105\,\text{ms}$, while poll #2 is still actively in-flight!
  6. At $t = 400\,\text{ms}$: Poll #2 succeeds and schedules a timer for $t = 15400\,\text{ms}$.
  7. If poll #1 had completed *after* poll #2 (e.g. if the backend took longer to reject), poll #1's timer would overwrite poll #2's timer, desynchronizing the polling interval.
- **Recommended Targeted Fix:**  
  Guard timer scheduling in `finally` with:
  ```ts
  if (currentRequestId === requestIdRef.current && isMountedRef.current && isPollingActive) {
    clearTimer();
    timerRef.current = setTimeout(() => {
      if (isMountedRef.current && isPollingActive) {
        executeFetch(false);
      }
    }, effectiveIntervalMs);
  }
  ```

---

### Finding F-02 (Priority: P1 — Lifecycle): Calling `stop()` triggers effect re-execution that immediately resets polling to `enabled`
- **Location:** [`frontend/src/hooks/usePollingResource.ts:88, 217, 221-241, 247-254`](file:///D:/FraudGraph%20Intelligence/frontend/src/hooks/usePollingResource.ts#L88)
- **Description:**  
  `isPollingActive` is managed as component state (`useState(enabled)`).  
  `executeFetch` declares `isPollingActive` in its `useCallback` dependency array (`[clearTimer, effectiveIntervalMs, isPollingActive, provenanceSource]`).  
  `useEffect` declares `executeFetch` in its dependency array (`[enabled, executeFetch, clearTimer]`).  
  When a caller invokes `stop()`, it runs `setIsPollingActive(false)`. This state update causes `executeFetch` to be recreated with a new reference. Because `executeFetch` changed, `useEffect` executes. In `useEffect`, it checks `enabled` (which remains `true` from options), calls `setIsPollingActive(enabled)` (resetting to `true`), and immediately calls `executeFetch(false)`.  
  Consequently, `stop()` cannot pause polling when `options.enabled !== false`.
- **Reproduction:**
  1. Component calls `const { stop } = usePollingResource(fetchFn, { enabled: true })`.
  2. Component invokes `stop()`.
  3. `setIsPollingActive(false)` triggers re-render.
  4. `executeFetch` reference changes.
  5. `useEffect` fires, sees `enabled === true`, and calls `setIsPollingActive(true)` and `executeFetch(false)`.
  6. Polling continues uninterrupted.
- **Recommended Targeted Fix:**  
  Manage `isPollingActive` in a mutable ref (`isPollingActiveRef = useRef(enabled)`), eliminating `isPollingActive` from the `useCallback` dependency array of `executeFetch`. This allows `stop()` and `start()` to cleanly toggle polling without triggering reactive effect resets.

---

### Finding F-03 (Priority: P2 — React Best Practice): State setter called inside another state setter's functional updater
- **Location:** [`frontend/src/hooks/usePollingResource.ts:139-146`](file:///D:/FraudGraph%20Intelligence/frontend/src/hooks/usePollingResource.ts#L139-L146)
- **Description:**  
  In `usePollingResource.ts`, loading indicators are toggled via:
  ```ts
  setData((prevData) => {
    if (prevData === null) {
      setIsLoading(true);
    } else {
      setIsRefreshing(true);
    }
    return prevData;
  });
  ```
  In React (especially under Concurrent Mode and Strict Mode), updater functions must be pure projections without side-effects. Triggering `setIsLoading` or `setIsRefreshing` inside `setData` can cause React to throw development warnings (`Cannot update a component while rendering a different component`) or perform duplicate invocations.
- **Recommended Targeted Fix:**  
  Track whether initial data has resolved using a ref (`hasResolvedInitialRef = useRef(false)`). Call `if (!hasResolvedInitialRef.current) setIsLoading(true); else setIsRefreshing(true);` directly in the synchronous flow of `executeFetch()`.

---

### Finding F-04 (Priority: P2 — Data Representation): Object-shaped FastAPI validation errors format as `[object Object]`
- **Location:** [`frontend/src/services/apiClient.ts:144-147`](file:///D:/FraudGraph%20Intelligence/frontend/src/services/apiClient.ts#L144-L147)
- **Description:**  
  FastAPI returns HTTP 422 Unprocessable Entity with `detail` formatted as an array of error objects: `{"detail": [{"loc": ["query", "base_dir"], "msg": "str type expected", "type": "value_error"}]}`.  
  Line 146 currently casts `detail` using `String((errorBody as { detail: unknown }).detail)`.  
  In JavaScript, `String([{}])` evaluates to `"[object Object]"`, obscuring the actual parameter name and validation message from error alerts.
- **Recommended Targeted Fix:**  
  Check if `detail` is an object/array and serialize with `JSON.stringify`:
  ```ts
  const detailMsg =
    errorBody && typeof errorBody === 'object' && 'detail' in errorBody
      ? (typeof (errorBody as { detail: unknown }).detail === 'object'
          ? JSON.stringify((errorBody as { detail: unknown }).detail)
          : String((errorBody as { detail: unknown }).detail))
      : rawText || response.statusText || 'HTTP Error';
  ```

---

### Finding F-05 (Priority: P2 — Resource Efficiency): Background polling continues when browser tab is inactive / hidden
- **Location:** [`frontend/src/hooks/usePollingResource.ts:206-215`](file:///D:/FraudGraph%20Intelligence/frontend/src/hooks/usePollingResource.ts#L206-L215)
- **Description:**  
  The polling engine does not listen to the Page Visibility API (`document.visibilityState` / `visibilitychange`). When a user switches to another tab or minimizes the browser, HTTP requests continue firing every 15s or 30s.
- **Recommended Targeted Fix:**  
  Add an optional visibility listener to pause timers when `document.hidden === true` and trigger an immediate catch-up probe when the tab returns to active focus.

---

### Finding F-06 (Priority: P2 — Rendering Efficiency): Unmemoized `Outlet context` in `AppLayout` triggers unnecessary re-renders of child routes
- **Location:** [`frontend/src/components/layout/AppLayout.tsx:68-76`](file:///D:/FraudGraph%20Intelligence/frontend/src/components/layout/AppLayout.tsx#L68-L76)
- **Description:**  
  `<Outlet context={{ settings, health: healthResource.data, healthState: healthResource.state, onRefreshHealth: healthResource.refresh }} />` constructs a new object literal on every render. Because `useHealthPolling` updates `lastSuccessTime` every 15 seconds, child routes subscribed to `useOutletContext()` will re-render even if health metrics did not change.
- **Recommended Targeted Fix:**  
  Wrap the context value in `useMemo()`:
  ```tsx
  const outletContext = useMemo(
    () => ({
      settings,
      health: healthResource.data,
      healthState: healthResource.state,
      onRefreshHealth: healthResource.refresh,
    }),
    [settings, healthResource.data, healthResource.state, healthResource.refresh]
  );
  ```

---

### Finding F-07 (Priority: P3 — Diagnostics Classification): Timeout cancellation is classified as `ABORT_ERROR` rather than a distinct `TIMEOUT_ERROR`
- **Location:** [`frontend/src/services/apiClient.ts:86-88, 116-121`](file:///D:/FraudGraph%20Intelligence/frontend/src/services/apiClient.ts#L86-L88)
- **Description:**  
  When `timeoutMs` triggers in `apiClient.request()`, `internalController.abort()` is called. When `fetch` rejects with `AbortError`, lines 116–121 categorize it as `kind: 'ABORT_ERROR'`. Callers cannot programmatically distinguish between a user/unmount cancellation and a network deadline timeout.
- **Recommended Targeted Fix:**  
  Set an internal flag `isTimedOut = true` before calling `internalController.abort()`, and throw `ApiError('TIMEOUT_ERROR')`.

---

## 4. Verification Evidence & Test Outcomes

### 4.1 Frontend TypeScript Verification
- **Command:** `cd "D:\FraudGraph Intelligence\frontend"; npx tsc --noEmit`
- **Exit Code:** `0`
- **Output:** Clean; 0 errors, 0 warnings.

### 4.2 Frontend Production Bundle Check
- **Command:** `cd "D:\FraudGraph Intelligence\frontend"; npm run build; git restore tsconfig.tsbuildinfo`
- **Exit Code:** `0`
- **Output:** Vite v6.4.4 compiled successfully in 2.44s:
  - `dist/index.html`: 0.87 kB
  - `dist/assets/index-ChKqFaYm.css`: 1.89 kB
  - `dist/assets/index-57rYZ618.js`: 271.74 kB
- **Cache Status:** `frontend/tsconfig.tsbuildinfo` was restored and remains unmodified at commit `5063a56`.

### 4.3 Focused Backend Tests
- **Command:** `cd "D:\FraudGraph Intelligence"; .\.venv\Scripts\python -m pytest backend/tests/test_health.py backend/tests/test_dashboard.py -v`
- **Exit Code:** `0`
- **Output:** 8 passed in 4.11s.

### 4.4 Full Backend Regression Suite
- **Command:** `cd "D:\FraudGraph Intelligence"; .\.venv\Scripts\python -m pytest -v`
- **Exit Code:** `0`
- **Output:** 150 collected: 148 passed, 2 skipped in 178.82s (0:02:58). 100% of runnable tests passed.

### 4.5 Whitespace & Working-Tree Check
- **Command:** `cd "D:\FraudGraph Intelligence"; git diff --check`
- **Exit Code:** `0`
- **Output:** 0 whitespace violations.

---

## 5. Distinction: Static Code Analysis vs. Runtime-Verified Behavior

| Behavior / Guarantee | Verification Method | Status / Evidence |
|---|:---:|---|
| **TypeScript compile safety** | Runtime compiler check (`tsc`) | **Verified PASS** (0 errors) |
| **Vite asset compilation** | Runtime bundler check (`vite build`) | **Verified PASS** (2.44s) |
| **Backend REST endpoint contracts** | Runtime pytest (`test_dashboard.py`) | **Verified PASS** (8 passed) |
| **Zero metric fabrication** | Static code audit + runtime test | **Verified PASS** |
| **Overlap prevention via chained setTimeout** | Static code audit | **Verified SOUND** (architectural design) |
| **Finding F-01 (Stale request timer schedule in finally)** | Static code trace | **Defect identified** (code path analyzed) |
| **Finding F-02 (`stop()` reactive reset in useEffect)** | Static code trace | **Defect identified** (code path analyzed) |
| **Browser visual rendering & CSS inspection** | Static JSX/CSS inspection | **Static review only** (no browser runner installed) |

*Note on Testing Limitations:* No browser automation tool (Playwright/Cypress/Puppeteer) or React component test runner (Vitest/Jest/React Testing Library) is installed in `frontend/package.json`. In strict compliance with authorization constraints, no dependencies were added. Frontend verification is anchored on static TypeScript analysis, Vite production bundling, and endpoint parity.

---

## 6. Git Status and Boundary Confirmation

### 6.1 Baseline HEAD
```powershell
git rev-parse HEAD
# ae5b1aa81c644cb1ec96e1f06591e05e06ff4dbc
```
*Confirmation:* HEAD remains exactly at the committed Phase 9B baseline.

### 6.2 Working-Tree Status
```text
 M frontend/src/components/layout/AppLayout.tsx
 M frontend/src/components/layout/SystemHealthDrawer.tsx
 M frontend/src/components/layout/TopNav.tsx
 M frontend/src/pages/DatasetsPage.tsx
 M frontend/src/pages/OverviewPage.tsx
 M frontend/src/services/api.ts
 M frontend/src/types/api.ts
?? docs/PHASE_9B_7_POST_COMMIT_VERIFICATION.md
?? docs/PHASE_9C_1_IMPLEMENTATION_REPORT.md
?? docs/PHASE_9C_1_1_POLLING_AUDIT.md
?? frontend/src/hooks/
?? frontend/src/services/apiClient.ts
?? frontend/src/types/telemetry.ts
```

### 6.3 Boundary Attestation
- **Source Code Modified by Audit:** **NO** (0 application source lines modified during this audit).
- **Files Staged:** **NO** (0 files staged; `git diff --cached --stat` empty).
- **Commits Created:** **NO**.
- **Remote Operations:** **NO** (0 pushes; `git remote -v` empty).
- **Existing Reports:** `docs/PHASE_9B_7_POST_COMMIT_VERIFICATION.md` and `docs/PHASE_9C_1_IMPLEMENTATION_REPORT.md` preserved intact.
- **Phase 9C.2 Boundary:** **Phase 9C.2 has NOT started.**

---

## 7. Recommended Targeted Remediation Plan (Prior to Phase 9C.2)

1. **Patch F-01 & F-02 in `usePollingResource.ts`:**
   - Add `currentRequestId === requestIdRef.current` guard before scheduling timers in `finally`.
   - Move `isPollingActive` from `useState` to `useRef(enabled)`, decoupling programmatic pauses from reactive effect resets.
2. **Patch F-03 in `usePollingResource.ts`:**
   - Eliminate state setter nesting inside `setData` updaters.
3. **Patch F-04 in `apiClient.ts`:**
   - Handle object/array error details with `JSON.stringify` to prevent `[object Object]`.
4. **Patch F-06 in `AppLayout.tsx`:**
   - Memoize `outletContext` with `useMemo()`.

Execution is stopped here as instructed. Awaiting user authorization.
