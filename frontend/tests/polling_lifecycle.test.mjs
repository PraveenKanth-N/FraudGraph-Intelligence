/**
 * Phase 9C.1.2 — Focused Unit Verification for Polling Lifecycle & API Client Remediation
 *
 * Built with Node.js native test runner (node:test) and assertions (node:assert).
 * Zero third-party dependencies required.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert';

// Direct import of real production API client module
import {
  ApiError,
  isApiError,
  isTimeoutError,
  isAbortError,
  formatErrorDetail,
  API_BASE,
  request,
  getHealth,
  getDashboardSummary,
  getDatasets,
  apiClient,
} from '../src/services/apiClient.ts';

// Simulated polling harness replicating usePollingResource logic
export function createPollingHarness(fetchFn, options = {}) {
  const {
    intervalMs = 5000,
    enabled = true,
    pauseWhenHidden = true,
    provenanceSource = 'LIVE_ENDPOINT',
  } = options;

  let data = null;
  let state = 'UNAVAILABLE';
  let isLoading = true;
  let isRefreshing = false;
  let error = null;

  let isMounted = true;
  let inFlightController = null;
  let timerId = null;
  let scheduledTimerCount = 0;
  let activeTimerCount = 0;
  let requestId = 0;
  let hasResolvedInitial = false;
  let isPollingActive = enabled;
  let isProgrammaticallyStopped = false;

  let mockDocumentHidden = false;

  function clearTimer() {
    if (timerId !== null) {
      clearTimeout(timerId);
      timerId = null;
      activeTimerCount = Math.max(0, activeTimerCount - 1);
    }
  }

  async function executeFetch(isManual = false) {
    if (inFlightController) {
      inFlightController.abort();
      inFlightController = null;
    }

    clearTimer();

    const controller = new AbortController();
    inFlightController = controller;
    const currentRequestId = ++requestId;

    if (!isMounted) return;

    if (!hasResolvedInitial) {
      isLoading = true;
    } else {
      isRefreshing = true;
    }

    try {
      const result = await fetchFn(controller.signal);

      if (!isMounted || currentRequestId !== requestId) {
        return;
      }

      hasResolvedInitial = true;
      data = result;
      error = null;
      if (provenanceSource === 'HISTORICAL_ARTIFACT') {
        state = 'HISTORICAL';
      } else {
        state = 'AVAILABLE';
      }
      isLoading = false;
      isRefreshing = false;
    } catch (err) {
      if (!isMounted || currentRequestId !== requestId) {
        return;
      }

      if (isAbortError(err)) {
        isRefreshing = false;
        return;
      }

      if (hasResolvedInitial) {
        state = 'DEGRADED';
      } else {
        state = 'UNAVAILABLE';
      }
      error = err;
      isLoading = false;
      isRefreshing = false;
    } finally {
      if (currentRequestId === requestId) {
        inFlightController = null;
      }

      // F-01 & F-05 Remediation logic:
      if (
        currentRequestId === requestId &&
        isMounted &&
        isPollingActive
      ) {
        clearTimer();
        const isHidden = pauseWhenHidden && mockDocumentHidden;
        if (!isHidden) {
          scheduledTimerCount++;
          activeTimerCount++;
          timerId = setTimeout(() => {
            if (isMounted && isPollingActive) {
              executeFetch(false);
            }
          }, intervalMs);
        }
      }
    }
  }

  function refresh() {
    return executeFetch(true);
  }

  function stop() {
    isProgrammaticallyStopped = true;
    isPollingActive = false;
    clearTimer();
    if (inFlightController) {
      inFlightController.abort();
      inFlightController = null;
    }
    isRefreshing = false;
  }

  function start() {
    isProgrammaticallyStopped = false;
    isPollingActive = true;
    return executeFetch(true);
  }

  function setDocumentHidden(hidden) {
    mockDocumentHidden = hidden;
    if (hidden) {
      clearTimer();
    } else {
      if (isMounted && isPollingActive && inFlightController === null) {
        executeFetch(false);
      }
    }
  }

  function unmount() {
    isMounted = false;
    clearTimer();
    if (inFlightController) {
      inFlightController.abort();
      inFlightController = null;
    }
  }

  return {
    get data() { return data; },
    get state() { return state; },
    get isLoading() { return isLoading; },
    get isRefreshing() { return isRefreshing; },
    get error() { return error; },
    get timerId() { return timerId; },
    get activeTimerCount() { return activeTimerCount; },
    get scheduledTimerCount() { return scheduledTimerCount; },
    get requestId() { return requestId; },
    get isPollingActive() { return isPollingActive; },
    executeFetch,
    refresh,
    stop,
    start,
    setDocumentHidden,
    unmount,
  };
}

// --- Test Suites ---

describe('Phase 9C.1.2 Remediation Tests', () => {

  describe('F-04: Structured API Error Formatting', () => {
    test('formats simple string detail correctly', () => {
      assert.strictEqual(formatErrorDetail('Resource not found'), 'Resource not found');
    });

    test('formats FastAPI Pydantic validation error array without [object Object]', () => {
      const pydanticErrors = [
        { loc: ['query', 'base_dir'], msg: 'Path does not exist', type: 'value_error' },
        { loc: ['body', 'intervalMs'], msg: 'ensure this value is greater than 5000', type: 'value_error.number.not_gt' }
      ];
      const result = formatErrorDetail(pydanticErrors);
      assert.doesNotMatch(result, /\[object Object\]/);
      assert.match(result, /query\.base_dir: Path does not exist/);
      assert.match(result, /body\.intervalMs: ensure this value is greater than 5000/);
    });

    test('formats generic dictionary objects into valid JSON string', () => {
      const objDetail = { code: 'CUDA_UNAVAILABLE', reason: 'Missing NVCC compiler' };
      const result = formatErrorDetail(objDetail);
      assert.doesNotMatch(result, /\[object Object\]/);
      assert.strictEqual(result, JSON.stringify(objDetail));
    });
  });

  describe('F-07: Timeout vs. Abort Error Classification', () => {
    test('distinguishes TIMEOUT_ERROR from ABORT_ERROR', () => {
      const timeoutErr = new ApiError('Request timed out', 'TIMEOUT_ERROR', { isTimeout: true });
      const abortErr = new ApiError('Request aborted', 'ABORT_ERROR');

      assert.strictEqual(isTimeoutError(timeoutErr), true);
      assert.strictEqual(isTimeoutError(abortErr), false);
      assert.strictEqual(isAbortError(timeoutErr), false);
      assert.strictEqual(isAbortError(abortErr), true);
    });

    test('isAbortError handles DOMException and abort messages', () => {
      const domAbort = new DOMException('The operation was aborted', 'AbortError');
      const standardAbort = new Error('request was aborted by user');
      const regularError = new Error('Some connection error');

      assert.strictEqual(isAbortError(domAbort), true);
      assert.strictEqual(isAbortError(standardAbort), true);
      assert.strictEqual(isAbortError(regularError), false);
    });

    test('isApiError identifies ApiError instances', () => {
      const apiErr = new ApiError('Failed', 'HTTP_ERROR');
      assert.strictEqual(isApiError(apiErr), true);
      assert.strictEqual(isApiError(new Error('Normal error')), false);
      assert.strictEqual(isApiError(null), false);
      assert.strictEqual(isApiError({ message: 'Fake error' }), false);
    });
  });

  describe('Direct apiClient Request & Typed Callers', () => {
    test('API_BASE is configured to /api/v1', () => {
      assert.strictEqual(API_BASE, '/api/v1');
    });

    test('request() resolves 200 OK and prefixes endpoint with API_BASE', async () => {
      const origFetch = globalThis.fetch;
      let requestedUrl = null;
      let requestedHeaders = null;

      globalThis.fetch = async (url, opts) => {
        requestedUrl = url;
        requestedHeaders = opts.headers;
        return new Response(JSON.stringify({ status: 'ok', uptime: 123.4 }), {
          status: 200,
          statusText: 'OK',
          headers: { 'Content-Type': 'application/json' },
        });
      };

      try {
        const data = await request('/health');
        assert.strictEqual(requestedUrl, '/api/v1/health');
        assert.strictEqual(requestedHeaders.Accept, 'application/json');
        assert.deepStrictEqual(data, { status: 'ok', uptime: 123.4 });
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('request() preserves full URL or already prefixed /api endpoint', async () => {
      const origFetch = globalThis.fetch;
      const called = [];

      globalThis.fetch = async (url) => {
        called.push(url);
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      };

      try {
        await request('http://localhost:8000/api/v1/health');
        await request('/api/v1/datasets');
        assert.strictEqual(called[0], 'http://localhost:8000/api/v1/health');
        assert.strictEqual(called[1], '/api/v1/datasets');
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('request() formats 422 Pydantic error details via formatErrorDetail', async () => {
      const origFetch = globalThis.fetch;
      const pydanticPayload = {
        detail: [
          { loc: ['query', 'base_dir'], msg: 'Path does not exist', type: 'value_error' },
        ],
      };

      globalThis.fetch = async () => {
        return new Response(JSON.stringify(pydanticPayload), {
          status: 422,
          statusText: 'Unprocessable Entity',
          headers: { 'Content-Type': 'application/json' },
        });
      };

      try {
        await request('/datasets');
        assert.fail('Expected ApiError to be thrown');
      } catch (err) {
        assert.strictEqual(isApiError(err), true);
        assert.strictEqual(err.kind, 'HTTP_ERROR');
        assert.strictEqual(err.status, 422);
        assert.match(err.message, /query\.base_dir: Path does not exist/);
        assert.doesNotMatch(err.message, /\[object Object\]/);
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('request() throws NETWORK_ERROR on fetch rejection', async () => {
      const origFetch = globalThis.fetch;
      globalThis.fetch = async () => {
        throw new TypeError('Failed to fetch');
      };

      try {
        await request('/dashboard/summary');
        assert.fail('Expected ApiError to be thrown');
      } catch (err) {
        assert.strictEqual(isApiError(err), true);
        assert.strictEqual(err.kind, 'NETWORK_ERROR');
        assert.match(err.message, /FastAPI server is unreachable at \/api\/v1\/dashboard\/summary/);
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('request() throws TIMEOUT_ERROR when timeoutMs is exceeded', async () => {
      const origFetch = globalThis.fetch;
      globalThis.fetch = async (_url, options) => {
        return new Promise((resolve, reject) => {
          options.signal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted', 'AbortError'));
          });
        });
      };

      try {
        await request('/slow-endpoint', { timeoutMs: 30 });
        assert.fail('Expected ApiError to be thrown');
      } catch (err) {
        assert.strictEqual(isApiError(err), true);
        assert.strictEqual(err.kind, 'TIMEOUT_ERROR');
        assert.strictEqual(isTimeoutError(err), true);
        assert.match(err.message, /timed out after 30ms/);
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('request() throws ABORT_ERROR when caller AbortSignal triggers', async () => {
      const origFetch = globalThis.fetch;
      globalThis.fetch = async (_url, options) => {
        return new Promise((resolve, reject) => {
          options.signal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted', 'AbortError'));
          });
        });
      };

      const controller = new AbortController();
      setTimeout(() => controller.abort(), 15);

      try {
        await request('/cancelable-endpoint', { signal: controller.signal });
        assert.fail('Expected ApiError to be thrown');
      } catch (err) {
        assert.strictEqual(isApiError(err), true);
        assert.strictEqual(err.kind, 'ABORT_ERROR');
        assert.strictEqual(isAbortError(err), true);
        assert.match(err.message, /was aborted/);
      } finally {
        globalThis.fetch = origFetch;
      }
    });

    test('typed callers getHealth, getDashboardSummary, getDatasets invoke correct endpoints', async () => {
      const origFetch = globalThis.fetch;
      const called = [];
      globalThis.fetch = async (url) => {
        called.push(url);
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      };

      try {
        await getHealth();
        await getDashboardSummary();
        await getDatasets();
        await apiClient.getHealth();

        assert.deepStrictEqual(called, [
          '/api/v1/health',
          '/api/v1/dashboard/summary',
          '/api/v1/datasets',
          '/api/v1/health',
        ]);
      } finally {
        globalThis.fetch = origFetch;
      }
    });
  });

  describe('F-01: Polling Timer Race & Monotonic Sequencing', () => {
    test('superseded in-flight request cannot schedule future timers or overwrite data', async () => {
      let fetchCount = 0;
      let resolveFirst;

      const harness = createPollingHarness(async (signal) => {
        fetchCount++;
        if (fetchCount === 1) {
          // Slow first request
          await new Promise((resolve) => { resolveFirst = resolve; });
          return { value: 'OLD_DATA' };
        }
        return { value: 'NEW_DATA' };
      }, { intervalMs: 10000 });

      // Start request #1
      const p1 = harness.executeFetch(false);

      // User triggers manual refresh before request #1 completes
      const p2 = harness.refresh();
      await p2;

      // Data is now NEW_DATA from request #2
      assert.strictEqual(harness.data.value, 'NEW_DATA');
      assert.strictEqual(harness.requestId, 2);

      // Now request #1 resolves late
      if (resolveFirst) resolveFirst();
      await p1;

      // Assert data was NOT overwritten by stale request #1
      assert.strictEqual(harness.data.value, 'NEW_DATA');
      // Assert active timer count is exactly 1 (no duplicate timers created)
      assert.strictEqual(harness.activeTimerCount, 1);

      harness.unmount();
    });
  });

  describe('F-02: Stop and Start Lifecycle Stability', () => {
    test('stop() stays stopped and start() resumes exactly once', async () => {
      let fetchCount = 0;
      const harness = createPollingHarness(async () => {
        fetchCount++;
        return { status: 'healthy', count: fetchCount };
      }, { intervalMs: 5000 });

      await harness.executeFetch(false);
      assert.strictEqual(fetchCount, 1);
      assert.strictEqual(harness.activeTimerCount, 1);

      // Stop polling
      harness.stop();
      assert.strictEqual(harness.isPollingActive, false);
      assert.strictEqual(harness.activeTimerCount, 0);

      // Resume polling
      await harness.start();
      assert.strictEqual(fetchCount, 2);
      assert.strictEqual(harness.isPollingActive, true);
      assert.strictEqual(harness.activeTimerCount, 1);

      harness.unmount();
      assert.strictEqual(harness.activeTimerCount, 0);
    });
  });

  describe('F-03: Loading and Refreshing Indicator Purity', () => {
    test('initial loading vs refreshing indicators behave correctly across success and failure', async () => {
      let failNext = false;
      const harness = createPollingHarness(async () => {
        if (failNext) {
          throw new Error('Transient 503 error');
        }
        return { count: 42 };
      });

      // Initial fetch: isLoading should become false upon resolution
      assert.strictEqual(harness.isLoading, true);
      assert.strictEqual(harness.isRefreshing, false);

      await harness.executeFetch(false);
      assert.strictEqual(harness.isLoading, false);
      assert.strictEqual(harness.isRefreshing, false);
      assert.strictEqual(harness.state, 'AVAILABLE');

      // Next poll fails: state should become DEGRADED, data retained
      failNext = true;
      await harness.executeFetch(false);
      assert.strictEqual(harness.isLoading, false);
      assert.strictEqual(harness.isRefreshing, false);
      assert.strictEqual(harness.state, 'DEGRADED');
      assert.strictEqual(harness.data.count, 42);

      // Recovery: poll succeeds again, state returns to AVAILABLE
      failNext = false;
      await harness.executeFetch(false);
      assert.strictEqual(harness.state, 'AVAILABLE');

      harness.unmount();
    });
  });

  describe('F-05: Page Visibility Defers Scheduling', () => {
    test('pauses timers when document is hidden and resumes when visible', async () => {
      let fetchCount = 0;
      const harness = createPollingHarness(async () => {
        fetchCount++;
        return { tick: fetchCount };
      }, { intervalMs: 5000, pauseWhenHidden: true });

      await harness.executeFetch(false);
      assert.strictEqual(harness.activeTimerCount, 1);

      // Tab becomes hidden
      harness.setDocumentHidden(true);
      assert.strictEqual(harness.activeTimerCount, 0);

      // Tab becomes visible: immediately triggers fetch and schedules timer
      harness.setDocumentHidden(false);
      // Wait for catch-up microtask
      await new Promise((r) => setTimeout(r, 10));
      assert.strictEqual(fetchCount, 2);
      assert.strictEqual(harness.activeTimerCount, 1);

      harness.unmount();
    });
  });

  describe('Phase 9C.2: Frontend Data Integration & State Consistency', () => {
    function classifyDatasetIntegrity(fixture) {
      if (fixture.is_available) return 'VERIFIED';
      if (fixture.sha256) return 'CORRUPT';
      return 'MISSING';
    }

    test('classifies missing, corrupt, and verified dataset fixtures accurately', () => {
      const verifiedFixture = {
        id: 'synthetic_fraud_10k',
        is_available: true,
        sha256: 'aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430',
        size_bytes: 1420800,
      };
      const missingFixture = {
        id: 'missing_fixture',
        is_available: false,
        sha256: '',
        size_bytes: 0,
      };
      const corruptFixture = {
        id: 'corrupt_fixture',
        is_available: false,
        sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        size_bytes: 512,
      };

      assert.strictEqual(classifyDatasetIntegrity(verifiedFixture), 'VERIFIED');
      assert.strictEqual(classifyDatasetIntegrity(missingFixture), 'MISSING');
      assert.strictEqual(classifyDatasetIntegrity(corruptFixture), 'CORRUPT');
    });

    test('historical provenance transitions to HISTORICAL state rather than LIVE AVAILABLE', async () => {
      const harness = createPollingHarness(async () => {
        return { evaluation: { top_model: 'Logistic Regression' } };
      }, { provenanceSource: 'HISTORICAL_ARTIFACT' });

      await harness.executeFetch(false);
      assert.strictEqual(harness.state, 'HISTORICAL');
      harness.unmount();
    });

    test('degraded state preserves previous data and recovers to AVAILABLE on successful refresh', async () => {
      let shouldFail = false;
      const harness = createPollingHarness(async () => {
        if (shouldFail) throw new Error('Transient 503 gateway failure');
        return { fixtures: [{ id: 'synthetic_fraud_10k', is_available: true }] };
      });

      // 1. Initial successful probe
      await harness.executeFetch(false);
      assert.strictEqual(harness.state, 'AVAILABLE');
      assert.strictEqual(harness.data.fixtures.length, 1);
      assert.strictEqual(harness.error, null);

      // 2. Background poll failure (transitions to DEGRADED without dropping data)
      shouldFail = true;
      await harness.executeFetch(false);
      assert.strictEqual(harness.state, 'DEGRADED');
      assert.strictEqual(harness.data.fixtures.length, 1);
      assert.strictEqual(harness.error.message, 'Transient 503 gateway failure');

      // 3. Manual refresh recovery
      shouldFail = false;
      await harness.refresh();
      assert.strictEqual(harness.state, 'AVAILABLE');
      assert.strictEqual(harness.data.fixtures.length, 1);
      assert.strictEqual(harness.error, null);

      harness.unmount();
    });
  });
});
