import { useState, useEffect, useRef, useCallback } from 'react';
import {
  TelemetryState,
  TelemetryProvenance,
  TelemetryProvenanceSource,
  PollingResourceState,
} from '../types/telemetry';
import { isAbortError } from '../services/apiClient';

/** Minimum enforced polling interval per Phase 9C requirements (5,000 ms). */
export const MIN_POLLING_INTERVAL_MS = 5000;

/** Default conservative polling interval (10,000 ms). */
export const DEFAULT_POLLING_INTERVAL_MS = 10000;

export interface UsePollingResourceOptions<T> {
  /**
   * Target polling interval in milliseconds.
   * Clamped to a minimum of 5,000 ms (5 seconds).
   * Note: This implements periodic HTTP polling, not a WebSocket or streaming connection.
   */
  intervalMs?: number;

  /** Whether automatic polling is active. Defaults to true. */
  enabled?: boolean;

  /**
   * Whether to pause background polling when document tab is hidden.
   * Resumes automatically when tab becomes visible. Defaults to true.
   */
  pauseWhenHidden?: boolean;

  /** Provenance attribution metadata for truthful state reporting. */
  provenanceSource?: TelemetryProvenanceSource;
  endpoint?: string;
  artifactPath?: string;
  hardwareContext?: string;
  caveats?: string[];
  description?: string;

  /** Lifecycle callbacks. */
  onSuccess?: (data: T) => void;
  onError?: (err: Error) => void;
}

export interface UsePollingResourceReturn<T> extends PollingResourceState<T> {
  /** Manually trigger an immediate out-of-band poll. */
  refresh: () => Promise<void>;
  /** Pause polling timers until explicitly started. */
  stop: () => void;
  /** Resume or start polling timers. */
  start: () => void;
}

/**
 * Reusable, lifecycle-safe polling hook for FraudGraph Intelligence.
 *
 * Guarantees (Phase 9C.1.2 Remediation):
 * 1. Clamped minimum interval (>= 5,000ms).
 * 2. Automatic request cancellation via AbortSignal on unmount or refresh.
 * 3. Prevention of overlapping requests via chained setTimeout scheduling.
 * 4. Stale response protection via monotonic request sequence counters.
 * 5. F-01 Fix: Only active request lifecycle can schedule timers in finally.
 * 6. F-02 Fix: Decoupled programmatic stop/start from reactive render state.
 * 7. F-03 Fix: Pure state setting without nested functional updaters.
 * 8. F-05 Fix: Safe Page Visibility detection pauses polling in hidden tabs.
 * 9. Honest telemetry states: AVAILABLE, DEGRADED (stale data retained), or UNAVAILABLE.
 */
export function usePollingResource<T>(
  fetchFn: (signal?: AbortSignal) => Promise<T>,
  options: UsePollingResourceOptions<T> = {}
): UsePollingResourceReturn<T> {
  const {
    intervalMs = DEFAULT_POLLING_INTERVAL_MS,
    enabled = true,
    pauseWhenHidden = true,
    provenanceSource = 'LIVE_ENDPOINT',
    endpoint,
    artifactPath,
    hardwareContext,
    caveats = [],
    description,
    onSuccess,
    onError,
  } = options;

  // Enforce the >= 5,000ms safety boundary
  const effectiveIntervalMs = Math.max(intervalMs, MIN_POLLING_INTERVAL_MS);

  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<TelemetryState>('UNAVAILABLE');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastSuccessTime, setLastSuccessTime] = useState<Date | null>(null);
  const [consecutiveFailures, setConsecutiveFailures] = useState<number>(0);

  // Synchronization refs (Decoupled from render cycle to avoid reactive resets)
  const isMountedRef = useRef<boolean>(true);
  const inFlightControllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestIdRef = useRef<number>(0);
  const hasResolvedInitialRef = useRef<boolean>(false);
  const isPollingActiveRef = useRef<boolean>(enabled);
  const isProgrammaticallyStoppedRef = useRef<boolean>(false);

  const fetchFnRef = useRef(fetchFn);
  const onSuccessRef = useRef(onSuccess);
  const onErrorRef = useRef(onError);

  fetchFnRef.current = fetchFn;
  onSuccessRef.current = onSuccess;
  onErrorRef.current = onError;

  const provenance: TelemetryProvenance = {
    source: provenanceSource,
    endpoint,
    artifactPath,
    hardwareContext,
    caveats,
    description:
      description ||
      (provenanceSource === 'HISTORICAL_ARTIFACT'
        ? 'Static verified repository artifact'
        : 'Periodic live HTTP probe (FastAPI backend)'),
  };

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const executeFetch = useCallback(
    async (_isManualRefresh: boolean = false): Promise<void> => {
      // 1. Cancel any prior in-flight request to prevent overlapping execution
      if (inFlightControllerRef.current) {
        inFlightControllerRef.current.abort();
        inFlightControllerRef.current = null;
      }

      clearTimer();

      // 2. Setup fresh AbortController and track monotonic request ID
      const controller = new AbortController();
      inFlightControllerRef.current = controller;
      const currentRequestId = ++requestIdRef.current;

      if (!isMountedRef.current) return;

      // F-03: Set loading/refreshing indicators directly without nested updater side-effects
      if (!hasResolvedInitialRef.current) {
        setIsLoading(true);
      } else {
        setIsRefreshing(true);
      }

      try {
        const result = await fetchFnRef.current(controller.signal);

        // Guard: check if component unmounted or a newer request superseded this one
        if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
          return;
        }

        hasResolvedInitialRef.current = true;
        setData(result);
        setError(null);
        setConsecutiveFailures(0);
        setLastSuccessTime(new Date());

        // Determine truthful telemetry state
        if (provenanceSource === 'HISTORICAL_ARTIFACT') {
          setState('HISTORICAL');
        } else {
          setState('AVAILABLE');
        }

        setIsLoading(false);
        setIsRefreshing(false);
        onSuccessRef.current?.(result);
      } catch (err: unknown) {
        // Discard if component unmounted or superseded
        if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
          return;
        }

        // Intentionally aborted requests do not trigger error state
        if (isAbortError(err)) {
          setIsRefreshing(false);
          return;
        }

        const normalizedError =
          err instanceof Error ? err : new Error(String(err));

        // If we had valid data previously, preserve it and mark DEGRADED
        if (hasResolvedInitialRef.current) {
          setState('DEGRADED');
        } else {
          setState('UNAVAILABLE');
        }

        setError(normalizedError);
        setConsecutiveFailures((prev) => prev + 1);
        setIsLoading(false);
        setIsRefreshing(false);
        onErrorRef.current?.(normalizedError);
      } finally {
        if (currentRequestId === requestIdRef.current) {
          inFlightControllerRef.current = null;
        }

        // F-01 Fix: Only the current, active request lifecycle can schedule the next timer
        // F-05 Fix: Defer scheduling if tab is currently hidden
        if (
          currentRequestId === requestIdRef.current &&
          isMountedRef.current &&
          isPollingActiveRef.current
        ) {
          clearTimer();
          const isHidden =
            pauseWhenHidden &&
            typeof document !== 'undefined' &&
            document.visibilityState === 'hidden';

          if (!isHidden) {
            timerRef.current = setTimeout(() => {
              if (isMountedRef.current && isPollingActiveRef.current) {
                executeFetch(false);
              }
            }, effectiveIntervalMs);
          }
        }
      }
    },
    [clearTimer, effectiveIntervalMs, pauseWhenHidden, provenanceSource]
  );

  // F-02 Fix: Initial mount and external enabled prop synchronization
  useEffect(() => {
    isMountedRef.current = true;

    if (enabled && !isProgrammaticallyStoppedRef.current) {
      isPollingActiveRef.current = true;
      executeFetch(false);
    } else {
      isPollingActiveRef.current = false;
      clearTimer();
      setIsLoading(false);
      setIsRefreshing(false);
    }

    return () => {
      isMountedRef.current = false;
      clearTimer();
      if (inFlightControllerRef.current) {
        inFlightControllerRef.current.abort();
        inFlightControllerRef.current = null;
      }
    };
  }, [enabled, executeFetch, clearTimer]);

  // F-05 Fix: Page visibility listener pauses timers when hidden, resumes on visible
  useEffect(() => {
    if (typeof document === 'undefined' || !pauseWhenHidden) return;

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // Tab became visible: resume polling if active, not stopped, and no request running
        if (
          isMountedRef.current &&
          isPollingActiveRef.current &&
          inFlightControllerRef.current === null
        ) {
          executeFetch(false);
        }
      } else {
        // Tab became hidden: cancel pending scheduled timer
        clearTimer();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [pauseWhenHidden, executeFetch, clearTimer]);

  const refresh = useCallback(async (): Promise<void> => {
    await executeFetch(true);
  }, [executeFetch]);

  // F-02 Fix: stop() remains stopped without reactive effect loops
  const stop = useCallback(() => {
    isProgrammaticallyStoppedRef.current = true;
    isPollingActiveRef.current = false;
    clearTimer();
    if (inFlightControllerRef.current) {
      inFlightControllerRef.current.abort();
      inFlightControllerRef.current = null;
    }
    setIsRefreshing(false);
  }, [clearTimer]);

  // F-02 Fix: start() resumes polling exactly once
  const start = useCallback(() => {
    isProgrammaticallyStoppedRef.current = false;
    isPollingActiveRef.current = true;
    executeFetch(true);
  }, [executeFetch]);

  return {
    data,
    state,
    isLoading,
    isRefreshing,
    error,
    lastSuccessTime,
    consecutiveFailures,
    provenance,
    refresh,
    stop,
    start,
  };
}

export default usePollingResource;
