import { HealthData } from '../types/api';
import { getHealth } from '../services/apiClient';
import { usePollingResource, UsePollingResourceReturn } from './usePollingResource';

export interface UseHealthPollingOptions {
  /** Polling interval in ms (minimum 5,000ms, default 15,000ms). */
  intervalMs?: number;
  /** Whether polling is active. Defaults to true. */
  enabled?: boolean;
}

/**
 * Hook for polling live system health and GPU diagnostic availability.
 *
 * Implements periodic HTTP polling against /api/v1/health with cancellation.
 * Does NOT simulate real-time GPU load, temperature, or dynamic streaming.
 */
export function useHealthPolling(
  options: UseHealthPollingOptions = {}
): UsePollingResourceReturn<HealthData> {
  const { intervalMs = 15000, enabled = true } = options;

  return usePollingResource<HealthData>(
    (signal) => getHealth({ signal }),
    {
      intervalMs,
      enabled,
      provenanceSource: 'LIVE_ENDPOINT',
      endpoint: '/api/v1/health',
      hardwareContext: 'Host runtime probe (sys.version + nvidia-smi fallback query)',
      caveats: [
        'Local GeForce MX450 lacks host NVCC compiler; local CUDA runs marked SKIPPED_CUDA_UNAVAILABLE',
        'Kernel execution latency (0.0679ms) is from remote Colab Tesla T4 fixture, not host GPU',
        'Does not infer live GPU utilization, memory usage, or clock speeds',
      ],
      description: 'Periodic live HTTP diagnostics probe against FastAPI backend',
    }
  );
}

export default useHealthPolling;
