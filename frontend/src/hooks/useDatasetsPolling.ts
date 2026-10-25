import { DatasetListResponse } from '../types/api';
import { getDatasets } from '../services/apiClient';
import { usePollingResource, UsePollingResourceReturn } from './usePollingResource';

export interface UseDatasetsPollingOptions {
  /** Polling interval in ms (minimum 5,000ms, default 30,000ms). */
  intervalMs?: number;
  /** Whether polling is active. Defaults to true. */
  enabled?: boolean;
}

/**
 * Hook for querying authentic repository dataset fixture metadata.
 *
 * Implements periodic HTTP polling against /api/v1/datasets.
 * Reflects authentic disk availability without hardcoded fallback fabrication.
 */
export function useDatasetsPolling(
  options: UseDatasetsPollingOptions = {}
): UsePollingResourceReturn<DatasetListResponse> {
  const { intervalMs = 30000, enabled = true } = options;

  return usePollingResource<DatasetListResponse>(
    (signal) => getDatasets({ signal }),
    {
      intervalMs,
      enabled,
      provenanceSource: 'LIVE_ENDPOINT',
      endpoint: '/api/v1/datasets',
      hardwareContext: 'Local host filesystem inspection in data/fixtures/',
      caveats: [
        'Fixtures are verified on disk; missing files return is_available=False, size_bytes=0, and sha256=""',
        'CSR dimensions (nodes, edges, feature dimension) are read from authentic NPZ arrays',
      ],
      description: 'Dynamic filesystem inspection of canonical dataset fixtures',
    }
  );
}

export default useDatasetsPolling;
