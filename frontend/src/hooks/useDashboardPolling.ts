import { DashboardSummary } from '../types/api';
import { getDashboardSummary } from '../services/apiClient';
import { usePollingResource, UsePollingResourceReturn } from './usePollingResource';

export interface UseDashboardPollingOptions {
  /** Polling interval in ms (minimum 5,000ms, default 30,000ms). */
  intervalMs?: number;
  /** Whether polling is active. Defaults to true. */
  enabled?: boolean;
}

/**
 * Hook for querying repository dashboard summary metrics.
 *
 * Implements periodic HTTP polling against /api/v1/dashboard/summary.
 * Explicitly marks metrics as HISTORICAL artifacts rather than live telemetry.
 */
export function useDashboardPolling(
  options: UseDashboardPollingOptions = {}
): UsePollingResourceReturn<DashboardSummary> {
  const { intervalMs = 30000, enabled = true } = options;

  return usePollingResource<DashboardSummary>(
    (signal) => getDashboardSummary({ signal }),
    {
      intervalMs,
      enabled,
      provenanceSource: 'HISTORICAL_ARTIFACT',
      endpoint: '/api/v1/dashboard/summary',
      artifactPath: 'experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv',
      caveats: [
        'Top speedup 44.60x is the arithmetic mean speedup for the MEAN operator (synthetic_tier_a_10k_mean)',
        'SUM operator speedup is 25.93x with documented HIGH_VARIABILITY (CV 189.0%)',
        'Model PR-AUC (0.3315) and ROC-AUC (0.7865) are frozen test metrics from Phase 7.5 under ADR-007',
        'Pure GPU kernel latency (0.0679ms) is from remote Colab Tesla T4 archive, not local execution',
      ],
      description: 'Repository evaluation and benchmark summary backed by authentic disk artifacts',
    }
  );
}

export default useDashboardPolling;
