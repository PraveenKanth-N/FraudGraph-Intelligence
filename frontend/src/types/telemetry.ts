/**
 * Phase 9C.1 — Truthful Telemetry State & Provenance Definitions
 *
 * Distinguishes explicitly between:
 * - AVAILABLE: Live endpoint returned valid data on current or recent probe
 * - DEGRADED: Polling request failed on refresh, retaining stale previous valid data
 * - UNAVAILABLE: No valid data is available (initial load failed or server unreachable)
 * - HISTORICAL: Grounded in recorded repository experiment/benchmark artifact, not a live execution
 * - NOT_IMPLEMENTED: Feature route or telemetry stream is not yet wired to an executable backend operation
 */

export type TelemetryState =
  | 'AVAILABLE'
  | 'DEGRADED'
  | 'UNAVAILABLE'
  | 'HISTORICAL'
  | 'NOT_IMPLEMENTED';

export type TelemetryProvenanceSource =
  | 'LIVE_ENDPOINT'
  | 'HISTORICAL_ARTIFACT'
  | 'UNAVAILABLE'
  | 'NOT_IMPLEMENTED';

export interface TelemetryProvenance {
  source: TelemetryProvenanceSource;
  endpoint?: string;
  artifactPath?: string;
  hardwareContext?: string;
  caveats?: string[];
  description?: string;
}

export interface PollingResourceState<T> {
  data: T | null;
  state: TelemetryState;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  lastSuccessTime: Date | null;
  consecutiveFailures: number;
  provenance: TelemetryProvenance;
}

export const TELEMETRY_STATE_DESCRIPTIONS: Record<TelemetryState, string> = {
  AVAILABLE: 'Live endpoint operational and returning verified data',
  DEGRADED: 'Background refresh failed; retaining last known valid data',
  UNAVAILABLE: 'Endpoint unreachable or no valid data available',
  HISTORICAL: 'Verified recorded experiment artifact (not a live execution)',
  NOT_IMPLEMENTED: 'Interface present; backend execution pipeline deferred',
};
