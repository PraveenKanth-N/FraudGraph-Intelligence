import { HealthData, DashboardSummary, DatasetListResponse } from '../types/api';

const API_BASE = '/api/v1';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const errorText = await res.text().catch(() => 'Unknown network error');
    throw new Error(`HTTP ${res.status}: ${errorText || res.statusText}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthData>(res);
}

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const res = await fetch(`${API_BASE}/dashboard/summary`);
  return handleResponse<DashboardSummary>(res);
}

export async function fetchDatasets(): Promise<DatasetListResponse> {
  const res = await fetch(`${API_BASE}/datasets`);
  return handleResponse<DatasetListResponse>(res);
}
