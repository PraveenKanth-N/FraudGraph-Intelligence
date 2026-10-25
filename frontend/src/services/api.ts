import {
  HealthData,
  DashboardSummary,
  DatasetListResponse,
  SubgraphResponse,
  TransactionListResponse,
  TransactionFilter,
} from '../types/api';
import {
  apiClient,
  ApiError,
  isApiError,
  isAbortError,
  isTimeoutError,
  formatErrorDetail,
  RequestOptions,
  getHealth,
  getDashboardSummary,
  getDatasets,
  getSubgraph,
  getTransactions,
  SubgraphParams,
  TransactionsParams,
  API_BASE,
} from './apiClient';

export {
  apiClient,
  ApiError,
  isApiError,
  isAbortError,
  isTimeoutError,
  formatErrorDetail,
  API_BASE,
};
export type { RequestOptions, SubgraphParams, TransactionsParams, TransactionFilter };

export async function fetchHealth(options?: RequestOptions): Promise<HealthData> {
  return getHealth(options);
}

export async function fetchDashboardSummary(options?: RequestOptions): Promise<DashboardSummary> {
  return getDashboardSummary(options);
}

export async function fetchDatasets(options?: RequestOptions): Promise<DatasetListResponse> {
  return getDatasets(options);
}

export async function fetchSubgraph(
  params: SubgraphParams,
  options?: RequestOptions
): Promise<SubgraphResponse> {
  return getSubgraph(params, options);
}

export async function fetchTransactions(
  params: TransactionsParams = {},
  options?: RequestOptions
): Promise<TransactionListResponse> {
  return getTransactions(params, options);
}

export default apiClient;
