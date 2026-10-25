import type {
  HealthData,
  DashboardSummary,
  DatasetListResponse,
  SubgraphResponse,
  TransactionListResponse,
  TransactionFilter,
  ApiErrorKind,
  ApiErrorDetails,
} from '../types/api';

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly statusText?: string;
  readonly url?: string;
  readonly details?: unknown;
  readonly rawText?: string;

  constructor(
    message: string,
    kind: ApiErrorKind,
    options: ApiErrorDetails = {}
  ) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = options.status;
    this.statusText = options.statusText;
    this.url = options.url;
    this.details = options.body;
    this.rawText = options.rawText;

    // Maintains proper stack trace for where error was thrown (V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

export function isTimeoutError(err: unknown): boolean {
  if (err instanceof ApiError) {
    return err.kind === 'TIMEOUT_ERROR';
  }
  return false;
}

export function isAbortError(err: unknown): boolean {
  if (err instanceof ApiError) {
    return err.kind === 'ABORT_ERROR';
  }
  if (err instanceof DOMException) {
    return err.name === 'AbortError';
  }
  if (err instanceof Error) {
    return err.name === 'AbortError' || err.message.toLowerCase().includes('abort');
  }
  return false;
}

export interface RequestOptions extends Omit<RequestInit, 'signal'> {
  signal?: AbortSignal;
  timeoutMs?: number;
  baseUrl?: string;
}

/**
 * Formats structured error detail payloads from FastAPI / Pydantic without [object Object].
 */
export function formatErrorDetail(detail: unknown): string {
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object') {
          const loc = Array.isArray((item as { loc?: unknown[] }).loc)
            ? (item as { loc: unknown[] }).loc.join('.')
            : String((item as { loc?: unknown }).loc || '');
          const msg =
            (item as { msg?: string }).msg ||
            (item as { message?: string }).message ||
            JSON.stringify(item);
          return loc ? `${loc}: ${msg}` : String(msg);
        }
        return String(item);
      })
      .join('; ');
  }
  if (detail !== null && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return String(detail);
}

/**
 * Executes a typed HTTP request against the FraudGraph Intelligence API.
 *
 * Ensures:
 * - Proper error classification (HTTP, Network, Parse, Abort, Timeout).
 * - Lifecycle cancellation via AbortSignal.
 * - Zero silent metric fabrication on failure.
 */
export async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const base = options.baseUrl ?? (typeof window !== 'undefined' ? '' : (typeof process !== 'undefined' && process.env?.API_BASE_URL ? process.env.API_BASE_URL : ''));
  const url = endpoint.startsWith('http')
    ? endpoint
    : endpoint.startsWith('/api')
      ? `${base}${endpoint}`
      : `${base}${API_BASE}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

  const { timeoutMs, signal: callerSignal, baseUrl: _unusedBase, ...fetchOptions } = options;

  let timerId: ReturnType<typeof setTimeout> | null = null;
  let internalController: AbortController | null = null;
  let isTimedOut = false;
  let activeSignal = callerSignal;

  if (timeoutMs && timeoutMs > 0) {
    internalController = new AbortController();
    timerId = setTimeout(() => {
      isTimedOut = true;
      internalController?.abort();
    }, timeoutMs);

    if (callerSignal) {
      if (callerSignal.aborted) {
        internalController.abort();
      } else {
        callerSignal.addEventListener(
          'abort',
          () => internalController?.abort(),
          { once: true }
        );
      }
      activeSignal = internalController.signal;
    } else {
      activeSignal = internalController.signal;
    }
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
      signal: activeSignal,
      headers: {
        Accept: 'application/json',
        ...fetchOptions.headers,
      },
    });
  } catch (err: unknown) {
    if (timerId) clearTimeout(timerId);

    if (isTimedOut) {
      throw new ApiError(
        `Request to ${url} timed out after ${timeoutMs}ms.`,
        'TIMEOUT_ERROR',
        { url, isTimeout: true }
      );
    }

    if (isAbortError(err)) {
      throw new ApiError(
        `Request to ${url} was aborted.`,
        'ABORT_ERROR',
        { url }
      );
    }

    const networkMessage = err instanceof Error ? err.message : 'Unknown network failure';
    throw new ApiError(
      `FastAPI server is unreachable at ${url}. Verify that the backend service is running. Details: ${networkMessage}`,
      'NETWORK_ERROR',
      { url, rawText: networkMessage }
    );
  } finally {
    if (timerId) clearTimeout(timerId);
  }

  if (!response.ok) {
    let errorBody: unknown = null;
    let rawText = '';
    try {
      rawText = await response.text();
      errorBody = JSON.parse(rawText);
    } catch {
      // Body was either plain text or empty
    }

    let detailMsg = '';
    if (errorBody && typeof errorBody === 'object' && 'detail' in errorBody) {
      detailMsg = formatErrorDetail((errorBody as { detail: unknown }).detail);
    } else {
      detailMsg = rawText || response.statusText || 'HTTP Error';
    }

    throw new ApiError(
      `HTTP ${response.status} (${response.statusText || 'Error'}) from ${url}: ${detailMsg}`,
      'HTTP_ERROR',
      {
        status: response.status,
        statusText: response.statusText,
        url,
        body: errorBody,
        rawText,
      }
    );
  }

  try {
    const data = await response.json();
    return data as T;
  } catch (err: unknown) {
    const parseMessage = err instanceof Error ? err.message : 'Invalid JSON';
    throw new ApiError(
      `Failed to parse JSON response from ${url}: ${parseMessage}`,
      'PARSE_ERROR',
      { status: response.status, statusText: response.statusText, url }
    );
  }
}

export async function get<T>(
  endpoint: string,
  options?: RequestOptions
): Promise<T> {
  return request<T>(endpoint, { ...options, method: 'GET' });
}

export async function getHealth(
  options?: RequestOptions
): Promise<HealthData> {
  return get<HealthData>('/health', options);
}

export async function getDashboardSummary(
  options?: RequestOptions
): Promise<DashboardSummary> {
  return get<DashboardSummary>('/dashboard/summary', options);
}

export async function getDatasets(
  options?: RequestOptions
): Promise<DatasetListResponse> {
  return get<DatasetListResponse>('/datasets', options);
}

export interface SubgraphParams {
  nodeId: number;
  hops?: number;
  maxNodes?: number;
  maxEdges?: number;
}

export interface TransactionsParams {
  offset?: number;
  limit?: number;
  filter?: TransactionFilter;
}

export async function getSubgraph(
  params: SubgraphParams,
  options?: RequestOptions
): Promise<SubgraphResponse> {
  const query = new URLSearchParams();
  query.set('node_id', String(params.nodeId));
  if (params.hops !== undefined) query.set('hops', String(params.hops));
  if (params.maxNodes !== undefined) query.set('max_nodes', String(params.maxNodes));
  if (params.maxEdges !== undefined) query.set('max_edges', String(params.maxEdges));
  return get<SubgraphResponse>(`/graph/subgraph?${query.toString()}`, options);
}

export async function getTransactions(
  params: TransactionsParams = {},
  options?: RequestOptions
): Promise<TransactionListResponse> {
  const query = new URLSearchParams();
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.filter !== undefined) query.set('filter', params.filter);
  const qStr = query.toString();
  return get<TransactionListResponse>(`/transactions${qStr ? `?${qStr}` : ''}`, options);
}

export const apiClient = {
  request,
  get,
  getHealth,
  getDashboardSummary,
  getDatasets,
  getSubgraph,
  getTransactions,
};

export default apiClient;
