export interface HealthComputeEngine {
  backend: string;
  cuda_available_locally: boolean;
  local_gpu_detected: string | null;
  vram_free_mb: number | null;
  cuda_compiler_status: string;
  colab_workflow_enabled: boolean;
}

export interface HealthSystem {
  python_version: string;
  timestamp: string;
}

export interface HealthData {
  status: string;
  project: string;
  project_id: string;
  version: string;
  compute_engine: HealthComputeEngine;
  system: HealthSystem;
}

export interface DatasetSummary {
  name: string;
  num_nodes: number;
  num_edges: number;
  feature_dim: number;
  fraud_count: number;
  licit_count: number;
  fraud_rate_pct: number;
  is_synthetic: boolean;
  provenance: string;
  relational_keys: string[];
  timespan_days: number;
  sha256: string;
}

export interface EvaluationSummary {
  artifact_state?: 'AVAILABLE' | 'MISSING' | 'INVALID';
  experiment_id: string;
  phase: string;
  top_model: string | null;
  top_test_pr_auc: number | null;
  top_test_roc_auc: number | null;
  top_test_f1: number | null;
  primary_policy: string;
  sensitivity_policy: string;
  num_models_evaluated: number;
  num_representations: number;
  adr_007_notice: string;
  error_message?: string | null;
}

export interface BenchmarkSummary {
  artifact_state?: 'AVAILABLE' | 'MISSING' | 'INVALID';
  session_id: string;
  phase: string;
  total_runs: number;
  measured_cpu_runs: number;
  skipped_gpu_runs: number;
  top_speedup_vs_cpu_ref: number | null;
  top_speedup_workload: string | null;
  local_gpu_status: string;
  high_variability_label: string;
  error_message?: string | null;
}

export interface HardwareSummary {
  local_gpu_detected: string | null;
  local_compiler_status: string;
  local_vram_free_mb: number | null;
  canonical_colab_gpu: string;
  colab_kernel_latency_ms: number;
}

export interface DashboardSummary {
  status: string;
  project: string;
  dataset: DatasetSummary;
  evaluation: EvaluationSummary;
  benchmarks: BenchmarkSummary;
  hardware: HardwareSummary;
  timestamp: string;
}

export interface DatasetFixtureItem {
  id: string;
  name: string;
  file_name: string;
  num_nodes: number;
  num_edges: number;
  feature_dim: number;
  size_bytes: number;
  is_synthetic: boolean;
  description: string;
  sha256: string;
  is_available?: boolean;
}

export type ApiErrorKind =
  | 'HTTP_ERROR'
  | 'NETWORK_ERROR'
  | 'PARSE_ERROR'
  | 'ABORT_ERROR'
  | 'TIMEOUT_ERROR';

export interface ApiErrorDetails {
  status?: number;
  statusText?: string;
  url?: string;
  body?: unknown;
  rawText?: string;
  isTimeout?: boolean;
}

export interface DatasetListResponse {
  status: string;
  total_count: number;
  fixtures: DatasetFixtureItem[];
}

export interface SubgraphNode {
  id: number;
  depth: number;
  is_root: boolean;
  timestamp_sec: number;
  fraud_label: number;
  is_fraud: boolean;
  degree: number;
}

export interface SubgraphEdge {
  source: number;
  target: number;
  temporal_delta_sec?: number | null;
}

export interface SubgraphQueryMeta {
  root_node_id: number;
  requested_hops: number;
  actual_hops: number;
  direction: string;
  max_nodes: number;
  max_edges: number;
}

export interface SubgraphSummary {
  total_nodes: number;
  total_edges: number;
  fraud_node_count: number;
  fraud_percentage: number;
  is_truncated: boolean;
  node_cap_hit: boolean;
  edge_cap_hit: boolean;
  truncation_reason?: string | null;
}

export interface SubgraphProvenance {
  fixture_name: string;
  sha256: string;
  causality_policy: string;
}

export interface SubgraphResponse {
  status: string;
  query: SubgraphQueryMeta;
  summary: SubgraphSummary;
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  provenance: SubgraphProvenance;
}

export type TransactionFilter = 'all' | 'fraud' | 'licit';

export interface TransactionItem {
  tx_id: number;
  timestamp_sec: number;
  fraud_label: number;
  is_fraud: boolean;
  risk_tier: string;
  in_degree: number;
  feature_dim: number;
  feature_sample: number[];
  source_card?: string | null;
  target_device?: string | null;
  notes: string;
}

export interface TransactionListResponse {
  status: string;
  total_dataset_records: number;
  total_fraud_records: number;
  total_licit_records: number;
  filtered_count: number;
  offset: number;
  limit: number;
  returned_count: number;
  has_next: boolean;
  has_prev: boolean;
  current_filter: TransactionFilter;
  ordering: string;
  items: TransactionItem[];
  schema_notes: string;
  provenance_sha256: string;
}
