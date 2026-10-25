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
  experiment_id: string;
  phase: string;
  top_model: string;
  top_test_pr_auc: number;
  top_test_roc_auc: number;
  top_test_f1: number;
  primary_policy: string;
  sensitivity_policy: string;
  num_models_evaluated: number;
  num_representations: number;
  adr_007_notice: string;
}

export interface BenchmarkSummary {
  session_id: string;
  phase: string;
  total_runs: number;
  measured_cpu_runs: number;
  skipped_gpu_runs: number;
  top_speedup_vs_cpu_ref: number;
  top_speedup_workload: string;
  local_gpu_status: string;
  high_variability_label: string;
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
}

export interface DatasetListResponse {
  status: string;
  total_count: number;
  fixtures: DatasetFixtureItem[];
}
