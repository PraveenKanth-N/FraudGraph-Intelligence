import os
import csv
import math
import tempfile
from datetime import datetime
from typing import List, Optional, Set
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.api.v1.endpoints.health import get_gpu_diagnostics

router = APIRouter()

# Base directory for resolving repository files safely (5 levels up from backend/app/api/v1/endpoints)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

CANONICAL_EVALUATION_PATH = os.path.join(
    WORKSPACE_ROOT, "experiments", "results", "evaluation_reporting_7_5", "summary_evaluation_metrics.csv"
)
CANONICAL_BENCHMARK_PATH = os.path.join(
    WORKSPACE_ROOT, "experiments", "results", "benchmark_runs", "session_20261024_230008_smoke", "summary_table.csv"
)

REQUIRED_EVAL_COLUMNS: Set[str] = {
    "context_policy",
    "model_name",
    "representation",
    "test_pr_auc",
    "test_roc_auc",
    "test_f1",
}

REQUIRED_BENCH_COLUMNS: Set[str] = {
    "workload_name",
    "operator",
    "engine",
    "measurement_state",
    "speedup_vs_cpu_ref",
}

def _is_safe_path(path: str) -> bool:
    """Ensure path is within WORKSPACE_ROOT or an authorized system temp directory."""
    abs_path = os.path.abspath(path)
    temp_dir = os.path.abspath(tempfile.gettempdir())

    in_workspace = False
    try:
        in_workspace = os.path.commonpath([abs_path, WORKSPACE_ROOT]) == WORKSPACE_ROOT
    except ValueError:
        pass

    in_temp = False
    try:
        in_temp = os.path.commonpath([abs_path, temp_dir]) == temp_dir
    except ValueError:
        pass

    return in_workspace or in_temp

class DatasetSummaryModel(BaseModel):
    name: str
    num_nodes: int
    num_edges: int
    feature_dim: int
    fraud_count: int
    licit_count: int
    fraud_rate_pct: float
    is_synthetic: bool
    provenance: str
    relational_keys: List[str]
    timespan_days: float
    sha256: str

class EvaluationSummaryModel(BaseModel):
    artifact_state: str  # "AVAILABLE", "MISSING", "INVALID"
    experiment_id: str
    phase: str
    top_model: Optional[str] = None
    top_test_pr_auc: Optional[float] = None
    top_test_roc_auc: Optional[float] = None
    top_test_f1: Optional[float] = None
    primary_policy: str
    sensitivity_policy: str
    num_models_evaluated: int
    num_representations: int
    adr_007_notice: str
    error_message: Optional[str] = None

class BenchmarkSummaryModel(BaseModel):
    artifact_state: str  # "AVAILABLE", "MISSING", "INVALID"
    session_id: str
    phase: str
    total_runs: int
    measured_cpu_runs: int
    skipped_gpu_runs: int
    top_speedup_vs_cpu_ref: Optional[float] = None
    top_speedup_workload: Optional[str] = None
    local_gpu_status: str
    high_variability_label: str
    error_message: Optional[str] = None

class HardwareSummaryModel(BaseModel):
    local_gpu_detected: Optional[str]
    local_compiler_status: str
    local_vram_free_mb: Optional[float]
    canonical_colab_gpu: str
    colab_kernel_latency_ms: float

class DashboardSummaryResponse(BaseModel):
    status: str
    project: str
    dataset: DatasetSummaryModel
    evaluation: EvaluationSummaryModel
    benchmarks: BenchmarkSummaryModel
    hardware: HardwareSummaryModel
    timestamp: str

def _load_evaluation_summary(csv_path: Optional[str] = None) -> EvaluationSummaryModel:
    """Read and strictly validate genuine evaluation metrics from the Phase 7.5 CSV artifact."""
    target_path = csv_path or CANONICAL_EVALUATION_PATH
    default_exp_id = "EXP-EVAL-PHASE7-4"
    default_phase = "Phase 7.4 / 7.5"
    default_adr_notice = "Continuous PR curve omitted per ADR-007; discrete validation thresholding policy applied."

    if not os.path.exists(target_path):
        return EvaluationSummaryModel(
            artifact_state="MISSING",
            experiment_id=default_exp_id,
            phase=default_phase,
            top_model=None,
            top_test_pr_auc=None,
            top_test_roc_auc=None,
            top_test_f1=None,
            primary_policy="causal_history",
            sensitivity_policy="train_only_context",
            num_models_evaluated=0,
            num_representations=0,
            adr_007_notice=default_adr_notice,
            error_message=f"Evaluation artifact file not found: {os.path.basename(target_path)}"
        )

    try:
        if os.path.getsize(target_path) == 0:
            return EvaluationSummaryModel(
                artifact_state="INVALID",
                experiment_id=default_exp_id,
                phase=default_phase,
                top_model=None,
                top_test_pr_auc=None,
                top_test_roc_auc=None,
                top_test_f1=None,
                primary_policy="causal_history",
                sensitivity_policy="train_only_context",
                num_models_evaluated=0,
                num_representations=0,
                adr_007_notice=default_adr_notice,
                error_message="Evaluation artifact file is empty"
            )

        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=0,
                    num_representations=0,
                    adr_007_notice=default_adr_notice,
                    error_message="Evaluation artifact contains no header row"
                )

            missing_cols = REQUIRED_EVAL_COLUMNS - set(reader.fieldnames)
            if missing_cols:
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=0,
                    num_representations=0,
                    adr_007_notice=default_adr_notice,
                    error_message=f"Evaluation artifact missing required columns: {sorted(missing_cols)}"
                )

            target_row = None
            models_evaluated = set()
            representations = set()

            for row in reader:
                model = row.get("model_name")
                rep = row.get("representation")
                if model:
                    models_evaluated.add(model)
                if rep:
                    representations.add(rep)

                if (
                    row.get("context_policy") == "causal_history"
                    and model == "logistic_regression"
                    and rep == "max"
                ):
                    target_row = row

            if target_row is None:
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=len(models_evaluated),
                    num_representations=len(representations),
                    adr_007_notice=default_adr_notice,
                    error_message="Target evaluation row (causal_history, logistic_regression, max) not found"
                )

            try:
                raw_pr = float(target_row["test_pr_auc"])
                raw_roc = float(target_row["test_roc_auc"])
                raw_f1 = float(target_row["test_f1"])
            except (ValueError, TypeError) as e:
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=len(models_evaluated),
                    num_representations=len(representations),
                    adr_007_notice=default_adr_notice,
                    error_message=f"Non-numeric metric value in target row: {e}"
                )

            if not (math.isfinite(raw_pr) and math.isfinite(raw_roc) and math.isfinite(raw_f1)):
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=len(models_evaluated),
                    num_representations=len(representations),
                    adr_007_notice=default_adr_notice,
                    error_message="Non-finite (NaN or Inf) metric value detected"
                )

            if not (0.0 <= raw_pr <= 1.0 and 0.0 <= raw_roc <= 1.0 and 0.0 <= raw_f1 <= 1.0):
                return EvaluationSummaryModel(
                    artifact_state="INVALID",
                    experiment_id=default_exp_id,
                    phase=default_phase,
                    top_model=None,
                    top_test_pr_auc=None,
                    top_test_roc_auc=None,
                    top_test_f1=None,
                    primary_policy="causal_history",
                    sensitivity_policy="train_only_context",
                    num_models_evaluated=len(models_evaluated),
                    num_representations=len(representations),
                    adr_007_notice=default_adr_notice,
                    error_message="Metric values outside valid probability interval [0.0, 1.0]"
                )

            return EvaluationSummaryModel(
                artifact_state="AVAILABLE",
                experiment_id=default_exp_id,
                phase=default_phase,
                top_model="Logistic Regression (max)",
                top_test_pr_auc=round(raw_pr, 4),
                top_test_roc_auc=round(raw_roc, 4),
                top_test_f1=round(raw_f1, 4),
                primary_policy="causal_history",
                sensitivity_policy="train_only_context",
                num_models_evaluated=len(models_evaluated),
                num_representations=len(representations),
                adr_007_notice=default_adr_notice,
                error_message=None
            )
    except Exception as e:
        return EvaluationSummaryModel(
            artifact_state="INVALID",
            experiment_id=default_exp_id,
            phase=default_phase,
            top_model=None,
            top_test_pr_auc=None,
            top_test_roc_auc=None,
            top_test_f1=None,
            primary_policy="causal_history",
            sensitivity_policy="train_only_context",
            num_models_evaluated=0,
            num_representations=0,
            adr_007_notice=default_adr_notice,
            error_message=f"Failed to read evaluation artifact: {e}"
        )

def _load_benchmark_summary(csv_path: Optional[str] = None) -> BenchmarkSummaryModel:
    """Read and strictly validate genuine benchmark summary from Phase 8 smoke session artifact."""
    target_path = csv_path or CANONICAL_BENCHMARK_PATH
    default_phase = "Phase 8"
    default_session_id = "session_20261024_230008_smoke"

    if not os.path.exists(target_path):
        return BenchmarkSummaryModel(
            artifact_state="MISSING",
            session_id=default_session_id,
            phase=default_phase,
            total_runs=0,
            measured_cpu_runs=0,
            skipped_gpu_runs=0,
            top_speedup_vs_cpu_ref=None,
            top_speedup_workload=None,
            local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
            high_variability_label="N/A",
            error_message=f"Benchmark artifact file not found: {os.path.basename(target_path)}"
        )

    try:
        if os.path.getsize(target_path) == 0:
            return BenchmarkSummaryModel(
                artifact_state="INVALID",
                session_id=default_session_id,
                phase=default_phase,
                total_runs=0,
                measured_cpu_runs=0,
                skipped_gpu_runs=0,
                top_speedup_vs_cpu_ref=None,
                top_speedup_workload=None,
                local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                high_variability_label="N/A",
                error_message="Benchmark artifact file is empty"
            )

        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if not reader or not reader[0]:
                return BenchmarkSummaryModel(
                    artifact_state="INVALID",
                    session_id=default_session_id,
                    phase=default_phase,
                    total_runs=0,
                    measured_cpu_runs=0,
                    skipped_gpu_runs=0,
                    top_speedup_vs_cpu_ref=None,
                    top_speedup_workload=None,
                    local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                    high_variability_label="N/A",
                    error_message="Benchmark artifact contains no valid rows or header"
                )

            fieldnames = set(reader[0].keys())
            missing_cols = REQUIRED_BENCH_COLUMNS - fieldnames
            if missing_cols:
                return BenchmarkSummaryModel(
                    artifact_state="INVALID",
                    session_id=default_session_id,
                    phase=default_phase,
                    total_runs=0,
                    measured_cpu_runs=0,
                    skipped_gpu_runs=0,
                    top_speedup_vs_cpu_ref=None,
                    top_speedup_workload=None,
                    local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                    high_variability_label="N/A",
                    error_message=f"Benchmark artifact missing required columns: {sorted(missing_cols)}"
                )

            total_runs = len(reader)
            measured = sum(1 for r in reader if r.get("measurement_state") == "MEASURED")
            skipped = sum(1 for r in reader if r.get("measurement_state") == "SKIPPED_CUDA_UNAVAILABLE")

            # Look for target speedup row: synthetic_tier_a_10k_mean + cpu_vectorized
            target_row = None
            for r in reader:
                if (
                    r.get("workload_name") == "synthetic_tier_a_10k_mean"
                    and r.get("engine") == "cpu_vectorized"
                ):
                    target_row = r
                    break

            if target_row is None:
                return BenchmarkSummaryModel(
                    artifact_state="INVALID",
                    session_id=default_session_id,
                    phase=default_phase,
                    total_runs=total_runs,
                    measured_cpu_runs=measured,
                    skipped_gpu_runs=skipped,
                    top_speedup_vs_cpu_ref=None,
                    top_speedup_workload=None,
                    local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                    high_variability_label="N/A",
                    error_message="Target benchmark workload synthetic_tier_a_10k_mean (cpu_vectorized) not found"
                )

            raw_speedup_str = target_row.get("speedup_vs_cpu_ref", "").rstrip("x")
            try:
                top_speedup = float(raw_speedup_str)
            except (ValueError, TypeError) as e:
                return BenchmarkSummaryModel(
                    artifact_state="INVALID",
                    session_id=default_session_id,
                    phase=default_phase,
                    total_runs=total_runs,
                    measured_cpu_runs=measured,
                    skipped_gpu_runs=skipped,
                    top_speedup_vs_cpu_ref=None,
                    top_speedup_workload=None,
                    local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                    high_variability_label="N/A",
                    error_message=f"Non-numeric speedup value in target row: {e}"
                )

            if not math.isfinite(top_speedup) or top_speedup <= 0.0:
                return BenchmarkSummaryModel(
                    artifact_state="INVALID",
                    session_id=default_session_id,
                    phase=default_phase,
                    total_runs=total_runs,
                    measured_cpu_runs=measured,
                    skipped_gpu_runs=skipped,
                    top_speedup_vs_cpu_ref=None,
                    top_speedup_workload=None,
                    local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                    high_variability_label="N/A",
                    error_message="Non-finite or non-positive speedup value detected"
                )

            high_var_rows = [r for r in reader if r.get("timing_stability") == "HIGH_VARIABILITY"]
            if high_var_rows:
                selected_var_row = high_var_rows[0]
                raw_op = selected_var_row.get("operator", "").strip()
                raw_engine = selected_var_row.get("engine", "").strip()
                raw_cv = selected_var_row.get("cv_percent", "").rstrip("%").strip()

                op_valid = bool(raw_op)
                cv_valid = False
                if raw_cv:
                    try:
                        cv_num = float(raw_cv)
                        if math.isfinite(cv_num) and cv_num >= 0.0:
                            cv_valid = True
                    except (ValueError, TypeError):
                        pass

                if not op_valid and not cv_valid:
                    high_var_label = "High variability documented (operator and CV unavailable)"
                elif not op_valid:
                    high_var_label = f"High variability documented (operator unavailable, CV {raw_cv}%)"
                elif not cv_valid:
                    engine_desc = raw_engine.replace("cpu_", "").strip()
                    desc = f"{raw_op.upper()} {engine_desc}".strip()
                    high_var_label = f"{desc} high variability documented (CV unavailable)"
                else:
                    engine_desc = raw_engine.replace("cpu_", "").strip()
                    desc = f"{raw_op.upper()} {engine_desc}".strip()
                    high_var_label = f"{desc} high variability documented (CV {raw_cv}%)"
            else:
                high_var_label = "All measured workloads STABLE"

            return BenchmarkSummaryModel(
                artifact_state="AVAILABLE",
                session_id=default_session_id,
                phase=default_phase,
                total_runs=total_runs,
                measured_cpu_runs=measured,
                skipped_gpu_runs=skipped,
                top_speedup_vs_cpu_ref=round(top_speedup, 2),
                top_speedup_workload=f"{target_row.get('workload_name', 'synthetic_tier_a_10k_mean')} ({target_row.get('engine', 'cpu_vectorized')})",
                local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
                high_variability_label=high_var_label,
                error_message=None
            )
    except Exception as e:
        return BenchmarkSummaryModel(
            artifact_state="INVALID",
            session_id=default_session_id,
            phase=default_phase,
            total_runs=0,
            measured_cpu_runs=0,
            skipped_gpu_runs=0,
            top_speedup_vs_cpu_ref=None,
            top_speedup_workload=None,
            local_gpu_status="SKIPPED_CUDA_UNAVAILABLE",
            high_variability_label="N/A",
            error_message=f"Failed to read benchmark artifact: {e}"
        )

@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    eval_csv_path: Optional[str] = None,
    bench_csv_path: Optional[str] = None,
):
    """Return consolidated platform metrics derived strictly from repository artifacts."""
    if eval_csv_path and not _is_safe_path(eval_csv_path):
        raise HTTPException(status_code=400, detail="eval_csv_path escapes authorized workspace boundaries")
    if bench_csv_path and not _is_safe_path(bench_csv_path):
        raise HTTPException(status_code=400, detail="bench_csv_path escapes authorized workspace boundaries")

    gpu_info = get_gpu_diagnostics()
    eval_summary = _load_evaluation_summary(eval_csv_path)
    bench_summary = _load_benchmark_summary(bench_csv_path)

    # Truthful overall status: operational only if all underlying artifacts are AVAILABLE
    if eval_summary.artifact_state == "AVAILABLE" and bench_summary.artifact_state == "AVAILABLE":
        status = "operational"
    else:
        status = "degraded"

    dataset_summary = DatasetSummaryModel(
        name="synthetic_fraud_10k",
        num_nodes=10000,
        num_edges=231984,
        feature_dim=32,
        fraud_count=350,
        licit_count=9650,
        fraud_rate_pct=3.5,
        is_synthetic=True,
        provenance="SYNTHETIC",
        relational_keys=["card_id", "device_id"],
        timespan_days=30.0,
        sha256="aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"
    )

    hardware_summary = HardwareSummaryModel(
        local_gpu_detected=gpu_info.get("name"),
        local_compiler_status=gpu_info.get("cuda_compiler_status", "UNAVAILABLE_LOCAL_COLAB_READY"),
        local_vram_free_mb=gpu_info.get("vram_free_mb"),
        canonical_colab_gpu="NVIDIA Tesla T4 (Phase 6 Archive)",
        colab_kernel_latency_ms=0.0679
    )

    return DashboardSummaryResponse(
        status=status,
        project="FraudGraph Intelligence",
        dataset=dataset_summary,
        evaluation=eval_summary,
        benchmarks=bench_summary,
        hardware=hardware_summary,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
