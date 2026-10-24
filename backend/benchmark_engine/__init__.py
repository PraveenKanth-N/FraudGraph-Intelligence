"""
Benchmark Engine Package — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Provides unified, reproducible, matrix-driven benchmarking across CPU and GPU aggregation engines.
"""

from typing import Any

from backend.benchmark_engine.benchmark_matrix import (
    BenchmarkSuiteProfile,
    ExecutionStatus,
    ImplementationRunConfig,
    WorkloadConfig,
    build_profile,
    get_profile_counts,
)
from backend.benchmark_engine.benchmark_verifier import (
    OperatorTolerance,
    NumericalVerificationResult,
    verify_aggregation_correctness,
)
from backend.benchmark_engine.benchmark_telemetry import (
    MemoryPreflightResult,
    check_host_ram,
    check_gpu_vram,
    compute_run_statistics,
    compute_throughput_and_bandwidth,
    get_hardware_telemetry,
)
from backend.benchmark_engine.experiment_manager import (
    ExperimentSession,
    SessionManifest,
    BenchmarkResultRecord,
)
from backend.benchmark_engine.benchmark_report import (
    BenchmarkReportGenerator,
)


def __getattr__(name: str) -> Any:
    """Lazy module attribute loading to prevent runpy circular execution warnings."""
    if name in ("BenchmarkRunner", "RunExecutionResult"):
        from backend.benchmark_engine.benchmark_runner import (
            BenchmarkRunner,
            RunExecutionResult,
        )
        return BenchmarkRunner if name == "BenchmarkRunner" else RunExecutionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkSuiteProfile",
    "ExecutionStatus",
    "ImplementationRunConfig",
    "WorkloadConfig",
    "build_profile",
    "get_profile_counts",
    "OperatorTolerance",
    "NumericalVerificationResult",
    "verify_aggregation_correctness",
    "MemoryPreflightResult",
    "check_host_ram",
    "check_gpu_vram",
    "compute_run_statistics",
    "compute_throughput_and_bandwidth",
    "get_hardware_telemetry",
    "ExperimentSession",
    "SessionManifest",
    "BenchmarkResultRecord",
    "BenchmarkRunner",
    "RunExecutionResult",
    "BenchmarkReportGenerator",
]
