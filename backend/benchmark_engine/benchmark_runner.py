"""
Matrix Benchmark Runner & Execution Orchestrator — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Orchestrates reproducible benchmarking of graph aggregation operations:
1. Loads graph fixtures or dynamically generates graphs.
2. Generates and caches ground-truth reference outputs using the Phase 4 CPU reference.
3. Executes host-RAM and GPU-VRAM preflight memory checks before execution.
4. Distinguishes timing boundaries:
   - CPU engines: monotonic high-resolution host timer (perf_counter_ns).
   - GPU engines: device CUDA event timers (cudaEventElapsedTime).
5. On Windows without local nvcc/CUDA binaries, faithfully records CPU runs as MEASURED
   and GPU runs as SKIPPED_CUDA_UNAVAILABLE without fabricating zero or fake numbers.
6. Enforces operator-specific numerical validation before accepting measurements.
"""

import argparse
from dataclasses import dataclass
import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure repository root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.cpu_engine.aggregators import (
    AggregationOp,
    aggregate_neighbors,
)
from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_generator import generate_synthetic_graph
from backend.gpu_engine.cuda_runner import (
    get_cuda_binary_path,
    is_nvcc_available,
    execute_cuda_aggregation,
)
from backend.benchmark_engine.benchmark_matrix import (
    ALL_ENGINES,
    CPU_ENGINES,
    CUDA_ENGINES,
    TIMING_BOUNDARIES,
    BenchmarkSuiteProfile,
    ExecutionStatus,
    ImplementationRunConfig,
    WorkloadConfig,
    build_profile,
)
from backend.benchmark_engine.benchmark_telemetry import (
    check_gpu_vram,
    check_host_ram,
    compute_run_statistics,
    compute_throughput_and_bandwidth,
    estimate_workload_memory_mb,
)
from backend.benchmark_engine.benchmark_verifier import (
    NumericalVerificationResult,
    verify_aggregation_correctness,
)
from backend.benchmark_engine.experiment_manager import (
    BenchmarkResultRecord,
    ExperimentSession,
)


@dataclass
class RunExecutionResult:
    """Outcome of a single implementation execution."""
    record: BenchmarkResultRecord
    output_tensor: Optional[np.ndarray] = None


class BenchmarkRunner:
    """
    Unified matrix benchmark execution orchestrator.
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        host_ram_safety_margin: float = 0.80,
        gpu_vram_safety_margin: float = 0.85,
        allow_bounded_fallback: bool = True,
        max_unverified_mb: float = 100.0,
    ):
        self.workspace_root = workspace_root or WORKSPACE_ROOT
        self.host_ram_margin = host_ram_safety_margin
        self.gpu_vram_margin = gpu_vram_safety_margin
        self.allow_bounded_fallback = allow_bounded_fallback
        self.max_unverified_mb = max_unverified_mb
        self._graph_cache: Dict[str, CSRGraph] = {}

    def get_or_load_graph(self, workload: WorkloadConfig) -> CSRGraph:
        """Loads graph from fixture file or generates synthetic graph dynamically."""
        cache_key = workload.name
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        if workload.graph_fixture:
            fixture_path = self.workspace_root / workload.graph_fixture
            if not fixture_path.exists():
                raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
            graph = CSRGraph.load(fixture_path)
        else:
            # Dynamically generated graph
            params = workload.generator_params or {}
            avg_deg = params.get("avg_degree", 10)
            gamma = params.get("gamma", 2.5 if workload.topology == "power_law" else None)
            seed = workload.seed or 42
            graph = generate_synthetic_graph(
                num_nodes=workload.num_nodes,
                avg_degree=avg_deg,
                feature_dim=workload.feature_dim,
                is_directed=workload.is_directed,
                seed=seed,
                gamma=gamma,
            )

        self._graph_cache[cache_key] = graph
        return graph

    def run_workload_implementation(
        self,
        workload: WorkloadConfig,
        run_config: ImplementationRunConfig,
        ref_output: Optional[np.ndarray] = None,
        loaded_graph: Optional[CSRGraph] = None,
    ) -> RunExecutionResult:
        """
        Executes a single engine implementation on a workload configuration.
        """
        graph = loaded_graph if loaded_graph is not None else self.get_or_load_graph(workload)
        engine = run_config.engine
        boundary = run_config.timing_boundary
        op = workload.operator
        warmup = run_config.warmup_iters
        repeat = run_config.repeat_iters

        # Read actual physical graph dimensions to ensure fidelity
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        feature_dim = graph.feature_dim

        mem_est = estimate_workload_memory_mb(num_nodes, num_edges, feature_dim)

        # ---------------------------------------------------------
        # Case A: CPU Engines
        # ---------------------------------------------------------
        if engine in CPU_ENGINES:
            # 1. Preflight RAM check
            preflight = check_host_ram(
                num_nodes,
                num_edges,
                feature_dim,
                safety_margin=self.host_ram_margin,
                allow_bounded_fallback=self.allow_bounded_fallback,
                max_unverified_mb=self.max_unverified_mb,
            )
            if not preflight.allowed:
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.SKIPPED_OOM_GUARD.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=[],
                    latency_stats_ms=compute_run_statistics([]),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={"status": ExecutionStatus.SKIPPED_OOM_GUARD.value, "reason": preflight.reason},
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight.to_dict(),
                    skip_or_failure_reason=preflight.reason,
                )
                return RunExecutionResult(record=record, output_tensor=None)

            cpu_method = "reference" if engine == "cpu_reference" else "vectorized"

            # 2. Warm-up iterations
            try:
                for _ in range(warmup):
                    _ = aggregate_neighbors(graph, op=op, method=cpu_method)
            except Exception as e:
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.FAILED_EXECUTION.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=[],
                    latency_stats_ms=compute_run_statistics([]),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={"status": ExecutionStatus.FAILED_EXECUTION.value, "reason": str(e)},
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight.to_dict(),
                    skip_or_failure_reason=f"Warmup execution failed: {e}",
                )
                return RunExecutionResult(record=record, output_tensor=None)

            # 3. Measured repetitions
            trial_times_ms: List[float] = []
            test_output: Optional[np.ndarray] = None
            try:
                for _ in range(repeat):
                    t0 = time.perf_counter_ns()
                    test_output = aggregate_neighbors(graph, op=op, method=cpu_method)
                    t1 = time.perf_counter_ns()
                    trial_times_ms.append((t1 - t0) / 1_000_000.0)
            except Exception as e:
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.FAILED_EXECUTION.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=trial_times_ms,
                    latency_stats_ms=compute_run_statistics(trial_times_ms),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={"status": ExecutionStatus.FAILED_EXECUTION.value, "reason": str(e)},
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight.to_dict(),
                    skip_or_failure_reason=f"Repetition execution failed: {e}",
                )
                return RunExecutionResult(record=record, output_tensor=None)

            # 4. Numerical verification
            if ref_output is None:
                # If this was cpu_reference itself, it is the reference
                ref_output = test_output

            verif = verify_aggregation_correctness(
                test_output=test_output,
                ref_output=ref_output,
                graph=graph,
                operator=op,
            )

            state = verif.status.value
            stats = compute_run_statistics(trial_times_ms)
            tp_bw = compute_throughput_and_bandwidth(
                num_nodes,
                num_edges,
                feature_dim,
                stats["mean"],
            )

            record = BenchmarkResultRecord(
                workload_name=workload.name,
                engine=engine,
                timing_boundary=boundary,
                measurement_state=state,
                operator=op,
                num_nodes=num_nodes,
                num_edges=num_edges,
                feature_dim=feature_dim,
                warmup_iters=warmup,
                repeat_iters=repeat,
                trial_times_ms=[round(t, 4) for t in trial_times_ms],
                latency_stats_ms=stats,
                throughput_and_bandwidth=tp_bw,
                numerical_verification=verif.to_dict(),
                memory_footprint_mb=mem_est,
                memory_guard=preflight.to_dict(),
                skip_or_failure_reason=verif.details if state != ExecutionStatus.MEASURED.value else None,
            )
            return RunExecutionResult(record=record, output_tensor=test_output)

        # ---------------------------------------------------------
        # Case B: CUDA Engines
        # ---------------------------------------------------------
        elif engine in CUDA_ENGINES:
            kernel_name = engine.replace("cuda_", "")
            binary_path = get_cuda_binary_path(kernel=kernel_name)
            cuda_available = is_nvcc_available() or binary_path.exists()

            if not cuda_available or not binary_path.exists():
                reason = (
                    f"CUDA execution unavailable on current host. "
                    f"Missing nvcc compiler or precompiled binary '{binary_path.name}'. "
                    f"Dispatched benchmarks must be executed via the Google Colab workflow "
                    f"('backend/gpu_engine/colab_runner/')."
                )
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.SKIPPED_CUDA_UNAVAILABLE.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=[],
                    latency_stats_ms=compute_run_statistics([]),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={
                        "status": ExecutionStatus.SKIPPED_CUDA_UNAVAILABLE.value,
                        "details": reason,
                    },
                    memory_footprint_mb=mem_est,
                    memory_guard={
                        "allowed": False,
                        "guard_status": "SKIPPED_CUDA_UNAVAILABLE",
                        "estimated_mb": mem_est["total_estimated_mb"],
                        "available_mb": 0.0,
                        "total_mb": 0.0,
                        "safety_margin": self.gpu_vram_margin,
                        "reason": reason,
                    },
                    skip_or_failure_reason=reason,
                )
                return RunExecutionResult(record=record, output_tensor=None)

            # Preflight GPU VRAM check
            preflight_vram = check_gpu_vram(
                num_nodes,
                num_edges,
                feature_dim,
                safety_margin=self.gpu_vram_margin,
            )
            if not preflight_vram.allowed:
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.SKIPPED_OOM_GUARD.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=[],
                    latency_stats_ms=compute_run_statistics([]),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={"status": ExecutionStatus.SKIPPED_OOM_GUARD.value, "reason": preflight_vram.reason},
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight_vram.to_dict(),
                    skip_or_failure_reason=preflight_vram.reason,
                )
                return RunExecutionResult(record=record, output_tensor=None)

            # Native CUDA execution via cuda_runner
            try:
                gpu_out, gpu_meta = execute_cuda_aggregation(
                    graph=graph,
                    operator=op,
                    kernel=kernel_name,
                    warmup=warmup,
                    repeat=repeat,
                    block_size=run_config.block_size,
                )
                # Verify against ref_output
                if ref_output is None:
                    ref_output = aggregate_neighbors(graph, op=op, method="reference")

                verif = verify_aggregation_correctness(
                    test_output=gpu_out,
                    ref_output=ref_output,
                    graph=graph,
                    operator=op,
                )
                state = verif.status.value
                timing = gpu_meta.get("timing_results_ms", {})
                kernel_mean = timing.get("kernel_mean", 0.0)
                stats = compute_run_statistics(timing.get("trial_times_ms", [kernel_mean]))
                tp_bw = compute_throughput_and_bandwidth(
                    num_nodes,
                    num_edges,
                    feature_dim,
                    stats["mean"],
                )

                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=state,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=timing.get("trial_times_ms", []),
                    latency_stats_ms=stats,
                    throughput_and_bandwidth=tp_bw,
                    numerical_verification=verif.to_dict(),
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight_vram.to_dict(),
                    skip_or_failure_reason=verif.details if state != ExecutionStatus.MEASURED.value else None,
                )
                return RunExecutionResult(record=record, output_tensor=gpu_out)

            except Exception as e:
                record = BenchmarkResultRecord(
                    workload_name=workload.name,
                    engine=engine,
                    timing_boundary=boundary,
                    measurement_state=ExecutionStatus.FAILED_EXECUTION.value,
                    operator=op,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=feature_dim,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                    trial_times_ms=[],
                    latency_stats_ms=compute_run_statistics([]),
                    throughput_and_bandwidth=compute_throughput_and_bandwidth(num_nodes, num_edges, feature_dim, 0.0),
                    numerical_verification={"status": ExecutionStatus.FAILED_EXECUTION.value, "reason": str(e)},
                    memory_footprint_mb=mem_est,
                    memory_guard=preflight_vram.to_dict(),
                    skip_or_failure_reason=f"CUDA execution failed: {e}",
                )
                return RunExecutionResult(record=record, output_tensor=None)

        else:
            raise ValueError(f"Unknown engine: {engine}")

    def run_profile(
        self,
        profile: BenchmarkSuiteProfile,
        session: Optional[ExperimentSession] = None,
        verbose: bool = True,
    ) -> ExperimentSession:
        """
        Executes an entire benchmark profile matrix across workloads and implementations.
        """
        if session is None:
            session = ExperimentSession(
                profile_name=profile.profile_name,
                workspace_root=self.workspace_root,
            )

        counts = profile.get_counts()
        if verbose:
            print(f"\n=======================================================")
            print(f"FGI Phase 8 Benchmark Automation Suite")
            print(f"Profile: {profile.profile_name} ({profile.description})")
            print(f"Workload Configurations: {counts['workload_count']}")
            print(f"Planned Runs: {counts['total_planned_runs']} ({counts['planned_cpu_runs']} CPU, {counts['planned_gpu_runs']} GPU)")
            print(f"Total Repetitions: {counts['total_repetitions']} ({counts['total_cpu_repetitions']} CPU, {counts['total_gpu_repetitions']} GPU)")
            print(f"Session Output: {session.session_dir}")
            print(f"=======================================================\n")

        for w_idx, workload in enumerate(profile.workloads, 1):
            if verbose:
                print(f"[{w_idx}/{len(profile.workloads)}] Workload: {workload.name} "
                      f"(N={workload.num_nodes}, M={workload.num_edges}, D={workload.feature_dim}, op={workload.operator})")

            # 0. Preflight host-RAM check BEFORE expensive graph loading or dynamic generation
            preflight_host = check_host_ram(
                workload.num_nodes,
                workload.num_edges,
                workload.feature_dim,
                safety_margin=self.host_ram_margin,
                allow_bounded_fallback=self.allow_bounded_fallback,
                max_unverified_mb=self.max_unverified_mb,
            )
            if not preflight_host.allowed:
                mem_est = estimate_workload_memory_mb(workload.num_nodes, workload.num_edges, workload.feature_dim)
                if verbose:
                    print(f"  [!] WORKLOAD BLOCKED BY HOST RAM GUARD ({preflight_host.guard_status}): {preflight_host.reason}")
                for eng in profile.implementations:
                    run_cfg = ImplementationRunConfig.for_engine(
                        engine=eng,
                        warmup_iters=profile.warmup_iters,
                        repeat_iters=profile.repeat_iters,
                    )
                    rec = BenchmarkResultRecord(
                        workload_name=workload.name,
                        engine=eng,
                        timing_boundary=run_cfg.timing_boundary,
                        measurement_state=ExecutionStatus.SKIPPED_OOM_GUARD.value,
                        operator=workload.operator,
                        num_nodes=workload.num_nodes,
                        num_edges=workload.num_edges,
                        feature_dim=workload.feature_dim,
                        warmup_iters=profile.warmup_iters,
                        repeat_iters=profile.repeat_iters,
                        trial_times_ms=[],
                        latency_stats_ms=compute_run_statistics([]),
                        throughput_and_bandwidth=compute_throughput_and_bandwidth(workload.num_nodes, workload.num_edges, workload.feature_dim, 0.0),
                        numerical_verification={"status": ExecutionStatus.SKIPPED_OOM_GUARD.value, "reason": preflight_host.reason},
                        memory_footprint_mb=mem_est,
                        memory_guard=preflight_host.to_dict(),
                        skip_or_failure_reason=preflight_host.reason,
                    )
                    session.save_run_result(workload, rec)
                continue

            # 1. Compute ground-truth reference output using Phase 4 CPU reference
            graph = self.get_or_load_graph(workload)
            ref_output = aggregate_neighbors(graph, op=workload.operator, method="reference")

            # 2. Iterate through implementations
            for eng in profile.implementations:
                run_cfg = ImplementationRunConfig.for_engine(
                    engine=eng,
                    warmup_iters=profile.warmup_iters,
                    repeat_iters=profile.repeat_iters,
                )

                res = self.run_workload_implementation(
                    workload=workload,
                    run_config=run_cfg,
                    ref_output=ref_output,
                    loaded_graph=graph,
                )

                # Save run result to session
                session.save_run_result(workload, res.record)

                if verbose:
                    state = res.record.measurement_state
                    if state == ExecutionStatus.MEASURED.value:
                        mean_ms = res.record.latency_stats_ms["mean"]
                        std_ms = res.record.latency_stats_ms["std"]
                        nodes_s = res.record.throughput_and_bandwidth["nodes_per_sec"]
                        print(f"  -> {eng:22s} [{state:8s}] Latency: {mean_ms:8.3f} ms +/- {std_ms:.3f} ms | {nodes_s:11,.1f} nodes/s")
                    elif "SKIPPED" in state:
                        print(f"  -> {eng:22s} [{state}] (Bypassed)")
                    else:
                        print(f"  -> {eng:22s} [{state}] Reason: {res.record.skip_or_failure_reason}")

        session.finalize_session("COMPLETED")
        if verbose:
            print(f"\n[OK] Benchmark session complete. Artifacts written to:")
            print(f"     {session.session_dir}\n")

        return session


def main() -> None:
    """CLI entrypoint for benchmark automation harness."""
    parser = argparse.ArgumentParser(description="FGI Phase 8 Benchmark Automation Suite")
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "scaling_features", "scaling_graphs", "power_law_vs_uniform"],
        help="Benchmark profile matrix to execute (default: smoke)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per run (default: 5)")
    parser.add_argument("--repeat", type=int, default=20, help="Measured repetitions per run (default: 20)")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=None,
        help="Subset of aggregation engines to execute (default: all)",
    )

    args = parser.parse_args()

    profile = build_profile(
        profile_name=args.profile,
        warmup_iters=args.warmup,
        repeat_iters=args.repeat,
        implementations=args.implementations,
    )

    runner = BenchmarkRunner()
    session = runner.run_profile(profile)

    # Generate summary report and CSV
    from backend.benchmark_engine.benchmark_report import BenchmarkReportGenerator
    reporter = BenchmarkReportGenerator(session)
    csv_path, md_path = reporter.generate_reports()
    print(f"[OK] Summary table generated: {csv_path}")
    print(f"[OK] Summary report generated: {md_path}")


if __name__ == "__main__":
    main()
