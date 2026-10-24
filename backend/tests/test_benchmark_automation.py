"""
Automated Test Suite for Benchmark Automation & Experiment Management — FraudGraph Intelligence (FGI)
Phase 8 Deliverable (FGI-MOD-08)

Tests:
1. Workload matrix and profile run count accounting (workloads vs runs vs repetitions).
2. Operator-specific numerical validation (mean, max, sum) and tolerance bounds.
3. Empty neighborhood zero-sentinel verification on isolated nodes.
4. Non-finite (NaN / Inf) and shape mismatch rejection.
5. Host-RAM and GPU-VRAM preflight memory guards.
6. Statistical latency metrics (mean, std ddof=1, p50, p95, min, max, CV%).
7. Atomic JSON serialization and session directory structure.
8. Experiment session manifest, provenance tracking, and partial recovery.
9. Colab ingestion schema compatibility for measured runs.
10. Transparent skipping of CUDA runs on hosts without nvcc / precompiled binaries.
11. Immutability of graph fixtures and reference aggregators.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import numpy as np
import pytest

from backend.benchmark_engine.benchmark_matrix import (
    ALL_ENGINES,
    CPU_ENGINES,
    CUDA_ENGINES,
    BenchmarkSuiteProfile,
    ExecutionStatus,
    ImplementationRunConfig,
    WorkloadConfig,
    build_profile,
    get_profile_counts,
)
from backend.benchmark_engine.benchmark_verifier import (
    OPERATOR_TOLERANCES,
    NumericalVerificationResult,
    get_operator_tolerance,
    verify_aggregation_correctness,
)
from backend.benchmark_engine.benchmark_telemetry import (
    check_gpu_vram,
    check_host_ram,
    compute_run_statistics,
    compute_throughput_and_bandwidth,
    estimate_algorithmic_traffic_bytes,
    estimate_compulsory_traffic_bytes,
    estimate_workload_memory_mb,
    get_hardware_telemetry,
    get_host_ram_info,
)
from backend.benchmark_engine.experiment_manager import (
    BenchmarkResultRecord,
    ExperimentSession,
    SessionManifest,
    atomic_write_json,
    compute_file_sha256,
)
from backend.benchmark_engine.benchmark_runner import BenchmarkRunner
from backend.benchmark_engine.benchmark_report import BenchmarkReportGenerator
from backend.cpu_engine.aggregators import aggregate_neighbors
from backend.cpu_engine.csr_graph import CSRGraph


# ==============================================================================
# 1. Profile Matrix & Count Accounting Tests
# ==============================================================================

def test_profile_workload_and_run_counts():
    """Validates rigorous separation of workloads, planned runs, and repetitions."""
    smoke = build_profile("smoke", warmup_iters=5, repeat_iters=20)

    # Smoke has 6 workloads: 3 on toy (mean, sum, max) + 3 on tier_a (mean, sum, max)
    assert len(smoke.workloads) == 6
    assert smoke.implementations == ALL_ENGINES

    counts = smoke.get_counts()
    assert counts["workload_count"] == 6
    assert counts["cpu_engine_count"] == 2
    assert counts["gpu_engine_count"] == 4
    assert counts["planned_cpu_runs"] == 12  # 6 workloads * 2 CPU engines
    assert counts["planned_gpu_runs"] == 24  # 6 workloads * 4 GPU engines
    assert counts["total_planned_runs"] == 36
    assert counts["total_cpu_repetitions"] == 240  # 12 * 20
    assert counts["total_gpu_repetitions"] == 480  # 24 * 20
    assert counts["total_repetitions"] == 720

    # Verify actual fixture dimensions in smoke profile
    toy_w = smoke.workloads[0]
    assert toy_w.num_nodes == 5
    assert toy_w.num_edges == 8
    assert toy_w.feature_dim == 4

    tier_a_w = smoke.workloads[3]
    assert tier_a_w.num_nodes == 10000
    assert tier_a_w.num_edges == 94945
    assert tier_a_w.feature_dim == 32

    # Test scaling_features profile counts
    scaling_feat = build_profile("scaling_features", warmup_iters=3, repeat_iters=10)
    feat_counts = scaling_feat.get_counts()
    assert feat_counts["workload_count"] == 10  # 5 dims * 2 ops
    assert feat_counts["total_planned_runs"] == 60  # 10 * 6

    # Test scaling_graphs profile counts
    scaling_graphs = build_profile("scaling_graphs", warmup_iters=3, repeat_iters=10)
    graph_counts = scaling_graphs.get_counts()
    assert graph_counts["workload_count"] == 6  # 3 tiers * 2 ops
    assert graph_counts["total_planned_runs"] == 36


# ==============================================================================
# 2. Operator Tolerance & Numerical Verification Tests
# ==============================================================================

def test_operator_tolerance_specification():
    """Verifies operator tolerances adhere to mathematical accumulator properties."""
    mean_tol = get_operator_tolerance("mean")
    assert mean_tol.atol == 1.0e-5
    assert mean_tol.rtol == 1.0e-5

    max_tol = get_operator_tolerance("max")
    assert max_tol.atol == 1.0e-5
    assert max_tol.rtol == 0.0

    sum_tol = get_operator_tolerance("sum")
    assert sum_tol.atol == 1.0e-4
    assert sum_tol.rtol == 1.0e-4


def test_numerical_verification_tolerances():
    """Tests numerical verification passes within bounds and fails beyond bounds."""
    # Build simple synthetic graph
    row_ptr = np.array([0, 2, 3], dtype=np.int64)
    col_idx = np.array([0, 1, 0], dtype=np.int64)
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    graph = CSRGraph(row_ptr=row_ptr, col_idx=col_idx, features=features)

    ref = aggregate_neighbors(graph, op="mean", method="reference")

    # 1. Exact match passes
    res_exact = verify_aggregation_correctness(ref, ref, graph, operator="mean")
    assert res_exact.status == ExecutionStatus.MEASURED
    assert res_exact.max_absolute_error == 0.0

    # 2. Small numerical perturbation within tolerance passes
    test_ok = ref + 5.0e-6
    res_ok = verify_aggregation_correctness(test_ok, ref, graph, operator="mean")
    assert res_ok.status == ExecutionStatus.MEASURED

    # 3. Deviation exceeding tolerance fails with FAILED_TOLERANCE
    test_bad = ref + 5.0e-4
    res_bad = verify_aggregation_correctness(test_bad, ref, graph, operator="mean")
    assert res_bad.status == ExecutionStatus.FAILED_TOLERANCE
    assert res_bad.max_absolute_error > 1.0e-5


def test_empty_neighborhood_zero_sentinel_verification():
    """Tests isolated nodes with degree 0 strictly produce zero vectors."""
    # Node 1 has degree 0
    row_ptr = np.array([0, 2, 2, 3], dtype=np.int64)
    col_idx = np.array([0, 2, 0], dtype=np.int64)
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    graph = CSRGraph(row_ptr=row_ptr, col_idx=col_idx, features=features)

    ref = aggregate_neighbors(graph, op="mean", method="reference")
    assert np.all(ref[1] == 0.0)

    # Valid candidate output
    res_ok = verify_aggregation_correctness(ref, ref, graph, operator="mean")
    assert res_ok.status == ExecutionStatus.MEASURED
    assert res_ok.empty_neighborhood_verified is True

    # Bad candidate where isolated node has non-zero values
    bad_cand = ref.copy()
    bad_cand[1, 0] = 0.001
    res_fail = verify_aggregation_correctness(bad_cand, ref, graph, operator="mean")
    assert res_fail.status == ExecutionStatus.FAILED_TOLERANCE
    assert "Zero-degree nodes failed" in res_fail.details


def test_non_finite_rejection():
    """Tests that NaNs and Infs are rejected immediately."""
    row_ptr = np.array([0, 1], dtype=np.int64)
    col_idx = np.array([0], dtype=np.int64)
    features = np.array([[1.0]], dtype=np.float32)
    graph = CSRGraph(row_ptr=row_ptr, col_idx=col_idx, features=features)
    ref = aggregate_neighbors(graph, op="mean", method="reference")

    # NaN test
    nan_cand = np.array([[np.nan]], dtype=np.float32)
    res_nan = verify_aggregation_correctness(nan_cand, ref, graph, operator="mean")
    assert res_nan.status == ExecutionStatus.FAILED_NON_FINITE
    assert res_nan.nan_detected is True

    # Inf test
    inf_cand = np.array([[np.inf]], dtype=np.float32)
    res_inf = verify_aggregation_correctness(inf_cand, ref, graph, operator="mean")
    assert res_inf.status == ExecutionStatus.FAILED_NON_FINITE
    assert res_inf.inf_detected is True


def test_shape_mismatch_rejection():
    """Tests that mismatched candidate tensor shapes are rejected."""
    row_ptr = np.array([0, 1], dtype=np.int64)
    col_idx = np.array([0], dtype=np.int64)
    features = np.array([[1.0, 2.0]], dtype=np.float32)
    graph = CSRGraph(row_ptr=row_ptr, col_idx=col_idx, features=features)
    ref = aggregate_neighbors(graph, op="mean", method="reference")

    wrong_shape = np.zeros((1, 5), dtype=np.float32)
    res = verify_aggregation_correctness(wrong_shape, ref, graph, operator="mean")
    assert res.status == ExecutionStatus.FAILED_SHAPE_MISMATCH
    assert res.shape_match is False


# ==============================================================================
# 3. Telemetry, Memory Preflight & Statistical Metrics Tests
# ==============================================================================

def test_statistical_metric_calculations():
    """Validates mean, std (ddof=1), median, p95, min, max, cv%, and stability flags."""
    # 5 samples: [10.0, 12.0, 14.0, 16.0, 18.0]
    latencies = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = compute_run_statistics(latencies)

    assert stats["sample_count"] == 5
    assert stats["mean"] == 14.0
    # Sample std of [10, 12, 14, 16, 18] with ddof=1 is sqrt(40/4) = sqrt(10) ~ 3.1623
    assert abs(stats["std"] - 3.1623) < 0.001
    assert stats["median_p50"] == 14.0
    assert stats["min"] == 10.0
    assert stats["max"] == 18.0
    assert stats["cv_percent"] > 0.0
    assert stats["is_stable"] is True
    assert stats["stability_label"] == "STABLE"

    # High variability series with outlier
    unstable_series = [1.0, 1.1, 1.2, 1.0, 30.0]
    unstable_stats = compute_run_statistics(unstable_series)
    assert unstable_stats["is_stable"] is False
    assert unstable_stats["stability_label"] == "HIGH_VARIABILITY"

    # Single sample: std must be 0.0
    stats_single = compute_run_statistics([42.0])
    assert stats_single["mean"] == 42.0
    assert stats_single["std"] == 0.0

    # Empty sample list: handles safely without exception
    stats_empty = compute_run_statistics([])
    assert stats_empty["mean"] == 0.0
    assert stats_empty["std"] == 0.0
    assert stats_empty["sample_count"] == 0


def test_throughput_and_bandwidth_calculations():
    """Validates nodes/sec, edges/sec, compulsory and algorithmic traffic, and model-derived bandwidth."""
    # Test byte calculation functions directly
    comp_bytes = estimate_compulsory_traffic_bytes(num_nodes=10, num_edges=20, feature_dim=4, index_bytes=8)
    # (10 + 1 + 20) * 8 + 8 * 10 * 4 = 31 * 8 + 320 = 248 + 320 = 568
    assert comp_bytes == 568

    algo_bytes = estimate_algorithmic_traffic_bytes(num_nodes=10, num_edges=20, feature_dim=4, index_bytes=8)
    # (10 + 1 + 20) * 8 + 4 * 20 * 4 + 4 * 10 * 4 = 248 + 320 + 160 = 728
    assert algo_bytes == 728

    tp_bw = compute_throughput_and_bandwidth(
        num_nodes=10000,
        num_edges=100000,
        feature_dim=64,
        mean_latency_ms=10.0,  # 0.01 seconds
        index_bytes=8,
    )
    assert tp_bw["nodes_per_sec"] == 10000 / 0.01  # 1,000,000
    assert tp_bw["edges_per_sec"] == 100000 / 0.01  # 10,000,000
    assert "compulsory_traffic_bytes" in tp_bw
    assert "algorithmic_traffic_bytes" in tp_bw
    assert "compulsory_bandwidth_gb_s" in tp_bw
    assert "algorithmic_bandwidth_gb_s" in tp_bw
    assert tp_bw["effective_bandwidth_gb_s"] == tp_bw["algorithmic_bandwidth_gb_s"]
    assert "ESTIMATED_ALGORITHMIC_MODEL" in tp_bw["bandwidth_nature"]

    # Zero latency handles safely
    zero_res = compute_throughput_and_bandwidth(10, 10, 10, 0.0)
    assert zero_res["nodes_per_sec"] == 0.0


def test_memory_preflight_guards():
    """Tests host-RAM and GPU-VRAM preflight checks across valid, unavailable, and invalid telemetry."""
    est = estimate_workload_memory_mb(num_nodes=10000, num_edges=100000, feature_dim=64)
    assert est["total_estimated_mb"] > 0.0

    # 1. Normal workload should pass on standard host RAM with valid telemetry
    res_host_ok = check_host_ram(num_nodes=1000, num_edges=5000, feature_dim=16)
    assert res_host_ok.allowed is True
    assert res_host_ok.guard_status in ("VERIFIED_PASS", "PERMITTED_BOUNDED_FALLBACK")

    # 2. Gigantic hypothetical workload exceeding any normal RAM must be blocked
    res_host_oom = check_host_ram(num_nodes=10**9, num_edges=10**10, feature_dim=1024)
    assert res_host_oom.allowed is False
    assert res_host_oom.guard_status in ("BLOCKED_CAPACITY_EXCEEDED", "BLOCKED_TELEMETRY_UNAVAILABLE")
    assert "exceeds" in (res_host_oom.reason or "") or "fail-closed" in (res_host_oom.reason or "")

    # 3. Explicit valid telemetry with insufficient physical memory (capacity exceeded)
    mock_ram_tight = {"available_mb": 5.0, "total_mb": 16000.0}
    res_tight = check_host_ram(
        num_nodes=50000,
        num_edges=500000,
        feature_dim=64,
        safety_margin=0.80,
        ram_info=mock_ram_tight,
    )
    assert res_tight.allowed is False
    assert res_tight.guard_status == "BLOCKED_CAPACITY_EXCEEDED"
    assert "exceeds" in (res_tight.reason or "")

    # 4. Host RAM telemetry completely unavailable (0.0 MB available)
    mock_ram_unavail = {"available_mb": 0.0, "total_mb": 0.0}

    # 4a. Large workload (> 100 MB) must fail closed under fail-closed safety policy
    res_unavail_large = check_host_ram(
        num_nodes=1000000,
        num_edges=5000000,
        feature_dim=64,
        allow_bounded_fallback=True,
        max_unverified_mb=100.0,
        ram_info=mock_ram_unavail,
    )
    assert res_unavail_large.allowed is False
    assert res_unavail_large.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"
    assert "fail-closed" in (res_unavail_large.reason or "")

    # 4b. Small workload (<= 100 MB) with bounded fallback enabled is permitted with explicit caution
    res_unavail_small_fallback = check_host_ram(
        num_nodes=1000,
        num_edges=5000,
        feature_dim=16,
        allow_bounded_fallback=True,
        max_unverified_mb=100.0,
        ram_info=mock_ram_unavail,
    )
    assert res_unavail_small_fallback.allowed is True
    assert res_unavail_small_fallback.guard_status == "PERMITTED_BOUNDED_FALLBACK"
    assert "Caution" in (res_unavail_small_fallback.reason or "")

    # 4c. Small workload with bounded fallback DISABLED must fail closed
    res_unavail_small_nofallback = check_host_ram(
        num_nodes=1000,
        num_edges=5000,
        feature_dim=16,
        allow_bounded_fallback=False,
        max_unverified_mb=100.0,
        ram_info=mock_ram_unavail,
    )
    assert res_unavail_small_nofallback.allowed is False
    assert res_unavail_small_nofallback.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"

    # 5. Invalid telemetry values (negative and NaN) must fail closed
    mock_ram_neg = {"available_mb": -128.0, "total_mb": -1024.0}
    res_neg = check_host_ram(
        num_nodes=1000000,
        num_edges=5000000,
        feature_dim=64,
        ram_info=mock_ram_neg,
    )
    assert res_neg.allowed is False
    assert res_neg.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"

    mock_ram_nan = {"available_mb": float("nan"), "total_mb": float("nan")}
    res_nan = check_host_ram(
        num_nodes=1000000,
        num_edges=5000000,
        feature_dim=64,
        ram_info=mock_ram_nan,
    )
    assert res_nan.allowed is False
    assert res_nan.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"

    # 6. GPU VRAM Preflight checks
    # 6a. GPU unavailable
    gpu_none = {"gpu_available": False}
    res_gpu_none = check_gpu_vram(1000, 5000, 16, gpu_info=gpu_none)
    assert res_gpu_none.allowed is False
    assert res_gpu_none.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"

    # 6b. GPU available but VRAM is 0 or negative
    gpu_zero = {"gpu_available": True, "free_vram_mb": 0.0, "total_vram_mb": 4096.0}
    res_gpu_zero = check_gpu_vram(1000, 5000, 16, gpu_info=gpu_zero)
    assert res_gpu_zero.allowed is False
    assert res_gpu_zero.guard_status == "BLOCKED_TELEMETRY_UNAVAILABLE"

    # 6c. GPU capacity exceeded
    gpu_tight = {"gpu_available": True, "free_vram_mb": 5.0, "total_vram_mb": 4096.0}
    res_gpu_tight = check_gpu_vram(50000, 500000, 64, safety_margin=0.85, gpu_info=gpu_tight)
    assert res_gpu_tight.allowed is False
    assert res_gpu_tight.guard_status == "BLOCKED_CAPACITY_EXCEEDED"

    # 6d. GPU sufficient VRAM
    gpu_ok = {"gpu_available": True, "free_vram_mb": 8000.0, "total_vram_mb": 8192.0}
    res_gpu_ok = check_gpu_vram(1000, 5000, 16, safety_margin=0.85, gpu_info=gpu_ok)
    assert res_gpu_ok.allowed is True
    assert res_gpu_ok.guard_status == "VERIFIED_PASS"


# ==============================================================================
# 4. Atomic Serialization & Session Manifest Tests
# ==============================================================================

def test_atomic_json_write_and_session_manifest():
    """Validates atomic writes and session manifest structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_out.json"
        data = {"hello": "world", "number": 123}

        atomic_write_json(test_file, data)
        assert test_file.exists()

        with open(test_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

        # Test session management
        session = ExperimentSession(
            profile_name="test_profile",
            session_id="test_sess_001",
            workspace_root=tmp_path,
            base_output_dir=tmp_path / "benchmark_runs",
        )

        assert session.session_dir.exists()
        assert session.raw_dir.exists()
        manifest_file = session.session_dir / "session_manifest.json"
        assert manifest_file.exists()

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)
        assert manifest_json["session_id"] == "test_sess_001"
        assert manifest_json["session_status"] == "IN_PROGRESS"

        session.finalize_session("COMPLETED")
        with open(manifest_file, "r", encoding="utf-8") as f:
            final_manifest = json.load(f)
        assert final_manifest["session_status"] == "COMPLETED"
        assert final_manifest["end_time"] is not None


# ==============================================================================
# 5. Colab Schema Compatibility & Transparent Skipping Tests
# ==============================================================================

def test_colab_schema_compatibility_and_reporting():
    """Validates that runner outputs comply with FGI experiment schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        session = ExperimentSession(
            profile_name="smoke",
            session_id="test_schema_sess",
            workspace_root=Path(__file__).resolve().parents[2],
            base_output_dir=tmp_path,
        )

        runner = BenchmarkRunner()
        toy_workload = WorkloadConfig(
            name="toy_graph_5nodes_mean",
            graph_fixture="data/fixtures/toy_graph_5nodes.npz",
            num_nodes=5,
            num_edges=8,
            feature_dim=4,
            operator="mean",
            topology="toy",
        )
        run_cfg = ImplementationRunConfig.for_engine("cpu_reference", warmup_iters=2, repeat_iters=3)
        res = runner.run_workload_implementation(toy_workload, run_cfg)

        assert res.record.measurement_state == ExecutionStatus.MEASURED.value
        json_path = session.save_run_result(toy_workload, res.record)
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            record_json = json.load(f)

        # Validate schema conformity
        required_keys = [
            "schema_version",
            "experiment_id",
            "session_id",
            "measurement_state",
            "hardware",
            "graph_configuration",
            "benchmark_parameters",
            "timing_results_ms",
            "numerical_verification",
            "throughput_and_bandwidth",
            "memory_footprint_mb",
            "memory_guard",
            "provenance",
        ]
        for k in required_keys:
            assert k in record_json, f"Missing required schema key '{k}'"

        assert record_json["measurement_state"] == "MEASURED"
        assert record_json["timing_results_ms"]["kernel_mean"] > 0.0
        assert record_json["numerical_verification"]["status"] == "MEASURED"
        assert "guard_status" in record_json["memory_guard"]

        # Test report generator on this session
        session.finalize_session("COMPLETED")
        reporter = BenchmarkReportGenerator(session)
        csv_path, md_path = reporter.generate_reports()
        assert csv_path.exists()
        assert md_path.exists()

        # Verify CSV has memory_guard_status
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
        assert "memory_guard_status" in header


def test_missing_cuda_transparent_skip():
    """Validates that CUDA runs are cleanly recorded as SKIPPED_CUDA_UNAVAILABLE on CPU hosts."""
    runner = BenchmarkRunner()
    toy_workload = WorkloadConfig(
        name="toy_graph_5nodes_mean",
        graph_fixture="data/fixtures/toy_graph_5nodes.npz",
        num_nodes=5,
        num_edges=8,
        feature_dim=4,
        operator="mean",
        topology="toy",
    )
    run_cfg = ImplementationRunConfig.for_engine("cuda_baseline", warmup_iters=2, repeat_iters=3)
    res = runner.run_workload_implementation(toy_workload, run_cfg)

    # On Windows without local nvcc binary, must be SKIPPED_CUDA_UNAVAILABLE
    assert res.record.measurement_state in (
        ExecutionStatus.SKIPPED_CUDA_UNAVAILABLE.value,
        ExecutionStatus.MEASURED.value,
    )
    if res.record.measurement_state == ExecutionStatus.SKIPPED_CUDA_UNAVAILABLE.value:
        assert res.record.trial_times_ms == []
        assert "CUDA" in (res.record.skip_or_failure_reason or "")


# ==============================================================================
# 6. Fixtures and Reference Immutability Tests
# ==============================================================================

def test_fixtures_immutability():
    """Guarantees that existing graph fixtures are never modified."""
    root = Path(__file__).resolve().parents[2]
    toy_fixture = root / "data" / "fixtures" / "toy_graph_5nodes.npz"
    tier_a_fixture = root / "data" / "fixtures" / "synthetic_tier_a_10k.npz"

    assert toy_fixture.exists()
    assert tier_a_fixture.exists()

    toy_hash = compute_file_sha256(toy_fixture)
    tier_a_hash = compute_file_sha256(tier_a_fixture)

    # Verify known SHA-256 hashes
    assert toy_hash == "3ea8f702bc1f944d9f164222bad637666307c7be8e4a9767f813e40edc137909"
    assert tier_a_hash == "584b18697688e006ab3cd0a3785783619658e5c03a409ec0f8a2e6705595844e"


def test_workload_preflight_guard_blocks_before_allocation(monkeypatch):
    """Verifies that an oversized workload is blocked before graph loading/allocation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        session = ExperimentSession(
            profile_name="oom_test",
            session_id="test_oom_preflight",
            workspace_root=tmp_path,
            base_output_dir=tmp_path / "benchmark_runs",
        )

        # Force check_host_ram to reject by setting safety_margin to 0.0 or a gigantic graph
        huge_workload = WorkloadConfig(
            name="gigantic_graph_workload",
            num_nodes=10**9,
            num_edges=10**10,
            feature_dim=1024,
            operator="mean",
            topology="synthetic",
        )
        profile = BenchmarkSuiteProfile(
            profile_name="oom_test",
            description="Profile testing preflight rejection before allocation",
            workloads=[huge_workload],
            implementations=["cpu_reference", "cpu_vectorized"],
            warmup_iters=1,
            repeat_iters=1,
        )

        runner = BenchmarkRunner(workspace_root=tmp_path)
        # Verify get_or_load_graph is NEVER called for this workload
        called_load = []
        def mock_get_or_load(w):
            called_load.append(w.name)
            raise RuntimeError("Should never be called for an OOM workload!")
        monkeypatch.setattr(runner, "get_or_load_graph", mock_get_or_load)

        # Run profile; should cleanly skip without error
        out_session = runner.run_profile(profile, session=session, verbose=False)
        assert len(called_load) == 0, "get_or_load_graph was called for an OOM workload!"
        assert len(out_session.results_index) == 2
        for r in out_session.results_index:
            assert r["measurement_state"] == ExecutionStatus.SKIPPED_OOM_GUARD.value


def test_workload_preflight_blocks_allocation_on_unavailable_telemetry(monkeypatch):
    """
    Verifies that when host RAM telemetry is unavailable, large workloads fail closed
    and are blocked before graph loading/allocation occurs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        session = ExperimentSession(
            profile_name="telemetry_unavail_test",
            session_id="test_unavail_preflight",
            workspace_root=tmp_path,
            base_output_dir=tmp_path / "benchmark_runs",
        )

        # Workload whose memory requirement exceeds max_unverified_mb (100 MB)
        # ~100k nodes, 1M edges, feature_dim=128 -> >150 MB
        large_workload = WorkloadConfig(
            name="large_unverified_workload",
            num_nodes=100000,
            num_edges=1000000,
            feature_dim=128,
            operator="mean",
            topology="synthetic",
        )
        profile = BenchmarkSuiteProfile(
            profile_name="unavail_test",
            description="Testing fail-closed allocation guard when telemetry is absent",
            workloads=[large_workload],
            implementations=["cpu_reference", "cpu_vectorized"],
            warmup_iters=1,
            repeat_iters=1,
        )

        # Mock get_host_ram_info to simulate system where RAM telemetry is unavailable
        from backend.benchmark_engine import benchmark_telemetry
        monkeypatch.setattr(benchmark_telemetry, "get_host_ram_info", lambda: {"available_mb": 0.0, "total_mb": 0.0})

        runner = BenchmarkRunner(workspace_root=tmp_path, max_unverified_mb=100.0)

        # Mock get_or_load_graph to assert that allocation is never attempted
        called_load = []
        def mock_get_or_load(w):
            called_load.append(w.name)
            raise RuntimeError("Allocation attempted despite unavailable telemetry!")
        monkeypatch.setattr(runner, "get_or_load_graph", mock_get_or_load)

        out_session = runner.run_profile(profile, session=session, verbose=False)
        assert len(called_load) == 0, "Graph allocation was attempted for blocked unverified workload!"
        assert len(out_session.results_index) == 2
        for r in out_session.results_index:
            assert r["measurement_state"] == ExecutionStatus.SKIPPED_OOM_GUARD.value
            assert r["memory_guard_status"] == "BLOCKED_TELEMETRY_UNAVAILABLE"
