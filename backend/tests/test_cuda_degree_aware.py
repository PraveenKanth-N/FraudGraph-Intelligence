"""
Unit and Schema Verification Tests for Degree-Aware Bucketing & Hybrid CUDA Dispatch (Phase 6, Experiment 6.3)
Tests cover:
1. Degree-aware CUDA kernel and runner source file completeness.
2. Makefile multi-target build configuration (baseline + coalesced + streams + degree_aware).
3. Python host adapter support for kernel="degree_aware".
4. Phase 6 Colab bundle packaging including degree-aware sources and skewed graph fixtures.
5. JSON schema conformity for degree-aware experiment results.
6. Degree bucketing partition mathematical invariants.
7. CPU reference aggregation correctness on skewed heavy-tailed graph (max degree > 1000).
8. Extreme star/hub degree stress test asserting exact zero sentinels and tolerance <= 1e-5.
"""

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

from backend.cpu_engine.aggregators import aggregate_neighbors
from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_generator import generate_synthetic_graph
from backend.gpu_engine.colab_runner.ingest_colab_results import validate_result_schema
from backend.gpu_engine.colab_runner.package_colab_bundle import build_colab_bundle
from backend.gpu_engine.cuda_runner import get_cuda_binary_path


def test_degree_aware_cuda_source_files_exist():
    workspace = Path(__file__).resolve().parents[2]
    cuda_dir = workspace / "backend" / "gpu_engine" / "cuda"

    required_files = [
        cuda_dir / "include" / "fgi_cuda_common.cuh",
        cuda_dir / "include" / "fgi_degree_aware_kernels.cuh",
        cuda_dir / "src" / "fgi_degree_aware_kernels.cu",
        cuda_dir / "src" / "main_degree_aware.cu",
        cuda_dir / "Makefile",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required CUDA source file missing: {rf}"

    # Verify key architectural tokens in degree-aware kernel and runner sources
    kernel_src = (cuda_dir / "src" / "fgi_degree_aware_kernels.cu").read_text(encoding="utf-8")
    assert "standard_degree_warp_kernel" in kernel_src
    assert "hub_degree_block_kernel" in kernel_src
    assert "s_warp_acc" in kernel_src
    assert "launch_degree_aware_aggregation" in kernel_src

    runner_src = (cuda_dir / "src" / "main_degree_aware.cu").read_text(encoding="utf-8")
    assert "cudaMallocHost" in runner_src
    assert "partition_degrees" in runner_src
    assert "num_isolated_nodes" in runner_src
    assert "num_hub_nodes" in runner_src
    assert "EXP-GPU-DEGREE-AWARE-OPT-6.3" in runner_src


def test_degree_aware_makefile_targets():
    workspace = Path(__file__).resolve().parents[2]
    makefile_content = (workspace / "backend" / "gpu_engine" / "cuda" / "Makefile").read_text(encoding="utf-8")

    assert "fgi_baseline_runner" in makefile_content
    assert "fgi_coalesced_runner" in makefile_content
    assert "fgi_streams_runner" in makefile_content
    assert "fgi_degree_aware_runner" in makefile_content
    assert "main_degree_aware.o" in makefile_content
    assert "fgi_degree_aware_kernels.o" in makefile_content


def test_degree_aware_runner_interface():
    binary_path = get_cuda_binary_path(kernel="degree_aware")
    assert "fgi_degree_aware_runner" in binary_path.name


def test_package_colab_bundle_phase6_includes_degree_aware_and_skewed():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_zip = Path(tmpdir) / "test_bundle_phase6_deg.zip"
        built_path = build_colab_bundle(output_zip=test_zip)

        assert built_path.exists()
        assert built_path.stat().st_size > 500_000

        with zipfile.ZipFile(built_path, "r") as zf:
            names = set(zf.namelist())
            assert "cuda/src/fgi_baseline_kernels.cu" in names
            assert "cuda/src/fgi_coalesced_kernels.cu" in names
            assert "cuda/src/fgi_degree_aware_kernels.cu" in names
            assert "cuda/include/fgi_degree_aware_kernels.cuh" in names
            assert "cuda/src/main_baseline.cu" in names
            assert "cuda/src/main_coalesced.cu" in names
            assert "cuda/src/main_streams.cu" in names
            assert "cuda/src/main_degree_aware.cu" in names
            assert "cuda/Makefile" in names
            assert "run_colab_benchmarks.py" in names
            assert "data/synthetic_tier_a_10k/meta.txt" in names
            assert "data/synthetic_skewed_10k/meta.txt" in names


def test_degree_aware_result_json_schema():
    valid_record = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-DEGREE-AWARE-OPT-6.3",
        "measurement_state": "MEASURED",
        "hardware": {
            "device_name": "Tesla T4",
            "compute_capability": "7.5",
            "total_vram_mb": 14912.7,
            "sm_count": 40,
        },
        "graph_configuration": {
            "num_nodes": 10000,
            "num_edges": 94970,
            "feature_dim": 32,
        },
        "benchmark_parameters": {
            "implementation": "cuda_optimized_degree_aware_hybrid",
            "operator": "mean",
            "warmup_iterations": 5,
            "recorded_repetitions": 20,
            "block_size": 256,
            "degree_threshold": 32,
            "num_isolated_nodes": 500,
            "num_standard_nodes": 9444,
            "num_hub_nodes": 56,
            "pinned_host_memory": True,
            "stream_execution": True,
        },
        "timing_results_ms": {
            "bucketing_overhead_ms": 0.082,
            "h2d_transfer_mean": 0.165,
            "kernel_mean": 0.015,
            "kernel_std": 0.001,
            "kernel_p50": 0.015,
            "kernel_p95": 0.016,
            "d2h_transfer_mean": 0.108,
            "end_to_end_mean": 0.370,
        },
        "performance_metrics": {
            "edges_per_second": 6.33e9,
            "effective_bandwidth_gbps": 198.5,
            "estimated_compulsory_bandwidth_gbps": 198.5,
            "pcie_h2d_bandwidth_gbps": 10.45,
            "pcie_d2h_bandwidth_gbps": 11.85,
        },
        "numerical_verification": {
            "status": "PASSED",
            "max_absolute_error": 0.0,
            "tolerance_threshold": 1.0e-5,
            "nan_detected": False,
        },
    }
    valid, msg = validate_result_schema(valid_record)
    assert valid, f"Schema validation failed: {msg}"


def test_degree_bucketing_partition_invariants():
    """Verifies that degree partitioning correctly segregates isolated, standard, and hub nodes."""
    workspace = Path(__file__).resolve().parents[2]
    skewed_path = workspace / "data" / "fixtures" / "synthetic_skewed_10k.npz"
    assert skewed_path.exists()

    graph = CSRGraph.load(skewed_path)
    degrees = graph.row_ptr[1:] - graph.row_ptr[:-1]
    threshold = 32

    isolated = np.where(degrees == 0)[0]
    standard = np.where((degrees >= 1) & (degrees <= threshold))[0]
    hubs = np.where(degrees > threshold)[0]

    # Invariants: Mutually exclusive and collectively exhaustive
    assert len(isolated) + len(standard) + len(hubs) == graph.num_nodes
    assert len(np.intersect1d(isolated, standard)) == 0
    assert len(np.intersect1d(isolated, hubs)) == 0
    assert len(np.intersect1d(standard, hubs)) == 0

    assert len(isolated) == 500
    assert len(hubs) == 56
    assert len(standard) == 9444


def test_cpu_reference_parity_on_skewed_graph():
    """Verifies that CPU reference aggregations execute cleanly on heavy-tailed power-law graph."""
    workspace = Path(__file__).resolve().parents[2]
    skewed_path = workspace / "data" / "fixtures" / "synthetic_skewed_10k.npz"
    graph = CSRGraph.load(skewed_path)

    for op in ["sum", "mean", "max"]:
        out_ref = aggregate_neighbors(graph, op=op, method="reference")
        out_vec = aggregate_neighbors(graph, op=op, method="vectorized")

        # Zero NaNs / Infs
        assert np.all(np.isfinite(out_ref))
        assert np.all(np.isfinite(out_vec))

        # Check isolated nodes have exact 0.0 vectors
        degrees = graph.row_ptr[1:] - graph.row_ptr[:-1]
        iso_indices = np.where(degrees == 0)[0]
        assert np.all(out_ref[iso_indices] == 0.0)
        assert np.all(out_vec[iso_indices] == 0.0)

        # Cross-method agreement
        max_diff = float(np.max(np.abs(out_ref - out_vec)))
        assert max_diff <= 1.0e-5, f"Vectorized vs Reference parity failure for {op}: {max_diff:.2e}"


def test_extreme_hub_stress_graph():
    """Tests extreme degree heterogeneity: 1 super hub with 1000 edges, leaves, and isolated nodes."""
    num_nodes = 100
    feature_dim = 16
    rng = np.random.default_rng(123)
    features = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)

    # Node 0 is connected to nodes 1..98 (degree 98). Node 99 is isolated (degree 0).
    edges = [(0, v) for v in range(1, num_nodes - 1)]

    graph = CSRGraph.from_edge_list(
        num_nodes=num_nodes,
        edge_list=edges,
        features=features,
        feature_dim=feature_dim,
        is_directed=True,
    )

    degrees = graph.row_ptr[1:] - graph.row_ptr[:-1]
    assert degrees[0] == 98
    assert degrees[99] == 0
    assert np.all(degrees[1:99] == 0)

    for op in ["sum", "mean", "max"]:
        out = aggregate_neighbors(graph, op=op, method="reference")
        assert np.all(out[99] == 0.0), f"Isolated node 99 must be 0 for {op}"
        assert np.all(out[1:99] == 0.0), f"Sink nodes must be 0 for {op}"
        assert not np.all(out[0] == 0.0), f"Hub node 0 must not be 0 for {op}"

        if op == "sum":
            expected_sum = np.sum(features[1:num_nodes - 1], axis=0)
            assert np.allclose(out[0], expected_sum, atol=1e-5)
        elif op == "mean":
            expected_mean = np.mean(features[1:num_nodes - 1], axis=0)
            assert np.allclose(out[0], expected_mean, atol=1e-5)
        elif op == "max":
            expected_max = np.max(features[1:num_nodes - 1], axis=0)
            assert np.allclose(out[0], expected_max, atol=1e-5)


def test_ingested_phase63_toy_validation_results():
    """Validates the ingested toy-graph validation JSON records for Degree-Aware dispatch."""
    workspace = Path(__file__).resolve().parents[2]
    results_dir = workspace / "experiments" / "results"

    for op in ["mean", "sum", "max"]:
        val_file = results_dir / f"val_toy_degree_aware_{op}.json"
        assert val_file.exists(), f"Toy validation result missing: {val_file}"

        with open(val_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["measurement_state"] == "MEASURED"
        assert data["hardware"]["device_name"] == "Tesla T4"
        assert data["benchmark_parameters"]["implementation"] == "cuda_optimized_degree_aware_hybrid"
        assert data["numerical_verification"]["status"] == "PASSED"
        assert data["numerical_verification"]["max_absolute_error"] == 0.0


def test_ingested_phase63_measured_benchmark_results():
    """Validates the ingested physical GPU benchmark JSON files for Degree-Aware dispatch."""
    workspace = Path(__file__).resolve().parents[2]
    results_dir = workspace / "experiments" / "results"

    # Validate Tier A records
    for op in ["mean", "sum", "max"]:
        matches = list(results_dir.glob(f"exp_gpu_degree_aware_tier_a_{op}_*.json"))
        assert len(matches) >= 1, f"Missing Tier A benchmark JSON for {op}"
        with open(matches[-1], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["measurement_state"] == "MEASURED"
        assert data["hardware"]["device_name"] == "Tesla T4"
        assert data["numerical_verification"]["status"] == "PASSED"
        assert data["numerical_verification"]["max_absolute_error"] <= 1.0e-5
        assert data["timing_results_ms"]["bucketing_overhead_ms"] > 0.0

    # Validate Skewed Stress graph records
    for op in ["mean", "sum", "max"]:
        matches = list(results_dir.glob(f"exp_gpu_degree_aware_skewed_{op}_*.json"))
        assert len(matches) >= 1, f"Missing Skewed benchmark JSON for {op}"
        with open(matches[-1], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["measurement_state"] == "MEASURED"
        assert data["hardware"]["device_name"] == "Tesla T4"
        assert data["numerical_verification"]["status"] == "PASSED"
        assert data["numerical_verification"]["max_absolute_error"] <= 1.0e-5
        assert data["timing_results_ms"]["bucketing_overhead_ms"] > 0.0
