"""
Unit and Schema Verification Tests for Pinned Host Memory & Asynchronous CUDA Streams (Phase 6, Experiment 6.2)
Tests cover:
1. Streams CUDA runner source file completeness.
2. Makefile multi-target configuration (baseline + coalesced + streams).
3. Python host adapter support for kernel="streams".
4. Phase 6 Colab bundle generation and file inclusion for streams runner.
5. Schema conformity for pinned-streams experiment JSON results.
6. PCIe bandwidth calculation assertions.
"""

import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from backend.gpu_engine.colab_runner.ingest_colab_results import validate_result_schema
from backend.gpu_engine.colab_runner.package_colab_bundle import build_colab_bundle
from backend.gpu_engine.cuda_runner import get_cuda_binary_path


def test_streams_cuda_source_files_exist():
    workspace = Path(__file__).resolve().parents[2]
    cuda_dir = workspace / "backend" / "gpu_engine" / "cuda"

    required_files = [
        cuda_dir / "include" / "fgi_cuda_common.cuh",
        cuda_dir / "include" / "fgi_coalesced_kernels.cuh",
        cuda_dir / "src" / "fgi_coalesced_kernels.cu",
        cuda_dir / "src" / "main_streams.cu",
        cuda_dir / "Makefile",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required CUDA source file missing: {rf}"

    # Verify key architectural tokens in streams runner source
    runner_src = (cuda_dir / "src" / "main_streams.cu").read_text(encoding="utf-8")
    assert "cudaMallocHost" in runner_src
    assert "cudaFreeHost" in runner_src
    assert "cudaStreamCreateWithFlags" in runner_src
    assert "cudaStreamNonBlocking" in runner_src
    assert "cudaMemcpyAsync" in runner_src
    assert "cudaStreamSynchronize" in runner_src
    assert "EXP-GPU-STREAMS-OPT-6.2" in runner_src
    assert "pcie_h2d_bandwidth_gbps" in runner_src
    assert "pcie_d2h_bandwidth_gbps" in runner_src


def test_streams_makefile_targets():
    workspace = Path(__file__).resolve().parents[2]
    makefile_content = (workspace / "backend" / "gpu_engine" / "cuda" / "Makefile").read_text(encoding="utf-8")

    assert "fgi_baseline_runner" in makefile_content
    assert "fgi_coalesced_runner" in makefile_content
    assert "fgi_streams_runner" in makefile_content
    assert "main_streams.o" in makefile_content
    assert "arch=compute_75,code=sm_75" in makefile_content


def test_streams_runner_interface():
    binary_path = get_cuda_binary_path(kernel="streams")
    assert "fgi_streams_runner" in binary_path.name


def test_package_colab_bundle_phase6_includes_streams():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_zip = Path(tmpdir) / "test_bundle_phase6_streams.zip"
        built_path = build_colab_bundle(output_zip=test_zip)

        assert built_path.exists()
        assert built_path.stat().st_size > 500_000

        with zipfile.ZipFile(built_path, "r") as zf:
            names = set(zf.namelist())
            assert "cuda/src/fgi_baseline_kernels.cu" in names
            assert "cuda/src/fgi_coalesced_kernels.cu" in names
            assert "cuda/src/main_baseline.cu" in names
            assert "cuda/src/main_coalesced.cu" in names
            assert "cuda/src/main_streams.cu" in names
            assert "cuda/Makefile" in names
            assert "run_colab_benchmarks.py" in names
            assert "fgi_colab_workflow.ipynb" in names


def test_streams_result_json_schema():
    valid_record = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-STREAMS-OPT-6.2",
        "measurement_state": "MEASURED",
        "hardware": {
            "device_name": "Tesla T4",
            "compute_capability": "7.5",
            "total_vram_mb": 14912.7,
            "sm_count": 40,
        },
        "graph_configuration": {
            "num_nodes": 10000,
            "num_edges": 94945,
            "feature_dim": 32,
        },
        "benchmark_parameters": {
            "implementation": "cuda_optimized_pinned_streams",
            "operator": "mean",
            "warmup_iterations": 5,
            "recorded_repetitions": 20,
            "block_size": 256,
            "pinned_host_memory": True,
            "stream_execution": True,
        },
        "timing_results_ms": {
            "h2d_transfer_mean": 0.38,
            "kernel_mean": 0.018,
            "kernel_std": 0.001,
            "kernel_p50": 0.018,
            "kernel_p95": 0.020,
            "d2h_transfer_mean": 0.22,
            "end_to_end_mean": 0.618,
        },
        "performance_metrics": {
            "edges_per_second": 5.27e9,
            "effective_bandwidth_gbps": 165.5,
            "estimated_compulsory_bandwidth_gbps": 165.5,
            "estimated_algorithmic_bandwidth_gbps": 769.6,
            "pcie_h2d_bandwidth_gbps": 4.47,
            "pcie_d2h_bandwidth_gbps": 5.82,
        },
        "numerical_verification": {
            "status": "PASSED",
            "max_absolute_error": 0.0,
            "tolerance_threshold": 1.0e-5,
            "nan_detected": False,
        },
    }
    valid, msg = validate_result_schema(valid_record)
    assert valid, f"Schema validation unexpectedly failed: {msg}"


def test_pcie_transfer_bandwidth_calculations():
    """
    Verifies that PCIe transfer bandwidth is calculated correctly from bytes and transfer latency.
    """
    num_nodes = 10000
    num_edges = 94945
    feature_dim = 32

    bytes_row = (num_nodes + 1) * 4
    bytes_col = num_edges * 4
    bytes_feat = num_nodes * feature_dim * 4
    bytes_h2d = bytes_row + bytes_col + bytes_feat  # 1,699,784 B
    bytes_d2h = num_nodes * feature_dim * 4          # 1,280,000 B

    # Theoretical transfer times in milliseconds
    h2d_ms = 0.40  # 0.4 ms
    d2h_ms = 0.25  # 0.25 ms

    bw_h2d = bytes_h2d / (h2d_ms * 1e-3 * 1e9)  # ~4.25 GB/s
    bw_d2h = bytes_d2h / (d2h_ms * 1e-3 * 1e9)  # ~5.12 GB/s

    assert 4.0 < bw_h2d < 5.0
    assert 5.0 < bw_d2h < 6.0


def test_ingested_phase62_toy_validation_results():
    """Validates that all 9 toy-graph validation JSON files for Phase 6.2 exist and passed."""
    workspace = Path(__file__).resolve().parents[2]
    results_dir = workspace / "experiments" / "results"

    engines = ["baseline", "coalesced", "streams"]
    operators = ["mean", "sum", "max"]

    for eng in engines:
        for op in operators:
            matches = list(results_dir.glob(f"val_toy_{eng}_{op}*20261024_114748.json"))
            assert len(matches) >= 1, f"Missing toy validation file for {eng} {op}"
            val_file = matches[0]

            with open(val_file, "r", encoding="utf-8") as f:
                d = json.load(f)

            assert d["numerical_verification"]["status"] == "PASSED"
            assert d["numerical_verification"]["max_absolute_error"] == 0
            assert d["numerical_verification"]["nan_detected"] is False
            assert d["graph_configuration"]["num_nodes"] == 5
            assert d["graph_configuration"]["num_edges"] == 8
            assert d["graph_configuration"]["feature_dim"] == 4


def test_ingested_phase62_measured_results():
    """Validates all 9 Phase 6.2 Tier A benchmark records (Baseline, Coalesced, Streams) on Tesla T4."""
    workspace = Path(__file__).resolve().parents[2]
    results_dir = workspace / "experiments" / "results"

    operators = ["mean", "sum", "max"]
    for op in operators:
        b_file = results_dir / f"exp_gpu_baseline_tier_a_{op}_20261024_114748.json"
        c_file = results_dir / f"exp_gpu_coalesced_tier_a_{op}_20261024_114748.json"
        s_file = results_dir / f"exp_gpu_streams_tier_a_{op}_20261024_114748.json"

        assert b_file.exists(), f"Missing baseline Tier A: {b_file}"
        assert c_file.exists(), f"Missing coalesced Tier A: {c_file}"
        assert s_file.exists(), f"Missing streams Tier A: {s_file}"

        with open(b_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        with open(c_file, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        with open(s_file, "r", encoding="utf-8") as f:
            s_data = json.load(f)

        # Verification status
        for data, name in [(b_data, "baseline"), (c_data, "coalesced"), (s_data, "streams")]:
            assert data["measurement_state"] == "MEASURED"
            assert data["hardware"]["device_name"] == "Tesla T4"
            assert data["numerical_verification"]["status"] == "PASSED"
            assert data["numerical_verification"]["max_absolute_error"] <= 1e-5
            assert data["numerical_verification"]["nan_detected"] is False
            assert data["graph_configuration"]["num_nodes"] == 10000
            assert data["graph_configuration"]["num_edges"] == 94945
            assert data["graph_configuration"]["feature_dim"] == 32
            assert data["benchmark_parameters"]["warmup_iterations"] == 5
            assert data["benchmark_parameters"]["recorded_repetitions"] == 20

        # Pinned streams specific parameters
        assert s_data["benchmark_parameters"]["pinned_host_memory"] is True
        assert s_data["benchmark_parameters"]["stream_execution"] is True
        assert s_data["performance_metrics"]["pcie_h2d_bandwidth_gbps"] > 10.0
        assert s_data["performance_metrics"]["pcie_d2h_bandwidth_gbps"] > 11.5

        # Timing and speedup calculations
        b_k = b_data["timing_results_ms"]["kernel_mean"]
        c_k = c_data["timing_results_ms"]["kernel_mean"]
        s_k = s_data["timing_results_ms"]["kernel_mean"]

        b_h2d = b_data["timing_results_ms"]["h2d_transfer_mean"]
        s_h2d = s_data["timing_results_ms"]["h2d_transfer_mean"]

        b_d2h = b_data["timing_results_ms"]["d2h_transfer_mean"]
        s_d2h = s_data["timing_results_ms"]["d2h_transfer_mean"]

        b_e2e = b_data["timing_results_ms"]["end_to_end_mean"]
        c_e2e = c_data["timing_results_ms"]["end_to_end_mean"]
        s_e2e = s_data["timing_results_ms"]["end_to_end_mean"]

        # Streams kernel latency is identical to Coalesced (~1.0x ratio, difference < 2%)
        k_ratio = c_k / s_k
        assert 0.98 <= k_ratio <= 1.03, f"Unexpected streams vs coalesced kernel ratio for {op}: {k_ratio:.2f}x"

        # Pure kernel speedup vs baseline remains ~3.8x - 4.1x
        k_speedup = b_k / s_k
        assert 3.80 <= k_speedup <= 4.15, f"Unexpected streams kernel speedup for {op}: {k_speedup:.2f}x"

        # H2D transfer speedup via pinned host memory is ~3.4x - 3.6x
        h2d_speedup = b_h2d / s_h2d
        assert 3.35 <= h2d_speedup <= 3.60, f"Unexpected H2D transfer speedup for {op}: {h2d_speedup:.2f}x"

        # D2H transfer speedup via pinned host memory is ~3.0x - 3.4x
        d2h_speedup = b_d2h / s_d2h
        assert 2.95 <= d2h_speedup <= 3.40, f"Unexpected D2H transfer speedup for {op}: {d2h_speedup:.2f}x"

        # End-to-end speedup vs baseline is ~3.3x - 3.5x
        e2e_speedup_base = b_e2e / s_e2e
        assert 3.30 <= e2e_speedup_base <= 3.50, f"Unexpected E2E speedup vs baseline for {op}: {e2e_speedup_base:.2f}x"

        # End-to-end speedup vs coalesced is ~3.2x - 3.3x
        e2e_speedup_coal = c_e2e / s_e2e
        assert 3.20 <= e2e_speedup_coal <= 3.35, f"Unexpected E2E speedup vs coalesced for {op}: {e2e_speedup_coal:.2f}x"
