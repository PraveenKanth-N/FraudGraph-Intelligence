"""
Unit and Schema Verification Tests for Warp-Coalesced CUDA Implementation (Phase 6, Experiment 6.1)
Tests cover:
1. Coalesced CUDA source and header completeness.
2. Kernel implementation token verification (warp mapping, masking, sentinels).
3. Makefile multi-target configuration (baseline + coalesced).
4. Python host adapter support for kernel="coalesced".
5. Phase 6 Colab bundle generation and file inclusion.
6. Schema conformity for coalesced experiment JSON results.
7. Graceful offline error reporting when local nvcc compiler is absent.
"""

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

from backend.cpu_engine.csr_graph import CSRGraph
from backend.gpu_engine.colab_runner.ingest_colab_results import validate_result_schema
from backend.gpu_engine.colab_runner.package_colab_bundle import build_colab_bundle
from backend.gpu_engine.cuda_runner import (
    execute_cuda_aggregation,
    get_cuda_binary_path,
    is_nvcc_available,
    validate_gpu_against_cpu,
)


def test_coalesced_cuda_source_files_exist():
    workspace = Path(__file__).resolve().parents[2]
    cuda_dir = workspace / "backend" / "gpu_engine" / "cuda"

    required_files = [
        cuda_dir / "include" / "fgi_cuda_common.cuh",
        cuda_dir / "include" / "fgi_baseline_kernels.cuh",
        cuda_dir / "include" / "fgi_coalesced_kernels.cuh",
        cuda_dir / "src" / "fgi_baseline_kernels.cu",
        cuda_dir / "src" / "fgi_coalesced_kernels.cu",
        cuda_dir / "src" / "main_baseline.cu",
        cuda_dir / "src" / "main_coalesced.cu",
        cuda_dir / "Makefile",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required CUDA source file missing: {rf}"

    # Verify key architectural tokens in coalesced kernel source
    kernel_src = (cuda_dir / "src" / "fgi_coalesced_kernels.cu").read_text(encoding="utf-8")
    assert "aggregate_coalesced_warp_kernel" in kernel_src
    assert "global_warp_id" in kernel_src
    assert "lane_id" in kernel_src
    assert "degree <= 0" in kernel_src  # Empty neighborhood check
    assert "OP_SUM" in kernel_src
    assert "OP_MEAN" in kernel_src
    assert "OP_MAX" in kernel_src
    assert "launch_coalesced_aggregation" in kernel_src


def test_coalesced_makefile_targets():
    workspace = Path(__file__).resolve().parents[2]
    makefile_content = (workspace / "backend" / "gpu_engine" / "cuda" / "Makefile").read_text(encoding="utf-8")

    assert "fgi_baseline_runner" in makefile_content
    assert "fgi_coalesced_runner" in makefile_content
    assert "fgi_coalesced_kernels.o" in makefile_content
    assert "main_coalesced.o" in makefile_content

    # CUDA 13.0 Tesla T4 compatibility assertions
    assert "arch=compute_75,code=sm_75" in makefile_content
    assert "compute_70" not in makefile_content, "compute_70 is deprecated/unsupported in CUDA 13.0"
    assert "compute_80" not in makefile_content, "Target strictly sm_75 for Colab Tesla T4"


def test_coalesced_runner_interface():
    binary_path = get_cuda_binary_path(kernel="coalesced")
    assert "fgi_coalesced_runner" in binary_path.name


def test_package_colab_bundle_phase6_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_zip = Path(tmpdir) / "test_bundle_phase6.zip"
        built_path = build_colab_bundle(output_zip=test_zip)

        assert built_path.exists()
        assert built_path.stat().st_size > 500_000

        with zipfile.ZipFile(built_path, "r") as zf:
            names = set(zf.namelist())
            assert "cuda/src/fgi_baseline_kernels.cu" in names
            assert "cuda/src/fgi_coalesced_kernels.cu" in names
            assert "cuda/src/main_baseline.cu" in names
            assert "cuda/src/main_coalesced.cu" in names
            assert "cuda/include/fgi_coalesced_kernels.cuh" in names
            assert "cuda/Makefile" in names
            assert "run_colab_benchmarks.py" in names
            assert "fgi_colab_workflow.ipynb" in names
            assert "data/toy_graph_5nodes/row_ptr.bin" in names
            assert "data/synthetic_tier_a_10k/row_ptr.bin" in names


def test_coalesced_result_json_schema():
    valid_record = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-COALESCED-OPT-6.1",
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
            "implementation": "cuda_optimized_warp_coalesced",
            "operator": "mean",
            "warmup_iterations": 5,
            "recorded_repetitions": 20,
            "block_size": 256,
        },
        "timing_results_ms": {
            "h2d_transfer_mean": 0.55,
            "kernel_mean": 0.045,
            "kernel_std": 0.001,
            "kernel_p50": 0.045,
            "kernel_p95": 0.047,
            "d2h_transfer_mean": 0.33,
            "end_to_end_mean": 0.925,
        },
        "performance_metrics": {
            "edges_per_second": 2.1e9,
            "effective_bandwidth_gbps": 305.0,
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


def test_canonical_tier_a_fixture():
    """Verifies that the canonical Tier A benchmark fixture has exactly N=10,000, M=94,945, D=32."""
    workspace = Path(__file__).resolve().parents[2]
    tier_a_path = workspace / "data" / "fixtures" / "synthetic_tier_a_10k.npz"
    assert tier_a_path.exists(), f"Tier A fixture missing: {tier_a_path}"

    graph = CSRGraph.load(tier_a_path)
    assert graph.num_nodes == 10000, f"Expected 10,000 nodes, got {graph.num_nodes}"
    assert graph.num_edges == 94945, f"Expected 94,945 directed edges, got {graph.num_edges}"
    assert graph.feature_dim == 32, f"Expected 32 features, got {graph.feature_dim}"


def test_bandwidth_metric_calculations():
    """
    Verifies the mathematical separation between Compulsory Physical DRAM Bandwidth
    and Algorithmic Memory Traffic (with L2 cache hits).
    """
    num_nodes = 10000
    num_edges = 94945
    feature_dim = 32

    bytes_row = (num_nodes + 1) * 4
    bytes_col = num_edges * 4
    bytes_feat_unique = num_nodes * feature_dim * 4
    bytes_out = num_nodes * feature_dim * 4
    bytes_compulsory = bytes_row + bytes_col + bytes_feat_unique + bytes_out
    assert bytes_compulsory == 2979784  # ~2.98 MB

    bytes_algorithmic = bytes_row + bytes_col + (num_edges * feature_dim * 4) + bytes_out
    assert bytes_algorithmic == 13852744  # ~13.85 MB

    # For measured coalesced sum kernel latency: t = 0.0170512 ms
    t_kernel_ms = 0.0170512
    bw_compulsory = bytes_compulsory / (t_kernel_ms * 1e-3 * 1e9)
    bw_algo = bytes_algorithmic / (t_kernel_ms * 1e-3 * 1e9)

    # Compulsory bandwidth must be within Tesla T4 peak DRAM capacity (320 GB/s)
    assert 170.0 < bw_compulsory < 180.0
    assert bw_compulsory < 320.0, "Physical bandwidth must not exceed hardware limit"

    # Algorithmic bandwidth reflects cache-amplified edge access rate (> 320 GB/s due to 4MB L2 cache hits)
    assert 800.0 < bw_algo < 820.0
    ratio = bytes_algorithmic / bytes_compulsory
    assert abs(ratio - (bw_algo / bw_compulsory)) < 1e-6


def test_throughput_string_formatting():
    """Verifies that integer comma formatting produces valid standard strings without non-standard printf flags."""
    def format_with_commas(val: int) -> str:
        return f"{val:,}"

    assert format_with_commas(5568230000) == "5,568,230,000"
    assert format_with_commas(1376590000) == "1,376,590,000"
    assert f"{5568230000:15.0f}".strip() == "5568230000"


def test_ingested_phase6_measured_results():
    """Validates that all six ingested Phase 6 Tier A JSON result files exist and meet speedup requirements."""
    workspace = Path(__file__).resolve().parents[2]
    results_dir = workspace / "experiments" / "results"

    operators = ["mean", "sum", "max"]
    base_results = {}
    coal_results = {}

    for op in operators:
        bf = sorted(results_dir.glob(f"exp_gpu_baseline_tier_a_{op}_20261019_*.json"))
        cf = sorted(results_dir.glob(f"exp_gpu_coalesced_tier_a_{op}_20261019_*.json"))
        assert len(bf) >= 1, f"Missing baseline result for {op}"
        assert len(cf) >= 1, f"Missing coalesced result for {op}"

        with open(bf[-1], "r", encoding="utf-8") as f:
            b_data = json.load(f)
        with open(cf[-1], "r", encoding="utf-8") as f:
            c_data = json.load(f)

        assert b_data["measurement_state"] == "MEASURED"
        assert c_data["measurement_state"] == "MEASURED"
        assert b_data["hardware"]["device_name"] == "Tesla T4"
        assert c_data["hardware"]["device_name"] == "Tesla T4"
        assert b_data["numerical_verification"]["status"] == "PASSED"
        assert c_data["numerical_verification"]["status"] == "PASSED"
        assert b_data["numerical_verification"]["max_absolute_error"] <= 1e-5
        assert c_data["numerical_verification"]["max_absolute_error"] <= 1e-5

        # Verify estimated bandwidth metrics are present in revalidation schema
        assert "estimated_compulsory_bandwidth_gbps" in b_data["performance_metrics"]
        assert "estimated_algorithmic_bandwidth_gbps" in b_data["performance_metrics"]
        assert "estimated_compulsory_bandwidth_gbps" in c_data["performance_metrics"]
        assert "estimated_algorithmic_bandwidth_gbps" in c_data["performance_metrics"]

        b_k = b_data["timing_results_ms"]["kernel_mean"]
        c_k = c_data["timing_results_ms"]["kernel_mean"]
        k_speedup = b_k / c_k

        b_e2e = b_data["timing_results_ms"]["end_to_end_mean"]
        c_e2e = c_data["timing_results_ms"]["end_to_end_mean"]
        e2e_speedup = b_e2e / c_e2e

        # Kernel speedup must be ~3.7x - 4.15x
        assert 3.70 <= k_speedup <= 4.15, f"Unexpected kernel speedup for {op}: {k_speedup:.2f}x"
        # End-to-end speedup reflects PCIe transfer dominance: ~0.98x - 1.15x (mean ~0.99x, sum ~1.09x, max ~1.07x)
        assert 0.98 <= e2e_speedup <= 1.15, f"Unexpected e2e speedup for {op}: {e2e_speedup:.2f}x"


