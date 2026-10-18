"""
Unit and Schema Verification Tests for Baseline CUDA & Colab Runner (Phase 5)
Tests cover:
1. CUDA source and header completeness.
2. Colab bundle generation and file inclusion.
3. Colab notebook JSON structure and syntax.
4. Experiment JSON schema validation (success and rejection paths).
5. Numerical equivalence and tolerance checking logic.
6. NaN / Inf / shape mismatch defenses.
7. Graceful offline error reporting when local nvcc compiler is absent.
8. Explicit skipping of live GPU execution when local nvcc is unavailable.
"""

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

from backend.gpu_engine.colab_runner.ingest_colab_results import validate_result_schema
from backend.gpu_engine.colab_runner.package_colab_bundle import build_colab_bundle
from backend.gpu_engine.cuda_runner import (
    execute_cuda_aggregation,
    is_nvcc_available,
    validate_gpu_against_cpu,
)


def test_cuda_source_files_exist():
    workspace = Path(__file__).resolve().parents[2]
    cuda_dir = workspace / "backend" / "gpu_engine" / "cuda"

    required_files = [
        cuda_dir / "include" / "fgi_cuda_common.cuh",
        cuda_dir / "include" / "fgi_baseline_kernels.cuh",
        cuda_dir / "src" / "fgi_baseline_kernels.cu",
        cuda_dir / "src" / "main_baseline.cu",
        cuda_dir / "Makefile",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required CUDA source file missing: {rf}"

    # Verify key tokens
    kernel_src = (cuda_dir / "src" / "fgi_baseline_kernels.cu").read_text(encoding="utf-8")
    assert "aggregate_baseline_kernel" in kernel_src
    assert "OP_SUM" in kernel_src
    assert "OP_MEAN" in kernel_src
    assert "OP_MAX" in kernel_src
    assert "degree <= 0" in kernel_src  # Empty neighborhood check


def test_package_colab_bundle_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_zip = Path(tmpdir) / "test_bundle.zip"
        built_path = build_colab_bundle(output_zip=test_zip)

        assert built_path.exists()
        assert built_path.stat().st_size > 500_000  # At least 500 KB with data

        with zipfile.ZipFile(built_path, "r") as zf:
            namelist = zf.namelist()
            assert "cuda/Makefile" in namelist
            assert "cuda/src/fgi_baseline_kernels.cu" in namelist
            assert "cuda/src/main_baseline.cu" in namelist
            assert "run_colab_benchmarks.py" in namelist
            assert "fgi_colab_workflow.ipynb" in namelist
            assert "data/toy_graph_5nodes/row_ptr.bin" in namelist
            assert "data/synthetic_tier_a_10k/row_ptr.bin" in namelist


def test_colab_workflow_notebook_validity():
    workspace = Path(__file__).resolve().parents[2]
    nb_path = workspace / "backend" / "gpu_engine" / "colab_runner" / "fgi_colab_workflow.ipynb"
    assert nb_path.exists()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat", 0) >= 4
    cells = nb.get("cells", [])
    assert len(cells) >= 4

    # Ensure critical commands exist
    code_sources = "".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")
    assert "nvidia-smi" in code_sources
    assert "nvcc" in code_sources
    assert "run_colab_benchmarks.py" in code_sources


def test_result_json_schema_validation_success():
    valid_payload = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-TEST-001",
        "measurement_state": "MEASURED",
        "hardware": {
            "device_name": "Tesla T4",
            "compute_capability": "7.5",
            "total_vram_mb": 15109.0,
        },
        "graph_configuration": {
            "num_nodes": 10000,
            "num_edges": 94945,
            "feature_dim": 32,
        },
        "benchmark_parameters": {
            "implementation": "cuda_baseline_thread_per_node",
            "operator": "mean",
            "warmup_iterations": 5,
            "recorded_repetitions": 20,
        },
        "timing_results_ms": {
            "h2d_transfer_mean": 1.25,
            "kernel_mean": 2.15,
            "d2h_transfer_mean": 0.45,
            "end_to_end_mean": 3.85,
        },
        "numerical_verification": {
            "status": "PASSED",
            "max_absolute_error": 1.5e-6,
            "tolerance_threshold": 1.0e-5,
        },
    }
    valid, msg = validate_result_schema(valid_payload)
    assert valid is True
    assert msg == "Valid"


def test_result_json_schema_rejection_unmeasured_state():
    invalid_payload = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-SAMPLE",
        "measurement_state": "SAMPLE",  # Should be rejected
        "hardware": {"device_name": "Tesla T4", "compute_capability": "7.5"},
        "graph_configuration": {},
        "benchmark_parameters": {},
        "timing_results_ms": {"kernel_mean": 1.0, "end_to_end_mean": 2.0},
        "numerical_verification": {"status": "PASSED", "max_absolute_error": 1e-6},
    }
    valid, msg = validate_result_schema(invalid_payload)
    assert valid is False
    assert "measurement_state must be 'MEASURED'" in msg


def test_result_json_schema_rejection_tolerance_failure():
    failing_payload = {
        "schema_version": "1.0.0",
        "experiment_id": "EXP-GPU-FAIL",
        "measurement_state": "MEASURED",
        "hardware": {"device_name": "Tesla T4", "compute_capability": "7.5"},
        "graph_configuration": {},
        "benchmark_parameters": {},
        "timing_results_ms": {"kernel_mean": 1.0, "end_to_end_mean": 2.0},
        "numerical_verification": {
            "status": "PASSED",
            "max_absolute_error": 0.05,  # Exceeds 1e-5
            "tolerance_threshold": 1.0e-5,
        },
    }
    valid, msg = validate_result_schema(failing_payload)
    assert valid is False
    assert "exceeds tolerance" in msg


def test_validate_gpu_against_cpu_passing():
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gpu = np.array([[1.0000001, 2.0], [2.9999998, 4.0]], dtype=np.float32)

    res = validate_gpu_against_cpu(gpu, cpu, tolerance=1e-5)
    assert res["status"] == "PASSED"
    assert res["max_absolute_error"] < 1e-5
    assert res["nan_detected"] is False


def test_validate_gpu_against_cpu_tolerance_failure():
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gpu = np.array([[1.1, 2.0], [3.0, 4.0]], dtype=np.float32)

    res = validate_gpu_against_cpu(gpu, cpu, tolerance=1e-5)
    assert res["status"] == "FAILED_TOLERANCE"
    assert np.isclose(res["max_absolute_error"], 0.1)


def test_validate_gpu_against_cpu_nan_detection():
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gpu = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

    res = validate_gpu_against_cpu(gpu, cpu, tolerance=1e-5)
    assert res["status"] == "FAILED_NAN_OR_INF"
    assert res["nan_detected"] is True


def test_validate_gpu_against_cpu_shape_mismatch():
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gpu = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)

    res = validate_gpu_against_cpu(gpu, cpu)
    assert res["status"] == "FAILED_SHAPE_MISMATCH"


def test_gpu_runner_graceful_missing_binary():
    from backend.cpu_engine.csr_graph import CSRGraph
    empty_g = CSRGraph(
        row_ptr=np.array([0], dtype=np.int32),
        col_idx=np.array([], dtype=np.int32),
        features=np.zeros((0, 4), dtype=np.float32),
        is_directed=True,
    )
    with pytest.raises(RuntimeError, match="Local compilation is unavailable"):
        execute_cuda_aggregation(empty_g, operator="mean")


@pytest.mark.skipif(not is_nvcc_available(), reason="nvcc compiler not installed locally - Colab workflow required")
def test_live_cuda_execution():
    """This test only executes if a live nvcc compiler and GPU runtime are present locally."""
    pass
