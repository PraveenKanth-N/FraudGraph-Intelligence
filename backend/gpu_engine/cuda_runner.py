"""
CUDA Engine Host Runner — FraudGraph Intelligence (FGI)
Provides a clean Python interface to the native CUDA aggregation binary.
Handles local execution when available, transparently reports when local compilation
is blocked, and provides verification utilities comparing GPU output against Phase 4 CPU reference.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.cpu_engine.csr_graph import CSRGraph


def is_nvcc_available() -> bool:
    """Checks if the NVIDIA CUDA compiler (nvcc) is available on system PATH."""
    return shutil.which("nvcc") is not None


def get_cuda_binary_path(kernel: str = "baseline") -> Path:
    """Returns the expected path to the compiled CUDA binary for baseline or coalesced engine."""
    workspace_root = Path(__file__).resolve().parents[2]
    binary_base = "fgi_coalesced_runner" if kernel == "coalesced" else "fgi_baseline_runner"
    binary_name = f"{binary_base}.exe" if sys.platform == "win32" else binary_base
    return workspace_root / "backend" / "gpu_engine" / "cuda" / "bin" / binary_name


def validate_gpu_against_cpu(
    gpu_output: np.ndarray,
    cpu_output: np.ndarray,
    tolerance: float = 1.0e-5,
) -> Dict[str, Any]:
    """
    Compares GPU output tensor against CPU reference tensor.
    
    Checks:
    1. Shape match
    2. Data type match
    3. Absence of NaNs and Infs
    4. Maximum absolute error <= tolerance
    """
    if gpu_output.shape != cpu_output.shape:
        return {
            "status": "FAILED_SHAPE_MISMATCH",
            "gpu_shape": list(gpu_output.shape),
            "cpu_shape": list(cpu_output.shape),
            "max_absolute_error": None,
        }

    nan_detected = bool(np.any(np.isnan(gpu_output)) or np.any(np.isinf(gpu_output)))
    if nan_detected:
        return {
            "status": "FAILED_NAN_OR_INF",
            "nan_detected": True,
            "max_absolute_error": None,
        }

    diff = np.abs(gpu_output - cpu_output)
    max_err = float(np.max(diff))
    passed = max_err <= tolerance

    return {
        "status": "PASSED" if passed else "FAILED_TOLERANCE",
        "max_absolute_error": max_err,
        "tolerance_threshold": tolerance,
        "nan_detected": False,
    }


def execute_cuda_aggregation(
    graph: CSRGraph,
    operator: str = "mean",
    kernel: str = "baseline",
    warmup: int = 5,
    repeat: int = 20,
    block_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes native CUDA aggregation if the compiled binary is present.
    If binary or nvcc is missing, raises RuntimeError detailing Colab workflow instructions.
    """
    binary_path = get_cuda_binary_path(kernel=kernel)
    if not binary_path.exists():
        raise RuntimeError(
            f"CUDA binary '{binary_path.name}' not found. "
            "Local compilation is unavailable due to missing nvcc/MSVC build tools. "
            "Please execute the Google Colab workflow: 'backend/gpu_engine/colab_runner/' "
            "(see docs/CUDA_BASELINE.md for instructions)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Dump binary arrays
        with open(tmp_path / "meta.txt", "w", encoding="utf-8") as f:
            f.write(f"{graph.num_nodes} {graph.num_edges} {graph.feature_dim}\n")
        graph.row_ptr.astype(np.int32).tofile(tmp_path / "row_ptr.bin")
        graph.col_idx.astype(np.int32).tofile(tmp_path / "col_idx.bin")
        graph.features.astype(np.float32).tofile(tmp_path / "features.bin")

        out_bin = tmp_path / "out.bin"
        json_out = tmp_path / "bench.json"

        cmd = [
            str(binary_path),
            "--graph", str(tmp_path),
            "--op", operator,
            "--warmup", str(warmup),
            "--repeat", str(repeat),
            "--out-bin", str(out_bin),
            "--export-json", str(json_out),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"CUDA execution failed (exit code {res.returncode}):\n{res.stderr}")

        output_arr = np.fromfile(out_bin, dtype=np.float32).reshape(graph.num_nodes, graph.feature_dim)
        with open(json_out, "r", encoding="utf-8") as f:
            bench_metadata = json.load(f)

        return output_arr, bench_metadata
