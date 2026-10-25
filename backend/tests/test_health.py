import subprocess
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.api.v1.endpoints.health import get_gpu_diagnostics
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "FraudGraph Intelligence" in data["service"]

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project_id"] == "FGI"
    assert "compute_engine" in data
    assert "local_gpu_detected" in data["compute_engine"]
    assert data["compute_engine"]["cuda_compiler_status"] in {
        "UNAVAILABLE_LOCAL_COLAB_READY",
        "AVAILABLE_LOCAL",
        "NO_GPU_DETECTED_COLAB_READY",
    }
    assert data["compute_engine"]["colab_workflow_enabled"] is True

def test_get_gpu_diagnostics_success_without_nvcc():
    with patch("subprocess.check_output", return_value="NVIDIA GeForce RTX 3080, 525.85, 8.6, 10240, 8192\n"), \
         patch("shutil.which", return_value=None):
        info = get_gpu_diagnostics()
        assert info["detected"] is True
        assert info["name"] == "NVIDIA GeForce RTX 3080"
        assert info["driver_version"] == "525.85"
        assert info["compute_capability"] == "8.6"
        assert info["vram_total_mb"] == 10240.0
        assert info["vram_free_mb"] == 8192.0
        assert info["local_nvcc_available"] is False
        assert info["cuda_compiler_status"] == "UNAVAILABLE_LOCAL_COLAB_READY"

def test_get_gpu_diagnostics_success_with_nvcc():
    with patch("subprocess.check_output", return_value="NVIDIA A100-SXM4-40GB, 535.54.03, 8.0, 40960, 40000\n"), \
         patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"):
        info = get_gpu_diagnostics()
        assert info["detected"] is True
        assert info["name"] == "NVIDIA A100-SXM4-40GB"
        assert info["driver_version"] == "535.54.03"
        assert info["compute_capability"] == "8.0"
        assert info["vram_total_mb"] == 40960.0
        assert info["vram_free_mb"] == 40000.0
        assert info["local_nvcc_available"] is True
        assert info["cuda_compiler_status"] == "AVAILABLE_LOCAL"

def test_get_gpu_diagnostics_missing_executable():
    with patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi not found")):
        info = get_gpu_diagnostics()
        assert info["detected"] is False
        assert info["name"] is None
        assert info["driver_version"] is None
        assert info["compute_capability"] is None
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["local_nvcc_available"] is False
        assert info["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"
        assert "MX450" not in str(info.values())

def test_get_gpu_diagnostics_command_failure():
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(returncode=127, cmd="nvidia-smi")):
        info = get_gpu_diagnostics()
        assert info["detected"] is False
        assert info["name"] is None
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"

def test_get_gpu_diagnostics_timeout():
    with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2)):
        info = get_gpu_diagnostics()
        assert info["detected"] is False
        assert info["name"] is None
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"

def test_get_gpu_diagnostics_malformed_empty():
    with patch("subprocess.check_output", return_value=""):
        info = get_gpu_diagnostics()
        assert info["detected"] is False
        assert info["name"] is None
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"

def test_get_gpu_diagnostics_malformed_whitespace():
    with patch("subprocess.check_output", return_value="   \n \t \n"):
        info = get_gpu_diagnostics()
        assert info["detected"] is False
        assert info["name"] is None
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"

def test_get_gpu_diagnostics_malformed_non_numeric_vram():
    with patch("subprocess.check_output", return_value="NVIDIA Custom GPU, 500.0, 7.0, N/A, invalid\n"), \
         patch("shutil.which", return_value=None):
        info = get_gpu_diagnostics()
        assert info["detected"] is True
        assert info["name"] == "NVIDIA Custom GPU"
        assert info["driver_version"] == "500.0"
        assert info["compute_capability"] == "7.0"
        assert info["vram_total_mb"] is None
        assert info["vram_free_mb"] is None
        assert info["cuda_compiler_status"] == "UNAVAILABLE_LOCAL_COLAB_READY"

def test_health_endpoint_no_gpu_mocked():
    no_gpu_payload = {
        "detected": False,
        "name": None,
        "driver_version": None,
        "compute_capability": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "local_nvcc_available": False,
        "cuda_compiler_status": "NO_GPU_DETECTED_COLAB_READY"
    }
    with patch("backend.app.api.v1.endpoints.health.get_gpu_diagnostics", return_value=no_gpu_payload):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        engine = data["compute_engine"]
        assert engine["local_gpu_detected"] is None
        assert engine["vram_free_mb"] is None
        assert engine["cuda_compiler_status"] == "NO_GPU_DETECTED_COLAB_READY"
        assert engine["cuda_available_locally"] is False
