from datetime import datetime
import shutil
import subprocess
import sys
from fastapi import APIRouter
from backend.app.config import settings

router = APIRouter()

def get_gpu_diagnostics() -> dict:
    """Safely query local GPU details without throwing exceptions."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version,compute_cap,memory.total,memory.free", "--format=csv,noheader,nounits"],
            timeout=2,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if not output:
            raise ValueError("nvidia-smi returned empty output")

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            raise ValueError("nvidia-smi returned empty lines")

        parts = [p.strip() for p in lines[0].split(",")]
        gpu_name = parts[0] if len(parts) > 0 and parts[0] else None
        if not gpu_name:
            raise ValueError("No GPU name parsed from nvidia-smi output")

        driver_version = parts[1] if len(parts) > 1 and parts[1] else None
        compute_capability = parts[2] if len(parts) > 2 and parts[2] else None

        vram_total_mb = None
        if len(parts) > 3 and parts[3]:
            try:
                vram_total_mb = float(parts[3])
            except ValueError:
                vram_total_mb = None

        vram_free_mb = None
        if len(parts) > 4 and parts[4]:
            try:
                vram_free_mb = float(parts[4])
            except ValueError:
                vram_free_mb = None

        local_nvcc = shutil.which("nvcc") is not None
        compiler_status = "AVAILABLE_LOCAL" if local_nvcc else "UNAVAILABLE_LOCAL_COLAB_READY"

        return {
            "detected": True,
            "name": gpu_name,
            "driver_version": driver_version,
            "compute_capability": compute_capability,
            "vram_total_mb": vram_total_mb,
            "vram_free_mb": vram_free_mb,
            "local_nvcc_available": local_nvcc,
            "cuda_compiler_status": compiler_status
        }
    except Exception:
        return {
            "detected": False,
            "name": None,
            "driver_version": None,
            "compute_capability": None,
            "vram_total_mb": None,
            "vram_free_mb": None,
            "local_nvcc_available": False,
            "cuda_compiler_status": "NO_GPU_DETECTED_COLAB_READY"
        }

@router.get("/health")
def check_health():
    gpu_info = get_gpu_diagnostics()
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "project_id": settings.PROJECT_ID,
        "version": settings.VERSION,
        "compute_engine": {
            "backend": settings.COMPUTE_BACKEND,
            "cuda_available_locally": gpu_info["local_nvcc_available"],
            "local_gpu_detected": gpu_info["name"] if gpu_info["detected"] else None,
            "vram_free_mb": gpu_info["vram_free_mb"] if gpu_info["detected"] else None,
            "cuda_compiler_status": gpu_info["cuda_compiler_status"],
            "colab_workflow_enabled": True
        },
        "system": {
            "python_version": sys.version.split()[0],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
