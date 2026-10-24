"""
Hardware Telemetry, Memory Preflight & Statistical Metrics — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Provides:
1. Native hardware telemetry collection (CPU model, cores, Host RAM, GPU model, VRAM).
2. Separate Host-RAM and GPU-VRAM preflight memory guards with configurable safety margins.
3. High-precision statistical calculations (mean, std ddof=1, p50, p95, min, max, CV%).
4. Throughput (nodes/s, edges/s) and effective memory bandwidth (GB/s) metrics.
"""

import ctypes
from dataclasses import asdict, dataclass
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MemoryPreflightResult:
    """Preflight evaluation of memory requirements before execution."""
    allowed: bool
    estimated_mb: float
    available_mb: float
    total_mb: float
    safety_margin: float
    guard_status: str = "VERIFIED_PASS"  # "VERIFIED_PASS", "BLOCKED_CAPACITY_EXCEEDED", "BLOCKED_TELEMETRY_UNAVAILABLE", "PERMITTED_BOUNDED_FALLBACK"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "guard_status": self.guard_status,
            "estimated_mb": round(self.estimated_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "total_mb": round(self.total_mb, 2),
            "safety_margin": self.safety_margin,
            "reason": self.reason,
        }


def get_host_ram_info() -> Dict[str, float]:
    """
    Retrieves total and available physical RAM in megabytes.
    Uses Win32 GlobalMemoryStatusEx on Windows, /proc/meminfo on Linux.
    """
    total_mb = 0.0
    avail_mb = 0.0

    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_mb = float(stat.ullTotalPhys) / (1024 * 1024)
                avail_mb = float(stat.ullAvailPhys) / (1024 * 1024)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = float(val) / 1024.0  # kB to MB
            total_mb = meminfo.get("MemTotal", 0.0)
            avail_mb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0.0))
        except Exception:
            pass

    return {
        "total_mb": round(total_mb, 2),
        "available_mb": round(avail_mb, 2),
    }


def get_gpu_info() -> Dict[str, Any]:
    """
    Queries GPU model and VRAM using nvidia-smi if available.
    Returns structured hardware metadata dictionary.
    """
    info: Dict[str, Any] = {
        "gpu_available": False,
        "device_name": None,
        "driver_version": None,
        "total_vram_mb": 0.0,
        "free_vram_mb": 0.0,
    }

    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return info

    try:
        cmd = [
            nvsmi,
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        if res:
            lines = res.splitlines()
            first_gpu = [item.strip() for item in lines[0].split(",")]
            if len(first_gpu) >= 4:
                info["gpu_available"] = True
                info["device_name"] = first_gpu[0]
                info["driver_version"] = first_gpu[1]
                info["total_vram_mb"] = float(first_gpu[2])
                info["free_vram_mb"] = float(first_gpu[3])
    except Exception:
        pass

    return info


def get_hardware_telemetry() -> Dict[str, Any]:
    """
    Gathers comprehensive host and GPU hardware telemetry for experiment records.
    """
    cpu_name = platform.processor() or "Unknown CPU"
    if sys.platform == "win32":
        try:
            res = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            if res:
                cpu_name = res
        except Exception:
            pass

    ram_info = get_host_ram_info()
    gpu_info = get_gpu_info()

    return {
        "platform": platform.platform(),
        "os": sys.platform,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu": {
            "model": cpu_name,
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "logical_cores": os.cpu_count() or 1,
            "host_ram_total_mb": ram_info["total_mb"],
            "host_ram_available_mb": ram_info["available_mb"],
        },
        "gpu": gpu_info,
    }


def estimate_workload_memory_mb(num_nodes: int, num_edges: int, feature_dim: int) -> Dict[str, float]:
    """
    Estimates memory consumption in MB for graph CSR structures and output tensors.
    """
    row_ptr_mb = ((num_nodes + 1) * 8) / (1024 * 1024)   # int64
    col_idx_mb = (num_edges * 8) / (1024 * 1024)           # int64
    features_mb = (num_nodes * feature_dim * 4) / (1024 * 1024)  # float32
    output_mb = (num_nodes * feature_dim * 4) / (1024 * 1024)    # float32

    graph_mb = row_ptr_mb + col_idx_mb + features_mb
    # Working buffer estimate: 2x output tensor for aggregation intermediates
    working_mb = output_mb * 2.0
    total_mb = graph_mb + output_mb + working_mb

    return {
        "graph_mb": round(graph_mb, 4),
        "output_mb": round(output_mb, 4),
        "working_mb": round(working_mb, 4),
        "total_estimated_mb": round(total_mb, 4),
    }


def check_host_ram(
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    safety_margin: float = 0.80,
    allow_bounded_fallback: bool = True,
    max_unverified_mb: float = 100.0,
    ram_info: Optional[Dict[str, float]] = None,
) -> MemoryPreflightResult:
    """
    Validates host RAM availability before launching CPU workloads.

    Safety Policy:
    1. Valid Telemetry (available_mb > 0, total_mb > 0):
       - If estimated_mb <= available_mb * safety_margin:
         allowed=True, guard_status='VERIFIED_PASS'.
       - Else:
         allowed=False, guard_status='BLOCKED_CAPACITY_EXCEEDED'.
    2. Unavailable or Invalid Telemetry (available_mb <= 0, total_mb <= 0, or NaN):
       - If allow_bounded_fallback is True AND estimated_mb <= max_unverified_mb:
         allowed=True, guard_status='PERMITTED_BOUNDED_FALLBACK'.
         (Workload is permitted under explicit bounded fallback policy; caution noted).
       - Else (workload exceeds max_unverified_mb or allow_bounded_fallback is False):
         allowed=False, guard_status='BLOCKED_TELEMETRY_UNAVAILABLE'.
         (Fail-closed policy protects system stability when available memory is unknown).
    """
    est = estimate_workload_memory_mb(num_nodes, num_edges, feature_dim)
    req_mb = est["total_estimated_mb"]

    ram = ram_info if ram_info is not None else get_host_ram_info()
    avail_mb = float(ram.get("available_mb", 0.0))
    tot_mb = float(ram.get("total_mb", 0.0))

    # Check for unavailable or invalid telemetry (e.g. 0, negative, NaN)
    telemetry_invalid = (
        avail_mb <= 0.0
        or tot_mb <= 0.0
        or np.isnan(avail_mb)
        or np.isnan(tot_mb)
    )

    if telemetry_invalid:
        if allow_bounded_fallback and req_mb <= max_unverified_mb:
            return MemoryPreflightResult(
                allowed=True,
                estimated_mb=req_mb,
                available_mb=max(avail_mb, 0.0) if not np.isnan(avail_mb) else 0.0,
                total_mb=max(tot_mb, 0.0) if not np.isnan(tot_mb) else 0.0,
                safety_margin=safety_margin,
                guard_status="PERMITTED_BOUNDED_FALLBACK",
                reason=(
                    f"Host RAM telemetry unavailable or invalid (available={avail_mb} MB, total={tot_mb} MB); "
                    f"permitted under explicit bounded fallback policy because estimated workload "
                    f"({req_mb:.2f} MB) <= unverified threshold ({max_unverified_mb:.1f} MB). "
                    f"Caution: Memory safety was NOT verified against physical RAM."
                ),
            )
        else:
            return MemoryPreflightResult(
                allowed=False,
                estimated_mb=req_mb,
                available_mb=max(avail_mb, 0.0) if not np.isnan(avail_mb) else 0.0,
                total_mb=max(tot_mb, 0.0) if not np.isnan(tot_mb) else 0.0,
                safety_margin=safety_margin,
                guard_status="BLOCKED_TELEMETRY_UNAVAILABLE",
                reason=(
                    f"Host RAM telemetry unavailable or invalid (available={avail_mb} MB, total={tot_mb} MB); "
                    f"blocked under fail-closed safety policy (estimated workload {req_mb:.2f} MB exceeds "
                    f"unverified threshold {max_unverified_mb:.1f} MB or bounded fallback is disabled)."
                ),
            )

    usable_mb = avail_mb * safety_margin
    if req_mb > usable_mb:
        return MemoryPreflightResult(
            allowed=False,
            estimated_mb=req_mb,
            available_mb=avail_mb,
            total_mb=tot_mb,
            safety_margin=safety_margin,
            guard_status="BLOCKED_CAPACITY_EXCEEDED",
            reason=(
                f"Workload requires estimated {req_mb:.1f} MB, which exceeds "
                f"{safety_margin*100:.0f}% safety budget of available host RAM ({usable_mb:.1f} MB / {avail_mb:.1f} MB)"
            ),
        )

    return MemoryPreflightResult(
        allowed=True,
        estimated_mb=req_mb,
        available_mb=avail_mb,
        total_mb=tot_mb,
        safety_margin=safety_margin,
        guard_status="VERIFIED_PASS",
        reason=None,
    )


def check_gpu_vram(
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    safety_margin: float = 0.85,
    gpu_info: Optional[Dict[str, Any]] = None,
) -> MemoryPreflightResult:
    """
    Validates GPU VRAM availability before launching CUDA workloads.
    """
    est = estimate_workload_memory_mb(num_nodes, num_edges, feature_dim)
    req_mb = est["total_estimated_mb"]

    gpu = gpu_info if gpu_info is not None else get_gpu_info()
    if not gpu.get("gpu_available", False):
        return MemoryPreflightResult(
            allowed=False,
            estimated_mb=req_mb,
            available_mb=0.0,
            total_mb=0.0,
            safety_margin=safety_margin,
            guard_status="BLOCKED_TELEMETRY_UNAVAILABLE",
            reason="No NVIDIA GPU detected or nvidia-smi unavailable",
        )

    free_mb = float(gpu.get("free_vram_mb", 0.0))
    tot_mb = float(gpu.get("total_vram_mb", 0.0))

    if free_mb <= 0.0 or tot_mb <= 0.0 or np.isnan(free_mb) or np.isnan(tot_mb):
        return MemoryPreflightResult(
            allowed=False,
            estimated_mb=req_mb,
            available_mb=max(free_mb, 0.0) if not np.isnan(free_mb) else 0.0,
            total_mb=max(tot_mb, 0.0) if not np.isnan(tot_mb) else 0.0,
            safety_margin=safety_margin,
            guard_status="BLOCKED_TELEMETRY_UNAVAILABLE",
            reason=f"GPU VRAM telemetry unavailable or non-positive (free={free_mb} MB, total={tot_mb} MB)",
        )

    usable_mb = free_mb * safety_margin
    if req_mb > usable_mb:
        return MemoryPreflightResult(
            allowed=False,
            estimated_mb=req_mb,
            available_mb=free_mb,
            total_mb=tot_mb,
            safety_margin=safety_margin,
            guard_status="BLOCKED_CAPACITY_EXCEEDED",
            reason=(
                f"Workload requires estimated {req_mb:.1f} MB, which exceeds "
                f"{safety_margin*100:.0f}% safety budget of free VRAM ({usable_mb:.1f} MB / {free_mb:.1f} MB)"
            ),
        )

    return MemoryPreflightResult(
        allowed=True,
        estimated_mb=req_mb,
        available_mb=free_mb,
        total_mb=tot_mb,
        safety_margin=safety_margin,
        guard_status="VERIFIED_PASS",
        reason=None,
    )


def compute_run_statistics(latencies_ms: List[float]) -> Dict[str, Any]:
    """
    Calculates statistical moments across repeated benchmark executions.
    Uses ddof=1 for sample standard deviation when sample count > 1.
    Evaluates timing stability to flag high variability (e.g. from GC / OS interrupts).
    """
    if not latencies_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median_p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "cv_percent": 0.0,
            "sample_count": 0,
            "is_stable": True,
            "stability_label": "STABLE",
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    p50_val = float(np.median(arr))
    p95_val = float(np.percentile(arr, 95))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    cv_val = float((std_val / mean_val) * 100.0) if mean_val > 1.0e-9 else 0.0

    # Stability criteria: CV <= 30% and max <= 3x median for sample sizes >= 5
    is_stable = bool(cv_val <= 30.0 and (p50_val <= 0.0 or max_val <= 3.0 * p50_val)) if n >= 5 else True
    stability_label = "STABLE" if is_stable else "HIGH_VARIABILITY"

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "median_p50": round(p50_val, 4),
        "p95": round(p95_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "cv_percent": round(cv_val, 2),
        "sample_count": n,
        "is_stable": is_stable,
        "stability_label": stability_label,
    }


def estimate_compulsory_traffic_bytes(
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    index_bytes: int = 8,
) -> int:
    """
    Estimates compulsory (minimum theoretical) data traffic assuming an ideal cache:
    - row_ptr: (num_nodes + 1) * index_bytes
    - col_idx: num_edges * index_bytes
    - features: num_nodes * feature_dim * 4 (each referenced node feature vector read once)
    - output: num_nodes * feature_dim * 4 (written once)
    """
    return int(
        (num_nodes + 1) * index_bytes +
        num_edges * index_bytes +
        num_nodes * feature_dim * 4 +
        num_nodes * feature_dim * 4
    )


def estimate_algorithmic_traffic_bytes(
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    index_bytes: int = 8,
) -> int:
    """
    Estimates algorithmic data traffic assuming un-cached edge-wise neighbor traversal:
    - row_ptr: (num_nodes + 1) * index_bytes
    - col_idx: num_edges * index_bytes
    - neighbor features: num_edges * feature_dim * 4 (every directed edge reads neighbor features)
    - output: num_nodes * feature_dim * 4 (written once)
    """
    return int(
        (num_nodes + 1) * index_bytes +
        num_edges * index_bytes +
        num_edges * feature_dim * 4 +
        num_nodes * feature_dim * 4
    )


def compute_throughput_and_bandwidth(
    num_nodes: int,
    num_edges: int,
    feature_dim: int,
    mean_latency_ms: float,
    index_bytes: int = 8,
) -> Dict[str, Any]:
    """
    Calculates execution throughput (nodes/sec, edges/sec) and model-estimated bandwidth (GB/s).

    IMPORTANT SCIENTIFIC DISTINCTION:
    - compulsory_bandwidth_gb_s: theoretical minimum traffic assuming perfect cache reuse.
    - algorithmic_bandwidth_gb_s: un-cached edge traversal traffic (M * D * 4 read).
    - effective_bandwidth_gb_s: alias to algorithmic_bandwidth_gb_s for backward compatibility.

    NOTE: These are MODEL-DERIVED ESTIMATES of algorithmic traffic, NOT measured hardware
    DRAM bandwidth counters from hardware performance profilers (e.g. Nsight/perf).
    """
    if mean_latency_ms <= 0:
        return {
            "nodes_per_sec": 0.0,
            "edges_per_sec": 0.0,
            "compulsory_traffic_bytes": 0,
            "algorithmic_traffic_bytes": 0,
            "compulsory_bandwidth_gb_s": 0.0,
            "algorithmic_bandwidth_gb_s": 0.0,
            "effective_bandwidth_gb_s": 0.0,
            "bandwidth_nature": "ESTIMATED_ALGORITHMIC_MODEL (Uncached CSR traversal model; not measured hardware DRAM counter)",
        }

    duration_sec = mean_latency_ms / 1000.0
    nodes_per_sec = float(num_nodes) / duration_sec
    edges_per_sec = float(num_edges) / duration_sec

    compulsory_bytes = estimate_compulsory_traffic_bytes(num_nodes, num_edges, feature_dim, index_bytes=index_bytes)
    algorithmic_bytes = estimate_algorithmic_traffic_bytes(num_nodes, num_edges, feature_dim, index_bytes=index_bytes)

    compulsory_gb_s = (compulsory_bytes / 1.0e9) / duration_sec
    algorithmic_gb_s = (algorithmic_bytes / 1.0e9) / duration_sec

    return {
        "nodes_per_sec": round(nodes_per_sec, 2),
        "edges_per_sec": round(edges_per_sec, 2),
        "compulsory_traffic_bytes": compulsory_bytes,
        "algorithmic_traffic_bytes": algorithmic_bytes,
        "compulsory_bandwidth_gb_s": round(compulsory_gb_s, 4),
        "algorithmic_bandwidth_gb_s": round(algorithmic_gb_s, 4),
        "effective_bandwidth_gb_s": round(algorithmic_gb_s, 4),
        "bandwidth_nature": "ESTIMATED_ALGORITHMIC_MODEL (Uncached CSR traversal model; not measured hardware DRAM counter)",
    }
