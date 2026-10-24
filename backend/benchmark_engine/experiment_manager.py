"""
Experiment Session & Provenance Manager — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Manages:
1. Isolated session directories: experiments/results/benchmark_runs/session_{session_id}/
2. Atomic JSON file serialization via temporary .tmp swap.
3. SHA-256 fixture checksumming, git environment provenance, and hardware telemetry capture.
4. Structured session manifests supporting interrupted / partial session recovery.
5. Schema compatibility with existing FGI Colab ingestion protocols.
"""

from dataclasses import asdict, dataclass, field
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from backend.benchmark_engine.benchmark_matrix import ExecutionStatus, WorkloadConfig
from backend.benchmark_engine.benchmark_telemetry import get_hardware_telemetry


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 cryptographic digest of a file."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info(workspace_root: Path) -> Dict[str, str]:
    """Retrieves current git commit hash and branch if available."""
    info = {"commit_hash": "unknown", "branch": "unknown", "dirty": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(workspace_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        info["commit_hash"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(workspace_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "-C", str(workspace_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        info["dirty"] = "true" if status else "false"
    except Exception:
        pass

    return info


def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Writes dictionary to JSON file atomically by writing to .tmp and replacing.
    Prevents corrupt files on process abort or system failure.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, file_path)


@dataclass
class SessionManifest:
    """Structured representation of an experiment session manifest."""
    schema_version: str
    session_id: str
    profile_name: str
    start_time: str
    end_time: Optional[str]
    session_status: str
    summary_counts: Dict[str, int]
    provenance: Dict[str, Any]
    runs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResultRecord:
    """Data record representing the outcome of a single engine run on a workload."""
    workload_name: str
    engine: str
    timing_boundary: str
    measurement_state: str  # from ExecutionStatus
    operator: str
    num_nodes: int
    num_edges: int
    feature_dim: int
    warmup_iters: int
    repeat_iters: int
    trial_times_ms: List[float]
    latency_stats_ms: Dict[str, Any]
    throughput_and_bandwidth: Dict[str, Any]
    numerical_verification: Dict[str, Any]
    memory_footprint_mb: Dict[str, float]
    memory_guard: Dict[str, Any] = field(default_factory=dict)
    skip_or_failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentSession:
    """
    Orchestrates an isolated benchmark experiment session.

    Creates:
        experiments/results/benchmark_runs/session_{session_id}/
            ├── session_manifest.json
            ├── raw_measurements/
            │   ├── {workload}_{engine}.json
            │   └── ...
            ├── summary_table.csv
            └── session_report.md
    """

    def __init__(
        self,
        profile_name: str,
        session_id: Optional[str] = None,
        workspace_root: Optional[Path] = None,
        base_output_dir: Optional[Path] = None,
    ):
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parents[2]
        else:
            self.workspace_root = Path(workspace_root)

        if session_id is None:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"{now_str}_{profile_name}"
        else:
            self.session_id = session_id

        self.profile_name = profile_name
        self.start_time = datetime.datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.status: str = "IN_PROGRESS"

        if base_output_dir is None:
            self.base_output_dir = self.workspace_root / "experiments" / "results" / "benchmark_runs"
        else:
            self.base_output_dir = Path(base_output_dir)

        self.session_dir = self.base_output_dir / f"session_{self.session_id}"
        self.raw_dir = self.session_dir / "raw_measurements"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.git_info = get_git_info(self.workspace_root)
        self.hardware = get_hardware_telemetry()
        self.fixture_hashes: Dict[str, str] = {}
        self.results_index: List[Dict[str, Any]] = []

        # If manifest already exists, load existing state; otherwise write initial manifest
        manifest_path = self.session_dir / "session_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                self.start_time = m.get("start_time", self.start_time)
                self.end_time = m.get("end_time")
                self.status = m.get("session_status", "IN_PROGRESS")
                self.results_index = m.get("runs", [])
                prov = m.get("provenance", {})
                self.fixture_hashes = prov.get("fixture_checksums", {})
                if "hardware" in prov:
                    self.hardware = prov["hardware"]
            except Exception:
                self.save_manifest()
        else:
            self.save_manifest()

    def record_fixture(self, fixture_rel_path: str) -> str:
        """Computes and records SHA-256 for a fixture file."""
        full_path = self.workspace_root / fixture_rel_path
        digest = compute_file_sha256(full_path)
        self.fixture_hashes[fixture_rel_path] = digest
        return digest

    def save_run_result(
        self,
        workload: WorkloadConfig,
        record: BenchmarkResultRecord,
    ) -> Path:
        """
        Saves individual run result to raw_measurements/ with full provenance and schema compliance.
        """
        filename = f"{workload.name}_{record.engine}.json"
        dest_path = self.raw_dir / filename

        # Record fixture hash if workload uses a fixture
        if workload.graph_fixture and workload.graph_fixture not in self.fixture_hashes:
            self.record_fixture(workload.graph_fixture)

        # Build schema-compliant JSON payload
        # Matches FGI Colab ingestion schema requirements
        payload = {
            "schema_version": "1.0.0",
            "experiment_id": f"EXP-{self.session_id}-{workload.name}-{record.engine}",
            "session_id": self.session_id,
            "measurement_state": record.measurement_state,
            "timestamp": datetime.datetime.now().isoformat(),
            "workload_name": workload.name,
            "engine": record.engine,
            "timing_boundary": record.timing_boundary,
            "skip_or_failure_reason": record.skip_or_failure_reason,
            "hardware": {
                "device_name": self.hardware["gpu"].get("device_name") or self.hardware["cpu"]["model"],
                "compute_capability": 8.0 if "cuda" in record.engine else "N/A (CPU)",
                "cpu_model": self.hardware["cpu"]["model"],
                "logical_cores": self.hardware["cpu"]["logical_cores"],
                "host_ram_mb": self.hardware["cpu"]["host_ram_total_mb"],
                "gpu_vram_mb": self.hardware["gpu"]["total_vram_mb"],
            },
            "graph_configuration": {
                "num_nodes": record.num_nodes,
                "num_edges": record.num_edges,
                "feature_dim": record.feature_dim,
                "operator": record.operator,
                "topology": workload.topology,
                "is_directed": workload.is_directed,
                "fixture_path": workload.graph_fixture,
                "fixture_sha256": self.fixture_hashes.get(workload.graph_fixture or "", ""),
            },
            "benchmark_parameters": {
                "warmup_iterations": record.warmup_iters,
                "recorded_repetitions": record.repeat_iters,
            },
            "timing_results_ms": {
                "kernel_mean": record.latency_stats_ms["mean"],
                "end_to_end_mean": record.latency_stats_ms["mean"],
                "std": record.latency_stats_ms["std"],
                "median_p50": record.latency_stats_ms["median_p50"],
                "p95": record.latency_stats_ms["p95"],
                "min": record.latency_stats_ms["min"],
                "max": record.latency_stats_ms["max"],
                "cv_percent": record.latency_stats_ms["cv_percent"],
                "is_stable": record.latency_stats_ms.get("is_stable", True),
                "stability_label": record.latency_stats_ms.get("stability_label", "STABLE"),
                "trial_times_ms": record.trial_times_ms,
            },
            "numerical_verification": record.numerical_verification,
            "throughput_and_bandwidth": record.throughput_and_bandwidth,
            "memory_footprint_mb": record.memory_footprint_mb,
            "memory_guard": record.memory_guard or {},
            "provenance": {
                "git_commit": self.git_info["commit_hash"],
                "git_branch": self.git_info["branch"],
                "git_dirty": self.git_info["dirty"],
                "python_version": self.hardware["python_version"],
                "numpy_version": self.hardware["numpy_version"],
            },
        }

        atomic_write_json(dest_path, payload)

        # Update session results index
        self.results_index.append({
            "workload_name": workload.name,
            "engine": record.engine,
            "measurement_state": record.measurement_state,
            "memory_guard_status": record.memory_guard.get("guard_status", "UNKNOWN") if record.memory_guard else "UNKNOWN",
            "mean_ms": record.latency_stats_ms["mean"],
            "std_ms": record.latency_stats_ms["std"],
            "nodes_per_sec": record.throughput_and_bandwidth.get("nodes_per_sec", 0.0),
            "effective_bandwidth_gb_s": record.throughput_and_bandwidth.get("effective_bandwidth_gb_s", 0.0),
            "verification_status": record.numerical_verification.get("status"),
            "file_name": filename,
        })

        self.save_manifest()
        return dest_path

    def finalize_session(self, status: str = "COMPLETED") -> None:
        """Marks the session as completed or partial and updates manifest."""
        self.status = status
        self.end_time = datetime.datetime.now().isoformat()
        self.save_manifest()

    def save_manifest(self) -> Path:
        """Writes session_manifest.json atomically."""
        manifest_path = self.session_dir / "session_manifest.json"

        completed = sum(1 for r in self.results_index if r["measurement_state"] == ExecutionStatus.MEASURED.value)
        skipped = sum(1 for r in self.results_index if "SKIPPED" in r["measurement_state"])
        failed = sum(1 for r in self.results_index if "FAILED" in r["measurement_state"])

        manifest_data = {
            "schema_version": "1.0.0",
            "session_id": self.session_id,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "session_status": self.status,
            "summary_counts": {
                "total_runs_recorded": len(self.results_index),
                "measured_runs": completed,
                "skipped_runs": skipped,
                "failed_runs": failed,
            },
            "provenance": {
                "git_commit": self.git_info["commit_hash"],
                "git_branch": self.git_info["branch"],
                "git_dirty": self.git_info["dirty"],
                "hardware": self.hardware,
                "fixture_checksums": self.fixture_hashes,
            },
            "runs": self.results_index,
        }

        atomic_write_json(manifest_path, manifest_data)
        return manifest_path
