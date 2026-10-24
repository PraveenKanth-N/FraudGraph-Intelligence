"""
Benchmark Reporting & Table Generator — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Generates:
1. summary_table.csv: Comprehensive tabular export of all recorded workload-engine combinations.
2. session_report.md: Academic-grade Markdown report detailing hardware environment,
   workload configurations, measured timings, throughputs, verification outcomes, and timing boundaries.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.benchmark_engine.benchmark_matrix import ExecutionStatus
from backend.benchmark_engine.experiment_manager import ExperimentSession


class BenchmarkReportGenerator:
    """
    Generates summary tabular and markdown reports for an experiment session.
    """

    def __init__(self, session: ExperimentSession):
        self.session = session
        self.session_dir = session.session_dir
        self.raw_dir = session.raw_dir

    def load_all_run_records(self) -> List[Dict[str, Any]]:
        """Reads all individual raw JSON run files from raw_measurements/."""
        records: List[Dict[str, Any]] = []
        if not self.raw_dir.exists():
            return records

        for json_path in sorted(self.raw_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records.append(data)
            except Exception:
                pass
        return records

    def generate_reports(self) -> Tuple[Path, Path]:
        """
        Generates both summary_table.csv and session_report.md in session directory.
        Returns: (csv_path, markdown_path)
        """
        records = self.load_all_run_records()
        csv_path = self.generate_csv_table(records)
        md_path = self.generate_markdown_report(records)
        return csv_path, md_path

    def generate_csv_table(self, records: List[Dict[str, Any]]) -> Path:
        """Writes summary_table.csv."""
        csv_path = self.session_dir / "summary_table.csv"

        fieldnames = [
            "workload_name",
            "operator",
            "num_nodes",
            "num_edges",
            "feature_dim",
            "topology",
            "engine",
            "timing_boundary",
            "measurement_state",
            "memory_guard_status",
            "timing_stability",
            "mean_latency_ms",
            "std_latency_ms",
            "median_p50_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "cv_percent",
            "nodes_per_sec",
            "edges_per_sec",
            "compulsory_traffic_bytes",
            "algorithmic_traffic_bytes",
            "compulsory_bandwidth_gb_s",
            "algorithmic_bandwidth_gb_s",
            "effective_bandwidth_gb_s",
            "verification_status",
            "max_absolute_error",
            "speedup_vs_cpu_ref",
        ]

        # First, find CPU reference mean latency for each workload to compute speedup
        ref_timings: Dict[str, float] = {}
        for r in records:
            if r.get("engine") == "cpu_reference" and r.get("measurement_state") == ExecutionStatus.MEASURED.value:
                w_name = r.get("workload_name", "")
                ref_timings[w_name] = r.get("timing_results_ms", {}).get("kernel_mean", 0.0)

        rows: List[Dict[str, Any]] = []
        for r in records:
            w_name = r.get("workload_name", "")
            g_cfg = r.get("graph_configuration", {})
            timing = r.get("timing_results_ms", {})
            tp_bw = r.get("throughput_and_bandwidth", {})
            verif = r.get("numerical_verification", {})

            mean_ms = timing.get("kernel_mean", 0.0)
            state = r.get("measurement_state", "")

            # Compute speedup vs cpu_reference if both are measured
            ref_ms = ref_timings.get(w_name, 0.0)
            if state == ExecutionStatus.MEASURED.value and ref_ms > 0 and mean_ms > 0:
                speedup_str = f"{ref_ms / mean_ms:.2f}x"
            else:
                speedup_str = "N/A"

            row = {
                "workload_name": w_name,
                "operator": g_cfg.get("operator", ""),
                "num_nodes": g_cfg.get("num_nodes", 0),
                "num_edges": g_cfg.get("num_edges", 0),
                "feature_dim": g_cfg.get("feature_dim", 0),
                "topology": g_cfg.get("topology", ""),
                "engine": r.get("engine", ""),
                "timing_boundary": r.get("timing_boundary", ""),
                "measurement_state": state,
                "memory_guard_status": r.get("memory_guard", {}).get("guard_status", "UNKNOWN"),
                "timing_stability": timing.get("stability_label", "STABLE"),
                "mean_latency_ms": round(mean_ms, 4),
                "std_latency_ms": round(timing.get("std", 0.0), 4),
                "median_p50_ms": round(timing.get("median_p50", 0.0), 4),
                "p95_ms": round(timing.get("p95", 0.0), 4),
                "min_ms": round(timing.get("min", 0.0), 4),
                "max_ms": round(timing.get("max", 0.0), 4),
                "cv_percent": round(timing.get("cv_percent", 0.0), 2),
                "nodes_per_sec": round(tp_bw.get("nodes_per_sec", 0.0), 1),
                "edges_per_sec": round(tp_bw.get("edges_per_sec", 0.0), 1),
                "compulsory_traffic_bytes": tp_bw.get("compulsory_traffic_bytes", 0),
                "algorithmic_traffic_bytes": tp_bw.get("algorithmic_traffic_bytes", 0),
                "compulsory_bandwidth_gb_s": round(tp_bw.get("compulsory_bandwidth_gb_s", 0.0), 4),
                "algorithmic_bandwidth_gb_s": round(tp_bw.get("algorithmic_bandwidth_gb_s", 0.0), 4),
                "effective_bandwidth_gb_s": round(tp_bw.get("effective_bandwidth_gb_s", 0.0), 4),
                "verification_status": verif.get("status", "N/A"),
                "max_absolute_error": verif.get("max_absolute_error", "N/A"),
                "speedup_vs_cpu_ref": speedup_str,
            }
            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    def generate_markdown_report(self, records: List[Dict[str, Any]]) -> Path:
        """Writes session_report.md."""
        md_path = self.session_dir / "session_report.md"
        hw = self.session.hardware
        git = self.session.git_info

        # Partition records
        measured = [r for r in records if r.get("measurement_state") == ExecutionStatus.MEASURED.value]
        skipped = [r for r in records if "SKIPPED" in r.get("measurement_state", "")]
        failed = [r for r in records if "FAILED" in r.get("measurement_state", "")]

        lines: List[str] = [
            f"# FraudGraph Intelligence — Phase 8 Benchmark Session Report",
            f"",
            f"**Session ID:** `{self.session.session_id}`  ",
            f"**Profile:** `{self.session.profile_name}`  ",
            f"**Execution Status:** `{self.session.status}`  ",
            f"**Timestamp:** `{self.session.start_time}`  ",
            f"",
            f"---",
            f"",
            f"## 1. System & Environment Metadata",
            f"",
            f"- **Git Commit:** `{git.get('commit_hash', 'unknown')}` (branch: `{git.get('branch', 'unknown')}`)  ",
            f"- **Host Platform:** {hw.get('platform', 'unknown')}  ",
            f"- **Python Version:** {hw.get('python_version', 'unknown')} | **NumPy:** {hw.get('numpy_version', 'unknown')}  ",
            f"- **CPU Model:** {hw['cpu'].get('model', 'unknown')} ({hw['cpu'].get('logical_cores', 1)} logical cores)  ",
            f"- **Host RAM:** {hw['cpu'].get('host_ram_total_mb', 0):,.0f} MB total ({hw['cpu'].get('host_ram_available_mb', 0):,.0f} MB available at preflight)  ",
            f"- **GPU Model:** {hw['gpu'].get('device_name', 'None detected')} (Driver: {hw['gpu'].get('driver_version', 'N/A')}, VRAM: {hw['gpu'].get('total_vram_mb', 0):,.0f} MB)  ",
            f"",
            f"---",
            f"",
            f"## 2. Benchmark Execution Summary",
            f"",
            f"| Metric | Count | Description |",
            f"| :--- | :--- | :--- |",
            f"| **Total Recorded Runs** | `{len(records)}` | Total workload-engine instances recorded |",
            f"| **Measured Runs** | `{len(measured)}` | Completed executions meeting numerical tolerances |",
            f"| **Skipped Runs** | `{len(skipped)}` | Bypassed due to toolchain or OOM guards |",
            f"| **Failed Runs** | `{len(failed)}` | Tolerance, non-finite, or execution errors |",
            f"",
            f"---",
            f"",
            f"## 3. Measured Performance Results",
            f"",
            f"| Workload | Operator | Engine | Timing Boundary | Latency (ms) | Throughput (nodes/s) | Bandwidth (GB/s) | Verif Status |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in measured:
            w_name = r.get("workload_name", "")
            g_cfg = r.get("graph_configuration", {})
            timing = r.get("timing_results_ms", {})
            tp = r.get("throughput_and_bandwidth", {})
            verif = r.get("numerical_verification", {})

            mean_ms = timing.get("kernel_mean", 0.0)
            std_ms = timing.get("std", 0.0)
            nodes_s = tp.get("nodes_per_sec", 0.0)
            bw = tp.get("effective_bandwidth_gb_s", 0.0)
            status = verif.get("status", "MEASURED")

            lines.append(
                f"| `{w_name}` | `{g_cfg.get('operator')}` | `{r.get('engine')}` | "
                f"`{r.get('timing_boundary')}` | {mean_ms:.3f} ± {std_ms:.3f} | "
                f"{nodes_s:,.1f} | {bw:.3f} | `{status}` |"
            )

        if skipped:
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 4. Skipped Implementation Runs",
                f"",
                f"The following planned engine implementations were safely bypassed:",
                f"",
                f"| Workload | Engine | Skip State | Reason |",
                f"| :--- | :--- | :--- | :--- |",
            ])
            for r in skipped:
                lines.append(
                    f"| `{r.get('workload_name')}` | `{r.get('engine')}` | "
                    f"`{r.get('measurement_state')}` | {r.get('skip_or_failure_reason', 'Bypassed')} |"
                )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Timing Boundary & Scientific Integrity Disclosures",
            f"",
            f"1. **Boundary Isolation:** CPU measurements use high-resolution host monotonic timers (`time.perf_counter_ns`), "
            f"capturing CPU kernel execution and intermediate Python function dispatch. GPU measurements (when executed on CUDA devices) "
            f"use device-side CUDA events (`cudaEventElapsedTime`) capturing pure kernel execution time.",
            f"2. **Zero Metric Fabrication:** If CUDA compilers (`nvcc`) or compiled binaries are unavailable on the local host, "
            f"GPU runs are explicitly recorded as `SKIPPED_CUDA_UNAVAILABLE`. No synthetic or zero latency numbers are accepted as measurements.",
            f"3. **Tolerance Enforcement:** All measurements undergo operator-specific validation against the Phase 4 CPU reference. "
            f"Any deviation exceeding operator tolerances causes immediate status rejection (`FAILED_TOLERANCE`).",
            f"",
            f"Generated automatically by FraudGraph Intelligence (FGI-MOD-08).",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return md_path
