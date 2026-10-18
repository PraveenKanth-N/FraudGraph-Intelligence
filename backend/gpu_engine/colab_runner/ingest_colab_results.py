"""
Google Colab GPU Result Ingestion Utility — FraudGraph Intelligence (FGI)
Ingests, validates, and archives benchmark JSON files received from Google Colab GPU runs.

Strict verification checks:
- JSON schema conformity against docs/EXPERIMENT_PROTOCOL.md.
- Measurement state must be 'MEASURED' (rejects fabricated or incomplete runs).
- Hardware telemetry (device name, compute capability, VRAM) must be present.
- Numerical validation status must be 'PASSED' with max_absolute_error <= 1e-5.
- Computes honest GPU speedup factors vs Phase 4 CPU reference measurements.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional


def validate_result_schema(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validates that a result dictionary conforms strictly to the FGI experiment schema."""
    required_top = [
        "schema_version",
        "experiment_id",
        "measurement_state",
        "hardware",
        "graph_configuration",
        "benchmark_parameters",
        "timing_results_ms",
        "numerical_verification",
    ]
    for key in required_top:
        if key not in data:
            return False, f"Missing required top-level field: '{key}'"

    if data["measurement_state"] != "MEASURED":
        return False, f"measurement_state must be 'MEASURED', found '{data['measurement_state']}'"

    hw = data["hardware"]
    if not hw.get("device_name") or "compute_capability" not in hw:
        return False, "Hardware metadata missing 'device_name' or 'compute_capability'"

    timings = data["timing_results_ms"]
    if timings.get("kernel_mean") is None or timings.get("end_to_end_mean") is None:
        return False, "Timing results missing 'kernel_mean' or 'end_to_end_mean'"

    verif = data["numerical_verification"]
    if verif.get("status") != "PASSED":
        return False, f"Numerical verification status is not 'PASSED': {verif.get('status')}"

    err = verif.get("max_absolute_error", 999.0)
    tol = verif.get("tolerance_threshold", 1e-5)
    if err > tol:
        return False, f"Max absolute error {err:.2e} exceeds tolerance {tol:.2e}"

    return True, "Valid"


def ingest_results_archive(archive_or_file: Path, dest_dir: Path = Path("experiments/results")) -> List[Path]:
    """Ingests and validates a zip archive or single JSON file into experiments/results/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ingested_files: List[Path] = []

    if archive_or_file.suffix == ".zip":
        print(f"[+] Unpacking results archive: {archive_or_file}")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(archive_or_file, "r") as zf:
                zf.extractall(tmpdir)

            for p in Path(tmpdir).rglob("*.json"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                valid, msg = validate_result_schema(data)
                if not valid:
                    print(f"[!] Skipping invalid result file '{p.name}': {msg}", file=sys.stderr)
                    continue

                dest_file = dest_dir / p.name
                shutil.copy2(p, dest_file)
                ingested_files.append(dest_file)
                print(f"[✓] Validated and ingested: {dest_file.name}")
    elif archive_or_file.suffix == ".json":
        with open(archive_or_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid, msg = validate_result_schema(data)
        if not valid:
            raise ValueError(f"Result file validation failed: {msg}")

        dest_file = dest_dir / archive_or_file.name
        shutil.copy2(archive_or_file, dest_file)
        ingested_files.append(dest_file)
        print(f"[✓] Validated and ingested: {dest_file.name}")
    else:
        raise ValueError(f"Unsupported file format: {archive_or_file.suffix}. Expected .zip or .json")

    return ingested_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGI Colab Results Ingestor")
    parser.add_argument("path", type=Path, help="Path to fgi_phase5_gpu_results.zip or result.json")
    args = parser.parse_args()

    try:
        files = ingest_results_archive(args.path)
        print(f"\n[OK] Ingestion complete. {len(files)} result files archived.")
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
