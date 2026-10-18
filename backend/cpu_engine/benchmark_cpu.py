"""
CPU Benchmark Harness — FraudGraph Intelligence (FGI)
Executes reproducible, high-precision timing measurements of CPU graph aggregation.

Adheres strictly to the FGI Experiment Protocol:
- Monotonic high-resolution timing (time.perf_counter_ns).
- Graph loading/generation isolated from timing.
- Configurable warm-up iterations (default: 5) and recorded repetitions (default: 20).
- Detailed statistical reporting (mean, std, median, p95, min, max).
- Throughput definitions: nodes/sec and directed CSR adjacency entries/sec.
- Automatic output export to experiments/results/*.json with 'MEASURED' status.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.cpu_engine.aggregators import (
    AggregationOp,
    aggregate_neighbors,
)
from backend.cpu_engine.csr_graph import CSRGraph


def get_cpu_info() -> Dict[str, Any]:
    """Retrieves detailed CPU model and environment information on Windows/Linux."""
    cpu_name = platform.processor() or "Unknown CPU"
    # Attempt PowerShell CIM query on Windows for friendly CPU name
    if platform.system() == "Windows":
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

    return {
        "cpu_model": cpu_name,
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "logical_cores": os.cpu_count() or 1,
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def benchmark_graph_aggregation(
    graph: CSRGraph,
    graph_name: str,
    operator: str = "mean",
    method: str = "reference",
    warmup_iters: int = 5,
    repeat_iters: int = 20,
    verify_against_ref: bool = True,
) -> Dict[str, Any]:
    """
    Executes a high-precision benchmark on a given graph and operator.
    
    Returns:
        Structured observation dictionary conforming to FGI experiment schema.
    """
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    feature_dim = graph.feature_dim
    stats = graph.get_stats()

    # 1. Warm-up phase
    for _ in range(warmup_iters):
        _ = aggregate_neighbors(graph, op=operator, method=method)

    # 2. Measured repetitions phase
    latencies_ms: List[float] = []
    output = None
    for _ in range(repeat_iters):
        t0 = time.perf_counter_ns()
        output = aggregate_neighbors(graph, op=operator, method=method)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    latencies_arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(latencies_arr))
    std_ms = float(np.std(latencies_arr, ddof=1)) if repeat_iters > 1 else 0.0
    p50_ms = float(np.median(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))

    mean_sec = mean_ms / 1000.0
    nodes_per_sec = float(num_nodes / mean_sec) if mean_sec > 0 else 0.0
    edges_per_sec = float(num_edges / mean_sec) if mean_sec > 0 else 0.0

    # 3. Numerical verification
    verif_status = "PASSED"
    max_abs_err = 0.0
    if verify_against_ref and method != "reference":
        ref_out = aggregate_neighbors(graph, op=operator, method="reference")
        max_abs_err = float(np.max(np.abs(output - ref_out)))
        if max_abs_err > 1e-5:
            verif_status = "FAILED_TOLERANCE"
    elif output is not None and not np.all(np.isfinite(output)):
        verif_status = "FAILED_NON_FINITE"

    # Memory footprint in MB
    footprint_mb = stats.get("memory_footprint_mb", 0.0)
    output_footprint_mb = float((num_nodes * feature_dim * 4) / (1024 * 1024))

    return {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "feature_dim": feature_dim,
        "is_directed": graph.is_directed,
        "operator": operator,
        "method": method,
        "warmup_iterations": warmup_iters,
        "recorded_repetitions": repeat_iters,
        "trial_times_ms": [round(t, 4) for t in latencies_ms],
        "latency_stats_ms": {
            "mean": round(mean_ms, 4),
            "std": round(std_ms, 4),
            "median_p50": round(p50_ms, 4),
            "p95": round(p95_ms, 4),
            "min": round(min_ms, 4),
            "max": round(max_ms, 4),
        },
        "throughput": {
            "nodes_per_second": round(nodes_per_sec, 2),
            "edges_per_second": round(edges_per_sec, 2),
        },
        "memory_footprint_mb": {
            "graph": round(footprint_mb, 4),
            "output": round(output_footprint_mb, 4),
            "total": round(footprint_mb + output_footprint_mb, 4),
        },
        "numerical_verification": {
            "status": verif_status,
            "max_absolute_error": max_abs_err,
            "tolerance_threshold": 1e-5,
            "nan_detected": bool(output is not None and np.any(np.isnan(output))),
        },
    }


def run_cpu_benchmarks(
    fixtures: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    warmup: int = 5,
    repeat: int = 20,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Runs full matrix of CPU benchmarks across specified fixtures and operators."""
    if fixtures is None:
        fixtures = ["toy_graph_5nodes.npz", "synthetic_tier_a_10k.npz"]
    if operators is None:
        operators = ["mean", "sum", "max"]
    if methods is None:
        methods = ["reference", "vectorized"]

    hardware = get_cpu_info()
    timestamp = datetime.datetime.now().isoformat()
    experiment_id = f"EXP-CPU-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("=" * 80)
    print(f" FRAUDGRAPH INTELLIGENCE — CPU BENCHMARK HARNESS")
    print(f" Experiment ID: {experiment_id} | State: MEASURED")
    print(f" CPU: {hardware['cpu_model']} ({hardware['logical_cores']} cores)")
    print(f" OS: {hardware['os']} | Python: {hardware['python_version']}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    fixtures_dir = Path("data/fixtures")

    for fix_name in fixtures:
        fix_path = fixtures_dir / fix_name
        if not fix_path.exists():
            print(f"[!] Fixture {fix_path} not found. Skipping.")
            continue

        print(f"\n[+] Loading graph fixture: {fix_name}")
        graph = CSRGraph.load(fix_path)
        print(f"    Nodes: {graph.num_nodes:,} | Edges: {graph.num_edges:,} | Dim: {graph.feature_dim} | Directed: {graph.is_directed}")

        for op in operators:
            for method in methods:
                # Max is only reference
                if op == "max" and method == "vectorized":
                    continue

                bench = benchmark_graph_aggregation(
                    graph=graph,
                    graph_name=fix_name,
                    operator=op,
                    method=method,
                    warmup_iters=warmup,
                    repeat_iters=repeat,
                )
                results.append(bench)
                stats = bench["latency_stats_ms"]
                th = bench["throughput"]
                verif = bench["numerical_verification"]

                print(
                    f"    [{op.upper():4s} | {method:10s}]  "
                    f"Mean: {stats['mean']:8.3f} ms +/- {stats['std']:6.3f}  |  "
                    f"p50: {stats['median_p50']:8.3f} ms  |  "
                    f"Throughput: {th['edges_per_second']:12,.0f} edges/s  |  "
                    f"Status: {verif['status']}"
                )

    payload = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "measurement_state": "MEASURED",
        "timestamp": timestamp,
        "hardware": hardware,
        "benchmark_parameters": {
            "warmup_iterations": warmup,
            "recorded_repetitions": repeat,
            "operators": operators,
            "methods": methods,
            "fixtures": fixtures,
        },
        "observations": results,
    }

    if save_results:
        results_dir = Path("experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_file = results_dir / f"{experiment_id.lower()}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[OK] Benchmark observations written to: {out_file}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGI CPU Benchmark Runner")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=20, help="Number of recorded repetitions")
    parser.add_argument("--no-save", action="store_true", help="Do not write result JSON to disk")
    args = parser.parse_args()

    run_cpu_benchmarks(
        warmup=args.warmup,
        repeat=args.repeat,
        save_results=not args.no_save,
    )
