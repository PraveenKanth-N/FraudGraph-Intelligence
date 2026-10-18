#!/usr/bin/env python3
"""
Automated Google Colab GPU Benchmark and Validation Runner
Executes within a Google Colab GPU environment (e.g. NVIDIA T4 / V100 / A100).
1. Detects and logs physical GPU hardware, driver, and CUDA compiler versions.
2. Compiles the baseline CUDA C++ kernels.
3. Executes correctness verification against CPU reference outputs (tol <= 1e-5).
4. Runs high-precision multi-repetition benchmark matrix.
5. Emits machine-readable JSON observation files with 'MEASURED' state.
"""

import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run_command(cmd: str, check: bool = True) -> str:
    print(f"\n[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res.stdout


def main():
    print("=" * 80)
    print(" FRAUDGRAPH INTELLIGENCE — COLAB GPU RUNNER (PHASE 5 BASELINE)")
    print("=" * 80)

    # 1. Environment & Hardware Detection
    print("\n[+] Inspecting Host and GPU Environment:")
    try:
        run_command("nvidia-smi")
        run_command("nvcc --version")
    except Exception as e:
        print(f"[ERROR] GPU environment inspection failed: {e}", file=sys.stderr)
        sys.exit(1)

    cuda_dir = Path("cuda")
    bin_path = cuda_dir / "bin" / "fgi_baseline_runner"

    # 2. Compile Baseline CUDA Executable
    print("\n[+] Compiling Baseline CUDA Engine:")
    run_command("make -C cuda clean")
    run_command("make -C cuda")

    if not bin_path.exists():
        print(f"[FATAL] Binary not found after compilation: {bin_path}", file=sys.stderr)
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Correctness Gate (Toy Graph Fixture)
    print("\n" + "=" * 80)
    print(" STEP 1: CORRECTNESS VERIFICATION GATE (toy_graph_5nodes)")
    print("=" * 80)

    toy_dir = Path("data/toy_graph_5nodes")
    operators = ["mean", "sum", "max"]

    for op in operators:
        print(f"\n[*] Testing operator '{op}' on toy fixture...")
        ref_bin = toy_dir / f"ref_{op}.bin"
        json_out = results_dir / f"val_toy_{op}.json"

        cmd = (
            f"./{bin_path} --graph {toy_dir} --op {op} --warmup 2 --repeat 5 "
            f"--validate {ref_bin} --export-json {json_out}"
        )
        run_command(cmd)

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
            verif = data.get("numerical_verification", {})
            if verif.get("status") != "PASSED":
                print(f"[FATAL] Correctness gate failed for operator '{op}' on toy fixture: {verif}", file=sys.stderr)
                sys.exit(2)
            print(f"[✓] Operator '{op}' PASSED correctness gate (Max Abs Error: {verif.get('max_absolute_error'):.2e})")

    # 4. Benchmark Matrix (Tier A 10K Node Graph)
    print("\n" + "=" * 80)
    print(" STEP 2: BENCHMARK MATRIX (synthetic_tier_a_10k, N=10,000, M=94,945, D=32)")
    print("=" * 80)

    tier_a_dir = Path("data/synthetic_tier_a_10k")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_files = []

    for op in operators:
        print(f"\n[*] Running Benchmark for operator '{op}' (5 warmups, 20 repetitions)...")
        ref_bin = tier_a_dir / f"ref_{op}.bin"
        exp_id = f"exp_gpu_baseline_tier_a_{op}_{timestamp_str}"
        json_out = results_dir / f"{exp_id}.json"

        cmd = (
            f"./{bin_path} --graph {tier_a_dir} --op {op} --warmup 5 --repeat 20 "
            f"--validate {ref_bin} --export-json {json_out}"
        )
        run_command(cmd)
        experiment_files.append(json_out)

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
            perf = data.get("performance_metrics", {})
            timings = data.get("timing_results_ms", {})
            verif = data.get("numerical_verification", {})

            print(f"    Operator:         {op.upper()}")
            print(f"    Kernel Mean:      {timings.get('kernel_mean', 0.0):.3f} ms +/- {timings.get('kernel_std', 0.0):.3f} ms")
            print(f"    Kernel p50:       {timings.get('kernel_p50', 0.0):.3f} ms")
            print(f"    H2D Transfer:     {timings.get('h2d_transfer_mean', 0.0):.3f} ms")
            print(f"    D2H Transfer:     {timings.get('d2h_transfer_mean', 0.0):.3f} ms")
            print(f"    End-to-End:       {timings.get('end_to_end_mean', 0.0):.3f} ms")
            print(f"    Throughput:       {perf.get('edges_per_second', 0.0):,.0f} edges/sec")
            print(f"    Validation:       {verif.get('status')} (Max Abs Error: {verif.get('max_absolute_error', 0.0):.2e})")

    # 5. Archive Results for Download
    zip_out = Path("fgi_phase5_gpu_results.zip")
    shutil.make_archive("fgi_phase5_gpu_results", "zip", results_dir)
    print(f"\n[OK] All benchmarks completed successfully. Results archived in: {zip_out.resolve()}")


if __name__ == "__main__":
    main()
