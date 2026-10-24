#!/usr/bin/env python3
"""
Automated Google Colab GPU Benchmark and Validation Runner — Phase 6 (Experiment 6.1)
Executes within a Google Colab GPU environment (e.g. NVIDIA Tesla T4 / V100 / A100).
1. Detects and logs physical GPU hardware, driver, and CUDA compiler versions.
2. Compiles both Baseline and Warp-Coalesced (Experiment 6.1) CUDA engines.
3. Executes correctness verification gates on toy fixtures (tol <= 1e-5).
4. Runs high-precision multi-repetition benchmark matrices on Tier A (10K nodes, 95K edges, D=32).
5. Computes speedups of Warp-Coalesced vs Baseline under identical hardware conditions.
6. Emits machine-readable JSON observation files with 'MEASURED' state.
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
    print(" FRAUDGRAPH INTELLIGENCE — COLAB GPU RUNNER (PHASE 6: EXP 6.1, 6.2 & 6.3)")
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
    baseline_bin = cuda_dir / "bin" / "fgi_baseline_runner"
    coalesced_bin = cuda_dir / "bin" / "fgi_coalesced_runner"
    streams_bin = cuda_dir / "bin" / "fgi_streams_runner"
    degree_aware_bin = cuda_dir / "bin" / "fgi_degree_aware_runner"

    # 2. Compile All Four CUDA Executables
    print("\n[+] Compiling Baseline, Warp-Coalesced, Streams, and Degree-Aware CUDA Engines:")
    run_command("find cuda -exec touch {} +", check=False)
    run_command("make -C cuda clean")
    run_command("make -C cuda")

    for b in [baseline_bin, coalesced_bin, streams_bin, degree_aware_bin]:
        if not b.exists():
            print(f"[FATAL] Binary not found after compilation: {b}", file=sys.stderr)
            sys.exit(1)
        print(f"[✓] Verified compiled binary: {b}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    toy_dir = Path("data/toy_graph_5nodes")
    tier_a_dir = Path("data/synthetic_tier_a_10k")
    skewed_dir = Path("data/synthetic_skewed_10k")
    operators = ["mean", "sum", "max"]

    engine_list = [
        (baseline_bin, "baseline"),
        (coalesced_bin, "coalesced"),
        (streams_bin, "streams"),
        (degree_aware_bin, "degree_aware"),
    ]

    # 3. Correctness Gate (Toy Graph Fixture)
    print("\n" + "=" * 80)
    print(" STEP 1: CORRECTNESS VERIFICATION GATE (toy_graph_5nodes)")
    print("=" * 80)

    for bin_target, label in engine_list:
        print(f"\n--- Testing Engine: {label.upper()} ({bin_target.name}) ---")
        for op in operators:
            ref_bin = toy_dir / f"ref_{op}.bin"
            json_out = results_dir / f"val_toy_{label}_{op}.json"

            cmd = (
                f"./{bin_target} --graph {toy_dir} --op {op} --warmup 2 --repeat 5 "
                f"--validate {ref_bin} --export-json {json_out}"
            )
            run_command(cmd)

            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)
                verif = data.get("numerical_verification", {})
                if verif.get("status") != "PASSED":
                    print(f"[FATAL] Correctness gate failed for {label} operator '{op}': {verif}", file=sys.stderr)
                    sys.exit(2)
                print(f"[✓] {label.upper()} '{op}' PASSED (Max Abs Error: {verif.get('max_absolute_error'):.2e})")

    # 4. Benchmark Matrix (Tier A 10K Node Graph)
    print("\n" + "=" * 80)
    print(" STEP 2: BENCHMARK MATRIX (synthetic_tier_a_10k, N=10,000, M=94,945, D=32)")
    print("=" * 80)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_records = {}

    for bin_target, label in engine_list:
        print(f"\n--- Benchmarking Engine: {label.upper()} on Tier A (5 warmups, 20 repetitions) ---")
        perf_records[label] = {}

        for op in operators:
            ref_bin = tier_a_dir / f"ref_{op}.bin"
            exp_id = f"exp_gpu_{label}_tier_a_{op}_{timestamp_str}"
            json_out = results_dir / f"{exp_id}.json"

            cmd = (
                f"./{bin_target} --graph {tier_a_dir} --op {op} --warmup 5 --repeat 20 "
                f"--validate {ref_bin} --export-json {json_out}"
            )
            run_command(cmd)

            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)
                perf = data.get("performance_metrics", {})
                timings = data.get("timing_results_ms", {})
                verif = data.get("numerical_verification", {})

                k_mean = timings.get("kernel_mean", 0.0)
                e2e_mean = timings.get("end_to_end_mean", 0.0)
                h2d_mean = timings.get("h2d_transfer_mean", 0.0)
                d2h_mean = timings.get("d2h_transfer_mean", 0.0)

                perf_records[label][op] = {
                    "kernel_mean": k_mean,
                    "end_to_end_mean": e2e_mean,
                    "h2d_mean": h2d_mean,
                    "d2h_mean": d2h_mean,
                    "throughput": perf.get("edges_per_second", 0.0),
                    "bandwidth": perf.get("effective_bandwidth_gbps", 0.0),
                    "verif": verif.get("status"),
                    "error": verif.get("max_absolute_error", 0.0),
                }

                print(f"    Operator:         {op.upper()} ({label})")
                print(f"    Kernel Mean:      {k_mean:.3f} ms +/- {timings.get('kernel_std', 0.0):.3f} ms")
                print(f"    Kernel p50:       {timings.get('kernel_p50', 0.0):.3f} ms")
                print(f"    H2D Transfer:     {h2d_mean:.3f} ms")
                print(f"    D2H Transfer:     {d2h_mean:.3f} ms")
                print(f"    End-to-End:       {e2e_mean:.3f} ms")
                print(f"    Throughput:       {perf.get('edges_per_second', 0.0):,.0f} edges/sec")
                if "estimated_compulsory_bandwidth_gbps" in perf:
                    print(f"    Est. Comp. BW:    {perf['estimated_compulsory_bandwidth_gbps']:.2f} GB/s (unique footprint)")
                if "pcie_h2d_bandwidth_gbps" in perf:
                    print(f"    PCIe H2D BW:      {perf['pcie_h2d_bandwidth_gbps']:.2f} GB/s | D2H: {perf.get('pcie_d2h_bandwidth_gbps', 0.0):.2f} GB/s")
                print(f"    Validation:       {verif.get('status')} (Max Abs Error: {verif.get('max_absolute_error', 0.0):.2e})")

    # 5. Benchmark Matrix on Skewed Stress Graph (if present)
    perf_skewed = {}
    if skewed_dir.exists():
        print("\n" + "=" * 80)
        print(" STEP 3: BENCHMARK MATRIX (synthetic_skewed_10k, N=10,000, M=94,970, Max Deg=6,340)")
        print("=" * 80)

        for bin_target, label in engine_list:
            print(f"\n--- Benchmarking Engine: {label.upper()} on Skewed Graph (5 warmups, 20 repetitions) ---")
            perf_skewed[label] = {}

            for op in operators:
                ref_bin = skewed_dir / f"ref_{op}.bin"
                exp_id = f"exp_gpu_{label}_skewed_{op}_{timestamp_str}"
                json_out = results_dir / f"{exp_id}.json"

                cmd = (
                    f"./{bin_target} --graph {skewed_dir} --op {op} --warmup 5 --repeat 20 "
                    f"--validate {ref_bin} --export-json {json_out}"
                )
                run_command(cmd)

                with open(json_out, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    perf = data.get("performance_metrics", {})
                    timings = data.get("timing_results_ms", {})
                    verif = data.get("numerical_verification", {})

                    k_mean = timings.get("kernel_mean", 0.0)
                    e2e_mean = timings.get("end_to_end_mean", 0.0)
                    h2d_mean = timings.get("h2d_transfer_mean", 0.0)
                    d2h_mean = timings.get("d2h_transfer_mean", 0.0)

                    perf_skewed[label][op] = {
                        "kernel_mean": k_mean,
                        "end_to_end_mean": e2e_mean,
                        "h2d_mean": h2d_mean,
                        "d2h_mean": d2h_mean,
                        "throughput": perf.get("edges_per_second", 0.0),
                        "verif": verif.get("status"),
                        "error": verif.get("max_absolute_error", 0.0),
                    }
                    print(f"    Operator:         {op.upper()} ({label})")
                    print(f"    Kernel Mean:      {k_mean:.3f} ms")
                    print(f"    End-to-End:       {e2e_mean:.3f} ms")
                    print(f"    Throughput:       {perf.get('edges_per_second', 0.0):,.0f} edges/sec")
                    print(f"    Validation:       {verif.get('status')} (Max Abs Error: {verif.get('max_absolute_error', 0.0):.2e})")

    # 6. Comparative Speedup Summary
    print("\n" + "=" * 80)
    print(" STEP 4: COMPARATIVE SPEEDUP SUMMARY (TIER A CANONICAL GRAPH)")
    print("=" * 80)
    print(f"{'Operator':<8} | {'Engine':<13} | {'Kernel (ms)':<12} | {'Kernel Spdup':<13} | {'Transfer (ms)':<14} | {'E2E (ms)':<10} | {'E2E Spdup':<10}")
    print("-" * 92)
    for op in operators:
        b_k = perf_records["baseline"][op]["kernel_mean"]
        b_e2e = perf_records["baseline"][op]["end_to_end_mean"]
        b_xfer = perf_records["baseline"][op]["h2d_mean"] + perf_records["baseline"][op]["d2h_mean"]

        for eng in ["baseline", "coalesced", "streams", "degree_aware"]:
            ek = perf_records[eng][op]["kernel_mean"]
            ee = perf_records[eng][op]["end_to_end_mean"]
            ex = perf_records[eng][op]["h2d_mean"] + perf_records[eng][op]["d2h_mean"]
            k_spdup = f"{(b_k / ek if ek > 0 else 0):.2f}x"
            e_spdup = f"{(b_e2e / ee if ee > 0 else 0):.2f}x"
            lbl = eng.replace("_", " ").title()
            print(f"{op.upper() if eng == 'baseline' else '':<8} | {lbl:<13} | {ek:10.3f}   | {k_spdup:<13} | {ex:12.3f}   | {ee:8.3f} | {e_spdup:<10}")
        print("-" * 92)

    if perf_skewed:
        print("\n" + "=" * 80)
        print(" STEP 5: COMPARATIVE SPEEDUP SUMMARY (SKEWED DEGREE STRESS GRAPH)")
        print("=" * 80)
        print(f"{'Operator':<8} | {'Engine':<13} | {'Kernel (ms)':<12} | {'Kernel Spdup':<13} | {'Transfer (ms)':<14} | {'E2E (ms)':<10} | {'E2E Spdup':<10}")
        print("-" * 92)
        for op in operators:
            b_k = perf_skewed["baseline"][op]["kernel_mean"]
            b_e2e = perf_skewed["baseline"][op]["end_to_end_mean"]
            b_xfer = perf_skewed["baseline"][op]["h2d_mean"] + perf_skewed["baseline"][op]["d2h_mean"]

            for eng in ["baseline", "coalesced", "streams", "degree_aware"]:
                ek = perf_skewed[eng][op]["kernel_mean"]
                ee = perf_skewed[eng][op]["end_to_end_mean"]
                ex = perf_skewed[eng][op]["h2d_mean"] + perf_skewed[eng][op]["d2h_mean"]
                k_spdup = f"{(b_k / ek if ek > 0 else 0):.2f}x"
                e_spdup = f"{(b_e2e / ee if ee > 0 else 0):.2f}x"
                lbl = eng.replace("_", " ").title()
                print(f"{op.upper() if eng == 'baseline' else '':<8} | {lbl:<13} | {ek:10.3f}   | {k_spdup:<13} | {ex:12.3f}   | {ee:8.3f} | {e_spdup:<10}")
            print("-" * 92)

    # 6. Archive Results for Download
    zip_out = Path("fgi_phase6_gpu_results.zip")
    shutil.make_archive("fgi_phase6_gpu_results", "zip", results_dir)
    print(f"\n[OK] Benchmarks complete. Archived in: {zip_out.resolve()}")


if __name__ == "__main__":
    main()
