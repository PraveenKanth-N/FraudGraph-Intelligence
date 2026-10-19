"""
Audit & Speedup Calculation Utility for Phase 6 (Experiment 6.1: Warp-Coalesced Aggregation)
Reads raw ingested JSON telemetry from experiments/results/ and computes exact mathematical
comparisons, pure kernel speedups, end-to-end speedups, and compulsory vs. algorithmic memory bandwidths.
"""

import glob
import json
from pathlib import Path


def main():
    results_dir = Path("experiments/results")
    operators = ["mean", "sum", "max"]

    N = 10000
    M = 94945
    D = 32

    # Compulsory unique bytes: row_ptr + col_idx + unique features + output
    bytes_compulsory = (N + 1) * 4 + M * 4 + N * D * 4 + N * D * 4  # 2,979,784 B
    # Algorithmic memory traffic: row_ptr + col_idx + (M * D * 4) + output
    bytes_algorithmic = (N + 1) * 4 + M * 4 + M * D * 4 + N * D * 4  # 13,852,744 B

    print("=" * 100)
    print(" FRAUDGRAPH INTELLIGENCE — PHASE 6.1 (WARP-COALESCED) AUDIT & SPEEDUP REPORT")
    print(f" Target Graph: synthetic_tier_a_10k (N={N:,}, M={M:,}, D={D})")
    print(f" Compulsory DRAM Footprint: {bytes_compulsory / 1e6:.2f} MB ({bytes_compulsory:,} bytes)")
    print(f" Algorithmic Memory Traffic: {bytes_algorithmic / 1e6:.2f} MB ({bytes_algorithmic:,} bytes)")
    print("=" * 100)

    records = {}

    for op in operators:
        bf_matches = sorted(glob.glob(str(results_dir / f"exp_gpu_baseline_tier_a_{op}_20261019_*.json")))
        cf_matches = sorted(glob.glob(str(results_dir / f"exp_gpu_coalesced_tier_a_{op}_20261019_*.json")))

        if not bf_matches or not cf_matches:
            print(f"[!] Missing telemetry for operator {op}")
            continue

        with open(bf_matches[-1], "r", encoding="utf-8") as f:
            b_data = json.load(f)
        with open(cf_matches[-1], "r", encoding="utf-8") as f:
            c_data = json.load(f)

        records[op] = {
            "baseline": b_data,
            "coalesced": c_data,
        }

    print("\n--- Summary Table ---")
    print(f"{'Operator':<8} | {'Base Kernel':<12} | {'Opt Kernel':<12} | {'Kernel Speedup':<15} | {'Base E2E':<12} | {'Opt E2E':<12} | {'E2E Speedup':<12}")
    print("-" * 92)

    for op in operators:
        b = records[op]["baseline"]
        c = records[op]["coalesced"]

        bk = b["timing_results_ms"]["kernel_mean"]
        ck = c["timing_results_ms"]["kernel_mean"]
        k_sp = bk / ck

        be = b["timing_results_ms"]["end_to_end_mean"]
        ce = c["timing_results_ms"]["end_to_end_mean"]
        e_sp = be / ce

        print(f"{op.upper():<8} | {bk:8.5f} ms | {ck:8.5f} ms | {k_sp:13.4f}x | {be:8.5f} ms | {ce:8.5f} ms | {e_sp:10.4f}x")

    print("\n--- Detailed Operator Audit & Memory Bandwidth ---")
    for op in operators:
        b = records[op]["baseline"]
        c = records[op]["coalesced"]

        bk = b["timing_results_ms"]["kernel_mean"]
        ck = c["timing_results_ms"]["kernel_mean"]
        k_sp = bk / ck

        be = b["timing_results_ms"]["end_to_end_mean"]
        ce = c["timing_results_ms"]["end_to_end_mean"]
        e_sp = be / ce

        b_tp = b["performance_metrics"]["edges_per_second"]
        c_tp = c["performance_metrics"]["edges_per_second"]

        b_bw_comp = bytes_compulsory / (bk * 1e-3 * 1e9)
        c_bw_comp = bytes_compulsory / (ck * 1e-3 * 1e9)

        b_bw_algo = bytes_algorithmic / (bk * 1e-3 * 1e9)
        c_bw_algo = bytes_algorithmic / (ck * 1e-3 * 1e9)

        print(f"\n[Operator: {op.upper()}]")
        print(f"  Validation Status:       Baseline={b['numerical_verification']['status']} (err={b['numerical_verification']['max_absolute_error']}), Coalesced={c['numerical_verification']['status']} (err={c['numerical_verification']['max_absolute_error']})")
        print(f"  Kernel Latency:          Baseline={bk:.5f} ms (p50: {b['timing_results_ms']['kernel_p50']:.5f} ms) -> Coalesced={ck:.5f} ms (p50: {c['timing_results_ms']['kernel_p50']:.5f} ms)")
        print(f"  Pure Kernel Speedup:     {k_sp:.4f}x (~{k_sp:.2f}x)")
        print(f"  H2D Transfer Mean:       Baseline={b['timing_results_ms']['h2d_transfer_mean']:.5f} ms | Coalesced={c['timing_results_ms']['h2d_transfer_mean']:.5f} ms")
        print(f"  D2H Transfer Mean:       Baseline={b['timing_results_ms']['d2h_transfer_mean']:.5f} ms | Coalesced={c['timing_results_ms']['d2h_transfer_mean']:.5f} ms")
        print(f"  End-to-End Latency:      Baseline={be:.5f} ms -> Coalesced={ce:.5f} ms")
        print(f"  End-to-End Speedup:      {e_sp:.4f}x (~{e_sp:.2f}x)")
        print(f"  Throughput:              Baseline={b_tp:,.0f} edges/s -> Coalesced={c_tp:,.0f} edges/s ({c_tp / b_tp:.2f}x)")
        print(f"  Compulsory DRAM BW:      Baseline={b_bw_comp:.2f} GB/s ({b_bw_comp / 320.0 * 100:.1f}% peak) -> Coalesced={c_bw_comp:.2f} GB/s ({c_bw_comp / 320.0 * 100:.1f}% peak)")
        print(f"  Algorithmic BW (L2 Hit): Baseline={b_bw_algo:.2f} GB/s -> Coalesced={c_bw_algo:.2f} GB/s")


if __name__ == "__main__":
    main()
