"""
Google Colab Bundle Packaging Utility — FraudGraph Intelligence (FGI)
Packages CUDA C++ source, fixtures, raw binary arrays, CPU reference outputs,
and automated execution scripts into a self-contained archive for Google Colab GPU execution.
"""

import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.cpu_engine.aggregators import aggregate_neighbors
from backend.cpu_engine.csr_graph import CSRGraph


def dump_raw_graph_and_references(graph_path: Path, out_dir: Path) -> dict:
    """
    Unpacks a CSRGraph .npz file into raw binary arrays and precomputed CPU reference outputs
    for C++ direct loading.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = CSRGraph.load(graph_path)

    # 1. Dump metadata
    meta_path = out_dir / "meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"{graph.num_nodes} {graph.num_edges} {graph.feature_dim}\n")

    # 2. Dump raw binary arrays
    graph.row_ptr.astype(np.int32).tofile(out_dir / "row_ptr.bin")
    graph.col_idx.astype(np.int32).tofile(out_dir / "col_idx.bin")
    graph.features.astype(np.float32).tofile(out_dir / "features.bin")

    # 3. Compute and dump CPU reference outputs for sum, mean, max
    ref_paths = {}
    for op in ["mean", "sum", "max"]:
        ref_out = aggregate_neighbors(graph, op=op, method="reference")
        bin_path = out_dir / f"ref_{op}.bin"
        ref_out.astype(np.float32).tofile(bin_path)
        ref_paths[op] = str(bin_path.name)

    stats = {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "feature_dim": graph.feature_dim,
        "references": ref_paths,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def build_colab_bundle(output_zip: Path = Path("experiments/colab_bundle_fgi_phase6.zip")) -> Path:
    """Builds the complete self-contained Google Colab zip bundle."""
    workspace_root = Path(__file__).resolve().parents[3]
    stage_dir = workspace_root / "build" / "colab_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Packaging Colab bundle in staging directory: {stage_dir}")

    # 1. Copy CUDA source tree
    cuda_dest = stage_dir / "cuda"
    shutil.copytree(workspace_root / "backend" / "gpu_engine" / "cuda", cuda_dest)

    # 2. Dump raw graph arrays and references for fixtures
    data_dest = stage_dir / "data"
    toy_src = workspace_root / "data" / "fixtures" / "toy_graph_5nodes.npz"
    tier_a_src = workspace_root / "data" / "fixtures" / "synthetic_tier_a_10k.npz"

    if toy_src.exists():
        print("    Dumping raw binary arrays for toy_graph_5nodes...")
        dump_raw_graph_and_references(toy_src, data_dest / "toy_graph_5nodes")

    if tier_a_src.exists():
        print("    Dumping raw binary arrays for synthetic_tier_a_10k...")
        dump_raw_graph_and_references(tier_a_src, data_dest / "synthetic_tier_a_10k")

    # 3. Copy execution runner scripts
    colab_runner_src = workspace_root / "backend" / "gpu_engine" / "colab_runner"
    shutil.copy2(colab_runner_src / "run_colab_benchmarks.py", stage_dir / "run_colab_benchmarks.py")
    if (colab_runner_src / "fgi_colab_workflow.ipynb").exists():
        shutil.copy2(colab_runner_src / "fgi_colab_workflow.ipynb", stage_dir / "fgi_colab_workflow.ipynb")

    # 4. Create ZIP archive
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    print(f"[+] Compressing into archive: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage_dir):
            for file in files:
                abs_file = Path(root) / file
                rel_path = abs_file.relative_to(stage_dir)
                zf.write(abs_file, rel_path)

    # Clean up stage dir
    shutil.rmtree(stage_dir)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Colab bundle created successfully: {output_zip} ({size_mb:.2f} MB)")
    return output_zip


if __name__ == "__main__":
    build_colab_bundle()
