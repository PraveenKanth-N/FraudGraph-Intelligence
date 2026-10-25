"""
Deterministic Synthetic Labeled Fraud Transaction Graph Generator
Module: FGI-MOD-07 (Fraud Scoring & Classifier Baseline)
Course: VIT Vellore — CSI4007 GPU Programming

Generates reproducible, labeled synthetic transaction graphs designed for evaluating
downstream graph-based fraud classification models.

RESEARCH INTEGRITY & DATA PROVENANCE NOTICE:
This generator creates a SYNTHETIC benchmark fixture. The 3.5% fraud prevalence
(exactly 350 fraud nodes out of 10,000 transactions) is a controlled design choice
reflecting real-world class imbalance (such as in the IEEE-CIS benchmark). It does NOT
claim to reproduce the full complexity, distribution, or empirical realities of live
financial fraud. All data is synthetically generated under controlled random seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Tuple

# Ensure project root is in sys.path when executed directly
_root_dir = str(Path(__file__).resolve().parents[2])
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import numpy as np

from backend.cpu_engine.csr_graph import CSRGraph, GraphValidationError

GENERATOR_VERSION: str = "1.0.0"


def generate_synthetic_fraud_graph(
    num_nodes: int = 10000,
    feature_dim: int = 32,
    target_fraud_rate: float = 0.035,
    seed: int = 42,
    num_cards: int = 2500,
    num_devices: int = 3000,
    num_fraud_rings: int = 20,
    time_span_days: float = 30.0,
    temporal_window_days: float = 7.0,
) -> tuple[CSRGraph, np.ndarray, np.ndarray]:
    """
    Generate a deterministic synthetic transaction graph with controlled fraud labels,
    chronological timestamps, and leakage-safe temporal CSR adjacency.

    Data Generating Process:
    1. Timestamps (t): Generated from a Poisson arrival process across time_span_days
       and strictly sorted in ascending order (t_0 <= t_1 <= ... <= t_{N-1}).
    2. Entity Identifiers: Power-law / Zipfian distribution of transactions per payment
       card (num_cards) and device terminal (num_devices).
    3. Fraud Labels (y): Exactly target_fraud_rate * num_nodes (default 350) positive labels:
       - Fraud Rings (~70% = 245 nodes): Clustered across compromised cards/devices where
         multiple fraudulent attempts share relational keys.
       - Opportunistic Fraud (~30% = 105 nodes): Standalone fraudulent transactions
         distributed across otherwise normal accounts.
    4. Numerical Features (X): 32 continuous dimensions (float32).
       - Licit nodes: Gaussian N(0, 1).
       - Fraud nodes: Controlled subtle multivariate shifts with substantial overlap
         (Features 0-3: amount/velocity proxies; Features 4-7: risk/behavior proxies;
          Features 8-31: uncorrelated noise descriptors).
       - Zero target leakage: Labels are stored separately and NEVER embedded into X.
    5. Temporal Graph Edges (E): Transactions sharing card_id or device_id within a sliding
       window (temporal_window_days) form directed edges (i, j) where j < i (t_j <= t_i).
       When node i aggregates neighbor features, it gathers strictly from historical transactions.

    Args:
        num_nodes: Total transaction count (default 10,000).
        feature_dim: Feature vector dimension per node (default 32).
        target_fraud_rate: Fraction of positive fraud nodes (default 0.035 = 350 nodes).
        seed: Random seed for 100% deterministic reproducibility (default 42).
        num_cards: Number of simulated payment card entities (default 2,500).
        num_devices: Number of simulated terminal / device entities (default 3,000).
        num_fraud_rings: Number of compromised entity clusters (default 20).
        time_span_days: Total simulated observation period in days (default 30.0).
        temporal_window_days: Maximum lookback window for relational edges (default 7.0).

    Returns:
        Tuple of (graph, labels, timestamps):
        - graph: Validated CSRGraph meeting all CSR invariants.
        - labels: 1D int32 array of shape (num_nodes,) with values in {0, 1}.
        - timestamps: 1D float64 array of shape (num_nodes,) in seconds from start.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if not (0.0 < target_fraud_rate < 1.0):
        raise ValueError(f"target_fraud_rate must be in (0, 1), got {target_fraud_rate}")

    rng = np.random.default_rng(seed)
    target_fraud_count = int(round(num_nodes * target_fraud_rate))

    # -------------------------------------------------------------------------
    # 1. Chronological Timestamps (Poisson Arrival Process)
    # -------------------------------------------------------------------------
    time_span_seconds = float(time_span_days * 86400.0)
    temporal_window_seconds = float(temporal_window_days * 86400.0)

    # Exponential inter-arrival intervals
    intervals = rng.exponential(scale=time_span_seconds / num_nodes, size=num_nodes)
    raw_timestamps = np.cumsum(intervals)
    # Normalize strictly to [0.0, time_span_seconds]
    timestamps = (
        (raw_timestamps - raw_timestamps[0])
        / (raw_timestamps[-1] - raw_timestamps[0])
        * time_span_seconds
    ).astype(np.float64)

    # -------------------------------------------------------------------------
    # 2. Entity Identifiers (Zipfian Frequency Distribution)
    # -------------------------------------------------------------------------
    card_weights = 1.0 / (np.arange(1, num_cards + 1) ** 0.8)
    card_probs = card_weights / card_weights.sum()
    card_ids = rng.choice(num_cards, size=num_nodes, p=card_probs)

    device_weights = 1.0 / (np.arange(1, num_devices + 1) ** 0.8)
    device_probs = device_weights / device_weights.sum()
    device_ids = rng.choice(num_devices, size=num_nodes, p=device_probs)

    # -------------------------------------------------------------------------
    # 3. Label Allocation (Exactly target_fraud_count = 350)
    # -------------------------------------------------------------------------
    labels = np.zeros(num_nodes, dtype=np.int32)

    ring_fraud_target = int(target_fraud_count * 0.70)
    isolated_fraud_target = target_fraud_count - ring_fraud_target

    # Select compromised entity clusters for fraud rings
    fraud_ring_cards = rng.choice(num_cards, size=min(num_fraud_rings, num_cards), replace=False)
    fraud_ring_devices = rng.choice(num_devices, size=min(num_fraud_rings, num_devices), replace=False)

    candidate_ring_txs = []
    for r in range(len(fraud_ring_cards)):
        rcard = fraud_ring_cards[r]
        rdev = fraud_ring_devices[r]
        matches = np.where((card_ids == rcard) | (device_ids == rdev))[0]
        if len(matches) > 1:
            candidate_ring_txs.extend(matches)

    candidate_ring_txs = sorted(list(set(candidate_ring_txs)))

    if len(candidate_ring_txs) >= ring_fraud_target:
        selected_ring_frauds = rng.choice(candidate_ring_txs, size=ring_fraud_target, replace=False)
    else:
        selected_ring_frauds = candidate_ring_txs
        ring_fraud_target = len(selected_ring_frauds)
        isolated_fraud_target = target_fraud_count - ring_fraud_target

    labels[selected_ring_frauds] = 1

    # Remaining isolated frauds chosen uniformly across licit transactions
    licit_indices = np.where(labels == 0)[0]
    selected_isolated_frauds = rng.choice(licit_indices, size=isolated_fraud_target, replace=False)
    labels[selected_isolated_frauds] = 1

    actual_fraud_count = int(np.sum(labels))
    if actual_fraud_count != target_fraud_count:
        raise ValueError(
            f"Internal allocation error: expected {target_fraud_count} frauds, got {actual_fraud_count}"
        )

    # -------------------------------------------------------------------------
    # 4. Numerical Node Feature Matrix (float32, 100% Finite, Zero Target Column)
    # -------------------------------------------------------------------------
    # Base licit transactions: standard Gaussian N(0, 1)
    features = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)

    # Fraud transactions: subtle multivariate distribution shift
    # Features 0..3: Amount / Velocity proxies (slight positive shift, higher variance)
    features[labels == 1, 0:4] += rng.normal(loc=0.55, scale=0.6, size=(target_fraud_count, 4)).astype(np.float32)
    # Features 4..7: Consistency / Profile match proxies (slight negative shift)
    features[labels == 1, 4:8] -= rng.normal(loc=0.45, scale=0.5, size=(target_fraud_count, 4)).astype(np.float32)
    # Features 8..31: Background noise / non-discriminative features (identical distribution)

    # Ensure absolute finiteness
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 5. Build Temporal Graph Edges (Strict Causality: t_neighbor <= t_query)
    # -------------------------------------------------------------------------
    # In FGI CSRGraph, row i contains the neighbors that node i aggregates from.
    # To prevent future leakage, every neighbor j in row i must have t_j <= t_i (j <= i).
    edge_list: list[tuple[int, int]] = []

    for key_arr in [card_ids, device_ids]:
        # Group transaction indices by entity key
        sorted_order = np.lexsort((timestamps, key_arr))
        sorted_keys = key_arr[sorted_order]
        sorted_txs = sorted_order

        key_diffs = np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
        splits = np.split(sorted_txs, key_diffs)

        for group in splits:
            if len(group) <= 1:
                continue
            for idx_curr in range(len(group)):
                i = group[idx_curr]
                t_i = timestamps[i]
                for idx_prev in range(idx_curr - 1, -1, -1):
                    j = group[idx_prev]
                    t_j = timestamps[j]
                    if (t_i - t_j) <= temporal_window_seconds:
                        edge_list.append((i, j))
                    else:
                        break  # Remaining prior transactions exceed sliding window

    # -------------------------------------------------------------------------
    # 6. Assemble and Validate CSRGraph
    # -------------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "dataset_name": "synthetic_fraud_10k",
        "dataset_provenance": "SYNTHETIC",
        "is_synthetic": True,
        "seed": seed,
        "target_fraud_rate": target_fraud_rate,
        "fraud_count": target_fraud_count,
        "licit_count": num_nodes - target_fraud_count,
        "time_span_days": time_span_days,
        "temporal_window_days": temporal_window_days,
        "relational_keys": ["card_id", "device_id"],
        "disclaimer": (
            "Controlled synthetic benchmark fixture for evaluating graph-based neighbor aggregation. "
            "The 3.5% fraud prevalence reflects real-world class imbalance, but does NOT represent "
            "live banking fraud distributions or empirical records."
        ),
    }

    graph = CSRGraph.from_edge_list(
        num_nodes=num_nodes,
        edge_list=edge_list,
        features=features,
        feature_dim=feature_dim,
        is_directed=True,
        allow_self_loops=False,
        deduplicate=True,
        sort_neighbors=True,
        metadata=metadata,
    )

    # Invariant verification: ensure 100% of CSR edges adhere to temporal causality
    row_ptr = graph.row_ptr
    col_idx = graph.col_idx
    for i in range(num_nodes):
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[p]
            if timestamps[j] > timestamps[i]:
                raise GraphValidationError(
                    f"Temporal leakage detected: neighbor {j} (t={timestamps[j]}) > query node {i} (t={timestamps[i]})"
                )

    return graph, labels, timestamps


def save_labeled_graph(
    destination: str | Path,
    graph: CSRGraph,
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> Path:
    """
    Serialize labeled CSRGraph to an .npz archive following FGI contract conventions.

    Archive Schema:
    - row_ptr: (num_nodes + 1,) int32
    - col_idx: (num_edges,) int32
    - features: (num_nodes, feature_dim) float32
    - labels: (num_nodes,) int32 binary {0, 1}
    - timestamps: (num_nodes,) float64 seconds from start
    - manifest: JSON string with stats, metadata, and provenance
    """
    dest_path = Path(destination)
    if dest_path.suffix != ".npz":
        dest_path = dest_path.with_suffix(".npz")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    labels_arr = np.asarray(labels, dtype=np.int32)
    timestamps_arr = np.asarray(timestamps, dtype=np.float64)

    if len(labels_arr) != graph.num_nodes:
        raise ValueError(f"labels length {len(labels_arr)} != num_nodes {graph.num_nodes}")
    if len(timestamps_arr) != graph.num_nodes:
        raise ValueError(f"timestamps length {len(timestamps_arr)} != num_nodes {graph.num_nodes}")

    fraud_count = int(np.sum(labels_arr))
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "feature_dim": graph.feature_dim,
        "is_directed": graph.is_directed,
        "stats": graph.get_stats(),
        "metadata": graph.metadata,
        "fraud_stats": {
            "fraud_count": fraud_count,
            "licit_count": graph.num_nodes - fraud_count,
            "fraud_percentage": round((fraud_count / graph.num_nodes) * 100.0, 4) if graph.num_nodes > 0 else 0.0,
        },
        "temporal_stats": {
            "min_timestamp_sec": float(timestamps_arr[0]) if len(timestamps_arr) > 0 else 0.0,
            "max_timestamp_sec": float(timestamps_arr[-1]) if len(timestamps_arr) > 0 else 0.0,
        },
        "array_schema": {
            "row_ptr": "1D np.int32 array (N + 1,) CSR row offsets",
            "col_idx": "1D np.int32 array (M,) neighbor node indices",
            "features": "2D np.float32 array (N, D) continuous node feature vectors",
            "labels": "1D np.int32 array (N,) binary fraud targets (0=licit, 1=fraud)",
            "timestamps": "1D np.float64 array (N,) chronological transaction timestamps in seconds",
            "manifest": "Embedded JSON string documenting metadata and schemas",
        },
        "idx_to_node_id": graph.idx_to_node_id,
    }

    np.savez_compressed(
        dest_path,
        row_ptr=graph.row_ptr,
        col_idx=graph.col_idx,
        features=graph.features,
        labels=labels_arr,
        timestamps=timestamps_arr,
        manifest=json.dumps(manifest),
    )

    return dest_path


def load_labeled_graph(source_path: str | Path) -> tuple[CSRGraph, np.ndarray, np.ndarray]:
    """
    Load a labeled CSRGraph from an .npz archive and validate its integrity.

    Returns:
        Tuple of (graph, labels, timestamps).
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Graph archive not found at: {src}")

    with np.load(src, allow_pickle=False) as data:
        row_ptr = np.array(data["row_ptr"])
        col_idx = np.array(data["col_idx"])
        features = np.array(data["features"])
        labels = data["labels"].astype(np.int32)
        timestamps = data["timestamps"].astype(np.float64)
        manifest_str = str(data["manifest"])
        manifest = json.loads(manifest_str)

    idx_to_node_id = manifest.get("idx_to_node_id")
    node_id_to_idx = {nid: i for i, nid in enumerate(idx_to_node_id)} if idx_to_node_id else None

    metadata = dict(manifest.get("metadata", {}))
    if "generator_version" in manifest and "generator_version" not in metadata:
        metadata["generator_version"] = manifest["generator_version"]

    graph = CSRGraph(
        row_ptr=row_ptr,
        col_idx=col_idx,
        features=features,
        is_directed=manifest.get("is_directed", True),
        node_id_to_idx=node_id_to_idx,
        idx_to_node_id=idx_to_node_id,
        metadata=metadata,
        validate=True,
    )

    return graph, labels, timestamps


def temporal_train_test_split(
    num_nodes: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological index split utility for temporal transaction graphs.

    Because transaction nodes are strictly sorted chronologically by timestamp,
    a temporal split partitions indices by continuous time windows:
    - Train: [0, train_end)
    - Val:   [train_end, val_end)
    - Test:  [val_end, num_nodes)

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    train_end = int(round(num_nodes * train_ratio))
    val_end = int(round(num_nodes * (train_ratio + val_ratio)))

    train_indices = np.arange(0, train_end, dtype=np.int32)
    val_indices = np.arange(train_end, val_end, dtype=np.int32)
    test_indices = np.arange(val_end, num_nodes, dtype=np.int32)

    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic labeled fraud graph fixture.")
    parser.add_argument("--out", type=str, default="data/fixtures/synthetic_fraud_10k.npz")
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[+] Generating synthetic labeled fraud graph (N={args.nodes}, D={args.dim}, rate={args.rate}, seed={args.seed})...")
    g, y, t = generate_synthetic_fraud_graph(
        num_nodes=args.nodes,
        feature_dim=args.dim,
        target_fraud_rate=args.rate,
        seed=args.seed,
    )
    out_path = save_labeled_graph(args.out, g, y, t)
    print(f"[OK] Saved labeled fixture to: {out_path}")
    print(f"     Nodes: {g.num_nodes:,}, Edges: {g.num_edges:,}, Dim: {g.feature_dim}")
    print(f"     Frauds: {int(np.sum(y)):,} ({np.sum(y)/len(y)*100:.2f}%), Licit: {len(y)-int(np.sum(y)):,}")
    print(f"     Memory: {g.get_stats()['memory_footprint_mb']:.4f} MB")
