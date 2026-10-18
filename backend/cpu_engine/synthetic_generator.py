"""
Deterministic Synthetic Graph Generator for Benchmarking and Correctness Testing
Module: FGI-MOD-02 & FGI-MOD-03
Course: VIT Vellore — CSI4007 GPU Programming
"""

from __future__ import annotations
from typing import Any
import numpy as np
from backend.cpu_engine.csr_graph import CSRGraph


def generate_synthetic_graph(
    num_nodes: int = 1000,
    avg_degree: float = 10.0,
    feature_dim: int = 32,
    topology: str = "power_law",
    is_directed: bool = True,
    seed: int = 42,
    allow_self_loops: bool = False,
    isolated_node_ratio: float = 0.05,
    power_law_gamma: float = 2.5,
) -> CSRGraph:
    """
    Generate a deterministic synthetic graph with controlled topological properties
    and guaranteed finite feature matrices.
    
    Args:
        num_nodes: Total number of vertices in the generated graph.
        avg_degree: Average neighbor count per node.
        feature_dim: Dimensionality of generated node feature vectors.
        topology: "power_law", "uniform", "star", or "empty".
        is_directed: Whether edges are directed (True) or undirected (False).
        seed: Random seed for 100% deterministic reproducibility.
        allow_self_loops: Policy for self-loops (u, u).
        isolated_node_ratio: Fraction of nodes deliberately left with degree 0.
        power_law_gamma: Exponent for power-law degree skew.
        
    Returns:
        CSRGraph instance meeting all CSR invariants.
    """
    if num_nodes <= 0:
        return CSRGraph(
            row_ptr=np.array([0], dtype=np.int32),
            col_idx=np.empty(0, dtype=np.int32),
            features=np.empty((0, feature_dim), dtype=np.float32),
            is_directed=is_directed,
            metadata={"generator": "empty_zero_nodes", "seed": seed},
            validate=True,
        )

    rng = np.random.default_rng(seed)

    # 1. Generate Finite Feature Matrix (Standard Gaussian)
    features = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)
    # Ensure zero NaNs or Infs
    np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

    # 2. Handle isolated nodes partition
    num_isolated = int(num_nodes * isolated_node_ratio)
    connectable_nodes = num_nodes - num_isolated
    if connectable_nodes <= 1 and topology != "empty":
        connectable_nodes = num_nodes

    edges: list[tuple[int, int]] = []

    if topology == "empty":
        edges = []

    elif topology == "star":
        # Node 0 is the super-hub connected to all other connectable nodes
        hub = 0
        for target in range(1, connectable_nodes):
            edges.append((hub, target))
            if not is_directed:
                edges.append((target, hub))

    elif topology == "uniform":
        # Erdős-Rényi style random edge allocation
        target_edges = int((connectable_nodes * avg_degree) / (1.0 if is_directed else 2.0))
        for _ in range(target_edges):
            u = int(rng.integers(0, connectable_nodes))
            v = int(rng.integers(0, connectable_nodes))
            if not allow_self_loops and u == v:
                continue
            edges.append((u, v))
            if not is_directed and u != v:
                edges.append((v, u))

    elif topology == "power_law":
        # Vectorized Barabási-Albert preferential attachment model
        m = max(1, int(round(avg_degree / (2.0 if not is_directed else 1.0))))
        initial_nodes = min(max(m + 1, 5), connectable_nodes)

        # Preallocate repeated nodes buffer for O(1) sampling
        max_repeated = int(initial_nodes * (initial_nodes - 1) + 2 * m * connectable_nodes) + 100
        repeated_arr = np.empty(max_repeated, dtype=np.int32)
        ptr = 0

        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j))
                repeated_arr[ptr] = i
                repeated_arr[ptr + 1] = j
                ptr += 2
                if not is_directed:
                    edges.append((j, i))

        # Preferential attachment growth using vectorized index sampling
        for new_node in range(initial_nodes, connectable_nodes):
            if ptr == 0:
                targets = [0]
            else:
                sample_k = min(m * 3, ptr)
                rand_indices = rng.integers(0, ptr, size=sample_k)
                unique_sampled = np.unique(repeated_arr[rand_indices])
                targets = unique_sampled[:m].tolist()

            for target in targets:
                edges.append((new_node, target))
                repeated_arr[ptr] = new_node
                repeated_arr[ptr + 1] = target
                ptr += 2
                if not is_directed:
                    edges.append((target, new_node))
    else:
        raise ValueError(f"Unknown topology: {topology}. Choose from 'power_law', 'uniform', 'star', 'empty'.")

    metadata: dict[str, Any] = {
        "generator": "generate_synthetic_graph",
        "topology": topology,
        "seed": seed,
        "avg_degree_target": avg_degree,
        "is_directed": is_directed,
        "allow_self_loops": allow_self_loops,
        "isolated_node_ratio": isolated_node_ratio,
        "power_law_gamma": power_law_gamma if topology == "power_law" else None,
    }

    return CSRGraph.from_edge_list(
        num_nodes=num_nodes,
        edge_list=edges,
        features=features,
        feature_dim=feature_dim,
        is_directed=is_directed,
        allow_self_loops=allow_self_loops,
        deduplicate=True,
        sort_neighbors=True,
        metadata=metadata,
    )
