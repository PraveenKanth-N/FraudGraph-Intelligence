"""
Compressed Sparse Row (CSR) Graph Representation & Invariants Validator
Module: FGI-MOD-03 (Graph Representation)
Course: VIT Vellore — CSI4007 GPU Programming
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np


class GraphValidationError(Exception):
    """Raised when CSR graph invariants or feature constraints are violated."""
    pass


class CSRGraph:
    """
    Canonical Compressed Sparse Row (CSR) graph representation for high-throughput
    parallel neighbor aggregation on CPU and CUDA.
    
    Attributes:
        row_ptr (np.ndarray): 1D array of shape (num_nodes + 1,) indicating starting offsets in col_idx.
        col_idx (np.ndarray): 1D array of shape (num_edges,) containing neighbor node indices.
        features (np.ndarray): 2D array of shape (num_nodes, feature_dim) of node feature vectors.
        num_nodes (int): Total number of vertices in the graph.
        num_edges (int): Total number of directed edges (equal to len(col_idx)).
        feature_dim (int): Dimensionality of each node feature vector.
        is_directed (bool): True if edges are directed; False if undirected (stored symmetrically).
        node_id_to_idx (dict): Mapping from source entity identifier (e.g. TransactionID) to node index.
        idx_to_node_id (list): List mapping node index [0, num_nodes - 1] back to source entity identifier.
        metadata (dict): Arbitrary construction provenance, policies, and configuration parameters.
    """

    def __init__(
        self,
        row_ptr: np.ndarray | Sequence[int],
        col_idx: np.ndarray | Sequence[int],
        features: np.ndarray | Sequence[Sequence[float]],
        is_directed: bool = True,
        node_id_to_idx: dict[Any, int] | None = None,
        idx_to_node_id: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> None:
        # Enforce canonical data types for CUDA compatibility
        self.row_ptr = np.asarray(row_ptr, dtype=np.int32)
        self.col_idx = np.asarray(col_idx, dtype=np.int32)
        self.features = np.asarray(features, dtype=np.float32)

        if self.features.ndim == 1 and len(self.row_ptr) > 1:
            # Reshape single-column features to 2D
            self.features = self.features.reshape(-1, 1)

        self.num_nodes = len(self.row_ptr) - 1 if len(self.row_ptr) > 0 else 0
        self.num_edges = len(self.col_idx)
        self.feature_dim = self.features.shape[1] if self.features.ndim == 2 else 0
        self.is_directed = bool(is_directed)

        # Node index mapping
        if node_id_to_idx is not None and idx_to_node_id is not None:
            self.node_id_to_idx = dict(node_id_to_idx)
            self.idx_to_node_id = list(idx_to_node_id)
        else:
            self.node_id_to_idx = {i: i for i in range(self.num_nodes)}
            self.idx_to_node_id = list(range(self.num_nodes))

        self.metadata = dict(metadata or {})

        if validate:
            self.validate()

    def validate(self) -> None:
        """
        Validate all fundamental CSR invariants.
        Raises GraphValidationError if any invariant is violated.
        """
        # Invariant 0: Dimension shapes
        if self.row_ptr.ndim != 1:
            raise GraphValidationError(f"row_ptr must be 1D, got ndim={self.row_ptr.ndim}")
        if self.col_idx.ndim != 1:
            raise GraphValidationError(f"col_idx must be 1D, got ndim={self.col_idx.ndim}")
        if self.features.ndim != 2:
            raise GraphValidationError(f"features must be 2D of shape (num_nodes, feature_dim), got ndim={self.features.ndim}")

        # Invariant 1: Empty graph boundary
        if self.num_nodes == 0:
            if len(self.row_ptr) != 1 or self.row_ptr[0] != 0:
                raise GraphValidationError("For an empty graph (num_nodes=0), row_ptr must be [0]")
            if len(self.col_idx) != 0:
                raise GraphValidationError("For an empty graph, col_idx must be empty")
            if self.features.shape[0] != 0:
                raise GraphValidationError("For an empty graph, features must have shape (0, feature_dim)")
            return

        # Invariant 2: Base offset
        if self.row_ptr[0] != 0:
            raise GraphValidationError(f"CSR invariant violated: row_ptr[0] must be 0, got {self.row_ptr[0]}")

        # Invariant 3: Monotonicity
        diffs = np.diff(self.row_ptr)
        if np.any(diffs < 0):
            bad_indices = np.where(diffs < 0)[0]
            raise GraphValidationError(
                f"CSR invariant violated: row_ptr must be monotonically non-decreasing. "
                f"Decreases at indices: {bad_indices.tolist()}"
            )

        # Invariant 4: Edge count consistency
        if self.row_ptr[-1] != self.num_edges:
            raise GraphValidationError(
                f"CSR invariant violated: row_ptr[-1] ({self.row_ptr[-1]}) != len(col_idx) ({self.num_edges})"
            )

        # Invariant 5: Node index bounds
        if self.num_edges > 0:
            min_col = np.min(self.col_idx)
            max_col = np.max(self.col_idx)
            if min_col < 0 or max_col >= self.num_nodes:
                raise GraphValidationError(
                    f"CSR invariant violated: col_idx entries must be in range [0, {self.num_nodes - 1}], "
                    f"got min={min_col}, max={max_col}"
                )

        # Invariant 6: Feature shape and finiteness
        if self.features.shape[0] != self.num_nodes:
            raise GraphValidationError(
                f"Feature row count mismatch: features.shape[0] ({self.features.shape[0]}) != num_nodes ({self.num_nodes})"
            )
        if self.features.shape[1] != self.feature_dim:
            raise GraphValidationError(
                f"Feature dimension mismatch: features.shape[1] ({self.features.shape[1]}) != feature_dim ({self.feature_dim})"
            )
        if not np.all(np.isfinite(self.features)):
            nan_count = int(np.isnan(self.features).sum())
            inf_count = int(np.isinf(self.features).sum())
            raise GraphValidationError(
                f"Features contain non-finite values (NaN: {nan_count}, Inf: {inf_count}). "
                "All feature values must be finite floating-point numbers."
            )

        # Invariant 7: Mapping consistency
        if len(self.idx_to_node_id) != self.num_nodes:
            raise GraphValidationError(
                f"Node mapping length mismatch: len(idx_to_node_id)={len(self.idx_to_node_id)} != num_nodes={self.num_nodes}"
            )
        if len(self.node_id_to_idx) != self.num_nodes:
            raise GraphValidationError(
                f"Node mapping dict mismatch: len(node_id_to_idx)={len(self.node_id_to_idx)} != num_nodes={self.num_nodes}"
            )

        # Invariant 8: Undirected symmetry check (if marked undirected)
        if not self.is_directed and self.num_edges > 0:
            edge_set = set(map(tuple, self.to_edge_list().tolist()))
            for u, v in edge_set:
                if (v, u) not in edge_set:
                    raise GraphValidationError(
                        f"Undirected graph symmetry invariant violated: edge ({u}, {v}) exists but reverse edge ({v}, {u}) is missing."
                    )

    def get_degree(self, node_idx: int) -> int:
        """Return the degree (neighbor count) of a single node."""
        if node_idx < 0 or node_idx >= self.num_nodes:
            raise IndexError(f"Node index {node_idx} out of range [0, {self.num_nodes - 1}]")
        return int(self.row_ptr[node_idx + 1] - self.row_ptr[node_idx])

    def get_degrees(self) -> np.ndarray:
        """Return a 1D array of degrees for all nodes."""
        if self.num_nodes == 0:
            return np.empty(0, dtype=np.int32)
        return np.diff(self.row_ptr)

    def get_neighbors(self, node_idx: int) -> np.ndarray:
        """Return a 1D slice of col_idx representing the neighbors of node_idx."""
        if node_idx < 0 or node_idx >= self.num_nodes:
            raise IndexError(f"Node index {node_idx} out of range [0, {self.num_nodes - 1}]")
        start = self.row_ptr[node_idx]
        end = self.row_ptr[node_idx + 1]
        return self.col_idx[start:end]

    def to_edge_list(self) -> np.ndarray:
        """
        Convert CSR arrays back to a 2D edge list array of shape (num_edges, 2).
        Guarantees round-trip equivalence.
        """
        if self.num_edges == 0:
            return np.empty((0, 2), dtype=np.int32)

        degrees = self.get_degrees()
        sources = np.repeat(np.arange(self.num_nodes, dtype=np.int32), degrees)
        targets = self.col_idx
        return np.column_stack((sources, targets))

    @classmethod
    def from_edge_list(
        cls,
        num_nodes: int,
        edge_list: Sequence[tuple[int, int]] | np.ndarray,
        features: np.ndarray | None = None,
        feature_dim: int = 32,
        is_directed: bool = True,
        allow_self_loops: bool = False,
        deduplicate: bool = True,
        sort_neighbors: bool = True,
        node_id_to_idx: dict[Any, int] | None = None,
        idx_to_node_id: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CSRGraph:
        """
        Construct a validated CSRGraph from an edge list.
        
        Args:
            num_nodes: Number of nodes.
            edge_list: Iterable of (src, dst) index pairs.
            features: Optional (num_nodes, feature_dim) array. If None, initialized with zeros.
            feature_dim: Dimensionality for generated default features.
            is_directed: If False, reverse edges (dst, src) are automatically added.
            allow_self_loops: If False, self-loops (u, u) are discarded.
            deduplicate: If True, duplicate edges are coalesced into a single edge.
            sort_neighbors: If True, neighbors in each row are sorted in ascending order.
        """
        if num_nodes < 0:
            raise ValueError(f"num_nodes cannot be negative, got {num_nodes}")

        meta = dict(metadata or {})
        meta["allow_self_loops"] = allow_self_loops
        meta["deduplicate"] = deduplicate
        meta["sort_neighbors"] = sort_neighbors

        edges_in = np.asarray(edge_list, dtype=np.int32)
        if edges_in.size == 0:
            edges_in = np.empty((0, 2), dtype=np.int32)
        elif edges_in.ndim != 2 or edges_in.shape[1] != 2:
            raise ValueError(f"edge_list must have shape (M, 2), got {edges_in.shape}")

        clean_edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for u, v in edges_in:
            u, v = int(u), int(v)
            if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) contains index out of bounds for num_nodes={num_nodes}"
                )

            # Self-loop policy
            if not allow_self_loops and u == v:
                continue

            pairs = [(u, v)]
            if not is_directed and u != v:
                pairs.append((v, u))

            for pair in pairs:
                if deduplicate:
                    if pair not in seen:
                        seen.add(pair)
                        clean_edges.append(pair)
                else:
                    clean_edges.append(pair)

        # Build CSR structures
        row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
        if len(clean_edges) == 0:
            col_idx = np.empty(0, dtype=np.int32)
        else:
            # Group edges by source
            buckets: list[list[int]] = [[] for _ in range(num_nodes)]
            for u, v in clean_edges:
                buckets[u].append(v)

            if sort_neighbors:
                for b in buckets:
                    b.sort()

            # Flatten buckets into col_idx and compute row_ptr
            flat_cols: list[int] = []
            for i in range(num_nodes):
                row_ptr[i] = len(flat_cols)
                flat_cols.extend(buckets[i])
            row_ptr[num_nodes] = len(flat_cols)
            col_idx = np.asarray(flat_cols, dtype=np.int32)

        # Handle features
        if features is None:
            features_arr = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        else:
            features_arr = np.asarray(features, dtype=np.float32)

        return cls(
            row_ptr=row_ptr,
            col_idx=col_idx,
            features=features_arr,
            is_directed=is_directed,
            node_id_to_idx=node_id_to_idx,
            idx_to_node_id=idx_to_node_id,
            metadata=meta,
            validate=True,
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Compute descriptive topological and memory metrics for the graph.
        """
        degrees = self.get_degrees()
        num_isolated = int(np.sum(degrees == 0)) if self.num_nodes > 0 else 0
        total_possible = (self.num_nodes * (self.num_nodes - 1)) if self.num_nodes > 1 else 1
        density = float(self.num_edges / total_possible) if self.num_nodes > 1 else 0.0

        memory_bytes = int(self.row_ptr.nbytes + self.col_idx.nbytes + self.features.nbytes)
        memory_mb = float(memory_bytes / (1024 * 1024))

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "undirected_equivalent_edges": self.num_edges if self.is_directed else self.num_edges // 2,
            "is_directed": self.is_directed,
            "feature_dim": self.feature_dim,
            "num_isolated_nodes": num_isolated,
            "isolated_node_percentage": round((num_isolated / self.num_nodes * 100.0), 2) if self.num_nodes > 0 else 0.0,
            "density": density,
            "degree_min": int(np.min(degrees)) if self.num_nodes > 0 else 0,
            "degree_max": int(np.max(degrees)) if self.num_nodes > 0 else 0,
            "degree_mean": float(np.mean(degrees)) if self.num_nodes > 0 else 0.0,
            "degree_median": float(np.median(degrees)) if self.num_nodes > 0 else 0.0,
            "degree_p90": float(np.percentile(degrees, 90)) if self.num_nodes > 0 else 0.0,
            "degree_p99": float(np.percentile(degrees, 99)) if self.num_nodes > 0 else 0.0,
            "memory_footprint_bytes": memory_bytes,
            "memory_footprint_mb": round(memory_mb, 4),
        }

    def save(self, destination: str | Path) -> Path:
        """
        Serialize graph arrays to an .npz file with accompanying JSON manifest metadata.
        """
        dest_path = Path(destination)
        if dest_path.suffix != ".npz":
            dest_path = dest_path.with_suffix(".npz")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "feature_dim": self.feature_dim,
            "is_directed": self.is_directed,
            "stats": self.get_stats(),
            "metadata": self.metadata,
            "idx_to_node_id": self.idx_to_node_id,
        }

        np.savez_compressed(
            dest_path,
            row_ptr=self.row_ptr,
            col_idx=self.col_idx,
            features=self.features,
            manifest=json.dumps(manifest),
        )
        return dest_path

    @classmethod
    def load(cls, source_path: str | Path) -> CSRGraph:
        """
        Load a CSRGraph from an .npz archive and validate its invariants.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Graph archive not found at: {src}")

        with np.load(src, allow_pickle=False) as data:
            row_ptr = data["row_ptr"]
            col_idx = data["col_idx"]
            features = data["features"]
            manifest_str = str(data["manifest"])
            manifest = json.loads(manifest_str)

        idx_to_node_id = manifest.get("idx_to_node_id")
        node_id_to_idx = {nid: i for i, nid in enumerate(idx_to_node_id)} if idx_to_node_id else None

        return cls(
            row_ptr=row_ptr,
            col_idx=col_idx,
            features=features,
            is_directed=manifest.get("is_directed", True),
            node_id_to_idx=node_id_to_idx,
            idx_to_node_id=idx_to_node_id,
            metadata=manifest.get("metadata", {}),
            validate=True,
        )
