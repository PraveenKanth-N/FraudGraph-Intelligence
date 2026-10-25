import os
import hashlib
import tempfile
from collections import deque
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import numpy as np

router = APIRouter()

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
CANONICAL_FIXTURE_PATH = os.path.join(WORKSPACE_ROOT, "data", "fixtures", "synthetic_fraud_10k.npz")


def _is_safe_path(path: str) -> bool:
    """Ensure path is within WORKSPACE_ROOT or an authorized system temp directory."""
    abs_path = os.path.abspath(path)
    temp_dir = os.path.abspath(tempfile.gettempdir())

    in_workspace = False
    try:
        in_workspace = os.path.commonpath([abs_path, WORKSPACE_ROOT]) == WORKSPACE_ROOT
    except ValueError:
        pass

    in_temp = False
    try:
        in_temp = os.path.commonpath([abs_path, temp_dir]) == temp_dir
    except ValueError:
        pass

    return in_workspace or in_temp


class SubgraphNode(BaseModel):
    id: int
    depth: int
    is_root: bool
    timestamp_sec: float
    fraud_label: int
    is_fraud: bool
    degree: int


class SubgraphEdge(BaseModel):
    source: int
    target: int
    temporal_delta_sec: Optional[float] = None


class SubgraphQueryMeta(BaseModel):
    root_node_id: int
    requested_hops: int
    actual_hops: int
    direction: str = "directed_csr"
    max_nodes: int
    max_edges: int


class SubgraphSummary(BaseModel):
    total_nodes: int
    total_edges: int
    fraud_node_count: int
    fraud_percentage: float
    is_truncated: bool
    node_cap_hit: bool
    edge_cap_hit: bool
    truncation_reason: Optional[str] = None


class SubgraphProvenance(BaseModel):
    fixture_name: str
    sha256: str
    causality_policy: str = "causal_history (directed CSR)"


class SubgraphResponse(BaseModel):
    status: str
    query: SubgraphQueryMeta
    summary: SubgraphSummary
    nodes: List[SubgraphNode]
    edges: List[SubgraphEdge]
    provenance: SubgraphProvenance


def _load_npz_arrays(path: str):
    """Safely loads and validates NPZ arrays for graph traversal."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset fixture file not found: {os.path.basename(path)}")

    try:
        with open(path, "rb") as fp:
            file_bytes = fp.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Load NPZ from disk and copy arrays into memory so file handle is closed
        with np.load(path) as data:
            required_keys = ["row_ptr", "col_idx", "timestamps", "labels"]
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"Missing required key '{k}' in NPZ file")

            row_ptr = np.array(data["row_ptr"])
            col_idx = np.array(data["col_idx"])
            timestamps = np.array(data["timestamps"])
            labels = np.array(data["labels"])

        if len(row_ptr) < 1:
            raise ValueError("row_ptr array is empty")

        return row_ptr, col_idx, timestamps, labels, sha256
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Dataset fixture file is corrupt or invalid: {str(e)}")


@router.get("/subgraph", response_model=SubgraphResponse)
def get_subgraph(
    node_id: int = Query(..., description="Root transaction node ID"),
    hops: int = Query(1, ge=1, le=3, description="Traversal depth (1 to 3)"),
    max_nodes: int = Query(150, ge=1, le=300, description="Maximum nodes to return (1 to 300)"),
    max_edges: int = Query(500, ge=1, le=1000, description="Maximum edges to return (1 to 1000)"),
    npz_path: Optional[str] = Query(None, description="Optional path to NPZ dataset fixture"),
):
    """Extract an authentic directed k-hop subgraph centered around node_id using CSR topology."""
    target_path = npz_path or CANONICAL_FIXTURE_PATH
    if not _is_safe_path(target_path):
        raise HTTPException(status_code=400, detail="npz_path escapes authorized workspace boundaries")

    row_ptr, col_idx, timestamps, labels, sha256 = _load_npz_arrays(target_path)
    num_nodes = len(row_ptr) - 1

    # Validate node_id within bounds
    if node_id < 0 or node_id >= num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"node_id {node_id} is out of bounds. Valid range is 0 to {num_nodes - 1}."
        )

    # Breadth-First Search (directed CSR traversal)
    # Strictly label-quarantined: traversal reads only row_ptr and col_idx
    visited_depth = {node_id: 0}
    queue = deque([(node_id, 0)])
    node_cap_hit = False

    while queue:
        curr, depth = queue.popleft()
        if depth >= hops:
            continue

        start_edge = int(row_ptr[curr])
        end_edge = int(row_ptr[curr + 1])
        neighbors = col_idx[start_edge:end_edge]

        for nbr in neighbors:
            nbr_id = int(nbr)
            if nbr_id not in visited_depth:
                if len(visited_depth) >= max_nodes:
                    node_cap_hit = True
                    break
                visited_depth[nbr_id] = depth + 1
                queue.append((nbr_id, depth + 1))
        if node_cap_hit:
            break

    # Assemble directed edges strictly between nodes in visited_depth
    # Preserves directed CSR orientation without adding synthetic reverse edges
    raw_edges = []
    for u in visited_depth:
        start_edge = int(row_ptr[u])
        end_edge = int(row_ptr[u + 1])
        for v in col_idx[start_edge:end_edge]:
            v_id = int(v)
            if v_id in visited_depth:
                raw_edges.append((u, v_id))

    # Deterministic deduplication and sorting
    unique_edges = sorted(list(set(raw_edges)))
    edge_cap_hit = False
    if len(unique_edges) > max_edges:
        edge_cap_hit = True
        selected_edges = unique_edges[:max_edges]
    else:
        selected_edges = unique_edges

    is_truncated = node_cap_hit or edge_cap_hit
    truncation_reasons = []
    if node_cap_hit:
        truncation_reasons.append("MAX_NODES_LIMIT_REACHED")
    if edge_cap_hit:
        truncation_reasons.append("MAX_EDGES_LIMIT_REACHED")
    truncation_reason_str = " | ".join(truncation_reasons) if truncation_reasons else None

    # Deterministic node sorting by (depth, id)
    sorted_node_ids = sorted(visited_depth.keys(), key=lambda nid: (visited_depth[nid], nid))

    node_objects: List[SubgraphNode] = []
    fraud_count = 0
    actual_max_depth = 0

    for nid in sorted_node_ids:
        d = visited_depth[nid]
        if d > actual_max_depth:
            actual_max_depth = d

        label_val = int(labels[nid])
        is_fraud_val = (label_val == 1)
        if is_fraud_val:
            fraud_count += 1

        node_deg = int(row_ptr[nid + 1] - row_ptr[nid])
        ts_val = float(timestamps[nid])

        node_objects.append(
            SubgraphNode(
                id=nid,
                depth=d,
                is_root=(nid == node_id),
                timestamp_sec=round(ts_val, 4),
                fraud_label=label_val,
                is_fraud=is_fraud_val,
                degree=node_deg,
            )
        )

    edge_objects: List[SubgraphEdge] = []
    for u, v in selected_edges:
        dt = round(float(timestamps[v] - timestamps[u]), 4)
        edge_objects.append(
            SubgraphEdge(
                source=u,
                target=v,
                temporal_delta_sec=dt,
            )
        )

    total_n = len(node_objects)
    fraud_pct = round((fraud_count / total_n * 100.0), 2) if total_n > 0 else 0.0

    return SubgraphResponse(
        status="success",
        query=SubgraphQueryMeta(
            root_node_id=node_id,
            requested_hops=hops,
            actual_hops=actual_max_depth,
            direction="directed_csr",
            max_nodes=max_nodes,
            max_edges=max_edges,
        ),
        summary=SubgraphSummary(
            total_nodes=total_n,
            total_edges=len(edge_objects),
            fraud_node_count=fraud_count,
            fraud_percentage=fraud_pct,
            is_truncated=is_truncated,
            node_cap_hit=node_cap_hit,
            edge_cap_hit=edge_cap_hit,
            truncation_reason=truncation_reason_str,
        ),
        nodes=node_objects,
        edges=edge_objects,
        provenance=SubgraphProvenance(
            fixture_name=os.path.basename(target_path),
            sha256=sha256,
            causality_policy="causal_history (directed CSR)",
        ),
    )
