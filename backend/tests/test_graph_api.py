import os
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

CANONICAL_NPZ_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "fixtures", "synthetic_fraud_10k.npz")
)


def test_isolated_node_degree_and_subgraph():
    """Verify node #1 degree in canonical fixture and confirm isolated subgraph behavior."""
    with np.load(CANONICAL_NPZ_PATH) as data:
        row_ptr = data["row_ptr"]
        deg_1 = int(row_ptr[2] - row_ptr[1])
        assert deg_1 == 0, f"Expected node 1 degree to be 0, got {deg_1}"

    res = client.get("/api/v1/graph/subgraph?node_id=1&hops=1")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "success"
    assert payload["query"]["root_node_id"] == 1
    assert payload["query"]["requested_hops"] == 1
    assert payload["query"]["actual_hops"] == 0

    assert payload["summary"]["total_nodes"] == 1
    assert payload["summary"]["total_edges"] == 0
    assert payload["summary"]["is_truncated"] is False
    assert payload["summary"]["node_cap_hit"] is False
    assert payload["summary"]["edge_cap_hit"] is False

    nodes = payload["nodes"]
    assert len(nodes) == 1
    root = nodes[0]
    assert root["id"] == 1
    assert root["depth"] == 0
    assert root["is_root"] is True
    assert root["degree"] == 0
    assert len(payload["edges"]) == 0


def test_hub_node_traversal_directed_semantics():
    """Verify hub node #5407 (degree 252) traversal and directed CSR edge semantics."""
    res = client.get("/api/v1/graph/subgraph?node_id=5407&hops=1&max_nodes=300")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "success"
    assert payload["query"]["root_node_id"] == 5407
    assert payload["query"]["direction"] == "directed_csr"

    with np.load(CANONICAL_NPZ_PATH) as data:
        row_ptr = data["row_ptr"]
        col_idx = data["col_idx"]
        expected_neighbors = set(int(v) for v in col_idx[row_ptr[5407] : row_ptr[5408]])

    nodes = payload["nodes"]
    returned_ids = {n["id"] for n in nodes}
    assert 5407 in returned_ids
    assert len(nodes) == len(expected_neighbors) + 1

    # Check directed edges: every edge must strictly originate or exist in canonical CSR
    edges = payload["edges"]
    for e in edges:
        u, v = e["source"], e["target"]
        assert u in returned_ids
        assert v in returned_ids
        actual_u_nbrs = set(int(x) for x in col_idx[row_ptr[u] : row_ptr[u + 1]])
        assert v in actual_u_nbrs, f"Directed edge ({u}, {v}) not found in CSR slice for node {u}"


def test_subgraph_multi_hop_depth_and_actual_hops():
    """Verify 2-hop traversal depth and actual_hops calculation."""
    res = client.get("/api/v1/graph/subgraph?node_id=5407&hops=2&max_nodes=150")
    assert res.status_code == 200
    payload = res.json()
    assert payload["query"]["requested_hops"] == 2
    assert payload["query"]["actual_hops"] <= 2
    for n in payload["nodes"]:
        assert n["depth"] in (0, 1, 2)


def test_deterministic_deduplication_and_ordering():
    """Verify that returned nodes and edges are deterministically deduplicated and ordered."""
    res = client.get("/api/v1/graph/subgraph?node_id=5407&hops=1&max_nodes=100")
    assert res.status_code == 200
    payload = res.json()

    node_ids = [n["id"] for n in payload["nodes"]]
    assert len(node_ids) == len(set(node_ids)), "Duplicate node IDs detected in response"

    edges = [(e["source"], e["target"]) for e in payload["edges"]]
    assert len(edges) == len(set(edges)), "Duplicate directed edges detected in response"
    assert edges == sorted(edges), "Edges must be sorted deterministically by (source, target)"


def test_invalid_inputs_client_errors():
    """Verify HTTP 400 or 422 for boundary violations and invalid types."""
    # Out of bounds node_id (< 0)
    res = client.get("/api/v1/graph/subgraph?node_id=-1")
    assert res.status_code == 400
    assert "out of bounds" in res.json()["detail"].lower()

    # Out of bounds node_id (>= num_nodes)
    res = client.get("/api/v1/graph/subgraph?node_id=10000")
    assert res.status_code == 400
    assert "out of bounds" in res.json()["detail"].lower()

    # Invalid hops (< 1 or > 3)
    res = client.get("/api/v1/graph/subgraph?node_id=10&hops=0")
    assert res.status_code == 422

    res = client.get("/api/v1/graph/subgraph?node_id=10&hops=4")
    assert res.status_code == 422

    # Invalid max_nodes
    res = client.get("/api/v1/graph/subgraph?node_id=10&max_nodes=0")
    assert res.status_code == 422

    res = client.get("/api/v1/graph/subgraph?node_id=10&max_nodes=301")
    assert res.status_code == 422

    # Invalid max_edges
    res = client.get("/api/v1/graph/subgraph?node_id=10&max_edges=0")
    assert res.status_code == 422

    res = client.get("/api/v1/graph/subgraph?node_id=10&max_edges=1001")
    assert res.status_code == 422


def test_node_and_edge_caps_with_truncation_metadata():
    """Verify that node and edge caps produce honest truncation flags and metadata."""
    # Node cap
    res = client.get("/api/v1/graph/subgraph?node_id=5407&hops=1&max_nodes=10")
    assert res.status_code == 200
    payload = res.json()
    assert payload["summary"]["is_truncated"] is True
    assert payload["summary"]["node_cap_hit"] is True
    assert payload["summary"]["total_nodes"] <= 10
    assert "MAX_NODES_LIMIT_REACHED" in payload["summary"]["truncation_reason"]

    # Edge cap
    res = client.get("/api/v1/graph/subgraph?node_id=5407&hops=1&max_nodes=100&max_edges=5")
    assert res.status_code == 200
    payload = res.json()
    assert payload["summary"]["is_truncated"] is True
    assert payload["summary"]["edge_cap_hit"] is True
    assert payload["summary"]["total_edges"] <= 5
    assert "MAX_EDGES_LIMIT_REACHED" in payload["summary"]["truncation_reason"]


def test_label_quarantine():
    """Verify that mutating node fraud labels does not alter graph traversal topology."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple deterministic 5-node graph: 0 -> 1 -> 2, 0 -> 3 -> 4
        row_ptr = np.array([0, 2, 3, 3, 4, 4], dtype=np.int32)
        col_idx = np.array([1, 3, 2, 4], dtype=np.int32)
        timestamps = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        labels_a = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        labels_b = np.array([1, 1, 1, 1, 1], dtype=np.int32)

        path_a = os.path.join(tmpdir, "graph_a.npz")
        path_b = os.path.join(tmpdir, "graph_b.npz")

        np.savez_compressed(path_a, row_ptr=row_ptr, col_idx=col_idx, timestamps=timestamps, labels=labels_a)
        np.savez_compressed(path_b, row_ptr=row_ptr, col_idx=col_idx, timestamps=timestamps, labels=labels_b)

        res_a = client.get(f"/api/v1/graph/subgraph?node_id=0&hops=2&npz_path={path_a}")
        res_b = client.get(f"/api/v1/graph/subgraph?node_id=0&hops=2&npz_path={path_b}")

        assert res_a.status_code == 200
        assert res_b.status_code == 200

        data_a = res_a.json()
        data_b = res_b.json()

        # Topology and edges must be 100% identical
        assert [(n["id"], n["depth"]) for n in data_a["nodes"]] == [(n["id"], n["depth"]) for n in data_b["nodes"]]
        assert [(e["source"], e["target"]) for e in data_a["edges"]] == [(e["source"], e["target"]) for e in data_b["edges"]]

        # Labels must reflect actual distinct values
        assert all(n["fraud_label"] == 0 for n in data_a["nodes"])
        assert all(n["fraud_label"] == 1 for n in data_b["nodes"])


def test_dataset_missing_and_corrupt():
    """Verify safe error responses for missing and malformed datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "non_existent.npz")
        res = client.get(f"/api/v1/graph/subgraph?node_id=0&npz_path={non_existent}")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

        corrupt = os.path.join(tmpdir, "corrupt.npz")
        np.savez_compressed(corrupt, dummy=np.array([1, 2, 3]))
        res2 = client.get(f"/api/v1/graph/subgraph?node_id=0&npz_path={corrupt}")
        assert res2.status_code == 503
        assert "corrupt or invalid" in res2.json()["detail"].lower()
