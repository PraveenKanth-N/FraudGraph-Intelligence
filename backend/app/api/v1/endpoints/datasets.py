import os
import hashlib
from typing import List, Optional
from fastapi import APIRouter
import numpy as np
from pydantic import BaseModel

router = APIRouter()

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

class DatasetFixtureItem(BaseModel):
    id: str
    name: str
    file_name: str
    num_nodes: int
    num_edges: int
    feature_dim: int
    size_bytes: int
    is_synthetic: bool
    description: str
    sha256: str
    is_available: bool = True

class DatasetListResponse(BaseModel):
    status: str
    total_count: int
    fixtures: List[DatasetFixtureItem]

CATALOG_FIXTURES = [
    {
        "id": "synthetic_fraud_10k",
        "name": "Synthetic Temporal Fraud Graph (10k)",
        "file_name": "synthetic_fraud_10k.npz",
        "is_synthetic": True,
        "description": "Temporal transaction graph with 3.5% fraud prevalence (350 frauds), card/device relational keys, and 30-day timeline.",
    },
    {
        "id": "synthetic_tier_a_10k",
        "name": "Synthetic Tier A Benchmark Graph (10k)",
        "file_name": "synthetic_tier_a_10k.npz",
        "is_synthetic": True,
        "description": "Controlled graph benchmark fixture with uniform degree distribution for baseline neighbor aggregation.",
    },
    {
        "id": "synthetic_skewed_10k",
        "name": "Synthetic Power-Law Skewed Graph (10k)",
        "file_name": "synthetic_skewed_10k.npz",
        "is_synthetic": True,
        "description": "Power-law degree distributed graph designed to stress-test warp divergence and degree-aware load balancing.",
    },
    {
        "id": "toy_graph_5nodes",
        "name": "Toy Verification Graph (5 Nodes)",
        "file_name": "toy_graph_5nodes.npz",
        "is_synthetic": True,
        "description": "Minimal deterministic test fixture for operator mathematical tolerance validation and unit tests.",
    },
]

def inspect_fixture_on_disk(file_name: str, base_dir: Optional[str] = None) -> tuple[bool, int, str, int, int, int]:
    """Inspect a dataset fixture file on disk without fallback fabrication.
    
    Returns:
        tuple of (is_available, size_bytes, sha256_hash, num_nodes, num_edges, feature_dim).
        If the file is absent or inaccessible, returns (False, 0, "", 0, 0, 0).
    """
    target_dir = base_dir or os.path.join(WORKSPACE_ROOT, "data", "fixtures")
    f_path = os.path.join(target_dir, file_name)

    if not os.path.exists(f_path):
        return False, 0, "", 0, 0, 0

    try:
        size_bytes = os.path.getsize(f_path)
        with open(f_path, "rb") as fp:
            sha256 = hashlib.sha256(fp.read()).hexdigest()

        # Load CSR arrays to verify integrity and extract genuine dimensions
        with np.load(f_path) as data:
            num_nodes = int(len(data["row_ptr"]) - 1)
            num_edges = int(len(data["col_idx"]))
            features = data["features"]
            feature_dim = int(features.shape[1]) if features.ndim > 1 else 1

        return True, size_bytes, sha256, num_nodes, num_edges, feature_dim
    except Exception:
        # If file exists on disk but is unreadable as a valid NPZ
        try:
            size_bytes = os.path.getsize(f_path)
            with open(f_path, "rb") as fp:
                sha256 = hashlib.sha256(fp.read()).hexdigest()
        except Exception:
            size_bytes = 0
            sha256 = ""
        return False, size_bytes, sha256, 0, 0, 0

@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(base_dir: Optional[str] = None):
    """List the authentic dataset fixtures verified in the repository."""
    fixtures = []
    for spec in CATALOG_FIXTURES:
        is_avail, size_b, sha, nodes, edges, f_dim = inspect_fixture_on_disk(
            spec["file_name"], base_dir=base_dir
        )
        fixtures.append(
            DatasetFixtureItem(
                id=spec["id"],
                name=spec["name"],
                file_name=spec["file_name"],
                num_nodes=nodes,
                num_edges=edges,
                feature_dim=f_dim,
                size_bytes=size_b,
                is_synthetic=spec["is_synthetic"],
                description=spec["description"],
                sha256=sha,
                is_available=is_avail,
            )
        )

    return DatasetListResponse(
        status="success",
        total_count=len(fixtures),
        fixtures=fixtures,
    )
