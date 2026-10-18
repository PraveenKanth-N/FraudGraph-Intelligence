"""
Fixture Generation Script
Generates standard deterministic test and benchmark fixtures into data/fixtures/.
"""

import json
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.cpu_engine.csr_graph import CSRGraph
from backend.cpu_engine.synthetic_generator import generate_synthetic_graph


def build_fixtures():
    fixtures_dir = Path("data/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Toy 5-node graph fixture
    toy = CSRGraph.from_edge_list(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1)],
        feature_dim=4,
        metadata={"name": "toy_graph_5nodes", "description": "5-node graph with 1 isolated node for unit testing"},
    )
    toy_path = toy.save(fixtures_dir / "toy_graph_5nodes.npz")
    print(f"Generated {toy_path.name}: {toy.get_stats()}")

    # 2. Tier A benchmark fixture (10,000 nodes, power-law, D=32)
    tier_a = generate_synthetic_graph(
        num_nodes=10000,
        avg_degree=10.0,
        feature_dim=32,
        topology="power_law",
        is_directed=True,
        seed=42,
    )
    tier_a_path = tier_a.save(fixtures_dir / "synthetic_tier_a_10k.npz")
    print(f"Generated {tier_a_path.name}: {tier_a.get_stats()}")


if __name__ == "__main__":
    build_fixtures()
