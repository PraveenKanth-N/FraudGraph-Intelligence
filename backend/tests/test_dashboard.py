from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["service"] == "FraudGraph Intelligence"

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "compute_engine" in data
    assert "local_gpu_detected" in data["compute_engine"]

def test_dashboard_summary():
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["project"] == "FraudGraph Intelligence"

    # Verify genuine dataset fields
    ds = data["dataset"]
    assert ds["name"] == "synthetic_fraud_10k"
    assert ds["num_nodes"] == 10000
    assert ds["num_edges"] == 231984
    assert ds["feature_dim"] == 32
    assert ds["fraud_count"] == 350
    assert ds["fraud_rate_pct"] == 3.5
    assert ds["is_synthetic"] is True

    # Verify evaluation summary
    ev = data["evaluation"]
    assert ev["top_model"] == "Logistic Regression (max)"
    assert ev["top_test_pr_auc"] > 0
    assert ev["top_test_roc_auc"] > 0
    assert "ADR-007" in ev["adr_007_notice"]

    # Verify benchmark summary
    bm = data["benchmarks"]
    assert bm["session_id"] == "session_20261024_230008_smoke"
    assert bm["total_runs"] == 36
    assert bm["measured_cpu_runs"] == 12
    assert bm["skipped_gpu_runs"] == 24
    assert bm["top_speedup_vs_cpu_ref"] == 44.60

    # Verify hardware summary
    hw = data["hardware"]
    assert hw["local_compiler_status"] == "UNAVAILABLE_LOCAL_COLAB_READY"
    assert "Tesla T4" in hw["canonical_colab_gpu"]

def test_list_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_count"] == 4
    fixtures = {item["id"]: item for item in data["fixtures"]}
    assert "synthetic_fraud_10k" in fixtures
    assert "synthetic_tier_a_10k" in fixtures
    assert "synthetic_skewed_10k" in fixtures
    assert "toy_graph_5nodes" in fixtures

    # Verify authentic fixture values derived from actual files
    fraud_10k = fixtures["synthetic_fraud_10k"]
    assert fraud_10k["is_available"] is True
    assert fraud_10k["size_bytes"] == 1429459
    assert fraud_10k["num_nodes"] == 10000
    assert fraud_10k["num_edges"] == 231984
    assert fraud_10k["feature_dim"] == 32
    assert fraud_10k["sha256"] == "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"

    toy = fixtures["toy_graph_5nodes"]
    assert toy["is_available"] is True
    assert toy["size_bytes"] == 1406
    assert toy["num_nodes"] == 5
    assert toy["num_edges"] == 8
    assert toy["feature_dim"] == 4
    assert toy["sha256"] == "3ea8f702bc1f944d9f164222bad637666307c7be8e4a9767f813e40edc137909"

def test_list_datasets_missing_file_honest_representation(tmp_path):
    # Query datasets pointing to an empty temporary directory where no fixtures exist
    response = client.get(f"/api/v1/datasets?base_dir={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_count"] == 4

    for item in data["fixtures"]:
        # Unavailability must be honestly represented without hardcoded fallbacks
        assert item["is_available"] is False
        assert item["size_bytes"] == 0
        assert item["sha256"] == ""
        assert item["num_nodes"] == 0
        assert item["num_edges"] == 0
        assert item["feature_dim"] == 0

def test_list_datasets_corrupt_file_handling(tmp_path):
    # Create a corrupt fixture file (valid on filesystem, but unreadable as NPZ)
    corrupt_file = tmp_path / "toy_graph_5nodes.npz"
    corrupt_file.write_bytes(b"CORRUPTED_BINARY_CONTENT_NOT_AN_NPZ")

    response = client.get(f"/api/v1/datasets?base_dir={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    fixtures = {item["id"]: item for item in data["fixtures"]}

    toy = fixtures["toy_graph_5nodes"]
    assert toy["is_available"] is False
    assert toy["size_bytes"] == len(b"CORRUPTED_BINARY_CONTENT_NOT_AN_NPZ")
    assert len(toy["sha256"]) == 64
    assert toy["num_nodes"] == 0
    assert toy["num_edges"] == 0
    assert toy["feature_dim"] == 0
