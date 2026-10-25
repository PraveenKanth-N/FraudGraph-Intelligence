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


def test_default_pagination_and_metadata():
    """Verify default parameters: offset=0, limit=50, filter=all against canonical fixture."""
    res = client.get("/api/v1/transactions")
    assert res.status_code == 200
    payload = res.json()

    assert payload["status"] == "success"
    assert payload["total_dataset_records"] == 10000
    assert payload["total_fraud_records"] == 350
    assert payload["total_licit_records"] == 9650
    assert payload["filtered_count"] == 10000
    assert payload["offset"] == 0
    assert payload["limit"] == 50
    assert payload["returned_count"] == 50
    assert payload["has_next"] is True
    assert payload["has_prev"] is False
    assert payload["current_filter"] == "all"
    assert "source_card and target_device are omitted" in payload["schema_notes"]

    items = payload["items"]
    assert len(items) == 50
    assert items[0]["tx_id"] == 0
    assert items[49]["tx_id"] == 49


def test_all_three_filters():
    """Verify that all three filters (all, fraud, licit) derive correct counts and isolate labels."""
    # Filter: fraud
    res_fraud = client.get("/api/v1/transactions?filter=fraud&limit=50")
    assert res_fraud.status_code == 200
    data_fraud = res_fraud.json()
    assert data_fraud["filtered_count"] == 350
    assert data_fraud["returned_count"] == 50
    for itm in data_fraud["items"]:
        assert itm["fraud_label"] == 1
        assert itm["is_fraud"] is True
        assert itm["risk_tier"] == "CONFIRMED_FRAUD"

    # Filter: licit
    res_licit = client.get("/api/v1/transactions?filter=licit&limit=50")
    assert res_licit.status_code == 200
    data_licit = res_licit.json()
    assert data_licit["filtered_count"] == 9650
    assert data_licit["returned_count"] == 50
    for itm in data_licit["items"]:
        assert itm["fraud_label"] == 0
        assert itm["is_fraud"] is False
        assert itm["risk_tier"] == "LICIT_CLEARED"

    # Filter: all
    res_all = client.get("/api/v1/transactions?filter=all&limit=20")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["filtered_count"] == 10000
    assert data_all["returned_count"] == 20


def test_filter_before_pagination_slicing():
    """Verify that label filtering occurs strictly before pagination slicing."""
    with np.load(CANONICAL_NPZ_PATH) as data:
        labels = data["labels"]
        expected_fraud_indices = list(np.where(labels == 1)[0])

    res_p1 = client.get("/api/v1/transactions?filter=fraud&offset=0&limit=20")
    res_p2 = client.get("/api/v1/transactions?filter=fraud&offset=20&limit=20")

    assert res_p1.status_code == 200
    assert res_p2.status_code == 200

    p1_ids = [itm["tx_id"] for itm in res_p1.json()["items"]]
    p2_ids = [itm["tx_id"] for itm in res_p2.json()["items"]]

    assert p1_ids == expected_fraud_indices[:20]
    assert p2_ids == expected_fraud_indices[20:40]
    assert set(p1_ids).isdisjoint(set(p2_ids))


def test_pagination_boundaries_and_empty_pages():
    """Verify boundary conditions: last partial page, exact boundary, and offset past end."""
    # Last partial page of fraud records (350 total: offset 340, limit 20 -> 10 items)
    res_last = client.get("/api/v1/transactions?filter=fraud&offset=340&limit=20")
    assert res_last.status_code == 200
    d_last = res_last.json()
    assert d_last["returned_count"] == 10
    assert d_last["has_next"] is False
    assert d_last["has_prev"] is True

    # Offset at exact boundary
    res_boundary = client.get("/api/v1/transactions?filter=fraud&offset=350&limit=20")
    assert res_boundary.status_code == 200
    d_b = res_boundary.json()
    assert d_b["returned_count"] == 0
    assert d_b["has_next"] is False

    # Offset well past end
    res_past = client.get("/api/v1/transactions?filter=all&offset=50000&limit=50")
    assert res_past.status_code == 200
    d_past = res_past.json()
    assert d_past["returned_count"] == 0
    assert d_past["has_next"] is False


def test_invalid_inputs_client_errors():
    """Verify client error codes (400 / 422) for invalid parameters."""
    # Negative offset
    res = client.get("/api/v1/transactions?offset=-1")
    assert res.status_code == 422

    # Limit zero
    res = client.get("/api/v1/transactions?limit=0")
    assert res.status_code == 422

    # Limit > 100
    res = client.get("/api/v1/transactions?limit=101")
    assert res.status_code == 422

    # Unknown filter
    res = client.get("/api/v1/transactions?filter=invalid_filter")
    assert res.status_code == 400
    assert "invalid filter" in res.json()["detail"].lower()


def test_honest_entity_key_and_feature_presentation():
    """Verify that source_card and target_device are honestly null and features are presented accurately."""
    res = client.get("/api/v1/transactions?limit=5")
    assert res.status_code == 200
    items = res.json()["items"]

    with np.load(CANONICAL_NPZ_PATH) as data:
        row_ptr = data["row_ptr"]
        features = data["features"]

    for itm in items:
        assert itm["source_card"] is None
        assert itm["target_device"] is None
        assert "discarded" in itm["notes"].lower()

        tx_id = itm["tx_id"]
        expected_deg = int(row_ptr[tx_id + 1] - row_ptr[tx_id])
        assert itm["in_degree"] == expected_deg

        expected_feats = [round(float(v), 4) for v in features[tx_id, :4]]
        assert itm["feature_sample"] == expected_feats


def test_dataset_missing_and_corrupt():
    """Verify safe error responses for missing and malformed datasets in transaction endpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "missing.npz")
        res = client.get(f"/api/v1/transactions?npz_path={non_existent}")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

        corrupt = os.path.join(tmpdir, "corrupt.npz")
        np.savez_compressed(corrupt, labels=np.array([]))  # empty labels
        res2 = client.get(f"/api/v1/transactions?npz_path={corrupt}")
        assert res2.status_code == 503
