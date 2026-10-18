from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "FraudGraph Intelligence" in data["service"]

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project_id"] == "FGI"
    assert "compute_engine" in data
    assert "local_gpu_detected" in data["compute_engine"]
    assert data["compute_engine"]["cuda_compiler_status"] == "UNAVAILABLE_LOCAL_COLAB_READY"
    assert data["compute_engine"]["colab_workflow_enabled"] is True
