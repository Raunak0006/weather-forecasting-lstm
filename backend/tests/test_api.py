"""Automated API Endpoint Tests for Weather Forecasting Service."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /api/health responds with 200 and expected schema keys."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "dataset_loaded" in data
    assert "version" in data


def test_dataset_info_endpoint():
    """Verify GET /api/dataset/info returns valid metadata and statistics."""
    response = client.get("/api/dataset/info")
    assert response.status_code == 200
    data = response.json()
    assert "record_count" in data
    assert "columns" in data
    assert "temperature_stats" in data
    stats = data["temperature_stats"]
    assert "min" in stats and "max" in stats and "mean" in stats


def test_history_endpoint():
    """Verify GET /api/history returns time-series observations."""
    response = client.get("/api/history?limit=50")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "total_records" in data
    assert isinstance(data["data"], list)
    if len(data["data"]) > 0:
        point = data["data"][0]
        assert "timestamp" in point
        assert "temperature" in point


def test_training_status_endpoint():
    """Verify GET /api/model/status returns active state."""
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["idle", "training", "completed", "failed"]
    assert "progress" in data
    assert "message" in data


def test_metrics_before_training():
    """Verify GET /api/model/metrics handles untrained state."""
    response = client.get("/api/model/metrics")
    # If not trained, should return 404
    assert response.status_code in [200, 404]


def test_predict_validation():
    """Verify POST /api/predict validates horizon input."""
    # Invalid horizon < 1
    response = client.post("/api/predict", json={"horizon": 0})
    assert response.status_code == 422  # Pydantic validation error


def test_upload_invalid_file():
    """Verify POST /api/data/upload rejects non-CSV files."""
    files = {"file": ("test.txt", b"invalid text data", "text/plain")}
    response = client.post("/api/data/upload", files=files)
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_column_mapping_missing_cols():
    """Verify POST /api/data/columns rejects nonexistent columns."""
    response = client.post(
        "/api/data/columns",
        json={
            "timestamp_column": "nonexistent_ts",
            "temperature_column": "nonexistent_temp",
        },
    )
    assert response.status_code == 400
