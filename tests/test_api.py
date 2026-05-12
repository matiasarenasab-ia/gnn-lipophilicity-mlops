import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"]   == "ok"
    assert "model"          in data
    assert "val_rmse"       in data


def test_predict_valid_molecule():
    payload = {
        "node_features": [[0.1]*9, [0.2]*9, [0.3]*9, [0.4]*9],
        "edge_index":    [[0,1,1,2,2,3], [1,0,2,1,3,2]]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "lipophilicity_logD" in data
    assert "model"              in data
    assert "num_atoms"          in data
    assert data["num_atoms"]    == 4
    assert isinstance(data["lipophilicity_logD"], float)


def test_predict_wrong_features():
    payload = {
        "node_features": [[0.1]*5],   # ← 5 features en vez de 9
        "edge_index":    [[], []]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_molecule():
    payload = {
        "node_features": [],           # ← molécula vacía
        "edge_index":    [[], []]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_edge_index():
    payload = {
        "node_features": [[0.1]*9, [0.2]*9],
        "edge_index":    [[0,1]]       # ← solo una lista en vez de dos
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422