from fastapi.testclient import TestClient
from app.api import app, feature_order, model

client = TestClient(app)

def test_health_endpoint():
    """Verifica que el endpoint /health funcione correctamente."""
    resp = client.get("/health")
    if model is None or feature_order is None:
        assert resp.status_code == 503
    else:
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "features_loaded" in data


def test_home_endpoint():
    """Verifica que el endpoint raíz devuelva información básica."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "version" in data


def test_predict_endpoint_ok_or_unavailable():
    """Prueba el endpoint /predict según disponibilidad del modelo."""
    payload = {"features": [1] * (len(feature_order) if feature_order else 10)}  # dummy input

    resp = client.post("/predict", json=payload)

    if model is None or feature_order is None:
        # Si el modelo no está cargado, debe dar 503
        assert resp.status_code == 503
    else:
        # Si el modelo está cargado, debe responder 200
        assert resp.status_code == 200
        data = resp.json()
        assert "heart_disease_probability" in data
        assert "prediction" in data
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["heart_disease_probability"] <= 1.0


def test_predict_endpoint_length_mismatch():
    """Debe devolver 400 si la cantidad de features no coincide."""
    if feature_order is None:
        return  # no aplica si no se cargó feature_order
    payload = {"features": [1, 2, 3]}  # menos columnas que las esperadas
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
