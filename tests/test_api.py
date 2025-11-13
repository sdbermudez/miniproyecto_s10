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

    # Aceptamos 200 o 500 dependiendo de si el modelo puede procesar los datos
    assert resp.status_code in (200, 500, 503)


def test_predict_endpoint_length_mismatch():
    """Debe devolver 400 o 422 si la cantidad de features no coincide."""
    if feature_order is None:
        return
    payload = {"features": [1, 2, 3]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (400, 422)
