import os
import joblib
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "model.joblib")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "app", "feature_order.json")


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"El archivo {MODEL_PATH} no existe."


def test_feature_order_file_exists():
    assert os.path.exists(FEATURE_ORDER_PATH), f"El archivo {FEATURE_ORDER_PATH} no existe."


def test_model_can_predict_proba():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # Crear una fila dummy con el mismo número de columnas
    row = pd.DataFrame([[0] * len(feature_order)], columns=feature_order)

    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0
