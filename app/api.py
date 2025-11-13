# app/api.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import List, Any
from pathlib import Path
import joblib
import pandas as pd
import json
import logging

# Logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

# Rutas absolutas basadas en este archivo
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.json"

# Cargar artefactos al iniciar (fallará rápido si falta algo)
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Modelo cargado desde {MODEL_PATH}")
except Exception as e:
    logger.exception(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")
    model = None

try:
    with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
        feature_order = json.load(f)
    logger.info(f"Orden de features cargado ({len(feature_order)} columnas).")
except Exception as e:
    logger.exception(f"No se pudo cargar feature_order.json desde {FEATURE_ORDER_PATH}: {e}")
    feature_order = None

if model is None or feature_order is None:
    logger.warning("Modelo o feature_order no están disponibles. La API devolverá 503 en /predict hasta corregirlo.")

app = FastAPI(
    title="Heart Failure Prediction API",
    description="API para predecir riesgo de falla cardíaca. Envía `features` en el orden esperado.",
    version="1.0"
)

class Input(BaseModel):
    # Validar longitud exacta si feature_order existe; usamos conlist para validar tipos homogéneos,
    # pero aquí permitimos heterogeneidad (Any) y validamos longitud y tipos en validator.
    features: List[Any] = Field(..., example=[57, "Male", "ATA", 140, 240, 0, "Normal", 160, "N", 1.0, "Up"])

    @validator("features")
    def check_length(cls, v):
        if feature_order is None:
            # No conocemos el orden aún; permitimos pasar, pero esto es raro en producción
            return v
        if len(v) != len(feature_order):
            raise ValueError(f"Se esperaban {len(feature_order)} valores según feature_order, llegaron {len(v)}.")
        return v

@app.get("/health")
def health():
    status = {"model_loaded": model is not None, "features_loaded": feature_order is not None}
    if not all(status.values()):
        raise HTTPException(status_code=503, detail=status)
    return {"status": "ok", **status}

@app.get("/")
def home():
    return {
        "message": "API de Predicción de Falla Cardíaca",
        "version": "1.0",
        "n_features": len(feature_order) if feature_order else None,
    }

def coerce_row_to_dtypes(row_list, feature_order, reference_df=None):
    """
    Convierte tipos básicos intentando inferir. 
    Si reference_df (ej. train X) está disponible, se puede usar para tipos más robustos.
    Aquí hacemos conversiones simples: números str->float/int si posible.
    """
    coerced = []
    for v in row_list:
        # if already numeric-like list -> leave it
        if isinstance(v, (int, float, bool)) or v is None:
            coerced.append(v)
            continue
        # try int
        try:
            iv = int(v)
            coerced.append(iv)
            continue
        except Exception:
            pass
        # try float
        try:
            fv = float(v)
            coerced.append(fv)
            continue
        except Exception:
            pass
        # fallback: keep string
        coerced.append(v)
    return coerced

@app.post("/predict")
def predict(data: Input, request: Request):
    if model is None or feature_order is None:
        raise HTTPException(status_code=503, detail="Modelo o metadata no disponible en el servidor.")

    # Validación de longitud ya hecha por Pydantic, redundante por seguridad:
    if len(data.features) != len(feature_order):
        raise HTTPException(status_code=400, detail=f"Length mismatch: expected {len(feature_order)} features.")

    # Coerción simple de tipos
    row = coerce_row_to_dtypes(data.features, feature_order)

    # Construir DataFrame con el orden correcto
    try:
        X_df = pd.DataFrame([row], columns=feature_order)
    except Exception as e:
        logger.exception("Error construyendo DataFrame de entrada: %s", e)
        raise HTTPException(status_code=400, detail=f"Error construyendo DataFrame: {e}")

    # Predecir (manejar excepciones del modelo)
    try:
        proba = model.predict_proba(X_df)[0][1]
        prediction = int(proba > 0.5)
    except Exception as e:
        logger.exception("Error en predict_proba: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    return {
        "heart_disease_probability": round(float(proba), 4),
        "prediction": prediction,
        "prediction_label": "Enfermedad" if prediction == 1 else "No Enfermedad",
        "features_received": len(data.features)
    }
