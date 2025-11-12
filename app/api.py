# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json



# Cargar el modelo entrenado
model = joblib.load("app/model.joblib")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Heart Failure Prediction API",
    description="Predice el riesgo de falla cardíaca a partir de datos clínicos",
    version="1.0"
)

# Definir el esquema de entrada
class Input(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Falla Cardíaca"}

with open("app/feature_order.json") as f:
    feature_order = json.load(f)


@app.post("/predict")
def predict(data: Input):
    """
    Endpoint de predicción: recibe una lista de features numéricas y devuelve probabilidad.
    """
    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    return {
        "heart_disease_probability": round(float(proba), 4),
        "prediction": int(proba > 0.5)
    }
