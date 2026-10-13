from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ACI606 Cloud API",
    description="Servicio remoto para clasificacion de movimiento",
    version="1.0.0",
)

MODEL_PATH = Path("models/modelo_giro.joblib")
model = joblib.load(MODEL_PATH)

class MovementInput(BaseModel):
    u_left: float
    u_right: float
    speed: float
    theta: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "modelo_giro.joblib",
    }

@app.post("/predict")
def predict(data: MovementInput):
    X = pd.DataFrame([{
        "u_left": data.u_left,
        "u_right": data.u_right,
        "speed": data.speed,
        "theta": data.theta,
    }])
    
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    
    return {
        "movement": "turn" if prediction == 1 else "straight",
        "class": prediction,
        "probability_turn": round(probability, 4),
    }