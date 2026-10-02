cat << 'EOF' > app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="HR Analytics Prediction Service",
    version="1.0"
)

MODEL_PATH = "models/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "HR Analytics API",
        "model_loaded": model is not None
    }

class EmployeeData(BaseModel):
    satisfaction_level: float = 0.5
    last_evaluation: float = 0.7
    number_project: int = 3
    average_montly_hours: int = 200
    time_spend_company: int = 3
    work_accident: int = 0
    promotion_last_5years: int = 0

@app.post("/predict")
def predict(data: EmployeeData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready or not loaded.")
    
    features = [[
        data.satisfaction_level,
        data.last_evaluation,
        data.number_project,
        data.average_montly_hours,
        data.time_spend_company,
        data.work_accident,
        data.promotion_last_5years
    ]]
    
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(proba) if proba is not None else None
    }
EOF