from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import os

app = FastAPI(title="Quick MLOps API")

MODEL_NAME = "quick-mlops-model"
MODEL_ALIAS = "champion"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

preprocessor = joblib.load("models/preprocessor.joblib")

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)

model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array([[data.feature1, data.feature2, data.feature3]])
    arr_t = preprocessor.transform(arr)
    pred = model.predict(arr_t)[0]
    return {"prediction": int(pred)}
