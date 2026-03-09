import pickle
import numpy as np
import os
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# ======================================================
# PATHS
# ======================================================

BASE_PATH = "D:/ML Lab/projects"

MODEL_PATH = os.path.join(BASE_PATH,"model.pkl")
SCALER_PATH = os.path.join(BASE_PATH,"scaler.pkl")
SELECTOR_PATH = os.path.join(BASE_PATH,"selector.pkl")
FEATURE_PATH = os.path.join(BASE_PATH,"selected_features.pkl")

# ======================================================
# LOAD ARTIFACTS
# ======================================================

model = pickle.load(open(MODEL_PATH,"rb"))
scaler = pickle.load(open(SCALER_PATH,"rb"))
selector = pickle.load(open(SELECTOR_PATH,"rb"))
selected_features = pickle.load(open(FEATURE_PATH,"rb"))

# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(
    title="Stock ML Prediction API",
    description="Backend API for stock prediction model",
    version="1.0"
)

# ======================================================
# INPUT SCHEMA
# ======================================================

class StockFeatures(BaseModel):

    features: list


# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/")
def home():

    return {"message":"Stock ML API Running"}


# ======================================================
# PREDICTION ENDPOINT
# ======================================================

@app.post("/predict")

def predict(data: StockFeatures):

    X = np.array(data.features).reshape(1,-1)

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    return {
        "prediction": int(pred),
        "signal": "UP" if pred == 1 else "DOWN"
    }