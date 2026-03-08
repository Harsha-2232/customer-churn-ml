from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("../model/churn_model.pkl")


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict-churn")
def predict_churn(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability > 0.7:
        risk = "High Risk"
    elif probability > 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_category": risk
    }