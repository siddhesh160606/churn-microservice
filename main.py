import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

# Load the saved Scikit-Learn Pipeline model
model = joblib.load("model_pipeline.pkl")


# Define data schema for input validation using Pydantic
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    contract: str


@app.get("/")
def home():
    return {
        "status": "API is online",
        "message": "Send POST request to /predict for churn risk calculation.",
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert Pydantic model payload to pandas DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    # Get churn probability from the pipeline model
    churn_prob = model.predict_proba(input_df)[0][1]
    prediction = int(churn_prob > 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(churn_prob), 2),
        "risk_level": "High" if churn_prob > 0.5 else "Low",
    }