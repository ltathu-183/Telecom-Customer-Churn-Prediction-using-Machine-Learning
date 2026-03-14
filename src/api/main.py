"""
FastAPI application for churn prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="Telco Churn Prediction API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pkl')
model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    """Customer data schema."""
    customer_id: str = Field(..., example="CUST_001")
    tenure: int = Field(..., ge=0, example=24)
    monthly_charges: float = Field(..., ge=0, example=79.85)
    contract: str = Field(..., example="Month-to-month")


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    customer_id: str
    churn_probability: float
    risk_level: str
    recommended_action: str


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn for a customer."""
    try:
        # Convert to dataframe
        df = pd.DataFrame([customer.dict()])
        
        # Predict (simplified - in production use full pipeline)
        features = pd.DataFrame({
            'tenure': [customer.tenure],
            'MonthlyCharges': [customer.monthly_charges]
        })
        
        churn_prob = model.predict_proba(features)[0][1]
        
        # Determine risk
        if churn_prob > 0.7:
            risk = "High Risk"
            action = "VIP retention call"
        elif churn_prob > 0.3:
            risk = "Medium Risk"
            action = "Send promotion"
        else:
            risk = "Low Risk"
            action = "No action"
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            risk_level=risk,
            recommended_action=action
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}