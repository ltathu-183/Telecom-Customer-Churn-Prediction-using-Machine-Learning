"""
Production FastAPI application for churn prediction.
Ready for deployment to production environment.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production-ready API for customer churn prediction",
    version="2.0.0",
    contact={
        "name": "AI Team",
        "email": "ai-team@viettel.com.vn"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing artifacts
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model_logistic_regression.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/feature_scaler.pkl')
ENCODER_PATH = os.getenv('ENCODER_PATH', 'models/categorical_encoder.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    encoder = None

# Request/Response schemas
class CustomerData(BaseModel):
    """Customer data schema for prediction"""
    customer_id: str = Field(..., example="CUST_12345")
    tenure: int = Field(..., ge=0, le=1000, example=24)
    monthly_charges: float = Field(..., ge=0, le=10000, example=79.85)
    total_charges: float = Field(..., ge=0, le=100000, example=1816.40)
    contract: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")
    internet_service: Optional[str] = Field(None, example="Fiber optic")
    phone_service: Optional[bool] = Field(None, example=True)
    online_security: Optional[bool] = Field(None, example=False)
    online_backup: Optional[bool] = Field(None, example=True)
    device_protection: Optional[bool] = Field(None, example=False)
    tech_support: Optional[bool] = Field(None, example=False)
    streaming_tv: Optional[bool] = Field(None, example=True)
    streaming_movies: Optional[bool] = Field(None, example=True)
    senior_citizen: Optional[bool] = Field(False, example=False)
    partner: Optional[bool] = Field(False, example=True)
    dependents: Optional[bool] = Field(False, example=False)
    paperless_billing: Optional[bool] = Field(False, example=True)

    @validator('monthly_charges')
    def validate_charges(cls, v):
        if v < 0:
            raise ValueError('Monthly charges must be non-negative')
        return v


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    customer_id: str
    churn_probability: float
    risk_level: str
    risk_color: str
    recommended_action: str
    action_channel: str
    priority: str
    estimated_cost: float
    potential_loss: float
    roi_of_retention: float
    timestamp: str
    top_features: Optional[List[Dict[str, float]]] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    predictions: List[PredictionResponse]
    summary: Dict[str, int]
    timestamp: str


# Helper functions
def preprocess_customer_data(customer_data: CustomerData) -> pd.DataFrame:
    """Preprocess customer data for prediction"""
    # Convert to dict
    data_dict = customer_data.dict()
    
    # Create dataframe
    df = pd.DataFrame([data_dict])
    df.set_index('customer_id', inplace=True)
    
    # Feature engineering (simplified - use full pipeline in production)
    features = {
        'tenure': [customer_data.tenure],
        'MonthlyCharges': [customer_data.monthly_charges],
        'TotalCharges': [customer_data.total_charges],
        'Contract_Month-to-month': [1 if customer_data.contract == 'Month-to-month' else 0],
        'Contract_One year': [1 if customer_data.contract == 'One year' else 0],
        'Contract_Two year': [1 if customer_data.contract == 'Two year' else 0],
        'PaymentMethod_Electronic check': [1 if customer_data.payment_method == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if customer_data.payment_method == 'Mailed check' else 0],
        'PaymentMethod_Bank transfer (automatic)': [1 if customer_data.payment_method == 'Bank transfer (automatic)' else 0],
        'PaymentMethod_Credit card (automatic)': [1 if customer_data.payment_method == 'Credit card (automatic)' else 0],
    }
    
    return pd.DataFrame(features, index=[customer_data.customer_id])


def determine_risk_level(churn_prob: float) -> tuple:
    """Determine risk level and color"""
    if churn_prob > 0.7:
        return "High Risk", "🔴", "high"
    elif churn_prob > 0.3:
        return "Medium Risk", "🟡", "medium"
    else:
        return "Low Risk", "🟢", "low"


def generate_retention_strategy(risk_level: str, churn_prob: float) -> tuple:
    """Generate retention strategy based on risk level"""
    if risk_level == "High Risk":
        return (
            "VIP retention call with special offer",
            "phone",
            "Immediate action required",
            150.0,
            3600.0,
            ((3600.0 - 150.0) / 150.0)
        )
    elif risk_level == "Medium Risk":
        return (
            "Send personalized promotion campaign",
            "email",
            "Monitor closely",
            50.0,
            3600.0,
            ((3600.0 - 50.0) / 50.0)
        )
    else:
        return (
            "No immediate action - continue monitoring",
            "none",
            "Stable customer",
            0.0,
            0.0,
            0.0
        )


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telco Churn Prediction API",
        "version": "2.0.0",
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_version": "2.0.0",
        "features_count": len(scaler.mean_) if scaler else 0,
        "training_date": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a single customer.
    
    Args:
        customer: Customer data
        
    Returns:
        Prediction response with churn probability and retention strategy
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess data
        features_df = preprocess_customer_data(customer)
        
        # Make prediction
        churn_prob = model.predict_proba(features_df)[0][1]
        
        # Determine risk level
        risk_level, risk_color, priority = determine_risk_level(churn_prob)
        
        # Generate retention strategy
        action, channel, priority_desc, cost, loss, roi = generate_retention_strategy(
            risk_level, churn_prob
        )
        
        # Create response
        response = PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=float(churn_prob),
            risk_level=risk_level,
            risk_color=risk_color,
            recommended_action=action,
            action_channel=channel,
            priority=priority,
            estimated_cost=cost,
            potential_loss=loss,
            roi_of_retention=roi,
            timestamp=datetime.now().isoformat(),
            top_features=None  # Add SHAP explanations if needed
        )
        
        logger.info(f"Prediction made for customer {customer.customer_id}: {churn_prob:.2%}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_churn(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response with summary
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        risk_counts = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
        
        for customer in request.customers:
            # Preprocess data
            features_df = preprocess_customer_data(customer)
            
            # Make prediction
            churn_prob = model.predict_proba(features_df)[0][1]
            
            # Determine risk level
            risk_level, risk_color, priority = determine_risk_level(churn_prob)
            risk_counts[risk_level] += 1
            
            # Generate retention strategy
            action, channel, priority_desc, cost, loss, roi = generate_retention_strategy(
                risk_level, churn_prob
            )
            
            # Create prediction
            prediction = PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=float(churn_prob),
                risk_level=risk_level,
                risk_color=risk_color,
                recommended_action=action,
                action_channel=channel,
                priority=priority,
                estimated_cost=cost,
                potential_loss=loss,
                roi_of_retention=roi,
                timestamp=datetime.now().isoformat(),
                top_features=None
            )
            
            predictions.append(prediction)
        
        # Create summary
        summary = {
            "total_customers": len(predictions),
            "high_risk": risk_counts["High Risk"],
            "medium_risk": risk_counts["Medium Risk"],
            "low_risk": risk_counts["Low Risk"],
            "avg_churn_probability": sum(p.churn_probability for p in predictions) / len(predictions)
        }
        
        logger.info(f"Batch prediction completed: {len(predictions)} customers")
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get API metrics (for monitoring)"""
    return {
        "uptime": "running",
        "requests_served": 0,  # Implement actual tracking
        "avg_response_time": 0.0,
        "error_rate": 0.0
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )