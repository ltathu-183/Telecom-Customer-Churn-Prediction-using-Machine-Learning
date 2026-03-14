"""
Churn prediction service for business logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class ChurnService:
    """Business logic for churn prediction and retention."""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7
        }
    
    def predict_churn(self, customer_data: pd.DataFrame) -> Dict:
        """
        Predict churn probability for a customer.
        
        Args:
            customer_data: Customer dataframe
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        features = self.preprocessor.transform(customer_data)
        
        # Predict
        churn_prob = self.model.predict_proba(features)[0][1]
        
        # Determine risk level
        if churn_prob > self.risk_thresholds['medium']:
            risk_level = 'High Risk'
        elif churn_prob > self.risk_thresholds['low']:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'
        
        return {
            'churn_probability': float(churn_prob),
            'risk_level': risk_level,
            'customer_id': customer_data.index[0] if hasattr(customer_data.index, '__getitem__') else 'unknown'
        }
    
    def generate_retention_strategy(self, prediction: Dict, customer_features: Dict) -> Dict:
        """
        Generate retention strategy based on prediction.
        
        Args:
            prediction: Prediction results
            customer_features: Customer features
            
        Returns:
            Retention strategy
        """
        churn_prob = prediction['churn_probability']
        risk_level = prediction['risk_level']
        
        if churn_prob > 0.7:
            return {
                'action': 'VIP retention call',
                'channel': 'phone',
                'priority': 'high',
                'budget': 150,
                'timeline': '24 hours'
            }
        elif churn_prob > 0.3:
            return {
                'action': 'Send promotion campaign',
                'channel': 'email',
                'priority': 'medium',
                'budget': 50,
                'timeline': '3 days'
            }
        else:
            return {
                'action': 'No immediate action',
                'channel': 'none',
                'priority': 'low',
                'budget': 0,
                'timeline': 'monitor'
            }