"""
Telco Churn AI System - Production Package

This package contains all production-ready code for the churn prediction system.
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@company.com"

from src.data.preprocessing import DataPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.train_model import ModelTrainer
from src.services.churn_service import ChurnService
from src.api.main import app

__all__ = [
    "DataPreprocessor",
    "FeatureBuilder", 
    "ModelTrainer",
    "ChurnService",
    "app"
]