"""
Data preprocessing module for production use.
Handles data cleaning, validation, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Production-ready data preprocessing class."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.feature_names = None
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'numerical_cols': [
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ],
            'categorical_cols': [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ],
            'target_col': 'Churn'
        }
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessing transformers on training data.
        
        Args:
            df: Training dataframe
            
        Returns:
            self for chaining
        """
        logger.info("Fitting data preprocessor...")
        
        # Handle missing values for numerical columns
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(df[self.config['numerical_cols']])
        
        # Fit encoder for categorical columns
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            dtype=np.float32
        )
        self.encoder.fit(df[self.config['categorical_cols']])
        
        # Fit scaler
        numerical_data = self.imputer.transform(df[self.config['numerical_cols']])
        self.scaler = StandardScaler()
        self.scaler.fit(numerical_data)
        
        # Store feature names
        self.feature_names = (
            self.config['numerical_cols'] +
            self.encoder.get_feature_names_out(self.config['categorical_cols']).tolist()
        )
        
        logger.info(f"Preprocessor fitted. Total features: {len(self.feature_names)}")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed dataframe
        """
        if not all([self.imputer, self.encoder, self.scaler]):
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        logger.info(f"Transforming data with shape {df.shape}")
        
        # Handle numerical features
        numerical_data = self.imputer.transform(df[self.config['numerical_cols']])
        numerical_scaled = self.scaler.transform(numerical_data)
        
        # Handle categorical features
        categorical_encoded = self.encoder.transform(df[self.config['categorical_cols']])
        
        # Combine features
        features = np.hstack([numerical_scaled, categorical_encoded])
        
        # Create dataframe
        result_df = pd.DataFrame(features, columns=self.feature_names, index=df.index)
        
        logger.info(f"Transformation complete. Output shape: {result_df.shape}")
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def save(self, filepath: str) -> None:
        """Save preprocessor to file."""
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from file."""
        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)