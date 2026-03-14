"""
Feature engineering module for production use.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Production-ready feature engineering class."""
    
    def __init__(self):
        self.feature_groups = {
            'customer_behavior': self._customer_behavior_features,
            'spending_patterns': self._spending_features,
            'contract_stability': self._contract_features,
            'service_usage': self._service_features,
            'temporal_features': self._temporal_features,
            'risk_scores': self._risk_score_features
        }
    
    def _customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate customer behavior features."""
        features = {}
        
        # Service count
        services = ['PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        features['service_count'] = df[services].apply(
            lambda x: x.str.contains('Yes').sum(), axis=1
        )
        
        # Premium services
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        features['premium_service_count'] = df[premium_services].apply(
            lambda x: x.str.contains('Yes').sum(), axis=1
        )
        
        # Streaming services
        streaming_services = ['StreamingTV', 'StreamingMovies']
        features['streaming_service_count'] = df[streaming_services].apply(
            lambda x: x.str.contains('Yes').sum(), axis=1
        )
        
        # Flags
        features['has_internet'] = (df['InternetService'] != 'No').astype(int)
        features['has_multiple_lines'] = (df['MultipleLines'] == 'Yes').astype(int)
        features['paperless_billing'] = (df['PaperlessBilling'] == 'Yes').astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate spending pattern features."""
        features = {}
        
        features['monthly_charges'] = df['MonthlyCharges']
        features['total_charges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # Spend per tenure
        features['spend_per_tenure'] = features['total_charges'] / (df['tenure'].replace(0, 1))
        
        # Spend growth proxy
        features['spend_growth_proxy'] = features['monthly_charges'] / (features['spend_per_tenure'] + 1)
        
        # High spend flag
        features['high_spend_flag'] = (
            features['monthly_charges'] > features['monthly_charges'].median()
        ).astype(int)
        
        # Spend volatility
        features['spend_volatility'] = features['monthly_charges'] / (features['total_charges'] + 1)
        
        return pd.DataFrame(features, index=df.index)
    
    def _contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate contract stability features."""
        features = {}
        
        # Contract length score
        contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
        features['contract_length_score'] = df['Contract'].map(contract_map).fillna(1)
        
        # Auto-pay flag
        features['has_autopay'] = df['PaymentMethod'].isin([
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ]).astype(int)
        
        # Early termination risk
        features['early_termination_risk'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        # Tenure bucket
        features['tenure_bucket'] = pd.qcut(
            df['tenure'], q=5, labels=False, duplicates='drop'
        )
        
        # Tenure risk score
        features['tenure_risk_score'] = 1 - (df['tenure'] / df['tenure'].max())
        
        # Long-term customer flag
        features['long_term_customer'] = (df['tenure'] > df['tenure'].median()).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate service usage features."""
        features = {}
        
        # Internet service level
        internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        features['internet_service_level'] = df['InternetService'].map(internet_map).fillna(0)
        
        # Service bundle score
        features['service_bundle_score'] = (
            df[services].apply(lambda x: x.str.contains('Yes').sum(), axis=1) +
            features['has_internet'] * 2
        )
        
        # Tech support dependency
        features['tech_support_dependency'] = (
            features['has_internet'] * 
            df[premium_services].apply(lambda x: x.str.contains('Yes').sum(), axis=1)
        )
        
        # Streaming intensity
        features['streaming_intensity'] = (
            df[streaming_services].apply(lambda x: x.str.contains('Yes').sum(), axis=1) /
            (df[services].apply(lambda x: x.str.contains('Yes').sum(), axis=1) + 1)
        )
        
        return pd.DataFrame(features, index=df.index)
    
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        features = {}
        
        features['tenure_years'] = df['tenure'] / 12
        features['tenure_months'] = df['tenure'] % 12
        
        # Lifecycle stage
        features['lifecycle_stage'] = pd.cut(
            df['tenure'],
            bins=[0, 6, 24, 60, float('inf')],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _risk_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate risk score features."""
        features = {}
        
        # Payment risk score
        payment_risk_map = {
            'Electronic check': 1.0,
            'Mailed check': 0.8,
            'Bank transfer (automatic)': 0.3,
            'Credit card (automatic)': 0.2
        }
        features['payment_risk_score'] = df['PaymentMethod'].map(payment_risk_map).fillna(0.5)
        
        # Overall risk score
        features['overall_risk_score'] = (
            features.get('early_termination_risk', 0) * 0.3 +
            features.get('tenure_risk_score', 0) * 0.2 +
            features['payment_risk_score'] * 0.2 +
            (1 - features.get('contract_length_score', 1) / 3) * 0.2 +
            features.get('paperless_billing', 0) * 0.1
        )
        
        return pd.DataFrame(features, index=df.index)
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from raw data.
        
        Args:
            df: Input dataframe with raw customer data
            
        Returns:
            Dataframe with engineered features
        """
        logger.info(f"Building features for {len(df)} samples")
        
        all_features = []
        for name, func in self.feature_groups.items():
            try:
                features = func(df)
                all_features.append(features)
                logger.info(f"Built {name} features: {features.shape}")
            except Exception as e:
                logger.error(f"Error building {name} features: {e}")
                raise
        
        result = pd.concat(all_features, axis=1)
        logger.info(f"Total features built: {result.shape[1]}")
        
        return result