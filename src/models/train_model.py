"""
Model training module for production use.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import logging
import json

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Production-ready model training class."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'class_weight': 'balanced'
                }
            }
        }
    
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Optional[List[str]] = None
    ) -> Tuple[Dict, str]:
        """
        Train and evaluate multiple models.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_names: List of model names to train
            
        Returns:
            Dictionary of results and best model name
        """
        logger.info(f"Training models on {len(X)} samples")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Model definitions
        model_defs = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.config['models']['random_forest']['n_estimators'],
                max_depth=self.config['models']['random_forest']['max_depth'],
                class_weight=self.config['models']['random_forest']['class_weight'],
                random_state=self.config['random_state'],
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=self.config['models']['xgboost']['n_estimators'],
                max_depth=self.config['models']['xgboost']['max_depth'],
                learning_rate=self.config['models']['xgboost']['learning_rate'],
                random_state=self.config['random_state'],
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=self.config['models']['lightgbm']['n_estimators'],
                max_depth=self.config['models']['lightgbm']['max_depth'],
                learning_rate=self.config['models']['lightgbm']['learning_rate'],
                class_weight=self.config['models']['lightgbm']['class_weight'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        }
        
        # Train models
        results = {}
        model_names = model_names or list(model_defs.keys())
        
        for name in model_names:
            logger.info(f"Training {name}...")
            
            model = model_defs[name]
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'model': model
            }
            
            results[name] = metrics
            logger.info(f"{name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = model
                self.best_model_name = name
        
        return results, self.best_model_name
    
    def save_model(self, filepath: str) -> None:
        """Save best model to file."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        logger.info(f"Saving best model ({self.best_model_name}) to {filepath}")
        joblib.dump(self.best_model, filepath)
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'score': self.best_score,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath.replace('.pkl', '_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        logger.info(f"Loading model from {filepath}")
        self.best_model = joblib.load(filepath)
        self.best_model_name = 'loaded'