"""
Model monitoring and drift detection for production.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, reference_data_path: str):
        self.reference_data = pd.read_csv(reference_data_path)
        self.prediction_log = []
        self.drift_alerts = []
    
    def log_prediction(self, customer_id: str, churn_prob: float, actual: bool = None):
        """Log prediction for monitoring"""
        self.prediction_log.append({
            'timestamp': datetime.now(),
            'customer_id': customer_id,
            'churn_probability': churn_prob,
            'actual': actual,
            'predicted_class': int(churn_prob > 0.5)
        })
    
    def calculate_performance_metrics(self, window_hours: int = 24) -> Dict:
        """Calculate performance metrics for recent predictions"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_predictions = [
            p for p in self.prediction_log 
            if p['timestamp'] > cutoff_time and p['actual'] is not None
        ]
        
        if len(recent_predictions) < 10:
            return {'insufficient_data': True}
        
        # Calculate metrics
        y_true = [p['actual'] for p in recent_predictions]
        y_pred = [p['predicted_class'] for p in recent_predictions]
        y_prob = [p['churn_probability'] for p in recent_predictions]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        return {
            'timestamp': datetime.now().isoformat(),
            'window_hours': window_hours,
            'total_predictions': len(recent_predictions),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else None
        }
    
    def detect_data_drift(self, current_data: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """Detect data drift using statistical tests"""
        drift_detected = False
        drifted_features = []
        
        for col in current_data.columns:
            if col in self.reference_data.columns:
                # KS test for numerical features
                from scipy import stats
                
                ref_values = self.reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    stat, p_value = stats.ks_2samp(ref_values, curr_values)
                    
                    if p_value < threshold:
                        drift_detected = True
                        drifted_features.append({
                            'feature': col,
                            'p_value': p_value,
                            'statistic': stat
                        })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'drifted_features': drifted_features,
            'total_features_checked': len(current_data.columns)
        }
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        performance = self.calculate_performance_metrics(window_hours=24)
        recent_data = pd.DataFrame([p for p in self.prediction_log[-100:]])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance,
            'total_predictions_logged': len(self.prediction_log),
            'predictions_last_hour': len([
                p for p in self.prediction_log 
                if p['timestamp'] > datetime.now() - timedelta(hours=1)
            ])
        }
        
        return report
    
    def save_report(self, filepath: str):
        """Save monitoring report to file"""
        report = self.generate_monitoring_report()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitoring report saved to {filepath}")


# Example usage
if __name__ == "__main__":
    monitor = ModelMonitor('data/processed/features_scaled.csv')
    
    # Simulate predictions
    for i in range(100):
        monitor.log_prediction(
            customer_id=f"CUST_{i}",
            churn_prob=np.random.beta(2, 5),
            actual=np.random.choice([True, False], p=[0.2, 0.8])
        )
    
    # Generate report
    report = monitor.generate_monitoring_report()
    print(json.dumps(report, indent=2))
    
    # Save report
    monitor.save_report('reports/monitoring_report.json')