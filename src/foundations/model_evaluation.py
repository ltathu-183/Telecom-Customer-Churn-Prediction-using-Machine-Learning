# src/foundations/model_evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Comprehensive model evaluation with manual calculations"""
    
    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate confusion matrix manually"""
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        return {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'accuracy': (TP + TN) / (TP + TN + FP + FN),
            'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
            'recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'f1': 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0,
            'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0
        }
    
    @staticmethod
    def calculate_auc_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate AUC-ROC manually using trapezoidal rule"""
        # Sort by predicted probability
        sorted_idx = np.argsort(y_pred_proba)
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred_proba[sorted_idx]
        
        # Calculate TPR and FPR at each threshold
        thresholds = np.unique(y_pred_sorted)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_pred_sorted >= threshold).astype(int)
            
            TP = np.sum((y_true_sorted == 1) & (y_pred == 1))
            TN = np.sum((y_true_sorted == 0) & (y_pred == 0))
            FP = np.sum((y_true_sorted == 0) & (y_pred == 1))
            FN = np.sum((y_true_sorted == 1) & (y_pred == 0))
            
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            
            tpr_list.append(TPR)
            fpr_list.append(FPR)
        
        # Calculate AUC using trapezoidal rule
        auc_score = 0
        for i in range(len(fpr_list) - 1):
            auc_score += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i+1] + tpr_list[i]) / 2
        
        return abs(auc_score)
    
    @staticmethod
    def calculate_lift_chart(
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        n_buckets: int = 10
    ) -> pd.DataFrame:
        """Calculate lift chart for business impact analysis"""
        # Create dataframe
        df = pd.DataFrame({
            'actual': y_true,
            'predicted_prob': y_pred_proba
        })
        
        # Sort by predicted probability
        df = df.sort_values('predicted_prob', ascending=False)
        
        # Create buckets
        df['bucket'] = pd.qcut(df['predicted_prob'], q=n_buckets, labels=False, duplicates='drop')
        
        # Calculate metrics per bucket
        results = []
        total_positive = df['actual'].sum()
        total_customers = len(df)
        
        for bucket in range(n_buckets):
            bucket_data = df[df['bucket'] == bucket]
            bucket_positive = bucket_data['actual'].sum()
            bucket_customers = len(bucket_data)
            
            cumulative_customers = df[df['bucket'] <= bucket].shape[0]
            cumulative_positive = df[df['bucket'] <= bucket]['actual'].sum()
            
            results.append({
                'bucket': bucket + 1,
                'customers': bucket_customers,
                'positive_cases': bucket_positive,
                'response_rate': bucket_positive / bucket_customers if bucket_customers > 0 else 0,
                'cumulative_customers': cumulative_customers,
                'cumulative_positive': cumulative_positive,
                'capture_rate': cumulative_positive / total_positive if total_positive > 0 else 0,
                'lift': (cumulative_positive / cumulative_customers) / (total_positive / total_customers) if cumulative_customers > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_calibration_curve(
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate calibration curve (reliability diagram)"""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        
        bin_means = []
        bin_acc = []
        bin_weights = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_pred = y_pred_proba[bin_mask].mean()
                bin_actual = y_true[bin_mask].mean()
                bin_weight = np.sum(bin_mask) / len(y_true)
                
                bin_means.append(bin_pred)
                bin_acc.append(bin_actual)
                bin_weights.append(bin_weight)
        
        return np.array(bin_means), np.array(bin_acc), np.array(bin_weights)
    
    @staticmethod
    def calculate_business_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        cost_fp: float = 50,    # Cost of false positive (retention cost)
        cost_fn: float = 500,   # Cost of false negative (churn loss)
        revenue_per_customer: float = 1000
    ) -> Dict:
        """Calculate business-oriented metrics"""
        # Optimal threshold based on cost
        thresholds = np.linspace(0, 1, 100)
        total_costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            TP = np.sum((y_true == 1) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            
            # Total cost
            cost = FP * cost_fp + FN * cost_fn
            total_costs.append(cost)
        
        optimal_threshold = thresholds[np.argmin(total_costs)]
        min_cost = min(total_costs)
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm = ModelEvaluator.calculate_confusion_matrix(y_true, y_pred_optimal)
        
        # Expected value
        expected_value = (
            cm['TP'] * (revenue_per_customer - cost_fp) +
            cm['TN'] * revenue_per_customer -
            cm['FP'] * cost_fp -
            cm['FN'] * cost_fn
        )
        
        return {
            'optimal_threshold': optimal_threshold,
            'minimum_cost': min_cost,
            'expected_value': expected_value,
            'cost_fp': cost_fp,
            'cost_fn': cost_fn,
            'revenue_per_customer': revenue_per_customer
        }