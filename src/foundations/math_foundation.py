# src/foundations/math_foundation.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

class StatisticalAnalyzer:
    """Implement core statistical concepts from scratch"""
    
    @staticmethod
    def calculate_descriptive_stats(series: pd.Series) -> Dict:
        """Calculate descriptive statistics manually"""
        n = len(series)
        mean = series.sum() / n
        variance = ((series - mean) ** 2).sum() / (n - 1)
        std = np.sqrt(variance)
        skewness = ((series - mean) ** 3).sum() / (n * std ** 3)
        kurtosis = ((series - mean) ** 4).sum() / (n * std ** 4) - 3
        
        return {
            'count': n,
            'mean': mean,
            'median': series.median(),
            'std': std,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75)
        }
    
    @staticmethod
    def hypothesis_test_proportions(
        churn_group: pd.Series, 
        non_churn_group: pd.Series,
        alpha: float = 0.05
    ) -> Dict:
        """Two-proportion z-test for churn analysis"""
        p1 = churn_group.mean()
        p2 = non_churn_group.mean()
        n1 = len(churn_group)
        n2 = len(non_churn_group)
        
        # Pooled proportion
        p_pool = (churn_group.sum() + non_churn_group.sum()) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        # Z-statistic
        z_stat = (p1 - p2) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        margin_error = stats.norm.ppf(1 - alpha/2) * se
        ci_lower = (p1 - p2) - margin_error
        ci_upper = (p1 - p2) + margin_error
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': p1 - p2
        }
    
    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix from scratch"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        corr_matrix = np.zeros((n_cols, n_cols))
        
        for i in range(n_cols):
            for j in range(n_cols):
                x = df[numeric_cols[i]]
                y = df[numeric_cols[j]]
                
                # Pearson correlation
                x_mean = x.mean()
                y_mean = y.mean()
                
                numerator = ((x - x_mean) * (y - y_mean)).sum()
                denominator = np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
                
                corr_matrix[i, j] = numerator / denominator if denominator != 0 else 0
        
        return pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)


class LinearAlgebraToolkit:
    """Implement core linear algebra operations"""
    
    @staticmethod
    def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication from scratch"""
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match")
        
        result = np.zeros((A.shape[0], B.shape[1]))
        
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    result[i, j] += A[i, k] * B[k, j]
        
        return result
    
    @staticmethod
    def calculate_eigenvalues(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors"""
        # Using numpy for efficiency, but showing understanding
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def pca_from_scratch(X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Principal Component Analysis from scratch"""
        # 1. Standardize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_standardized = (X - X_mean) / X_std
        
        # 2. Calculate covariance matrix
        cov_matrix = np.cov(X_standardized.T)
        
        # 3. Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Select top n_components
        components = eigenvectors[:, :n_components]
        
        # 6. Transform data
        transformed = X_standardized @ components
        
        return transformed, eigenvalues, components


class AlgorithmImplementer:
    """Implement core ML algorithms from scratch"""
    
    @staticmethod
    def linear_regression_from_scratch(X: np.ndarray, y: np.ndarray):
        """Linear regression using normal equation"""
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        return theta
    
    @staticmethod
    def logistic_regression_gradient_descent(
        X: np.ndarray, 
        y: np.ndarray, 
        learning_rate: float = 0.01,
        iterations: int = 1000
    ):
        """Logistic regression using gradient descent"""
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples, n_features = X_b.shape
        
        # Initialize weights
        weights = np.zeros(n_features)
        
        # Gradient descent
        for i in range(iterations):
            # Linear combination
            z = X_b @ weights
            
            # Sigmoid activation
            y_pred = 1 / (1 + np.exp(-z))
            
            # Gradient calculation
            gradient = (1/n_samples) * X_b.T @ (y_pred - y)
            
            # Update weights
            weights -= learning_rate * gradient
        
        return weights
    
    @staticmethod
    def decision_tree_from_scratch(
        X: np.ndarray, 
        y: np.ndarray, 
        max_depth: int = 3,
        min_samples_split: int = 2
    ):
        """Simple decision tree implementation"""
        
        class Node:
            def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
                self.feature_idx = feature_idx
                self.threshold = threshold
                self.left = left
                self.right = right
                self.value = value
        
        def gini_impurity(y):
            """Calculate Gini impurity"""
            classes = np.unique(y)
            impurity = 1.0
            for cls in classes:
                p = np.sum(y == cls) / len(y)
                impurity -= p ** 2
            return impurity
        
        def best_split(X, y):
            """Find best split"""
            best_gini = float('inf')
            best_feature = None
            best_threshold = None
            
            n_features = X.shape[1]
            
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                
                for threshold in thresholds:
                    # Split data
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    
                    # Calculate weighted Gini
                    left_gini = gini_impurity(y[left_mask])
                    right_gini = gini_impurity(y[right_mask])
                    
                    weighted_gini = (
                        np.sum(left_mask) / len(y) * left_gini +
                        np.sum(right_mask) / len(y) * right_gini
                    )
                    
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_feature = feature_idx
                        best_threshold = threshold
            
            return best_feature, best_threshold
        
        def build_tree(X, y, depth=0):
            """Recursively build decision tree"""
            n_samples, n_classes = len(y), len(np.unique(y))
            
            # Stopping criteria
            if (depth >= max_depth or 
                n_classes == 1 or 
                n_samples < min_samples_split):
                return Node(value=np.bincount(y).argmax())
            
            # Find best split
            feature_idx, threshold = best_split(X, y)
            
            if feature_idx is None:
                return Node(value=np.bincount(y).argmax())
            
            # Split data
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # Build subtrees
            left = build_tree(X[left_mask], y[left_mask], depth + 1)
            right = build_tree(X[right_mask], y[right_mask], depth + 1)
            
            return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
        
        root = build_tree(X, y)
        return root