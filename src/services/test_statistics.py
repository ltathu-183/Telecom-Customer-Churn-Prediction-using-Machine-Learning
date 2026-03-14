"""
Unit tests for statistical implementations from scratch.
Critical for demonstrating production code quality to Viettel.
"""

import unittest
import numpy as np
import pandas as pd
from src.foundations.statistics import StatisticalAnalyzer


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test statistical implementations from scratch"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.normal_data = np.random.normal(loc=0, scale=1, size=1000)
        self.df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    def test_descriptive_stats(self):
        """Test descriptive statistics calculation"""
        analyzer = StatisticalAnalyzer()
        stats = analyzer.calculate_descriptive_stats(pd.Series(self.normal_data))
        
        # Check basic properties
        self.assertAlmostEqual(stats['mean'], 0.0, delta=0.1)
        self.assertAlmostEqual(stats['std'], 1.0, delta=0.1)
        self.assertAlmostEqual(stats['skewness'], 0.0, delta=0.2)
        self.assertAlmostEqual(stats['kurtosis'], 0.0, delta=0.5)
    
    def test_hypothesis_test(self):
        """Test hypothesis testing implementation"""
        analyzer = StatisticalAnalyzer()
        
        # Create two groups with known difference
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.5, 1, 100)  # Different mean
        
        result = analyzer.hypothesis_test_proportions(
            pd.Series(group1 > 0),
            pd.Series(group2 > 0)
        )
        
        # Should detect significant difference
        self.assertTrue(result['significant'])
        self.assertLess(result['p_value'], 0.05)
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation"""
        analyzer = StatisticalAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(self.df)
        
        # Check diagonal is 1.0
        for i in range(len(corr_matrix.columns)):
            self.assertAlmostEqual(corr_matrix.iloc[i, i], 1.0, delta=0.01)
        
        # Check symmetry
        self.assertTrue(np.allclose(corr_matrix.values, corr_matrix.values.T))
    
    def test_pca_implementation(self):
        """Test PCA implementation from scratch"""
        from src.foundations.linear_algebra import LinearAlgebraToolkit
        
        toolkit = LinearAlgebraToolkit()
        X = self.df[['feature1', 'feature2']].values
        
        # Standardize manually for comparison
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Our PCA
        transformed, eigenvalues, components = toolkit.pca_from_scratch(X, n_components=2)
        
        # Verify properties
        self.assertEqual(transformed.shape[1], 2)
        self.assertEqual(len(eigenvalues), 2)
        self.assertGreaterEqual(eigenvalues[0], eigenvalues[1])  # Sorted descending
        
        # Check orthogonality of components
        dot_product = np.dot(components[:, 0], components[:, 1])
        self.assertAlmostEqual(dot_product, 0.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()