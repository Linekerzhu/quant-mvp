"""
Feature importance module smoke tests.

OR5-CODE T4: Coverage gap fill - basic smoke tests for feature_importance.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock


class TestFeatureImportance:
    """Smoke tests for feature importance calculation."""

    def test_importance_returns_dict(self):
        """Feature importance should return a dictionary."""
        from src.features.feature_importance import calculate_feature_importance
        
        # Mock model with feature_importance method
        mock_model = Mock()
        mock_model.feature_importance = Mock(return_value=np.array([0.1, 0.3, 0.2, 0.4]))
        
        feature_names = ['returns_5d', 'rv_20d', 'sma_20', 'vol_ratio']
        
        result = calculate_feature_importance(mock_model, feature_names, method='gain')
        
        assert isinstance(result, dict), "Should return dictionary"
        assert len(result) == len(feature_names), "Should have entry for each feature"

    def test_dummy_noise_in_ranking(self):
        """dummy_noise should appear in feature ranking."""
        from src.features.feature_importance import calculate_feature_importance
        
        # Mock model
        mock_model = Mock()
        mock_model.feature_importance = Mock(return_value=np.array([0.1, 0.3, 0.05, 0.2, 0.35]))
        
        feature_names = ['returns_5d', 'rv_20d', 'dummy_noise', 'sma_20', 'vol_ratio']
        
        result = calculate_feature_importance(mock_model, feature_names, method='gain')
        
        assert 'dummy_noise' in result, "dummy_noise should be in importance dict"
        # dummy_noise importance should be relatively low for non-overfitted model
        # (This is a smoke test, actual check is in meta_trainer)

    def test_importance_values_non_negative(self):
        """Feature importance values should be non-negative."""
        from src.features.feature_importance import calculate_feature_importance
        
        mock_model = Mock()
        mock_model.feature_importance = Mock(return_value=np.array([0.1, 0.3, 0.2]))
        
        feature_names = ['feature_a', 'feature_b', 'feature_c']
        
        result = calculate_feature_importance(mock_model, feature_names, method='gain')
        
        for name, importance in result.items():
            assert importance >= 0, f"Feature {name} importance should be non-negative"

    def test_feature_ranking_order(self):
        """Feature ranking should order by importance descending."""
        from src.features.feature_importance import get_feature_ranking
        
        importance_dict = {
            'feature_a': 0.1,
            'feature_b': 0.5,
            'feature_c': 0.3
        }
        
        ranking = get_feature_ranking(importance_dict)
        
        assert ranking[0][0] == 'feature_b', "Highest importance should be first"
        assert ranking[-1][0] == 'feature_a', "Lowest importance should be last"
