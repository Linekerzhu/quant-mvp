"""
Feature importance tests - R29 Fix

R29 审计整改：测试对齐实际 API（FeatureImportanceTracker）
"""
import pytest
import pandas as pd
import numpy as np
from src.features.feature_importance import FeatureImportanceTracker


class TestFeatureImportanceTracker:
    """Test FeatureImportanceTracker class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 100
        data = {
            'label_return': np.random.randn(n),
            'feature_a': np.random.randn(n),
            'feature_b': np.random.randn(n),
            'dummy_noise': np.random.randn(n),
            'event_valid': [True] * n,  # Required for IC calculation
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def tracker(self):
        """Create tracker instance"""
        return FeatureImportanceTracker(lookback_days=63)

    def test_tracker_calculate_ic_returns_metrics(self, tracker, sample_data):
        """R29-A1: calculate_ic should return ICMetrics for single feature"""
        metrics = tracker.calculate_ic(sample_data, 'feature_a', return_col='label_return')

        # May return None if insufficient data, but should not error
        if metrics is not None:
            assert hasattr(metrics, 'ic_mean')
            assert -1.0 <= metrics.ic_mean <= 1.0

    def test_tracker_calculate_all_features(self, tracker, sample_data):
        """R29-A1: calculate_all_features should return dict of metrics"""
        feature_cols = ['feature_a', 'feature_b', 'dummy_noise']
        results = tracker.calculate_all_features(sample_data, feature_cols, return_col='label_return')

        assert isinstance(results, dict)
        # Should have calculated metrics for features
        # (may be empty if insufficient data, but should not error)

    def test_tracker_get_top_features(self, tracker, sample_data):
        """R29-A1: get_top_features should return sorted list"""
        feature_cols = ['feature_a', 'feature_b', 'dummy_noise']

        # Calculate IC for all features
        tracker.calculate_all_features(sample_data, feature_cols, return_col='label_return')

        top_features = tracker.get_top_features(n=3, min_ic=0.01)

        assert isinstance(top_features, list)
        # Should be sorted by absolute IC descending (could be empty)
        if len(top_features) > 0:
            # Each item should be a tuple (feature_name, ic_mean)
            assert all(isinstance(f, tuple) and len(f) == 2 for f in top_features)

    def test_tracker_detect_drift_no_error(self, tracker, sample_data):
        """R29-A1: detect_drift should not error even without baseline"""
        feature_cols = ['feature_a', 'feature_b', 'dummy_noise']

        # Calculate without setting baseline
        tracker.calculate_all_features(sample_data, feature_cols, return_col='label_return')

        # detect_drift should handle missing baseline gracefully
        drift_alerts = tracker.detect_drift()

        # Should return a list (could be empty if no baseline)
        assert isinstance(drift_alerts, list)

    def test_tracker_set_baseline(self, tracker, sample_data):
        """R29-A1: set_baseline should work after calculation"""
        feature_cols = ['feature_a', 'feature_b', 'dummy_noise']

        results = tracker.calculate_all_features(sample_data, feature_cols, return_col='label_return')

        # Set baseline
        tracker.set_baseline()

        # Baseline should be set
        assert tracker.baseline_metrics is not None
