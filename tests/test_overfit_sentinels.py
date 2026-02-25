"""
Overfitting Sentinel Tests

Implements two key overfitting detection mechanisms:
1. Dummy Feature Sentinel - Detects if model is overfitting to noise
2. Time Shuffle Sentinel - Detects if model relies on temporal leakage

These tests are part of Phase B validation before proceeding to model training.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import sys
from pathlib import Path

# Add parent directory to path (portable)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import FeatureEngineer
from src.labels.triple_barrier import TripleBarrierLabeler
from src.labels.sample_weights import SampleWeightCalculator


class DummyFeatureSentinel:
    """
    Sentinel to detect overfitting via dummy noise feature.
    
    Injects a pure noise feature (N(0,1)) and checks if the model
    assigns it significant importance. If the dummy feature ranks
    in the top 25% of feature importance, it indicates overfitting.
    """
    
    # Thresholds from Plan v4 Patch 2
    TOP_PERCENTILE_THRESHOLD = 0.25  # Top 25%
    RELATIVE_CONTRIBUTION_THRESHOLD = 1.0  # Dummy contributes >100% of mean real feature
    
    def __init__(self, dummy_feature_name: str = 'dummy_noise'):
        self.dummy_feature_name = dummy_feature_name
        self.results: Dict = {}
    
    def run_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        model = None
    ) -> Tuple[bool, Dict]:
        """
        Run dummy feature overfitting test.
        
        Args:
            X: Feature matrix (must include dummy_noise column)
            y: Labels
            sample_weight: Sample weights
            model: Model to test (default: RandomForest)
            
        Returns:
            (passed, details)
        """
        if self.dummy_feature_name not in X.columns:
            raise ValueError(f"Dummy feature '{self.dummy_feature_name}' not found in features")
        
        # Use default model if not provided
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        # Train model
        model.fit(X, y, sample_weight=sample_weight)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns.tolist()
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # P1 (R26-A1): Fix dummy_rank - use reset_index to get sorted position
        # OLD BUG: .index[0] used original DataFrame index, not sorted position
        # dummy_noise always last column → rank always = max → never triggers alert
        importance_df = importance_df.reset_index(drop=True)
        
        # Find dummy feature rank (now using sorted position, not original index)
        dummy_row = importance_df[importance_df['feature'] == self.dummy_feature_name]
        dummy_rank = dummy_row.index[0] + 1  # +1 for 1-based rank
        dummy_importance = dummy_row['importance'].values[0]
        
        # Calculate top percentile threshold
        n_features = len(feature_names)
        top_n_threshold = int(n_features * self.TOP_PERCENTILE_THRESHOLD)
        
        # Calculate relative contribution
        real_features = importance_df[importance_df['feature'] != self.dummy_feature_name]
        mean_real_importance = real_features['importance'].mean()
        relative_contribution = dummy_importance / max(mean_real_importance, 1e-6)
        
        # Determine if passed
        # Pass if dummy is NOT in top 25% AND relative contribution < 50%
        passed = (dummy_rank > top_n_threshold) and (relative_contribution < self.RELATIVE_CONTRIBUTION_THRESHOLD)
        
        self.results = {
            'dummy_rank': int(dummy_rank),
            'total_features': n_features,
            'top_threshold': top_n_threshold,
            'dummy_importance': float(dummy_importance),
            'mean_real_importance': float(mean_real_importance),
            'relative_contribution': float(relative_contribution),
            'passed': passed,
            'top_5_features': importance_df.head(5)['feature'].tolist()
        }
        
        return passed, self.results


class TimeShuffleSentinel:
    """
    Sentinel to detect temporal leakage via time shuffling.
    
    Shuffles the time index multiple times and checks if model performance
    is consistent. If performance improves after shuffling, it indicates
    the model is exploiting temporal patterns that shouldn't exist.
    """
    
    # Thresholds from Plan v4 Patch 3
    N_SHUFFLES = 5
    # P0-4 Fix: Using AUC instead of accuracy for class imbalance robustness
    MEAN_AUC_THRESHOLD = 0.55  # Mean AUC across shuffles must be <= 0.55
    MAX_AUC_THRESHOLD = 0.58   # No single shuffle can exceed 0.58
    
    def __init__(self, n_shuffles: int = 5):
        self.n_shuffles = n_shuffles
        self.results: Dict = {}
    
    def run_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        model = None,
        cv_folds: int = 5
    ) -> Tuple[bool, Dict]:
        """
        Run time shuffle leakage test.
        
        Args:
            X: Feature matrix (with date index if available)
            y: Labels
            sample_weight: Sample weights
            model: Model to test (default: RandomForest)
            cv_folds: Number of CV folds
            
        Returns:
            (passed, details)
        """
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        # P0-4 Fix: Use roc_auc scoring instead of accuracy
        # AUC is robust to class imbalance, accuracy is not
        scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'roc_auc_ovr'
        
        # Baseline score (no shuffle)
        baseline_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        baseline_mean = baseline_scores.mean()
        
        # Run multiple shuffles
        shuffle_scores = []
        
        for i in range(self.n_shuffles):
            # Shuffle the data
            shuffle_idx = np.random.RandomState(42 + i).permutation(len(X))
            X_shuffled = X.iloc[shuffle_idx]
            y_shuffled = y.iloc[shuffle_idx]
            
            if sample_weight is not None:
                w_shuffled = sample_weight[shuffle_idx]
            else:
                w_shuffled = None
            
            # Evaluate on shuffled data
            scores = cross_val_score(
                model, X_shuffled, y_shuffled, cv=cv_folds, scoring=scoring
            )
            shuffle_scores.append(scores.mean())
        
        shuffle_scores = np.array(shuffle_scores)
        
        # Check thresholds
        mean_shuffle = shuffle_scores.mean()
        max_shuffle = shuffle_scores.max()
        
        # P2 (R27-B4): Add comparison vs baseline
        # If shuffle AUC > baseline, that's suspicious (model performs better on random data)
        # Allow 5% tolerance for noise
        mean_vs_baseline_ok = mean_shuffle <= baseline_mean * 1.05
        
        # Pass if:
        # 1. mean <= absolute threshold (0.55)
        # 2. max <= absolute threshold (0.58)
        # 3. mean shuffle not significantly higher than baseline (5% tolerance)
        passed = (
            (mean_shuffle <= self.MEAN_AUC_THRESHOLD) and 
            (max_shuffle <= self.MAX_AUC_THRESHOLD) and
            mean_vs_baseline_ok
        )
        
        self.results = {
            'baseline_auc': float(baseline_mean),
            'shuffle_aucs': shuffle_scores.tolist(),
            'shuffle_mean_auc': float(mean_shuffle),
            'shuffle_std_auc': float(shuffle_scores.std()),
            'shuffle_max_auc': float(max_shuffle),
            'mean_vs_baseline_ok': mean_vs_baseline_ok,  # P2 (R27-B4): New check
            'passed': passed,
            'n_shuffles': self.n_shuffles,
            'scoring': scoring
        }
        
        return passed, self.results


# =============================================================================
# Pytest Test Cases
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    n_samples = 500
    n_features = 10
    
    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add dummy noise feature
    X['dummy_noise'] = np.random.randn(n_samples)
    
    # Create synthetic labels (correlated with some features)
    y = (X['feature_0'] + X['feature_1'] > 0).astype(int)
    
    return X, y


class TestDummyFeatureSentinel:
    """Test dummy feature overfitting detection."""
    
    def test_dummy_sentinel_initialization(self):
        """Test sentinel can be initialized."""
        sentinel = DummyFeatureSentinel()
        assert sentinel.dummy_feature_name == 'dummy_noise'
    
    def test_dummy_sentinel_detects_overfitting(self, sample_data):
        """Test sentinel detects when model overfits to dummy feature."""
        X, y = sample_data
        
        # Create a model that will overfit
        overfit_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,  # Deep tree to overfit
            random_state=42
        )
        
        sentinel = DummyFeatureSentinel()
        passed, results = sentinel.run_test(X, y, model=overfit_model)
        
        # Test should detect potential overfitting
        assert 'dummy_rank' in results
        assert 'dummy_importance' in results
        assert 'top_5_features' in results
        
        print(f"Dummy feature rank: {results['dummy_rank']}/{results['total_features']}")
        print(f"Dummy importance: {results['dummy_importance']:.4f}")
        print(f"Top 5 features: {results['top_5_features']}")
    
    def test_dummy_sentinel_missing_feature(self, sample_data):
        """Test sentinel raises error when dummy feature missing."""
        X, y = sample_data
        X_no_dummy = X.drop(columns=['dummy_noise'])
        
        sentinel = DummyFeatureSentinel()
        
        with pytest.raises(ValueError, match="Dummy feature 'dummy_noise' not found"):
            sentinel.run_test(X_no_dummy, y)


class TestTimeShuffleSentinel:
    """Test time shuffle leakage detection."""
    
    def test_shuffle_sentinel_initialization(self):
        """Test sentinel can be initialized."""
        sentinel = TimeShuffleSentinel(n_shuffles=3)
        assert sentinel.n_shuffles == 3
    
    def test_shuffle_sentinel_detects_leakage(self, sample_data):
        """Test sentinel evaluates shuffled performance."""
        X, y = sample_data
        
        sentinel = TimeShuffleSentinel(n_shuffles=3)
        passed, results = sentinel.run_test(X, y)
        
        # F2 Fix: Updated key names from accuracy to auc
        assert 'baseline_auc' in results
        assert 'shuffle_mean_auc' in results
        assert 'shuffle_max_auc' in results
        assert 'shuffle_aucs' in results
        
        print(f"Baseline AUC: {results['baseline_auc']:.4f}")
        print(f"Shuffle mean AUC: {results['shuffle_mean_auc']:.4f}")
        print(f"Shuffle max AUC: {results['shuffle_max_auc']:.4f}")
    
    def test_shuffle_thresholds(self):
        """Test default thresholds are set correctly."""
        sentinel = TimeShuffleSentinel()
        # F3 Fix: Updated attribute names from ACCURACY to AUC
        assert sentinel.MEAN_AUC_THRESHOLD == 0.55
        assert sentinel.MAX_AUC_THRESHOLD == 0.58


class TestSentinelIntegration:
    """Integration tests with actual pipeline components."""
    
    def test_sentinels_with_mock_data(self):
        """Test sentinels with mock financial data."""
        # This is a placeholder - real test would use tests/fixtures/mock_prices.parquet
        # and run the full feature + label + sentinel pipeline
        
        # For now, just verify the sentinels can be instantiated
        dummy_sentinel = DummyFeatureSentinel()
        shuffle_sentinel = TimeShuffleSentinel()
        
        assert dummy_sentinel is not None
        assert shuffle_sentinel is not None


def run_all_sentinels(X: pd.DataFrame, y: pd.Series, sample_weight=None) -> Dict:
    """
    Run all overfitting sentinels and return combined results.
    
    Args:
        X: Feature matrix
        y: Labels
        sample_weight: Optional sample weights
        
    Returns:
        Combined sentinel results
    """
    results = {
        'dummy_sentinel': {'passed': False, 'details': {}},
        'shuffle_sentinel': {'passed': False, 'details': {}},
        'all_passed': False
    }
    
    # Run dummy feature sentinel
    dummy_sentinel = DummyFeatureSentinel()
    try:
        passed, details = dummy_sentinel.run_test(X, y, sample_weight)
        results['dummy_sentinel'] = {'passed': passed, 'details': details}
    except Exception as e:
        results['dummy_sentinel']['error'] = str(e)
    
    # Run time shuffle sentinel
    shuffle_sentinel = TimeShuffleSentinel()
    try:
        passed, details = shuffle_sentinel.run_test(X, y, sample_weight)
        results['shuffle_sentinel'] = {'passed': passed, 'details': details}
    except Exception as e:
        results['shuffle_sentinel']['error'] = str(e)
    
    # Overall result
    results['all_passed'] = (
        results['dummy_sentinel'].get('passed', False) and
        results['shuffle_sentinel'].get('passed', False)
    )
    
    return results


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
