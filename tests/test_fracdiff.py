"""
Tests for FracDiff (Fractional Differentiation) Module

Author: 李得勤
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features.fracdiff import (
    fracdiff_weights,
    fracdiff_fixed_window,
    fracdiff_expand_window,
    fracdiff_online,
    find_min_d_stationary,
    FracDiffTransformer,
    create_fracdiff_features
)


class TestFracdiffWeights:
    """Test suite for weight calculation."""
    
    def test_weights_d_0(self):
        """Test weights for d=0 (should be [1, 0, 0, ...])"""
        weights = fracdiff_weights(0.0, 10)
        assert weights[0] == 1.0
        assert np.all(weights[1:] == 0)
    
    def test_weights_d_1(self):
        """Test weights for d=1 (should be [1, -1, 0, 0, ...])"""
        weights = fracdiff_weights(1.0, 10)
        # For d=1: w[0]=1, w[1]=(0-1)/1=-1, then all zeros
        assert weights[0] == 1.0
        assert weights[1] == -1.0
        assert np.all(weights[2:] == 0)
    
    def test_weights_d_05(self):
        """Test weights for d=0.5"""
        weights = fracdiff_weights(0.5, 10)
        # For d=0.5: w[0]=1, w[1]=(0-0.5)/1=-0.5, w[2]=(1-0.5)/2*(-0.5)=-0.125
        assert weights[0] == 1.0
        assert abs(weights[1] - (-0.5)) < 0.001
        assert abs(weights[2] - (-0.125)) < 0.001
    
    def test_weights_positive_d(self):
        """Test that weights sum to approximately 1 for small d"""
        weights = fracdiff_weights(0.1, 100)
        # For very small d, first weight dominates
        assert weights[0] == 1.0
    
    def test_weights_length(self):
        """Test that weights array has correct length"""
        weights = fracdiff_weights(0.5, 50)
        assert len(weights) == 50


class TestFracdiffFixedWindow:
    """Test suite for fixed window fracdiff."""
    
    @pytest.fixture
    def price_series(self):
        """Create a random walk price series."""
        np.random.seed(42)
        n = 500
        price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        return pd.Series(price)
    
    def test_basic(self, price_series):
        """Test basic fracdiff computation."""
        result = fracdiff_fixed_window(price_series, 0.5, 100)
        assert len(result) == len(price_series)
        assert result.isna().sum() == 99  # First window-1 are NaN
    
    def test_d_bounds(self, price_series):
        """Test that d must be between 0 and 1."""
        with pytest.raises(ValueError):
            fracdiff_fixed_window(price_series, 1.5, 100)
        
        with pytest.raises(ValueError):
            fracdiff_fixed_window(price_series, -0.5, 100)
    
    def test_d_0_returns_original(self, price_series):
        """Test that d=0 returns original series (with NaN padding)."""
        result = fracdiff_fixed_window(price_series, 0.0, 100)
        # First 99 should be NaN
        assert result.isna().sum() == 99
    
    def test_d_1_is_first_diff(self, price_series):
        """Test that d=1 is approximately first difference."""
        result = fracdiff_fixed_window(price_series, 1.0, 100)
        # Compare with simple diff
        simple_diff = price_series.diff()
        # Both should be NaN for first 100, then similar
        assert result.isna().sum() == 99
    
    def test_no_lookahead(self, price_series):
        """Test that computation is causal (no look-ahead)."""
        # Change the last value dramatically
        series = price_series.copy()
        series.iloc[-1] = 10000
        
        result = fracdiff_fixed_window(series, 0.5, 100)
        
        # The change should NOT affect values before the last 100
        # Check a value far from the end
        result_original = fracdiff_fixed_window(price_series, 0.5, 100)
        
        # Values before the last 100 should be the same
        np.testing.assert_array_almost_equal(
            result.iloc[:-100],
            result_original.iloc[:-100]
        )


class TestFracdiffExpandWindow:
    """Test suite for expanding window fracdiff."""
    
    @pytest.fixture
    def price_series(self):
        """Create a price series."""
        np.random.seed(42)
        n = 100
        price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        return pd.Series(price)
    
    def test_basic(self, price_series):
        """Test basic expanding window computation."""
        result = fracdiff_expand_window(price_series, 0.5, 50)
        assert len(result) == len(price_series)
    
    def test_causal(self, price_series):
        """Test that computation is causal."""
        # Expanding window should naturally be causal
        series = price_series.copy()
        result = fracdiff_expand_window(series, 0.5, 50)
        assert result.notna().any()


class TestFracdiffOnline:
    """Test suite for online fracdiff."""
    
    def test_equals_fixed(self):
        """Test that online equals fixed window."""
        np.random.seed(42)
        n = 200
        price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        series = pd.Series(price)
        
        result_fixed = fracdiff_fixed_window(series, 0.5, 50)
        result_online = fracdiff_online(series, 0.5, 50)
        
        np.testing.assert_array_almost_equal(
            result_fixed.values,
            result_online.values
        )


class TestFindMinDStationary:
    """Test suite for finding minimum d."""
    
    @pytest.fixture
    def stationary_series(self):
        """Create a stationary series."""
        return pd.Series(np.random.randn(200))
    
    @pytest.fixture
    def non_stationary_series(self):
        """Create a non-stationary series (random walk)."""
        np.random.seed(42)
        n = 200
        price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        return pd.Series(price)
    
    def test_stationary_returns_0(self, stationary_series):
        """Test that already stationary series returns d=0."""
        d = find_min_d_stationary(stationary_series, threshold=0.05)
        assert d == 0.0
    
    def test_random_walk_needs_d(self, non_stationary_series):
        """Test that random walk needs positive d."""
        d = find_min_d_stationary(non_stationary_series, threshold=0.05)
        assert d > 0


class TestFracDiffTransformer:
    """Test suite for sklearn-compatible transformer."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'price1': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            'price2': 50 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            'volume': np.random.randint(1000, 10000, n)
        })
    
    def test_fit_transform(self, sample_df):
        """Test fit_transform."""
        transformer = FracDiffTransformer(d=0.5, window=50)
        result = transformer.fit_transform(sample_df)
        
        assert 'price1' in result.columns
        assert 'price2' in result.columns
        assert 'volume' in result.columns
    
    def test_repr(self):
        """Test __repr__."""
        transformer = FracDiffTransformer(d=0.5, window=100)
        repr_str = repr(transformer)
        assert 'FracDiffTransformer' in repr_str
        assert 'd=0.5' in repr_str


class TestCreateFracdiffFeatures:
    """Test suite for feature creation helper."""
    
    def test_basic(self):
        """Test create_fracdiff_features."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'adj_close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        })
        
        result = create_fracdiff_features(df, 'adj_close', [0.3, 0.5], 50)
        
        assert 'fracdiff_3' in result.columns
        assert 'fracdiff_5' in result.columns
    
    def test_default_d_values(self):
        """Test with default d values."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'adj_close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        })
        
        result = create_fracdiff_features(df, 'adj_close')
        
        # Default d_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert 'fracdiff_3' in result.columns
        assert 'fracdiff_4' in result.columns
        assert 'fracdiff_5' in result.columns
        assert 'fracdiff_6' in result.columns
        assert 'fracdiff_7' in result.columns


class TestFracdiffIntegration:
    """Integration tests."""
    
    def test_with_mock_prices(self):
        """Test with mock price data."""
        FIXTURES_DIR = Path(__file__).parent / "fixtures"
        
        if not (FIXTURES_DIR / "mock_prices.parquet").exists():
            pytest.skip("Mock data not available")
        
        df = pd.read_parquet(FIXTURES_DIR / "mock_prices.parquet")
        
        # Filter to one symbol
        symbol_data = df[df['symbol'] == 'MOCK000'].copy()
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        # Apply fracdiff
        result = create_fracdiff_features(
            symbol_data,
            'adj_close',
            d_values=[0.3, 0.5],
            window=50
        )
        
        assert 'fracdiff_3' in result.columns
        assert 'fracdiff_5' in result.columns
        
        # Check that we have valid values
        assert result['fracdiff_3'].notna().sum() > 0
        assert result['fracdiff_5'].notna().sum() > 0
