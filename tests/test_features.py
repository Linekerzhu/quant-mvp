"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.build_features import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    @pytest.fixture
    def engineer(self):
        """Create feature engineer."""
        return FeatureEngineer()
    
    def test_momentum_features(self, engineer, mock_prices):
        """Test momentum feature calculation."""
        result = engineer._calc_momentum_features(mock_prices)
        
        # Check return features exist
        for window in [5, 10, 20, 60]:
            assert f'returns_{window}d' in result.columns
        
        # Check RSI exists
        assert 'rsi_14' in result.columns
        
        # Check MACD exists
        assert 'macd_line' in result.columns
        assert 'macd_signal' in result.columns
    
    def test_volatility_features(self, engineer, mock_prices):
        """Test volatility feature calculation."""
        result = engineer._calc_volatility_features(mock_prices)
        
        # Check realized volatility
        for window in [5, 20, 60]:
            assert f'rv_{window}d' in result.columns
        
        # Check ATR
        assert 'atr_14' in result.columns
    
    def test_volume_features(self, engineer, mock_prices):
        """Test volume feature calculation."""
        result = engineer._calc_volume_features(mock_prices)
        
        # Check relative volume
        assert 'relative_volume_20d' in result.columns
        
        # Check OBV
        assert 'obv' in result.columns
    
    def test_mean_reversion_features(self, engineer, mock_prices):
        """Test mean reversion feature calculation."""
        result = engineer._calc_mean_reversion_features(mock_prices)
        
        # Check z-scores
        assert 'price_vs_sma20_zscore' in result.columns
        assert 'price_vs_sma60_zscore' in result.columns
        assert 'price_vs_ema20_zscore' in result.columns
    
    def test_dummy_noise_injection(self, engineer, mock_prices):
        """Test dummy noise feature injection (Plan v4)."""
        result = engineer._inject_dummy_noise(mock_prices)
        
        # Check dummy_noise exists
        assert 'dummy_noise' in result.columns
        
        # Check distribution
        assert abs(result['dummy_noise'].mean()) < 0.5  # Should be ~0
        assert abs(result['dummy_noise'].std() - 1.0) < 0.5  # Should be ~1
    
    def test_full_feature_build(self, engineer, mock_prices):
        """Test full feature building pipeline."""
        result = engineer.build_features(mock_prices)
        
        # Check feature version
        assert 'feature_version' in result.columns
        
        # Check expected features exist
        assert 'returns_5d' in result.columns
        assert 'rsi_14' in result.columns
        assert 'dummy_noise' in result.columns
        
        # Check no NaN in key features (except initial periods)
        non_na_count = result['returns_20d'].notna().sum()
        assert non_na_count > 0
    
    def test_feature_metadata(self, engineer):
        """Test feature metadata generation."""
        metadata = engineer.get_feature_metadata()
        
        assert 'version' in metadata
        assert 'feature_count' in metadata
        assert 'dummy_feature' in metadata
        assert metadata['dummy_feature'] == 'dummy_noise'
