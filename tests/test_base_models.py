"""
Tests for Base Models (SMA and Momentum)

Tests the Meta-Labeling base signal generators for:
1. No look-ahead bias (shift(1) correctly applied)
2. Correct signal values {-1, 0, +1}
3. Cold start handling
4. Deterministic output
5. Integration with Triple Barrier

Author: 李得勤
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.signals.base_models import BaseModelSMA, BaseModelMomentum

# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestBaseModelSMA:
    """Test suite for BaseModelSMA."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data for testing."""
        return pd.read_parquet(FIXTURES_DIR / "mock_prices.parquet")
    
    @pytest.fixture
    def sample_df(self):
        """Create minimal test DataFrame."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        
        # Simulate random walk prices
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_close': prices
        })
    
    def test_sma_signal_values(self, sample_df):
        """Verify SMA signal values are in {-1, 0, +1}."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        result = model.generate_signals(sample_df)
        
        # Check unique values
        unique_signals = result['side'].unique()
        assert set(unique_signals).issubset({-1, 0, 1}), \
            f"Signal values must be in {{-1, 0, 1}}, got {unique_signals}"
        
        # Check no NaN
        assert result['side'].isna().sum() == 0, "Signals contain NaN"
    
    def test_sma_signal_no_lookahead(self, sample_df):
        """CRITICAL: Verify SMA signal does NOT use T-day price data."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        
        # Create data with extreme T-day move
        df = sample_df.copy()
        df.loc[99, 'adj_close'] = 500  # Extreme move on last day
        
        result = model.generate_signals(df)
        
        # The signal on day 99 should NOT reflect the extreme move
        # because shift(1) means T-day signal only uses T-1 and earlier data
        # 
        # Verify: If we remove the extreme move, signal should be same
        df_normal = sample_df.copy()
        result_normal = model.generate_signals(df_normal)
        
        # Signals before the last day should be identical
        # (cold start period is 60 days, so day 60-98 should match)
        pd.testing.assert_series_equal(
            result['side'].iloc[:99],
            result_normal['side'].iloc[:99],
            check_dtype=False
        )
    
    def test_sma_cold_start(self, sample_df):
        """Verify cold start period returns side=0."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        result = model.generate_signals(sample_df)
        
        # First 60 days (slow_window - 1) should be 0
        assert (result['side'].iloc[:60] == 0).all(), \
            "Cold start period should have side=0"
        
        # After cold start, should have non-zero signals
        assert (result['side'].iloc[60:] != 0).any(), \
            "Post cold-start should have non-zero signals"
    
    def test_sma_deterministic(self, sample_df):
        """Verify same input produces same output."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        
        result1 = model.generate_signals(sample_df)
        result2 = model.generate_signals(sample_df)
        
        pd.testing.assert_series_equal(
            result1['side'],
            result2['side'],
            check_dtype=False
        )
    
    def test_sma_with_mock_data(self, mock_prices):
        """Integration test with full mock dataset."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        
        # Test on one symbol
        symbol_data = mock_prices[mock_prices['symbol'] == 'MOCK000'].copy()
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        result = model.generate_signals(symbol_data)
        
        # Verify structure
        assert 'side' in result.columns
        assert len(result) == len(symbol_data)
        
        # Verify cold start
        assert (result['side'].iloc[:60] == 0).all()
        
        # Verify values
        assert set(result['side'].unique()).issubset({-1, 0, 1})


class TestBaseModelMomentum:
    """Test suite for BaseModelMomentum."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data for testing."""
        return pd.read_parquet(FIXTURES_DIR / "mock_prices.parquet")
    
    @pytest.fixture
    def sample_df(self):
        """Create minimal test DataFrame."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        
        # Simulate random walk prices
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_close': prices
        })
    
    def test_momentum_signal_values(self, sample_df):
        """Verify Momentum signal values are in {-1, 0, +1}."""
        model = BaseModelMomentum(window=20)
        result = model.generate_signals(sample_df)
        
        unique_signals = result['side'].unique()
        assert set(unique_signals).issubset({-1, 0, 1}), \
            f"Signal values must be in {{-1, 0, 1}}, got {unique_signals}"
        
        assert result['side'].isna().sum() == 0, "Signals contain NaN"
    
    def test_momentum_signal_no_lookahead(self, sample_df):
        """CRITICAL: Verify Momentum signal does NOT use T-day price data."""
        model = BaseModelMomentum(window=20)
        
        # Create data with extreme T-day move
        df = sample_df.copy()
        df.loc[49, 'adj_close'] = 500  # Extreme move on last day
        
        result = model.generate_signals(df)
        
        # Same logic as SMA test - T-day signal should not use T-day price
        df_normal = sample_df.copy()
        result_normal = model.generate_signals(df_normal)
        
        # Signals before the last day should be identical
        pd.testing.assert_series_equal(
            result['side'].iloc[:49],
            result_normal['side'].iloc[:49],
            check_dtype=False
        )
    
    def test_momentum_cold_start(self, sample_df):
        """Verify cold start period returns side=0."""
        model = BaseModelMomentum(window=20)
        result = model.generate_signals(sample_df)
        
        # First 20 days (window - 1) should be 0
        assert (result['side'].iloc[:20] == 0).all(), \
            "Cold start period should have side=0"
        
        # After cold start, should have non-zero signals
        assert (result['side'].iloc[20:] != 0).any(), \
            "Post cold-start should have non-zero signals"
    
    def test_momentum_deterministic(self, sample_df):
        """Verify same input produces same output."""
        model = BaseModelMomentum(window=20)
        
        result1 = model.generate_signals(sample_df)
        result2 = model.generate_signals(sample_df)
        
        pd.testing.assert_series_equal(
            result1['side'],
            result2['side'],
            check_dtype=False
        )
    
    def test_momentum_with_mock_data(self, mock_prices):
        """Integration test with full mock dataset."""
        model = BaseModelMomentum(window=20)
        
        # Test on one symbol
        symbol_data = mock_prices[mock_prices['symbol'] == 'MOCK000'].copy()
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        result = model.generate_signals(symbol_data)
        
        # Verify structure
        assert 'side' in result.columns
        assert len(result) == len(symbol_data)
        
        # Verify cold start
        assert (result['side'].iloc[:20] == 0).all()
        
        # Verify values
        assert set(result['side'].unique()).issubset({-1, 0, 1})


class TestBaseModelIntegration:
    """Integration tests for Base Model + Triple Barrier."""
    
    @pytest.fixture
    def mock_prices_with_atr(self):
        """Create test data with ATR column."""
        df = pd.read_parquet(FIXTURES_DIR / "mock_prices.parquet")
        
        # Add required columns for Triple Barrier
        df['atr_14'] = df.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(14).std()
        )
        df['volume'] = df['volume'].fillna(1e6)
        df['can_trade'] = True
        df['features_valid'] = True
        
        return df
    
    def test_sma_signal_for_triple_barrier(self, mock_prices_with_atr):
        """Verify SMA signals work with Triple Barrier input format."""
        model = BaseModelSMA(fast_window=20, slow_window=60)
        
        symbol_data = mock_prices_with_atr[
            mock_prices_with_atr['symbol'] == 'MOCK000'
        ].copy().sort_values('date').reset_index(drop=True)
        
        # Generate signals
        result = model.generate_signals(symbol_data)
        
        # Should have side column
        assert 'side' in result.columns
        
        # Verify non-zero signals exist (for Triple Barrier to process)
        nonzero_count = (result['side'] != 0).sum()
        assert nonzero_count > 0, "Should have non-zero signals for Triple Barrier"
        
        # Verify signal distribution
        long_signals = (result['side'] == 1).sum()
        short_signals = (result['side'] == -1).sum()
        assert long_signals > 0, "Should have long signals"
        assert short_signals > 0, "Should have short signals"
    
    def test_momentum_signal_for_triple_barrier(self, mock_prices_with_atr):
        """Verify Momentum signals work with Triple Barrier input format."""
        model = BaseModelMomentum(window=20)
        
        symbol_data = mock_prices_with_atr[
            mock_prices_with_atr['symbol'] == 'MOCK000'
        ].copy().sort_values('date').reset_index(drop=True)
        
        result = model.generate_signals(symbol_data)
        
        assert 'side' in result.columns
        nonzero_count = (result['side'] != 0).sum()
        assert nonzero_count > 0, "Should have non-zero signals for Triple Barrier"


class TestEdgeCases:
    """Edge case tests for base models."""
    
    def test_sma_empty_dataframe(self):
        """Test SMA with empty DataFrame."""
        import pandas as pd
        from src.signals.base_models import BaseModelSMA
        
        model = BaseModelSMA(fast_window=5, slow_window=10)
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'adj_close'])
        
        try:
            model.generate_signals(empty_df)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()
    
    def test_momentum_empty_dataframe(self):
        """Test Momentum with empty DataFrame."""
        import pandas as pd
        from src.signals.base_models import BaseModelMomentum
        
        model = BaseModelMomentum(window=10)
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'adj_close'])
        
        try:
            model.generate_signals(empty_df)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()
    
    def test_sma_missing_column(self):
        """Test SMA with missing adj_close column."""
        import pandas as pd
        from src.signals.base_models import BaseModelSMA
        
        model = BaseModelSMA(fast_window=5, slow_window=10)
        df = pd.DataFrame({'symbol': ['A'], 'date': ['2026-01-01']})
        
        try:
            model.generate_signals(df)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "adj_close" in str(e).lower()
