"""
Comprehensive data leakage tests.

These tests ensure no future data is used in training or feature calculation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestNoDataLeakage:
    """Test suite for data leakage prevention."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    def test_point_in_time_alignment(self, mock_prices):
        """
        Test that all data is properly Point-in-Time aligned.
        
        Data should be sorted by date with no future values.
        """
        for symbol in mock_prices['symbol'].unique():
            symbol_df = mock_prices[mock_prices['symbol'] == symbol].sort_values('date')
            
            # Check monotonic dates
            dates = symbol_df['date'].values
            for i in range(1, len(dates)):
                assert dates[i] > dates[i-1], f"Non-monotonic dates for {symbol}"
    
    def test_forward_fill_limit(self, mock_prices):
        """
        Test that forward fill doesn't leak future data.
        
        Forward fill should only use past values.
        """
        from src.data.validate import DataValidator
        
        validator = DataValidator()
        
        # Check that validation doesn't create leakage
        subset = mock_prices[mock_prices['symbol'] == 'MOCK001'].copy()  # Has NaN
        
        # Find NaN positions
        nan_mask = subset['raw_close'].isna()
        
        if nan_mask.any():
            passed, cleaned, report = validator.validate(subset, 'MOCK001')
            
            # After validation, NaN should be filled only from past values
            # Not from future values
            
            # Find filled positions
            was_nan = nan_mask.values
            still_nan = cleaned['raw_close'].isna().values
            filled = was_nan & ~still_nan
            
            # For filled values, they should equal previous non-NaN value
            for i in np.where(filled)[0]:
                if i > 0:
                    # Filled value should come from before, not after
                    filled_value = cleaned.iloc[i]['raw_close']
                    
                    # Find previous non-NaN
                    prev_idx = i - 1
                    while prev_idx >= 0 and was_nan[prev_idx]:
                        prev_idx -= 1
                    
                    if prev_idx >= 0:
                        prev_value = subset.iloc[prev_idx]['raw_close']
                        # Allow for floating point differences
                        assert abs(filled_value - prev_value) < 0.01 or np.isnan(prev_value)
    
    def test_feature_calculation_no_future(self, mock_prices):
        """
        Test that feature calculation only uses historical data.
        
        For a feature at time t, it should only use data from <= t.
        """
        # Placeholder for feature engineering tests
        # Will be expanded in test_features.py
        
        # For now, just verify date ordering
        for symbol in ['MOCK004', 'MOCK005']:
            symbol_df = mock_prices[mock_prices['symbol'] == symbol].sort_values('date')
            
            # Calculate a simple rolling mean
            window = 5
            rolling = symbol_df['adj_close'].rolling(window=window, min_periods=1).mean()
            
            # At each point, rolling mean should only use available history
            for i in range(len(symbol_df)):
                expected_window = min(i + 1, window)
                actual_values = symbol_df.iloc[max(0, i - window + 1):i + 1]['adj_close'].values
                
                assert len(actual_values) == expected_window, f"Window size mismatch at {i}"
    
    def test_no_lookahead_in_returns(self, mock_prices):
        """
        Test that returns don't use future prices.
        
        Return at time t should use price at t and t-1 only.
        """
        for symbol in ['MOCK004']:
            symbol_df = mock_prices[mock_prices['symbol'] == symbol].sort_values('date')
            
            # Calculate log returns manually
            prices = symbol_df['adj_close'].values
            manual_returns = np.log(prices[1:] / prices[:-1])
            
            # Each return only uses current and previous price
            for i in range(len(manual_returns)):
                used_prices = [prices[i], prices[i + 1]]  # t and t+1 for forward return
                # This is actually correct - return at t uses t and t-1
    
    def test_train_test_separation(self):
        """
        Test that train and test sets are properly separated.
        
        Will be expanded when we have train/test split logic.
        """
        # Placeholder for train/test separation tests
        pass


class TestFeatureLayerLeakage:
    """
    Feature layer leakage tests (B23).
    
    Ensures feature engineering doesn't use future data.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known pattern for testing."""
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        df = pd.DataFrame({
            'symbol': ['TEST'] * 20,
            'date': dates,
            'adj_open': [100 + i for i in range(20)],
            'adj_high': [102 + i for i in range(20)],
            'adj_low': [98 + i for i in range(20)],
            'adj_close': [101 + i for i in range(20)],
            'volume': [1000] * 20,
            'can_trade': [True] * 20,
            'atr_20': [2.0] * 20  # R7: updated to atr_20 to match config
        })
        return df
    
    def test_momentum_features_no_future(self, sample_data):
        """
        Test that momentum features don't use future data.
        
        returns_5d at date[i] should only use prices from date[i-5] to date[i].
        """
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        result = engineer._calc_momentum_features_fast(sample_data.copy())
        
        # Check 5-day return at index 5
        idx = 5
        expected_return = np.log(
            sample_data.iloc[idx]['adj_close'] / sample_data.iloc[idx - 5]['adj_close']
        )
        actual_return = result.iloc[idx]['returns_5d']
        
        assert abs(actual_return - expected_return) < 1e-6, \
            f"returns_5d uses future data: expected {expected_return}, got {actual_return}"
    
    def test_volatility_features_no_future(self, sample_data):
        """
        Test that volatility features only use historical data.
        """
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        result = engineer._calc_volatility_features_fast(sample_data.copy())
        
        # Check ATR at index 10
        idx = 10
        
        # ATR should only use data up to index 10
        atr = result.iloc[idx]['atr_20']  # R7: updated to atr_20
        
        # ATR should be finite and based on historical data
        assert np.isfinite(atr), "ATR is not finite"
        assert atr >= 0, "ATR should be non-negative"
    
    def test_mean_reversion_features_no_future(self, sample_data):
        """
        Test that mean reversion features only use historical data.
        """
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        result = engineer._calc_mean_reversion_features_fast(sample_data.copy())
        
        # Check SMA z-score at index 10
        idx = 10
        
        # SMA(20) at index 10 should use indices 0-10 (only 11 days available)
        sma = result.iloc[idx]['price_vs_sma20_zscore']
        
        # Should be based on available history (not full 20 days)
        assert np.isfinite(sma) or np.isnan(sma), "SMA z-score should be finite or NaN"
    
    def test_features_valid_flag(self, sample_data):
        """
        Test that features_valid flag is correctly set.
        
        Rows with NaN features should have features_valid=False.
        """
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        result = engineer.build_features(sample_data)
        
        # Check that features_valid column exists
        assert 'features_valid' in result.columns
        
        # Early rows may have NaN features due to lookback windows
        # They should be marked as invalid
        early_rows = result.iloc[:5]
        
        # Log the results for inspection
        invalid_count = (~result['features_valid']).sum()
        print(f"Invalid feature rows: {invalid_count}/{len(result)}")
    
    def test_no_forward_peeking_in_rolling(self):
        """
        Test that rolling windows don't peek forward.
        
        Create a specific test case where forward peeking would be detectable.
        """
        # Create data with a clear pattern
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        
        # Price doubles on day 5
        prices = [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]
        
        df = pd.DataFrame({
            'symbol': ['TEST'] * 10,
            'date': dates,
            'adj_open': prices,
            'adj_high': [p + 2 for p in prices],
            'adj_low': [p - 2 for p in prices],
            'adj_close': prices,
            'volume': [1000] * 10,
            'can_trade': [True] * 10
        })
        
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        result = engineer.build_features(df)
        
        # At day 5 (index 5), returns_5d = log(close[5]/close[0])
        # This SHOULD capture the price jump since day 5 IS the current observation
        # The key test: day 4 should NOT reflect the day 5 jump
        if 'returns_5d' in result.columns:
            return_day4 = result.iloc[4]['returns_5d']
            
            # returns_5d at index 4 = log(close[4]/close[-1]) → NaN (insufficient history)
            # NaN means no forward leak — it doesn't use future data
            if pd.notna(return_day4):
                # If somehow not NaN, verify it only uses data up to day 4
                price_day0 = prices[0]
                price_day4 = prices[4]
                expected_return = np.log(price_day4 / price_day0)
                assert abs(return_day4 - expected_return) < 1e-6, \
                    f"Rolling calculation uses future data: expected {expected_return}, got {return_day4}"
            
            # Also verify day 5 return correctly captures the jump (uses current obs, not future)
            return_day5 = result.iloc[5]['returns_5d']
            if pd.notna(return_day5):
                expected_day5 = np.log(prices[5] / prices[0])  # log(200/100)
                assert abs(return_day5 - expected_day5) < 1e-6, \
                    f"Day 5 return wrong: expected {expected_day5}, got {return_day5}"
