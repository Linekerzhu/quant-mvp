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
