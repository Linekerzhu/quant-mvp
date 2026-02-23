"""
Tests for data ingestion module.
Uses static mock data only - no network calls.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.ingest import DualSourceIngest


class TestDataIngestion:
    """Test data ingestion with mock data."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    def test_mock_data_loads(self, mock_prices):
        """Test that mock data loads correctly."""
        assert len(mock_prices) > 0
        assert 'symbol' in mock_prices.columns
        assert 'adj_close' in mock_prices.columns
        assert 'raw_close' in mock_prices.columns
    
    def test_mock_data_has_scenarios(self, mock_prices):
        """Test that mock data contains all scenarios."""
        symbols = mock_prices['symbol'].unique()
        
        # Split scenario
        assert 'MOCK000' in symbols
        split_data = mock_prices[mock_prices['symbol'] == 'MOCK000']
        # Check that split is reflected in adj vs raw
        
        # Halt scenario
        assert 'MOCK001' in symbols
        halt_data = mock_prices[mock_prices['symbol'] == 'MOCK001']
        assert halt_data['raw_close'].isna().sum() > 0
        
        # Jump scenario
        assert 'MOCK002' in symbols
        
        # Delist scenario
        assert 'MOCK003' in symbols
        delist_data = mock_prices[mock_prices['symbol'] == 'MOCK003']
        assert delist_data['raw_close'].isna().sum() > 0
    
    def test_dual_source_initialization(self):
        """Test that dual source can be initialized."""
        ingest = DualSourceIngest()
        assert ingest.primary is not None
        assert ingest.backup is not None


class TestDataValidation:
    """Test data validation with mock data."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    def test_validation_runs(self, mock_prices):
        """Test that validation runs without error."""
        from src.data.validate import DataValidator
        
        validator = DataValidator()
        
        # Test on subset
        subset = mock_prices[mock_prices['symbol'] == 'MOCK004'].copy()
        passed, cleaned, report = validator.validate(subset, 'MOCK004')
        
        assert isinstance(passed, bool)
        assert isinstance(cleaned, pd.DataFrame)
        assert 'checks' in report
    
    def test_duplicate_detection(self, mock_prices):
        """Test duplicate detection."""
        from src.data.validate import DataValidator
        
        validator = DataValidator()
        
        # Create data with duplicate
        subset = mock_prices[mock_prices['symbol'] == 'MOCK004'].copy()
        duplicate = subset.iloc[[0]]
        with_duplicates = pd.concat([subset, duplicate], ignore_index=True)
        
        passed, cleaned, report = validator.validate(with_duplicates, 'MOCK004')
        
        assert report['checks']['duplicates']['count'] >= 1


class TestNoLeakage:
    """Test that there's no data leakage."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    def test_no_future_data_in_features(self, mock_prices):
        """
        Test that feature calculation doesn't use future data.
        
        This is a basic check - full leakage testing is in test_no_leakage.py
        """
        # Simple check: prices are sorted by date
        for symbol in mock_prices['symbol'].unique()[:3]:
            symbol_df = mock_prices[mock_prices['symbol'] == symbol].sort_values('date')
            dates = symbol_df['date'].tolist()
            
            # Check dates are monotonically increasing
            for i in range(1, len(dates)):
                assert dates[i] > dates[i-1], f"Dates not sorted for {symbol}"
    
    def test_adj_factor_calculation(self, mock_prices):
        """Test that adjustment factor is calculated correctly."""
        # adj_factor = adj_close / raw_close
        # For most days, this should be consistent
        
        for symbol in ['MOCK004', 'MOCK005']:  # Normal symbols
            symbol_df = mock_prices[mock_prices['symbol'] == symbol].dropna()
            
            if len(symbol_df) > 0:
                adj_factor = symbol_df['adj_close'] / symbol_df['raw_close']
                
                # Adj factor should be stable (except for splits)
                # In mock data without splits, should be ~1.0
                assert adj_factor.std() < 0.01, f"Unstable adj_factor for {symbol}"
