"""
Tests for corporate actions module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.corporate_actions import CorporateActionsHandler


class TestCorporateActions:
    """Test corporate action detection and handling."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        return pd.read_parquet(path)
    
    @pytest.fixture
    def handler(self):
        """Create corporate actions handler."""
        return CorporateActionsHandler()
    
    def test_split_detection(self, handler, mock_prices):
        """Test split detection on MOCK000."""
        result = handler.detect_splits(mock_prices)
        
        # MOCK000 should have detected split
        mock000 = result[result['symbol'] == 'MOCK000']
        assert mock000['detected_split'].any()
    
    def test_split_ratio_calculation(self, handler, mock_prices):
        """Test that split ratio is calculated correctly."""
        result = handler.detect_splits(mock_prices)
        
        mock000 = result[result['symbol'] == 'MOCK000']
        split_rows = mock000[mock000['detected_split']]
        
        if not split_rows.empty:
            ratio = split_rows.iloc[0]['split_ratio']
            # Should be around 2.0 for 2:1 split
            assert 1.8 < ratio < 2.2
    
    def test_delisting_detection(self, handler, mock_prices):
        """Test delisting detection on MOCK003."""
        result, delist_info = handler.detect_delistings(mock_prices)
        
        # MOCK003 should be detected as delisted
        assert 'MOCK003' in delist_info
    
    def test_suspension_detection(self, handler, mock_prices):
        """Test suspension detection on MOCK001."""
        result = handler.detect_suspensions(mock_prices)
        
        # MOCK001 should have suspension marked
        mock001 = result[result['symbol'] == 'MOCK001']
        assert mock001['is_suspended'].any()
    
    def test_suspension_can_trade_flag(self, handler, mock_prices):
        """Test that suspended symbols are marked as non-tradable."""
        result = handler.detect_suspensions(mock_prices)
        
        mock001 = result[result['symbol'] == 'MOCK001']
        suspended = mock001[mock001['is_suspended']]
        
        # Suspended days should have can_trade = False
        assert not suspended['can_trade'].any()
    
    def test_apply_all_corporate_actions(self, handler, mock_prices):
        """Test applying all corporate actions at once."""
        result, info = handler.apply_all(mock_prices)
        
        assert 'splits_detected' in info
        assert 'delistings_detected' in info
        assert 'suspensions_detected' in info
        
        # Should detect our mock scenarios
        assert info['splits_detected'] >= 1
        assert info['delistings_detected'] >= 1
        assert info['suspensions_detected'] >= 1
    
    def test_no_false_split_detection(self, handler, mock_prices):
        """Test that normal price movements aren't flagged as splits."""
        result = handler.detect_splits(mock_prices)
        
        # Normal symbols (MOCK004+) shouldn't have splits
        normal = result[result['symbol'].isin(['MOCK004', 'MOCK005', 'MOCK006'])]
        assert not normal['detected_split'].any()
