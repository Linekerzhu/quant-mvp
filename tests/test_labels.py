"""
Tests for Triple Barrier labeling module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.labels.triple_barrier import TripleBarrierLabeler


class TestTripleBarrierLabeler:
    """Test Triple Barrier labeling."""
    
    @pytest.fixture
    def mock_prices(self):
        """Load mock price data."""
        path = Path("tests/fixtures/mock_prices.parquet")
        df = pd.read_parquet(path)
        
        # Add required columns for testing
        df['can_trade'] = True
        df['atr_14'] = df.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(14, min_periods=1).std()
        )
        
        return df
    
    @pytest.fixture
    def labeler(self):
        """Create labeler."""
        return TripleBarrierLabeler()
    
    def test_label_columns_created(self, labeler, mock_prices):
        """Test that label columns are created."""
        result = labeler.label_events(mock_prices)
        
        assert 'label' in result.columns
        assert 'label_barrier' in result.columns
        assert 'label_return' in result.columns
        assert 'label_holding_days' in result.columns
        assert 'event_valid' in result.columns
    
    def test_valid_events_marked(self, labeler, mock_prices):
        """Test that valid events are marked."""
        result = labeler.label_events(mock_prices)
        
        # Should have some valid events
        assert result['event_valid'].sum() > 0
    
    def test_barrier_values(self, labeler, mock_prices):
        """Test that barrier values are valid."""
        result = labeler.label_events(mock_prices)
        
        valid = result[result['event_valid'] == True]
        
        if len(valid) > 0:
            # Labels should be -1, 0, or 1 (unified semantics)
            assert valid['label'].isin([-1, 0, 1]).all()
            
            # Barrier should be one of profit/loss/time
            assert valid['label_barrier'].isin(['profit', 'loss', 'time']).all()
            
            # Holding days should be positive
            assert (valid['label_holding_days'] > 0).all()
    
    def test_label_distribution(self, labeler, mock_prices):
        """Test label distribution statistics."""
        result = labeler.label_events(mock_prices)
        
        dist = labeler.get_label_distribution(result)
        
        assert 'total_events' in dist
        assert 'by_label' in dist
        assert 'profit' in dist['by_label']  # Fixed: was 'positive'
        assert 'loss' in dist['by_label']    # Fixed: was 'negative'
        assert 'mean_return' in dist


class TestTripleBarrierLogic:
    """Test Triple Barrier internal logic."""
    
    def test_profit_barrier_hit(self):
        """Test profit barrier detection."""
        labeler = TripleBarrierLabeler()
        
        # Create synthetic data with clear profit
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        df = pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_open': [100] * 20,
            'adj_high': [120] + [110] * 19,  # First day hits profit barrier
            'adj_low': [100] * 20,
            'adj_close': [105] * 20,
            'atr_14': [5] * 20,
            'can_trade': [True] * 20
        })
        
        result = labeler.label_events(df)
        
        # First event should hit profit barrier
        if result['event_valid'].iloc[0]:
            assert result['label_barrier'].iloc[0] == 'profit'
    
    def test_loss_barrier_hit(self):
        """Test loss barrier detection."""
        labeler = TripleBarrierLabeler()
        
        # Create synthetic data with clear loss
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        df = pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_open': [100] * 20,
            'adj_high': [100] * 20,
            'adj_low': [80] + [90] * 19,  # First day hits loss barrier
            'adj_close': [95] * 20,
            'atr_14': [5] * 20,
            'can_trade': [True] * 20
        })
        
        result = labeler.label_events(df)
        
        # First event should hit loss barrier
        if result['event_valid'].iloc[0]:
            assert result['label_barrier'].iloc[0] == 'loss'
