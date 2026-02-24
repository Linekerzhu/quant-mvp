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
            'adj_high': [120] + [111] * 19,  # Days after entry hit profit barrier (>110)
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


class TestNonOverlappingConstraint:
    """Test Phase B §6.5: Non-overlapping event constraint (P1-1)."""
    
    def test_no_overlapping_events_per_symbol(self):
        """
        Test that same symbol cannot have overlapping events.
        
        When an event is active (entry to exit), no new event should
        be triggered for the same symbol.
        """
        labeler = TripleBarrierLabeler()
        
        # Create data where price stays flat - events will hit time barrier
        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        df = pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_open': [100] * 30,
            'adj_high': [101] * 30,
            'adj_low': [99] * 30,
            'adj_close': [100] * 30,
            'atr_14': [1.0] * 30,  # Small ATR
            'can_trade': [True] * 30
        })
        
        result = labeler.label_events(df)
        
        # Get valid events
        valid_events = result[result['event_valid'] == True]
        
        # Check that events don't overlap
        if len(valid_events) > 1:
            for i in range(len(valid_events) - 1):
                exit_date_i = valid_events.iloc[i]['date'] + pd.Timedelta(
                    days=int(valid_events.iloc[i]['label_holding_days'])
                )
                entry_date_next = valid_events.iloc[i + 1]['date']
                
                # Next event should start after current event exits
                assert entry_date_next > exit_date_i, \
                    f"Events overlap: event {i} exits {exit_date_i}, next starts {entry_date_next}"
    
    def test_overlapping_events_rejected(self):
        """
        Test that overlapping events are explicitly rejected.
        
        With 10-day holding period, we should see ~3 events in 30 days
        (not 30 events), demonstrating the overlap constraint.
        """
        labeler = TripleBarrierLabeler()
        
        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        df = pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'adj_open': [100] * 30,
            'adj_high': [101] * 30,
            'adj_low': [99] * 30,
            'adj_close': [100] * 30,
            'atr_14': [1.0] * 30,
            'can_trade': [True] * 30
        })
        
        result = labeler.label_events(df)
        valid_count = result['event_valid'].sum()
        
        # With 10-day holding and non-overlap constraint:
        # Max ~3 events in 30 days
        assert valid_count <= 3, \
            f"Expected <= 3 non-overlapping events in 30 days, got {valid_count}"
    
    def test_multi_symbol_events_independent(self):
        """
        Test that different symbols can have concurrent events.
        
        Overlap constraint is per-symbol, not global.
        """
        labeler = TripleBarrierLabeler()
        
        dates = pd.date_range('2024-01-01', periods=15, freq='B')
        
        # Create data for 2 symbols
        df = pd.concat([
            pd.DataFrame({
                'symbol': 'A',
                'date': dates,
                'adj_open': [100] * 15,
                'adj_high': [101] * 15,
                'adj_low': [99] * 15,
                'adj_close': [100] * 15,
                'atr_14': [1.0] * 15,
                'can_trade': [True] * 15
            }),
            pd.DataFrame({
                'symbol': 'B',
                'date': dates,
                'adj_open': [200] * 15,
                'adj_high': [201] * 15,
                'adj_low': [199] * 15,
                'adj_close': [200] * 15,
                'atr_14': [2.0] * 15,
                'can_trade': [True] * 15
            })
        ]).reset_index(drop=True)
        
        result = labeler.label_events(df)
        
        # Both symbols should have events
        for symbol in ['A', 'B']:
            symbol_events = result[(result['symbol'] == symbol) & (result['event_valid'] == True)]
            assert len(symbol_events) > 0, f"Symbol {symbol} should have valid events"
