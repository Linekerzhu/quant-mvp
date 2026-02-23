"""
Tests for sample weights module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.labels.sample_weights import SampleWeightCalculator


class TestSampleWeightCalculator:
    """Test sample weight calculation."""
    
    @pytest.fixture
    def calculator(self):
        """Create weight calculator."""
        return SampleWeightCalculator()
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        
        return pd.DataFrame({
            'symbol': ['A'] * 5 + ['B'] * 5,
            'date': list(dates[:5]) + list(dates[:5]),
            'event_valid': [True] * 10,
            'label_holding_days': [5] * 10,
            'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        })
    
    def test_weights_calculated(self, calculator, sample_events):
        """Test that weights are calculated."""
        result = calculator.calculate_weights(sample_events)
        
        assert 'sample_weight' in result.columns
        assert result['sample_weight'].notna().all()
    
    def test_weight_range(self, calculator, sample_events):
        """Test that weights are in valid range."""
        result = calculator.calculate_weights(sample_events)
        
        valid_weights = result[result['event_valid'] == True]['sample_weight']
        
        # All weights should be positive
        assert (valid_weights > 0).all()
        
        # Weights should be <= 1
        assert (valid_weights <= 1).all()
    
    def test_concurrent_events_lower_weight(self, calculator):
        """Test that concurrent events get lower weights."""
        # Create overlapping events
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'date': dates,
            'event_valid': [True] * 5,
            'label_holding_days': [10] * 5,  # All overlap
            'label': [1] * 5
        })
        
        result = calculator.calculate_weights(df)
        
        # All events overlap, so weights should be 1/5
        for weight in result['sample_weight']:
            assert abs(weight - 0.2) < 0.01
    
    def test_non_overlapping_events_full_weight(self, calculator):
        """Test that non-overlapping events get full weight."""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        
        df = pd.DataFrame({
            'symbol': ['A'] * 5,
            'date': dates,
            'event_valid': [True] * 5,
            'label_holding_days': [1] * 5,  # No overlap
            'label': [1] * 5
        })
        
        result = calculator.calculate_weights(df)
        
        # Same symbol events don't overlap (per protocol)
        # So weights should all be 1.0
        for weight in result['sample_weight']:
            assert weight == 1.0
    
    def test_weight_statistics(self, calculator, sample_events):
        """Test weight statistics generation."""
        result = calculator.calculate_weights(sample_events)
        
        stats = calculator.get_weight_statistics(result)
        
        assert 'mean' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
