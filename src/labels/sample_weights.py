"""
Sample Weights Module

Implements uniqueness-based sample weighting.
Concurrent events get lower weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from src.ops.event_logger import get_logger

logger = get_logger()


class SampleWeightCalculator:
    """
    Calculate sample weights based on event uniqueness.
    
    Weight = 1 / average_concurrent_events_during_lifetime
    """
    
    def __init__(self):
        pass
    
    def calculate_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sample weights for labeled events.
        
        Args:
            df: DataFrame with labeled events (must have event_valid, label_holding_days)
            
        Returns:
            DataFrame with sample_weight column added
        """
        df = df.copy()
        df['sample_weight'] = 1.0  # Default weight
        
        # Only calculate for valid events
        valid_mask = df['event_valid'] == True
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            logger.warn("no_valid_events_for_weights", {})
            return df
        
        # Calculate concurrency for each event
        for idx, row in valid_df.iterrows():
            symbol = row['symbol']
            entry_date = row['date']
            holding_days = int(row['label_holding_days'])
            
            # Find overlapping events
            overlap_count = self._count_overlapping_events(
                valid_df, symbol, entry_date, holding_days
            )
            
            # Weight = 1 / average concurrency
            weight = 1.0 / max(overlap_count, 1)
            df.loc[idx, 'sample_weight'] = weight
        
        logger.info("weights_calculated", {
            "valid_events": len(valid_df),
            "mean_weight": df.loc[valid_mask, 'sample_weight'].mean(),
            "min_weight": df.loc[valid_mask, 'sample_weight'].min(),
            "max_weight": df.loc[valid_mask, 'sample_weight'].max()
        })
        
        return df
    
    def _count_overlapping_events(
        self,
        valid_df: pd.DataFrame,
        current_symbol: str,
        entry_date: pd.Timestamp,
        holding_days: int
    ) -> int:
        """
        Count number of concurrent events during lifetime.
        
        Events are concurrent if:
        - Same symbol is not allowed (per event protocol)
        - Different symbols can overlap
        """
        # Exit date
        entry_idx = valid_df[valid_df['date'] == entry_date].index[0]
        
        # Find events that overlap in time
        overlapping = 0
        
        for idx, row in valid_df.iterrows():
            # Skip if same event
            if row['date'] == entry_date and row['symbol'] == current_symbol:
                continue
            
            # Skip if same symbol (no intra-symbol overlap allowed)
            if row['symbol'] == current_symbol:
                continue
            
            # Check temporal overlap
            other_entry = row['date']
            other_exit = other_entry + pd.Timedelta(days=int(row['label_holding_days']))
            
            current_exit = entry_date + pd.Timedelta(days=holding_days)
            
            # Overlap if: other_entry < current_exit AND other_exit > entry_date
            if other_entry < current_exit and other_exit > entry_date:
                overlapping += 1
        
        # Include self in count
        return overlapping + 1
    
    def get_weight_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about sample weights."""
        valid_mask = df['event_valid'] == True
        
        if valid_mask.sum() == 0:
            return {}
        
        weights = df.loc[valid_mask, 'sample_weight']
        
        return {
            'count': len(weights),
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'median': float(weights.median())
        }
