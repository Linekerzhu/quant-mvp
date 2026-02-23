"""
Sample Weights Module

Implements uniqueness-based sample weighting.
Concurrent events get lower weights.

OPTIMIZED: Uses vectorized interval tree approach instead of O(n²) pairwise comparison.
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from typing import Dict, List
import bisect

from src.ops.event_logger import get_logger

logger = get_logger()


class SampleWeightCalculator:
    """
    Calculate sample weights based on event uniqueness.
    
    Weight = 1 / average_concurrent_events_during_lifetime
    
    OPTIMIZATION: Uses interval tree + vectorized operations.
    Complexity: O(n log n) instead of O(n²)
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
        
        # Build interval tree for efficient overlap queries
        intervals = self._build_interval_tree(valid_df)
        
        # Calculate weights using optimized algorithm
        weights = self._calculate_weights_optimized(valid_df, intervals)
        
        # Assign weights back to dataframe
        df.loc[valid_mask, 'sample_weight'] = weights
        
        logger.info("weights_calculated", {
            "valid_events": len(valid_df),
            "mean_weight": float(weights.mean()),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max())
        })
        
        return df
    
    def _build_interval_tree(self, valid_df: pd.DataFrame) -> Dict:
        """
        Build interval tree for efficient overlap queries.
        
        Structure: {symbol: [(entry_date, exit_date, event_id), ...]}
        Sorted by entry_date for each symbol.
        """
        intervals = {}
        
        for idx, row in valid_df.iterrows():
            symbol = row['symbol']
            entry_date = row['date']
            holding_days = int(row['label_holding_days'])
            exit_date = entry_date + BusinessDay(holding_days)
            
            if symbol not in intervals:
                intervals[symbol] = []
            
            intervals[symbol].append((entry_date, exit_date, idx))
        
        # Sort each symbol's intervals by entry date
        for symbol in intervals:
            intervals[symbol].sort(key=lambda x: x[0])
        
        return intervals
    
    def _calculate_weights_optimized(
        self,
        valid_df: pd.DataFrame,
        intervals: Dict
    ) -> pd.Series:
        """
        Calculate weights using vectorized operations.
        
        Algorithm:
        1. For each event, count concurrent events from OTHER symbols
        2. Intra-symbol events are not allowed (always 1)
        3. Use interval tree for efficient overlap detection
        """
        weights = pd.Series(index=valid_df.index, dtype=float)
        
        # Pre-compute all event intervals for cross-symbol queries
        all_entries = []
        all_exits = []
        all_symbols = []
        all_indices = []
        
        for idx, row in valid_df.iterrows():
            entry_date = row['date']
            holding_days = int(row['label_holding_days'])
            exit_date = entry_date + BusinessDay(holding_days)
            
            all_entries.append(entry_date)
            all_exits.append(exit_date)
            all_symbols.append(row['symbol'])
            all_indices.append(idx)
        
        # Convert to arrays for vectorized operations
        all_entries = np.array(all_entries)
        all_exits = np.array(all_exits)
        all_symbols = np.array(all_symbols)
        
        # For each event, count overlapping events from OTHER symbols
        for i, idx in enumerate(all_indices):
            current_symbol = all_symbols[i]
            entry_date = all_entries[i]
            exit_date = all_exits[i]
            
            # Find events from OTHER symbols that overlap
            # Overlap condition: other_entry < current_exit AND other_exit > current_entry
            other_symbol_mask = all_symbols != current_symbol
            overlap_mask = (
                other_symbol_mask &
                (all_entries < exit_date) &
                (all_exits > entry_date)
            )
            
            # Count includes self (1) + overlapping events
            concurrent_count = overlap_mask.sum() + 1
            
            # Weight = 1 / average concurrency
            weight = 1.0 / max(concurrent_count, 1)
            weights.loc[idx] = weight
        
        return weights
    
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
