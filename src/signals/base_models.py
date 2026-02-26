"""
Base Models for Meta-Labeling Architecture

This module provides base signal generators that produce directional signals
for the Meta-Labeling framework. Each base model generates simple signals
(+1, -1, or 0) that are then evaluated by the Meta Model.

Author: 李得勤
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
from typing import Optional


class BaseModelSMA:
    """
    Dual Moving Average Crossover Signal Generator
    
    Generates signals based on fast/slow moving average crossover:
    - +1: Fast SMA > Slow SMA (bullish golden cross)
    - -1: Fast SMA < Slow SMA (bearish death cross)
    -  0: Insufficient data (cold start period)
    
    CRITICAL: Uses shift(1) to prevent look-ahead bias.
    T-day signals can only use data from T-1 and earlier.
    """
    
    def __init__(self, fast_window: int = 20, slow_window: int = 60):
        """
        Initialize SMA base model.
        
        Args:
            fast_window: Period for fast moving average (default: 20)
            slow_window: Period for slow moving average (default: 60)
        """
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA crossover signals.
        
        Args:
            df: DataFrame with at least columns [symbol, date, adj_close]
            
        Returns:
            DataFrame with added 'side' column:
            - +1: bullish (fast > slow)
            - -1: bearish (fast < slow)
            -  0: cold start (insufficient data)
        """
        result = df.copy()
        
        # CRITICAL: Use shift(1) to prevent look-ahead bias
        # T-day signal can only use T-1 and earlier data
        sma_fast = result['adj_close'].shift(1).rolling(self.fast_window).mean()
        sma_slow = result['adj_close'].shift(1).rolling(self.slow_window).mean()
        
        # Generate signals: +1 for bullish, -1 for bearish
        result['side'] = np.where(sma_fast > sma_slow, 1, -1)
        
        # Cold start: first slow_window days have insufficient data
        result.loc[:self.slow_window - 1, 'side'] = 0
        
        # Ensure integer type
        result['side'] = result['side'].astype(int)
        
        return result
    
    def __repr__(self):
        return f"BaseModelSMA(fast={self.fast_window}, slow={self.slow_window})"


class BaseModelMomentum:
    """
    Momentum Breakout Signal Generator
    
    Generates signals based on N-day momentum (log return):
    - +1: Positive momentum (price going up)
    - -1: Negative momentum (price going down)
    -  0: Insufficient data (cold start period)
    
    CRITICAL: Uses shift(1) to prevent look-ahead bias.
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize Momentum base model.
        
        Args:
            window: Period for momentum calculation (default: 20)
        """
        self.window = window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals.
        
        Args:
            df: DataFrame with at least columns [symbol, date, adj_close]
            
        Returns:
            DataFrame with added 'side' column:
            - +1: positive momentum (returns > 0)
            - -1: negative momentum (returns < 0)
            -  0: cold start (insufficient data)
        """
        result = df.copy()
        
        # CRITICAL: Use shift(1) to prevent look-ahead bias
        # Calculate log returns using T-1 data
        returns = np.log(result['adj_close'] / result['adj_close'].shift(1))
        returns_nd = returns.shift(1).rolling(self.window).sum()
        
        # Generate signals: +1 for positive momentum, -1 for negative
        result['side'] = np.where(returns_nd > 0, 1, -1)
        
        # Cold start: first window days have insufficient data
        result.loc[:self.window - 1, 'side'] = 0
        
        # Ensure integer type
        result['side'] = result['side'].astype(int)
        
        return result
    
    def __repr__(self):
        return f"BaseModelMomentum(window={self.window})"
