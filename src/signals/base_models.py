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

from src.signals.base import BaseSignalGenerator, SignalModelRegistry


@SignalModelRegistry.register('sma')
class BaseModelSMA(BaseSignalGenerator):
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
        # P0 Fix: Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        if 'adj_close' not in df.columns:
            raise ValueError("Missing required column: adj_close")
        
        result = df.copy()
        
        # CRITICAL: Use shift(1) to prevent look-ahead bias
        # T-day signal can only use T-1 and earlier data
        sma_fast = result['adj_close'].shift(1).rolling(self.fast_window).mean()
        sma_slow = result['adj_close'].shift(1).rolling(self.slow_window).mean()
        
        # Generate signals: +1 for bullish, -1 for bearish
        result['side'] = np.where(sma_fast > sma_slow, 1, -1)
        
        # R14-A4 Fix: Cold start - use position-based indexing instead of label-based
        # .iloc[:self.slow_window] selects first slow_window positions regardless of index
        result.iloc[:self.slow_window, result.columns.get_loc('side')] = 0
        
        # Ensure integer type
        result['side'] = result['side'].astype(int)
        
        return result
    
    def __repr__(self):
        return f"BaseModelSMA(fast={self.fast_window}, slow={self.slow_window})"


@SignalModelRegistry.register('momentum')
class BaseModelMomentum(BaseSignalGenerator):
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
    
    def _validate_price_data(self, price_series: pd.Series) -> pd.Series:
        """
        检查价格数据有效性。
        
        Args:
            price_series: Price series with index
        
        Returns:
            Boolean Series: True if valid, False otherwise
        """
        price_prev = price_series.shift(1)
        price_curr = price_series
        
        # Valid if: both prices > 0 and not NaN
        valid_mask = (
            (price_prev > 0) & 
            (price_curr > 0) & 
            price_prev.notna() & 
            price_curr.notna()
        )
        
        return valid_mask
    
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
        # P0 Fix: Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        if 'adj_close' not in df.columns:
            raise ValueError("Missing required column: adj_close")
        
        result = df.copy()
        
        # CRITICAL: Use shift(1) to prevent look-ahead bias
        # Calculate log returns using T-1 data
        
        # P0 Fix: Handle NaN/Inf in price data
        # Get T-1 price and verify it's valid
        price_prev = result['adj_close'].shift(1)
        price_curr = result['adj_close']
        
        # Use extracted validation method
        valid_mask = self._validate_price_data(result['adj_close'])
        
        # Calculate ratio safely - keep as Series to use .shift()
        result['price_ratio'] = price_curr / price_prev
        result.loc[~valid_mask, 'price_ratio'] = np.nan
        
        # Log return
        result['returns'] = np.log(result['price_ratio'])
        
        # Accumulate returns
        returns_nd = result['returns'].shift(1).rolling(self.window).sum()
        
        # Generate signals: +1 for positive momentum, -1 for negative
        result['side'] = np.where(returns_nd > 0, 1, -1)
        
        # R14-A4 Fix: Cold start - use position-based indexing instead of label-based
        result.iloc[:self.window, result.columns.get_loc('side')] = 0
        
        # Also set side=0 where returns are NaN (no valid signal)
        result.loc[returns_nd.isna(), 'side'] = 0
        
        # Ensure integer type
        result['side'] = result['side'].astype(int)
        
        return result
    
    def __repr__(self):
        return f"BaseModelMomentum(window={self.window})"
