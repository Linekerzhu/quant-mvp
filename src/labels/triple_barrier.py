"""
Triple Barrier Labeling Module

Implements Triple Barrier method for labeling events.
Follows the event generation protocol from config/event_protocol.yaml.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
import yaml

from src.ops.event_logger import get_logger

logger = get_logger()


class BarrierHit(Enum):
    """Which barrier was hit."""
    PROFIT = 1
    LOSS = -1
    TIME = 0


class TripleBarrierLabeler:
    """
    Triple Barrier event labeler.
    
    For each event:
    - Entry: T+1 open after trigger
    - Exit: First of (profit barrier, loss barrier, max holding days)
    """
    
    def __init__(self, config_path: str = "config/event_protocol.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        tb = self.config['triple_barrier']
        self.atr_window = tb['atr']['window']
        self.min_atr_pct = tb['atr']['min_atr_pct']
        self.tp_mult = tb['profit_take']['multiplier']
        self.sl_mult = tb['stop_loss']['multiplier']
        self.max_holding_days = tb['max_holding_days']
    
    def label_events(
        self,
        df: pd.DataFrame,
        trigger_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Generate Triple Barrier labels for all valid events.
        
        Args:
            df: DataFrame with OHLCV + ATR data
            trigger_col: Column indicating event trigger date
            
        Returns:
            DataFrame with label columns added
        """
        df = df.copy()
        
        # Initialize label columns
        df['label'] = np.nan
        df['label_barrier'] = None
        df['label_return'] = np.nan
        df['label_holding_days'] = np.nan
        df['event_valid'] = False
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask].sort_values('date').reset_index(drop=True)
            
            if len(symbol_df) < self.max_holding_days + 1:
                continue
            
            # Generate labels for each day (if valid)
            for i in range(len(symbol_df) - 1):
                # Skip if not tradable
                if not self._is_valid_event(symbol_df, i):
                    continue
                
                label, barrier, ret, holding_days = self._label_single_event(
                    symbol_df, i
                )
                
                # Store in original dataframe
                date = symbol_df.loc[i, 'date']
                df.loc[(df['symbol'] == symbol) & (df['date'] == date), 'label'] = label
                df.loc[(df['symbol'] == symbol) & (df['date'] == date), 'label_barrier'] = barrier
                df.loc[(df['symbol'] == symbol) & (df['date'] == date), 'label_return'] = ret
                df.loc[(df['symbol'] == symbol) & (df['date'] == date), 'label_holding_days'] = holding_days
                df.loc[(df['symbol'] == symbol) & (df['date'] == date), 'event_valid'] = True
        
        logger.info("events_labeled", {
            "total_events": df['event_valid'].sum(),
            "profit_hits": (df['label_barrier'] == 'profit').sum(),
            "loss_hits": (df['label_barrier'] == 'loss').sum(),
            "time_hits": (df['label_barrier'] == 'time').sum()
        })
        
        return df
    
    def _is_valid_event(self, symbol_df: pd.DataFrame, idx: int) -> bool:
        """Check if this is a valid event trigger."""
        # Must have ATR
        if pd.isna(symbol_df.loc[idx, 'atr_14']):
            return False
        
        # Must not be suspended
        if symbol_df.loc[idx, 'can_trade'] == False:
            return False
        
        # Must have enough data for max holding period
        if idx + self.max_holding_days >= len(symbol_df):
            return False
        
        # Check if there's an overlapping event for same symbol
        # (Handled by event generation protocol - one per symbol per day max)
        
        return True
    
    def _label_single_event(
        self,
        symbol_df: pd.DataFrame,
        entry_idx: int
    ) -> Tuple[int, str, float, int]:
        """
        Label a single event.
        
        Returns:
            (label, barrier_hit, return, holding_days)
        """
        # Entry price (T+1 open)
        entry_price = symbol_df.loc[entry_idx + 1, 'adj_open']
        
        # ATR at trigger
        atr = symbol_df.loc[entry_idx, 'atr_14']
        atr = max(atr, entry_price * self.min_atr_pct)
        
        # Set barriers
        profit_barrier = entry_price * (1 + self.tp_mult * atr / entry_price)
        loss_barrier = entry_price * (1 - self.sl_mult * atr / entry_price)
        
        # Check each day in holding period
        for day in range(1, min(self.max_holding_days + 1, len(symbol_df) - entry_idx)):
            idx = entry_idx + day
            
            # Get day's high and low
            day_high = symbol_df.loc[idx, 'adj_high']
            day_low = symbol_df.loc[idx, 'adj_low']
            
            # Check profit barrier
            if day_high >= profit_barrier:
                exit_price = profit_barrier
                ret = (exit_price - entry_price) / entry_price
                return (1, 'profit', ret, day)
            
            # Check loss barrier
            if day_low <= loss_barrier:
                exit_price = loss_barrier
                ret = (exit_price - entry_price) / entry_price
                return (-1, 'loss', ret, day)
        
        # Time barrier hit
        exit_idx = min(entry_idx + self.max_holding_days, len(symbol_df) - 1)
        exit_price = symbol_df.loc[exit_idx, 'adj_close']
        ret = (exit_price - entry_price) / entry_price
        
        # Label based on return
        label = 1 if ret > 0 else 0
        
        return (label, 'time', ret, self.max_holding_days)
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of labels."""
        valid_df = df[df['event_valid'] == True]
        
        return {
            'total_events': len(valid_df),
            'positive': (valid_df['label'] == 1).sum(),
            'negative': (valid_df['label'] == 0).sum(),
            'profit_barriers': (valid_df['label_barrier'] == 'profit').sum(),
            'loss_barriers': (valid_df['label_barrier'] == 'loss').sum(),
            'time_barriers': (valid_df['label_barrier'] == 'time').sum(),
            'mean_return': valid_df['label_return'].mean(),
            'mean_holding_days': valid_df['label_holding_days'].mean()
        }
