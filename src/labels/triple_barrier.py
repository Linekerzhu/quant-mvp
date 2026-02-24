"""
Triple Barrier Labeling Module

Implements Triple Barrier method for labeling events.
Follows the event generation protocol from config/event_protocol.yaml.

CRITICAL: Enforces non-overlapping event constraint per symbol.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
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
    
    EVENT PROTOCOL §6.5: Non-overlapping constraint
    - Only ONE active event per symbol at any time
    - New events can only trigger AFTER the previous event exits
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
        
        # FIX A1: Dynamic ATR column name based on config
        self.atr_col = f'atr_{self.atr_window}'
        
        # Track active events per symbol: symbol -> (entry_date, exit_date)
        self._active_events: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    
    def label_events(
        self,
        df: pd.DataFrame,
        trigger_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Generate Triple Barrier labels for all valid events.
        
        Enforces non-overlapping constraint: only one active event per symbol.
        
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
        
        # Reset active events tracking
        self._active_events = {}
        
        # Track rejected events for logging
        rejected_count = 0
        overlap_rejected = 0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask].sort_values('date').reset_index(drop=True)
            
            if len(symbol_df) < self.max_holding_days + 1:
                continue
            
            # Generate labels for each day (if valid)
            for i in range(len(symbol_df) - 1):
                trigger_date = symbol_df.loc[i, 'date']
                
                # Check if this is a valid event trigger
                is_valid, rejection_reason = self._is_valid_event(symbol_df, i, trigger_date)
                
                if not is_valid:
                    rejected_count += 1
                    if rejection_reason == 'overlap':
                        overlap_rejected += 1
                    continue
                
                # Label this event and get actual exit
                label, barrier, ret, actual_holding_days, exit_date = self._label_single_event_with_exit(
                    symbol_df, i, trigger_date
                )
                
                # Register this event as active
                self._active_events[symbol] = (trigger_date, exit_date)
                
                # Store in original dataframe
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label'] = label
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_barrier'] = barrier
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_return'] = ret
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_holding_days'] = actual_holding_days
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'event_valid'] = True
        
        valid_count = df['event_valid'].sum()
        
        logger.info("events_labeled", {
            "total_valid_events": int(valid_count),
            "rejected_events": rejected_count,
            "overlap_rejected": overlap_rejected,
            "profit_hits": (df['label_barrier'] == 'profit').sum(),
            "loss_hits": (df['label_barrier'] == 'loss').sum(),
            "time_hits": (df['label_barrier'] == 'time').sum()
        })
        
        return df
    
    def _is_valid_event(
        self,
        symbol_df: pd.DataFrame,
        idx: int,
        trigger_date: pd.Timestamp
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this is a valid event trigger.
        
        Returns:
            (is_valid, rejection_reason)
        """
        symbol = symbol_df.loc[idx, 'symbol']
        
        # Must have ATR
        if pd.isna(symbol_df.loc[idx, self.atr_col]):
            return False, 'missing_atr'
        
        # FIX B1: Defense-in-depth - check features_valid if available
        if 'features_valid' in symbol_df.columns and not symbol_df.loc[idx, 'features_valid']:
            return False, 'features_invalid'
        
        # Must not be suspended
        if symbol_df.loc[idx, 'can_trade'] == False:
            return False, 'suspended'
        
        # Must have enough data for max holding period
        if idx + self.max_holding_days >= len(symbol_df):
            return False, 'insufficient_data'
        
        # NON-OVERLAPPING CONSTRAINT (§6.5 Event Protocol)
        # Check if there's an active event for this symbol
        if symbol in self._active_events:
            _, active_exit_date = self._active_events[symbol]
            # New event can only start AFTER the active event exits
            if trigger_date <= active_exit_date:
                return False, 'overlap'
        
        return True, None
    
    def _label_single_event_with_exit(
        self,
        symbol_df: pd.DataFrame,
        entry_idx: int,
        trigger_date: pd.Timestamp
    ) -> Tuple[int, str, float, int, pd.Timestamp]:
        """
        Label a single event and return actual exit date.
        
        Returns:
            (label, barrier_hit, return, holding_days, exit_date)
        """
        # Entry price (T+1 open)
        entry_price = symbol_df.loc[entry_idx + 1, 'adj_open']
        entry_date = symbol_df.loc[entry_idx + 1, 'date']
        
        # ATR at trigger
        atr = symbol_df.loc[entry_idx, self.atr_col]
        atr = max(atr, entry_price * self.min_atr_pct)
        
        # Set barriers
        profit_barrier = entry_price * (1 + self.tp_mult * atr / entry_price)
        loss_barrier = entry_price * (1 - self.sl_mult * atr / entry_price)
        
        # Check each day in holding period
        max_day = min(self.max_holding_days + 1, len(symbol_df) - entry_idx)
        
        for day in range(1, max_day):
            idx = entry_idx + day
            
            # Get day's high and low
            day_high = symbol_df.loc[idx, 'adj_high']
            day_low = symbol_df.loc[idx, 'adj_low']
            exit_date = symbol_df.loc[idx, 'date']
            
            # Check profit barrier
            if day_high >= profit_barrier:
                exit_price = profit_barrier
                ret = np.log(exit_price / entry_price)  # B24: Log return
                # Unified label semantics: 1=profit, -1=loss, 0=time
                return (1, 'profit', ret, day, exit_date)
            
            # Check loss barrier
            if day_low <= loss_barrier:
                exit_price = loss_barrier
                ret = np.log(exit_price / entry_price)  # B24: Log return
                return (-1, 'loss', ret, day, exit_date)
        
        # Time barrier hit
        exit_idx = min(entry_idx + self.max_holding_days, len(symbol_df) - 1)
        exit_date = symbol_df.loc[exit_idx, 'date']
        exit_price = symbol_df.loc[exit_idx, 'adj_close']
        ret = np.log(exit_price / entry_price)  # B24: Log return
        
        # Time barrier label: use sign of return
        label = 1 if ret > 0 else -1 if ret < 0 else 0
        
        return (label, 'time', ret, self.max_holding_days, exit_date)
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Get distribution of labels.
        
        Unified label semantics:
        - 1: Profit barrier hit OR time barrier with positive return
        - -1: Loss barrier hit OR time barrier with negative return  
        - 0: Time barrier with zero return (rare)
        """
        valid_df = df[df['event_valid'] == True]
        
        if len(valid_df) == 0:
            return {
                'total_events': 0,
                'profit': 0,
                'loss': 0,
                'time_neutral': 0,
                'mean_return': 0.0
            }
        
        # Count by barrier type
        profit_barriers = (valid_df['label_barrier'] == 'profit').sum()
        loss_barriers = (valid_df['label_barrier'] == 'loss').sum()
        time_barriers = (valid_df['label_barrier'] == 'time').sum()
        
        # Count by unified label
        profit_label = (valid_df['label'] == 1).sum()
        loss_label = (valid_df['label'] == -1).sum()
        neutral_label = (valid_df['label'] == 0).sum()
        
        return {
            'total_events': len(valid_df),
            'by_barrier': {
                'profit': int(profit_barriers),
                'loss': int(loss_barriers),
                'time': int(time_barriers)
            },
            'by_label': {
                'profit': int(profit_label),
                'loss': int(loss_label),
                'neutral': int(neutral_label)
            },
            'mean_return': float(valid_df['label_return'].mean()),
            'mean_holding_days': float(valid_df['label_holding_days'].mean())
        }
    
    def check_class_imbalance(self, df: pd.DataFrame) -> Dict:
        """
        Check for class imbalance and recommend class weights (B20).
        
        Returns:
            Dict with class counts, ratios, and recommended class weights
        """
        valid_df = df[df['event_valid'] == True]
        
        if len(valid_df) == 0:
            return {'error': 'No valid events'}
        
        # Count by label
        profit = (valid_df['label'] == 1).sum()
        loss = (valid_df['label'] == -1).sum()
        neutral = (valid_df['label'] == 0).sum()
        total = profit + loss + neutral
        
        if total == 0:
            return {'error': 'No labeled events'}
        
        # Calculate ratios
        profit_ratio = profit / total
        loss_ratio = loss / total
        neutral_ratio = neutral / total
        
        # Check imbalance severity
        max_ratio = max(profit_ratio, loss_ratio, neutral_ratio)
        min_ratio = min(profit_ratio, loss_ratio, neutral_ratio)
        imbalance_ratio = max_ratio / max(min_ratio, 0.01)
        
        # Recommend class weights (inverse frequency)
        class_weights = {
            1: round(total / max(profit, 1), 2),
            -1: round(total / max(loss, 1), 2),
            0: round(total / max(neutral, 1), 2) if neutral > 0 else 0
        }
        
        result = {
            'class_counts': {
                'profit': int(profit),
                'loss': int(loss),
                'neutral': int(neutral),
                'total': int(total)
            },
            'class_ratios': {
                'profit': round(profit_ratio, 4),
                'loss': round(loss_ratio, 4),
                'neutral': round(neutral_ratio, 4)
            },
            'imbalance_severity': round(imbalance_ratio, 2),
            'is_imbalanced': imbalance_ratio > 2.0,
            'recommended_class_weights': class_weights
        }
        
        # Log warning if severely imbalanced
        if result['is_imbalanced']:
            logger.warn("class_imbalance_detected", {
                "profit_ratio": profit_ratio,
                "loss_ratio": loss_ratio,
                "imbalance_ratio": imbalance_ratio,
                "recommended_weights": class_weights
            })
        
        return result
