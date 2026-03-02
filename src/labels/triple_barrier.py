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
        
        Meta-Labeling Architecture (Phase C):
        ----------------------------------------
        If 'side' column exists (from Base Model):
        - Triple Barrier ONLY triggers when Base Model produces signal (side != 0)
        - label=1: Base Model signal was profitable (hit profit barrier first)
        - label=0: Base Model signal was unprofitable (hit loss barrier or time)
        - Meta Model learns "which Base Model signals are more reliable"
        
        Backward Compatibility:
        - If no 'side' column, generates labels for ALL valid events (original behavior)
        
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
        df['label_exit_date'] = pd.NaT  # FIX A1 (R17): Store actual exit date
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
                
                # FIX-2: Get side from signal
                event_side = int(symbol_df.loc[i, 'side']) if 'side' in symbol_df.columns else 1
                
                # Label this event and get actual exit
                label, barrier, ret, actual_holding_days, exit_date = self._label_single_event_with_exit(
                    symbol_df, i, trigger_date, side=event_side
                )
                
                # P1 (R27-A3): Check for invalid exit (NaN return)
                # Mark event as invalid if exit_price was NaN
                # P1 (R30-S1): Also filter invalid_atr (R29-B2 fix)
                event_is_valid = barrier not in ['time_invalid_exit', 'invalid_atr']
                
                # Register this event as active
                self._active_events[symbol] = (trigger_date, exit_date)
                
                # Store in original dataframe
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label'] = label
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_barrier'] = barrier
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_return'] = ret
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_holding_days'] = actual_holding_days
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'label_exit_date'] = exit_date  # FIX A1 (R17)
                df.loc[(df['symbol'] == symbol) & (df['date'] == trigger_date), 'event_valid'] = event_is_valid
        
        valid_count = df['event_valid'].sum()
        
        logger.info("events_labeled", {
            "total_valid_events": int(valid_count),
            "rejected_events": rejected_count,
            "overlap_rejected": overlap_rejected,
            "profit_hits": int((df['label_barrier'].isin(['profit', 'profit_gap'])).sum()),
            "loss_hits": int((df['label_barrier'].isin(['loss', 'loss_gap', 'loss_collision'])).sum()),
            "time_hits": int((df['label_barrier'] == 'time').sum()),
            "gap_events": int((df['label_barrier'].isin(['loss_gap', 'profit_gap'])).sum()),
            "collision_events": int((df['label_barrier'] == 'loss_collision').sum())
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
        
        # FIX B1 (R17): Check ATR with fallback for backup sources
        # When source_provides_adj_ohlc=False (e.g., Tiingo), ATR is NaN
        # But we can use raw OHLC as fallback for barrier calculation
        atr_value = symbol_df.loc[idx, self.atr_col] if self.atr_col in symbol_df.columns else np.nan
        
        if pd.isna(atr_value):
            # Check if we have raw OHLC for fallback ATR calculation
            has_raw_ohlc = all(col in symbol_df.columns for col in ['raw_high', 'raw_low'])
            if not has_raw_ohlc:
                return False, 'missing_atr'
            # If we have raw OHLC, allow the event (ATR will be calculated from raw data)
        
        # FIX B1: Defense-in-depth - check features_valid if available
        if 'features_valid' in symbol_df.columns and not symbol_df.loc[idx, 'features_valid']:
            return False, 'features_invalid'
        
        # =======================================================================
        # Meta-Labeling Support (Phase C)
        # =======================================================================
        # If 'side' column exists (from Base Model), only generate events
        # when Base Model produces a signal (side != 0).
        # 
        # Meta-Labeling Architecture:
        # - Triple Barrier only triggers when Base Model signals (side != 0)
        # - label=1: Base Model signal was profitable
        # - label=0: Base Model signal was unprofitable
        # - Meta Model learns "which Base Model signals are more reliable"
        # =======================================================================
        if 'side' in symbol_df.columns:
            if symbol_df.loc[idx, 'side'] == 0:
                return False, 'no_signal'  # Base Model has no signal in cold start
        
        # Must not be suspended
        # P1 (R26-A2): Use != True instead of == False for NaN-safety
        if symbol_df.loc[idx, 'can_trade'] != True:
            return False, 'suspended'
        
        # P1 (R26-A1): Check entry day (T+1) data quality
        # Trigger day features_valid doesn't guarantee entry day price is valid
        # IMPORTANT: Must mirror _label_single_event_with_exit fallback logic
        entry_idx = idx + 1
        if entry_idx >= len(symbol_df):
            return False, 'no_entry_day_data'
        
        # FIX: Use same fallback logic as _label_single_event_with_exit
        # Primary: adj_open, Fallback: adj_close (for Tiingo backup sources)
        entry_price = symbol_df.loc[entry_idx, 'adj_open']
        if pd.isna(entry_price):
            # Fallback to adj_close for backup sources (Tiingo)
            if 'adj_close' in symbol_df.columns:
                entry_price = symbol_df.loc[entry_idx, 'adj_close']
        
        if pd.isna(entry_price) or entry_price <= 0:
            return False, 'invalid_entry_price'
        
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
        trigger_idx: int,  # P2 (R27-B2): Renamed from entry_idx - this is trigger day, not entry day
        trigger_date: pd.Timestamp,
        side: int = 1  # FIX-2: +1 for long, -1 for short
    ) -> Tuple[int, str, float, int, pd.Timestamp]:
        """
        Label a single event and return actual exit date.
        
        P2 (R27-B2) FIX: Parameter renamed from entry_idx to trigger_idx.
        Entry day is trigger_idx + 1 (T+1).
        
        FIX-2: Added side parameter to handle asymmetric barriers.
        - Long (side=+1): profit barrier above, loss barrier below
        - Short (side=-1): profit barrier below, loss barrier above
        
        Returns:
            (label, barrier_hit, return, holding_days, exit_date)
        """
        # Actual entry day is T+1
        entry_idx = trigger_idx + 1
        
        # FIX B1 (R17): Handle backup sources without reliable adj OHLC
        # Check if we have adj_open or need to fall back to raw prices
        has_adj_ohlc = all(col in symbol_df.columns and pd.notna(symbol_df.loc[entry_idx, col]) 
                          for col in ['adj_open', 'adj_high', 'adj_low', 'adj_close'])
        
        # EXT2-Q3 Fix: 添加降级标记
        degraded_suffix = '' if has_adj_ohlc else '_degraded'
        
        if has_adj_ohlc:
            # Entry price (T+1 open)
            entry_price = symbol_df.loc[entry_idx, 'adj_open']
        else:
            # Fallback to raw close (Tiingo has adj_close but not adj_open)
            # Use adj_close as proxy for entry price
            entry_price = symbol_df.loc[entry_idx, 'adj_close']
        
        entry_date = symbol_df.loc[entry_idx, 'date']
        
        # ATR at trigger (trigger_idx is the original trigger day)
        atr = symbol_df.loc[trigger_idx, self.atr_col] if self.atr_col in symbol_df.columns else np.nan
        
        # FIX B1 (R17): Fallback ATR from raw OHLC if primary ATR unavailable
        if pd.isna(atr):
            if all(col in symbol_df.columns for col in ['raw_high', 'raw_low']):
                # Use raw high-low range as ATR approximation
                # This is less accurate but allows Tiingo failover to generate events
                # Calculate rolling mean of (high - low) for the ATR window
                start_idx = max(0, trigger_idx - self.atr_window + 1)
                atr = (symbol_df.loc[start_idx:trigger_idx, 'raw_high'] - 
                       symbol_df.loc[start_idx:trigger_idx, 'raw_low']).mean()
        
        atr = max(atr, entry_price * self.min_atr_pct)
        
        # P2 (R29-B2): Final ATR NaN check - defensive guard
        # If ATR is still NaN after all fallbacks, cannot compute barriers
        # features_valid should catch this upstream, but defensive programming
        if pd.isna(atr):
            return (0, 'invalid_atr', np.nan, 0, trigger_date)
        
        # FIX-2: Set barriers based on side
        # Long (side=+1): profit above, loss below
        # Short (side=-1): profit below, loss above
        if side >= 0:  # Long or neutral
            profit_barrier = entry_price * (1 + self.tp_mult * atr / entry_price)
            loss_barrier = entry_price * (1 - self.sl_mult * atr / entry_price)
        else:  # Short
            profit_barrier = entry_price * (1 - self.tp_mult * atr / entry_price)  # Below entry
            loss_barrier = entry_price * (1 + self.sl_mult * atr / entry_price)   # Above entry
        
        # Normalize: upper_barrier always above, lower_barrier always below
        upper_barrier = max(profit_barrier, loss_barrier)
        lower_barrier = min(profit_barrier, loss_barrier)
        # Track which barrier is profit for label semantics
        upper_is_profit = (side >= 0)  # Long: upper=profit, Short: upper=loss
        
        # Check each day in holding period
        # R33-A1: Start from day=0 (entry day T+1) instead of day=1
        # Entry day barrier check is critical for high-volatility scenarios
        max_day = min(self.max_holding_days + 1, len(symbol_df) - entry_idx)
        
        for day in range(0, max_day):  # R33-A1: Changed from range(1, max_day)
            idx = entry_idx + day
            
            # FIX B1 (R17): Use adj OHLC if available, else raw OHLC
            # P1 (R27-A2): Tiingo backup path - use adj_close for barrier comparison
            # When entry_price is adj_close, barriers must also use adj_close
            # Raw prices after split would trigger false barriers
            if has_adj_ohlc:
                day_high = symbol_df.loc[idx, 'adj_high']
                day_low = symbol_df.loc[idx, 'adj_low']
            else:
                # P1 (R27-A2): Use adj_close for barrier check (consistent with entry_price)
                # Fallback to raw only if adj_close unavailable
                day_high = symbol_df.loc[idx, 'adj_close']
                day_low = symbol_df.loc[idx, 'adj_close']
            
            # P2 (R27-B3): Skip day if price data is NaN (yfinance data quality issue)
            # NaN comparison always returns False, so barrier wouldn't trigger
            # This could cause wrong labels if day_high/day_low is NaN
            if pd.isna(day_high) or pd.isna(day_low):
                continue  # Skip this day, continue to next day in holding period
            
            exit_date = symbol_df.loc[idx, 'date']
            
            # ============================================================
            # OR5 HOTFIX: Maximum Pessimism Principle (HF-1)
            # ============================================================
            # Execution priority: Gap > Collision > Normal
            # Pessimism: loss > profit in all ambiguous cases
            # ============================================================
            
            # Get day open for gap detection
            if has_adj_ohlc:
                day_open = symbol_df.loc[idx, 'adj_open']
            else:
                day_open = symbol_df.loc[idx, 'adj_close']
            
            # R33-A1: Skip gap detection on day=0 (entry day)
            # On entry day, open IS the entry_price, so gap is physically impossible
            # Gap detection only applies to day >= 1
            if day >= 1:
                # Skip if day_open is NaN (can't determine gap)
                if pd.isna(day_open):
                    continue
                
                # FIX-2: Use normalized upper/lower barriers
                # label = price direction: +1 if price went UP, -1 if price went DOWN
                # STEP 1: Gap Execution (跳空执行 - 最优先)
                if day_open <= lower_barrier:
                    # Price跌破lower barrier → 价格跌了
                    exit_price = day_open
                    ret = side * np.log(exit_price / entry_price)  # FIX-2: Add side
                    label = -1  # Price went DOWN
                    barrier_type = 'loss_gap' if upper_is_profit else 'profit_gap'
                    return (label, barrier_type + degraded_suffix, ret, day, exit_date)
                
                if day_open >= upper_barrier:
                    # Price涨破upper barrier → 价格涨了
                    exit_price = day_open
                    ret = side * np.log(exit_price / entry_price)  # FIX-2: Add side
                    label = 1  # Price went UP
                    barrier_type = 'profit_gap' if upper_is_profit else 'loss_gap'
                    return (label, barrier_type + degraded_suffix, ret, day, exit_date)
            
            # STEP 2: Collision Detection (同日双穿检测)
            # R33-A1: Also applies to day=0
            # If both barriers hit same day, force loss (pessimism)
            
            if day_high >= upper_barrier and day_low <= lower_barrier:
                # 同日双穿 → 悲观原则用loss barrier退出
                # 但label是价格方向
                if upper_is_profit:  # 做多: loss在下方的lower_barrier
                    exit_price = lower_barrier
                    label = -1  # 价格跌了
                else:             # 做空: loss在上方的upper_barrier
                    exit_price = upper_barrier
                    label = 1   # 价格涨了
                ret = side * np.log(exit_price / entry_price)
                return (label, 'loss_collision' + degraded_suffix, ret, day, exit_date)
            
            # STEP 3: Normal Path (正常路径)
            # R33-A1: Also applies to day=0
            # Loss check before profit check (pessimism)
            
            if day_low <= lower_barrier:
                # 价格跌到下方barrier
                exit_price = lower_barrier
                ret = side * np.log(exit_price / entry_price)  # FIX-2: Add side
                label = -1  # Price went DOWN
                barrier_type = 'loss' if upper_is_profit else 'profit'
                return (label, barrier_type + degraded_suffix, ret, day, exit_date)
            
            if day_high >= upper_barrier:
                # 价格涨到上方barrier
                exit_price = upper_barrier
                ret = side * np.log(exit_price / entry_price)  # FIX-2: Add side
                label = 1  # Price went UP
                barrier_type = 'profit' if upper_is_profit else 'loss'
                return (label, barrier_type + degraded_suffix, ret, day, exit_date)
        
        # Time barrier hit
        exit_idx = min(entry_idx + self.max_holding_days, len(symbol_df) - 1)
        exit_date = symbol_df.loc[exit_idx, 'date']
        exit_price = symbol_df.loc[exit_idx, 'adj_close']
        
        # P1 (R27-A3): Check exit_price validity
        # NaN exit_price would produce NaN return → pollutes training data
        if pd.isna(exit_price) or exit_price <= 0:
            # Cannot compute valid return - mark as invalid event
            # This will be caught by label_events and set event_valid=False
            return (0, 'time_invalid_exit' + degraded_suffix, np.nan, self.max_holding_days, exit_date)
        
        ret = side * np.log(exit_price / entry_price)  # FIX-2: Add side
        
        # P0-A2: AFML Ch3.4 - time barrier = no clear outcome → neutral label
        # 35% of events hit time barrier, 38% with |return| < 1%
        # Using sign of return would conflate +0.0003 with +0.043
        label = 0
        
        return (label, 'time' + degraded_suffix, ret, self.max_holding_days, exit_date)
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Get distribution of labels.
        
        P0-A2: Updated label semantics (AFML Ch3.4):
        - 1: Profit barrier hit
        - -1: Loss barrier hit
        - 0: Time barrier hit (neutral - no clear outcome)
        
        Note: label_return still contains actual return for analysis
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
        
        # Count by barrier type (OR5: added gap/collision types)
        profit_barriers = (valid_df['label_barrier'] == 'profit').sum()
        loss_barriers = (valid_df['label_barrier'] == 'loss').sum()
        profit_gap = (valid_df['label_barrier'] == 'profit_gap').sum()
        loss_gap = (valid_df['label_barrier'] == 'loss_gap').sum()
        loss_collision = (valid_df['label_barrier'] == 'loss_collision').sum()
        time_barriers = (valid_df['label_barrier'] == 'time').sum()
        
        # Count by unified label
        profit_label = (valid_df['label'] == 1).sum()
        loss_label = (valid_df['label'] == -1).sum()
        neutral_label = (valid_df['label'] == 0).sum()
        
        return {
            'total_events': len(valid_df),
            'by_barrier': {
                'profit': int(profit_barriers),
                'profit_gap': int(profit_gap),
                'loss': int(loss_barriers),
                'loss_gap': int(loss_gap),
                'loss_collision': int(loss_collision),
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
