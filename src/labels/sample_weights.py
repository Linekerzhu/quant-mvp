"""
Sample Weights Module

Implements uniqueness-based sample weighting.
Concurrent events get lower weights.

OPTIMIZED: Uses interval stabbing query algorithm for true O(n log n).
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from src.ops.event_logger import get_logger

logger = get_logger()


class IntervalTreeNode:
    """Simple interval tree node for overlap queries."""
    
    def __init__(self, center):
        self.center = center
        self.intervals = []  # Intervals that contain center
        self.left = None
        self.right = None


class SampleWeightCalculator:
    """
    Calculate sample weights based on event uniqueness.
    
    Weight = 1 / (1 + number of other symbols with overlapping events)
    
    OPTIMIZATION: True O(n log n) using interval tree + symbol-level tracking.
    
    Key insight: Within a symbol, events never overlap (by protocol).
    So we only need to count how many OTHER symbols have ANY event
    overlapping with the current event's lifetime.
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
        
        # EXT2-Q4 Fix: 使用差分数组 (Sweep-line) 方法，O(N log N)
        weights = self._calculate_weights_sweep_line(valid_df)
        
        # Assign weights back to dataframe
        # P2-C2: Use weights.index for alignment instead of valid_mask + .values
        # This handles cases where valid_df has non-contiguous index
        df.loc[weights.index, 'sample_weight'] = weights
        
        logger.info("weights_calculated", {
            "valid_events": len(valid_df),
            "mean_weight": float(weights.mean()),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max())
        })
        
        return df
    
    def _calculate_weights_interval_tree(self, valid_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weights using interval tree - TRUE O(n log n).
        
        Algorithm:
        1. Group events by symbol
        2. For each symbol, build sorted list of intervals
        3. For each event, query how many OTHER symbols have ANY event overlapping
        4. Use binary search for efficient range queries
        
        Complexity: O(n log n) where n = number of events
        """
        # 新方法：差分数组 (Sweep-line) - O(N log N)
        # 核心思路：用 event counter 在时间线上的差分数组快速计算每日并发数
        return self._calculate_weights_sweep_line(valid_df)
    
    def _calculate_weights_sweep_line(self, valid_df: pd.DataFrame) -> pd.Series:
        """
        EXT2-Q4 Fix: 使用差分数组 (Sweep-line) 方法实现 O(N log N) 样本权重计算。
        
        算法步骤：
        1. 为每个事件计算 [entry_date, exit_date) 区间
        2. 用差分数组在时间线上标记每日并发数变化
        3. 累加得到每日并发数
        4. 对每个事件，计算其持仓期间的平均并发数
        5. weight = 1 / (1 + avg_concurrent_events)
        
        复杂度：O(N log N + N * D_avg) 其中 D_avg 是平均持仓天数
        """
        import bisect
        
        weights = pd.Series(1.0, index=valid_df.index)
        
        # Step 1: 构建事件区间
        from pandas.tseries.offsets import BDay
        
        events = []
        for idx, row in valid_df.iterrows():
            symbol = row['symbol']
            trigger_date = row['date']
            entry_date = trigger_date + BDay(1)
            
            if 'label_exit_date' in row and pd.notna(row['label_exit_date']):
                exit_date = row['label_exit_date']
            else:
                holding_days = int(row['label_holding_days'])
                exit_date = trigger_date + BDay(holding_days)
            
            events.append({
                'idx': idx,
                'symbol': symbol,
                'entry': entry_date,
                'exit': exit_date
            })
        
        if not events:
            return weights
        
        # Step 2: 收集所有唯一日期并构建差分数组
        all_dates_set = set()
        for e in events:
            # 使用 date_range 获取持仓期间的所有交易日
            e_dates = pd.date_range(e['entry'], e['exit'], freq='B')
            all_dates_set.update(e_dates)
        
        all_dates = sorted(all_dates_set)
        date_to_pos = {d: i for i, d in enumerate(all_dates)}
        
        # 差分数组：+1 在 entry，-1 在 exit 之后
        diff = np.zeros(len(all_dates) + 1)
        
        # Step 3: 用差分数组标记每个事件的起止
        event_date_ranges = []
        for e in events:
            entry_pos = date_to_pos.get(e['entry'])
            exit_pos = date_to_pos.get(e['exit'])
            
            if entry_pos is None or exit_pos is None:
                event_date_ranges.append(None)
                continue
            
            # 差分标记：entry +1，exit 后 -1
            diff[entry_pos] += 1
            if exit_pos + 1 < len(diff):
                diff[exit_pos + 1] -= 1
            
            event_date_ranges.append((entry_pos, exit_pos))
        
        # Step 4: 累加得到每日并发数
        daily_concurrent = np.cumsum(diff)
        
        # Step 5: 对每个事件计算平均并发数
        for i, e in enumerate(events):
            range_info = event_date_ranges[i]
            if range_info is None:
                continue
            
            entry_pos, exit_pos = range_info
            # 计算持仓期间的日均并发数（排除 exit 当天）
            if exit_pos > entry_pos:
                concurrent_days = daily_concurrent[entry_pos:exit_pos]
                avg_concurrent = concurrent_days.mean() if len(concurrent_days) > 0 else 0
            else:
                avg_concurrent = 0
            
            # 计算权重：1 / (1 + concurrent)
            weight = 1.0 / (1.0 + avg_concurrent)
            weights.loc[e['idx']] = weight
        
        return weights
    
    def _calculate_weights_interval_tree(self, valid_df: pd.DataFrame) -> pd.Series:
        
        # Build intervals: symbol -> list of (entry, exit, event_id)
        symbol_intervals = defaultdict(list)
        all_intervals = []
        
        for idx, row in valid_df.iterrows():
            symbol = row['symbol']
            # P1 (R29-A1) FIX: Use entry_date (T+1) not trigger_date
            # Active period is [entry, exit], not [trigger, exit]
            # This was causing 12% systematic underweighting
            # Option A: BusinessDay(1) offset (simple, minor holiday error)
            # Option B: Store label_entry_date in triple_barrier (precise, future work)
            from pandas.tseries.offsets import BDay
            trigger_date = row['date']
            entry_date = trigger_date + BDay(1)
            
            # FIX A1 (R17): Use stored exit_date from triple_barrier instead of BDay
            # BDay skips weekends but NOT market holidays, causing 26% date mismatch
            if 'label_exit_date' in row and pd.notna(row['label_exit_date']):
                exit_date = row['label_exit_date']
            else:
                # P2 (R25-B2): BDay fallback - should never execute in normal flow
                # triple_barrier always sets label_exit_date for valid events
                holding_days = int(row['label_holding_days'])
                exit_date = trigger_date + BusinessDay(holding_days)
                logger.warn("bday_exit_date_fallback", {
                    "symbol": symbol,
                    "trigger_date": str(trigger_date),
                    "holding_days": holding_days,
                    "note": "label_exit_date not set - using BDay approximation"
                })
            
            interval = (entry_date, exit_date, idx, symbol)
            symbol_intervals[symbol].append(interval)
            all_intervals.append(interval)
        
        # Sort intervals within each symbol by entry date
        for symbol in symbol_intervals:
            symbol_intervals[symbol].sort(key=lambda x: x[0])
        
        # Build a timeline of all entry and exit dates
        # This allows us to use binary search for overlap detection
        
        # FIX A1: AFML uniqueness weighting - count concurrent events per day
        # For each event, calculate average 1/(concurrent_events) over its lifetime
        # This is the correct AFML algorithm (not "how many symbols overlap")
        
        # FIX B2 (R17): Use actual trading days from data instead of BDay-generated timeline
        # BDay includes market holidays (MLK Day, etc.) as "ghost dates" that inflate concurrency
        # Solution: Build timeline from actual dates in dataset
        all_dates_set = set()
        
        for idx, row in valid_df.iterrows():
            # P1 (R29-A1): entry_date = trigger + BDay(1)
            from pandas.tseries.offsets import BDay
            trigger_date = row['date']
            entry_date = trigger_date + BDay(1)
            
            # Use stored exit_date if available
            if 'label_exit_date' in row and pd.notna(row['label_exit_date']):
                exit_date = row['label_exit_date']
            else:
                holding_days = int(row['label_holding_days'])
                exit_date = trigger_date + BusinessDay(holding_days)
            
            # Collect all dates this event spans
            # P1 (R25-A1): Use freq='B' (business days) to exclude weekends
            # This prevents weekend days from inflating concurrency counts
            event_dates = pd.date_range(entry_date, exit_date, freq='B')
            for d in event_dates:
                all_dates_set.add(d)
        
        # Convert to sorted DatetimeIndex
        all_dates = pd.DatetimeIndex(sorted(all_dates_set))
        
        # For each date, count active events
        daily_event_count = pd.Series(0, index=all_dates)
        
        for idx, row in valid_df.iterrows():
            # P1 (R29-A1): entry_date = trigger + BDay(1)
            trigger_date = row['date']
            entry_date = trigger_date + BDay(1)
            
            # FIX A1 (R17): Use stored exit_date
            if 'label_exit_date' in row and pd.notna(row['label_exit_date']):
                exit_date = row['label_exit_date']
            else:
                holding_days = int(row['label_holding_days'])
                exit_date = trigger_date + BusinessDay(holding_days)
            
            # Increment count for each day this event is active
            # P1 (R25-A1): Use freq='B' to exclude weekends
            # P1 (R25-A2): Remove inclusive='left' to include exit_date (last active day)
            active_days = pd.date_range(entry_date, exit_date, freq='B')
            # Only count days that exist in our timeline (actual trading days)
            valid_active_days = active_days.intersection(all_dates)
            daily_event_count[valid_active_days] += 1
        
        # Calculate uniqueness for each event
        for idx, row in valid_df.iterrows():
            # P1 (R29-A1): entry_date = trigger + BDay(1)
            trigger_date = row['date']
            entry_date = trigger_date + BDay(1)
            
            # FIX A1 (R17): Use stored exit_date
            if 'label_exit_date' in row and pd.notna(row['label_exit_date']):
                exit_date = row['label_exit_date']
            else:
                holding_days = int(row['label_holding_days'])
                exit_date = trigger_date + BusinessDay(holding_days)
            
            # Get active days for this event
            # P1 (R25-A1): Use freq='B' to exclude weekends
            # P1 (R25-A2): Remove inclusive='left' to include exit_date
            active_days = pd.date_range(entry_date, exit_date, freq='B')
            valid_active_days = active_days.intersection(all_dates)
            
            # Uniqueness = mean(1 / concurrent_count) over event lifetime
            if len(valid_active_days) > 0:
                concurrent_counts = daily_event_count[valid_active_days]
                uniqueness = (1.0 / concurrent_counts).mean()
            else:
                uniqueness = 1.0
            
            weights.loc[idx] = uniqueness
        
        return weights
    
    # P2 (R29-B3): DEAD CODE - These alternative algorithms are never called
    # Coverage: 0%. Either delete or add tests if needed as fallback.
    # Current implementation uses interval tree approach (above).
    
    def _has_overlap_binary_search(
        self,
        sorted_intervals: List[Tuple[pd.Timestamp, pd.Timestamp, any]],
        query_entry: pd.Timestamp,
        query_exit: pd.Timestamp
    ) -> bool:
        """
        Check if ANY interval in the symbol overlaps with query interval.
        
        Uses binary search for O(log n) query time.
        
        Overlap condition: entry < query_exit AND exit > query_entry
        
        FIX B3: Use paired intervals instead of separate entry/exit lists
        to ensure entry[i] and exit[i] correspond to the same event.
        """
        import bisect
        
        # FIX B3: Extract dates from paired intervals (same ordering guaranteed)
        entry_dates = [iv[0] for iv in sorted_intervals]
        exit_dates = [iv[1] for iv in sorted_intervals]
        
        # Find the first entry that is >= query_exit
        # All entries before this could potentially overlap
        first_after_query_exit = bisect.bisect_left(entry_dates, query_exit)
        
        # Check these candidate intervals
        for i in range(first_after_query_exit):
            entry_date = entry_dates[i]
            exit_date = exit_dates[i]
            
            # Check overlap: entry < query_exit AND exit > query_entry
            if exit_date > query_entry:
                return True
        
        return False
    
    def _calculate_weights_optimized(self, valid_df: pd.DataFrame) -> pd.Series:
        """
        Fallback: Vectorized approach with better cache efficiency.
        
        Complexity: O(n × k) where k = number of symbols (500 for S&P 500)
        For n = 126K events, this is ~63M operations, acceptable in practice.
        """
        weights = pd.Series(index=valid_df.index, dtype=float)
        
        # Group by symbol for efficient lookup
        symbol_groups = {}
        for symbol in valid_df['symbol'].unique():
            symbol_groups[symbol] = valid_df[valid_df['symbol'] == symbol]
        
        for idx, row in valid_df.iterrows():
            symbol = row['symbol']
            entry_date = row['date']
            holding_days = int(row['label_holding_days'])
            exit_date = entry_date + BusinessDay(holding_days)
            
            overlapping_symbols = 0
            
            for other_symbol, other_df in symbol_groups.items():
                if other_symbol == symbol:
                    continue
                
                # Vectorized overlap check for this symbol
                overlaps = (
                    (other_df['date'] < exit_date) &
                    (other_df['date'] + other_df['label_holding_days'].apply(
                        lambda x: BusinessDay(int(x))
                    ) > entry_date)
                )
                
                if overlaps.any():
                    overlapping_symbols += 1
            
            concurrent_count = 1 + overlapping_symbols
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
