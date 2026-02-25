"""
Corporate Actions Module

Handles splits, dividends, delistings, and suspensions.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.ops.event_logger import get_logger

logger = get_logger()


class CorporateActionsHandler:
    """Handles corporate actions and their effects on price data."""
    
    def __init__(
        self,
        split_detection_threshold: float = 0.50,
        adj_change_threshold: float = 0.05
    ):
        self.split_threshold = split_detection_threshold
        self.adj_threshold = adj_change_threshold
    
    def detect_splits(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect stock splits by comparing raw vs adjusted price changes.
        
        A split is detected when:
        - Raw close changes > 50% (up or down)
        - Adjusted close changes < 5%
        
        Returns:
            DataFrame with 'detected_split' column added
        """
        df = df.copy()
        df['detected_split'] = False
        df['split_ratio'] = 1.0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            # CRITICAL FIX A1: Reset index to ensure positional indexing works correctly
            symbol_df = df.loc[mask].sort_values('date').reset_index(drop=True)
            
            if len(symbol_df) < 2:
                continue
            
            # Calculate returns
            # P0-A1 (R20): ffill before pct_change to handle suspension gaps
            # pct_change(fill_method=None) returns NaN after gaps, missing splits on
            # first trading day after suspension. ffill preserves detection semantics.
            raw_returns = symbol_df['raw_close'].ffill().pct_change(fill_method=None).abs()
            adj_returns = symbol_df['adj_close'].ffill().pct_change(fill_method=None).abs()
            
            # Detect splits
            # FIX B3: Use >= to catch exact 2:1 splits (|return| = 0.50)
            split_mask = (
                (raw_returns >= self.split_threshold) &
                (adj_returns < self.adj_threshold)
            )
            
            if split_mask.any():
                split_dates = symbol_df.loc[split_mask, 'date'].tolist()
                
                for split_date in split_dates:
                    # Calculate split ratio
                    idx = symbol_df[symbol_df['date'] == split_date].index[0]
                    if idx > 0:
                        # P1 (R25): Use ffilled raw_close to handle gap scenarios
                        # If previous row is NaN (gap), ffill provides last valid value
                        raw_ffilled = symbol_df['raw_close'].ffill()
                        prev_raw = raw_ffilled.iloc[idx - 1]
                        curr_raw = symbol_df.iloc[idx]['raw_close']
                        
                        if curr_raw != 0 and pd.notna(prev_raw):
                            ratio = prev_raw / curr_raw
                            
                            df.loc[
                                (df['symbol'] == symbol) & (df['date'] == split_date),
                                ['detected_split', 'split_ratio']
                            ] = [True, ratio]
                            
                            logger.info("split_detected", {
                                "symbol": symbol,
                                "date": str(split_date),
                                "ratio": ratio
                            }, symbol)
        
        return df
    
    def process_dividends(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process dividends (already in adjusted prices).
        
        For MVP: Dividends are handled by using adjusted prices.
        No separate processing needed.
        """
        # Dividends are implicitly handled by adj_close
        return df
    
    def detect_delistings(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect delisted symbols (no data after certain date).
        
        Returns:
            (cleaned_df, delisting_info)
        """
        delist_info = {}
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            # OR4-P0-1 (R25): Sort by date to ensure last_valid_index works correctly
            symbol_df = df.loc[mask].sort_values('date')
            
            # Check for trailing NaN (delisting)
            last_valid = symbol_df['raw_close'].last_valid_index()
            
            if last_valid is not None:
                last_date = symbol_df.loc[last_valid, 'date']
                last_price = symbol_df.loc[last_valid, 'raw_close']
                
                # Check if data ends before max date
                # P1-B3 (R21): Use trading days instead of calendar days
                # 3 trading days across weekend = 5 calendar days, causing false positives
                max_date = df['date'].max()
                
                if last_date < max_date - pd.offsets.BDay(5):
                    delist_info[symbol] = {
                        'last_date': last_date.strftime('%Y-%m-%d'),
                        'last_price': float(last_price)
                    }
                    
                    logger.info("delisting_detected", {
                        "symbol": symbol,
                        "last_date": last_date.strftime('%Y-%m-%d'),
                        "last_price": float(last_price)
                    }, symbol)
        
        return df, delist_info
    
    def handle_delisting_exit(
        self,
        positions: Dict[str, float],
        delist_info: Dict
    ) -> List[Dict]:
        """
        Generate exit orders for delisted positions.
        
        Returns:
            List of exit orders
        """
        orders = []
        
        for symbol in positions:
            if symbol in delist_info:
                orders.append({
                    'symbol': symbol,
                    'action': 'exit',
                    'reason': 'delisting',
                    'exit_price': delist_info[symbol]['last_price'],
                    'exit_date': delist_info[symbol]['last_date'],
                    'order_type': 'market'  # Force exit at last available price
                })
                
                logger.info("delisting_exit_order", {
                    "symbol": symbol,
                    "exit_price": delist_info[symbol]['last_price']
                }, symbol)
        
        return orders
    
    def detect_suspensions(
        self,
        df: pd.DataFrame,
        min_consecutive: int = 5,
        resume_min_days: int = 5
    ) -> pd.DataFrame:
        """
        Detect and mark trading suspensions.
        
        Suspension = 5+ consecutive days of NaN
        Resume = Need 5+ days of valid data after suspension
        """
        df = df.copy()
        df['is_suspended'] = False
        df['suspension_start'] = None
        df['can_trade'] = True
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            # CRITICAL FIX A2: Reset index to prevent cross-symbol contamination
            symbol_df = df.loc[mask].sort_values('date').reset_index(drop=True)
            
            # Find NaN runs
            is_nan = symbol_df['raw_close'].isna()
            
            # Group consecutive NaN
            nan_groups = []
            in_nan = False
            start_idx = None
            
            for idx, nan in zip(symbol_df.index, is_nan):
                if nan and not in_nan:
                    in_nan = True
                    start_idx = idx
                elif not nan and in_nan:
                    in_nan = False
                    nan_groups.append((start_idx, idx))
            
            if in_nan:
                nan_groups.append((start_idx, None))
            
            # Mark suspensions (5+ consecutive days)
            for start, end in nan_groups:
                if end is None:
                    end_idx = symbol_df.index[-1]
                else:
                    # P0 (R25): Fix off-by-one - 'end' is first non-NaN row index
                    # loc[start:end] is inclusive, so we need end-1 to exclude it
                    end_idx = end - 1
                
                days = symbol_df.loc[start:end_idx].shape[0]
                
                if days >= min_consecutive:
                    # CRITICAL FIX A2: Get actual indices from symbol_df, then map back to original df
                    suspension_mask = symbol_df.index.isin(symbol_df.loc[start:end_idx].index)
                    suspension_indices = symbol_df.loc[suspension_mask].index
                    
                    # Map back to original df using symbol + date for safety
                    suspension_dates = symbol_df.loc[suspension_indices, 'date'].tolist()
                    original_mask = (df['symbol'] == symbol) & (df['date'].isin(suspension_dates))
                    
                    df.loc[original_mask, 'is_suspended'] = True
                    df.loc[original_mask, 'suspension_start'] = symbol_df.loc[start, 'date']
                    df.loc[original_mask, 'can_trade'] = False
                    
                    logger.warn("suspension_marked", {
                        "symbol": symbol,
                        "start": symbol_df.loc[start, 'date'].strftime('%Y-%m-%d'),
                        "days": days
                    }, symbol)
            
            # Mark post-suspension cold start period
            for start, end in nan_groups:
                if end is None:
                    continue
                
                # P0 (R25): Fix off-by-one - same as suspension marking above
                end_idx = end - 1
                days = symbol_df.loc[start:end_idx].shape[0]
                
                if days >= min_consecutive:
                    # Find resume point
                    resume_idx = symbol_df.index.get_loc(end)
                    post_suspension = symbol_df.iloc[resume_idx:resume_idx + resume_min_days]
                    
                    if len(post_suspension) > 0:
                        # CRITICAL FIX A2: Map back to original df using symbol + date
                        cold_start_dates = post_suspension['date'].tolist()
                        original_cold_mask = (df['symbol'] == symbol) & (df['date'].isin(cold_start_dates))
                        df.loc[original_cold_mask, 'can_trade'] = False
        
        return df
    
    def apply_all(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply all corporate action processing.
        
        Returns:
            (processed_df, info_dict)
        """
        info = {
            'splits_detected': 0,
            'delistings_detected': 0,
            'suspensions_detected': 0
        }
        
        # Detect splits
        df = self.detect_splits(df)
        info['splits_detected'] = df['detected_split'].sum()
        
        # Detect delistings
        df, delist_info = self.detect_delistings(df)
        info['delistings_detected'] = len(delist_info)
        info['delistings'] = delist_info
        
        # Detect suspensions
        df = self.detect_suspensions(df)
        info['suspensions_detected'] = df['is_suspended'].sum()
        
        return df, info
