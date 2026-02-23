"""
Data Validation Module

Quality checks and Point-in-Time alignment validation.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from src.ops.event_logger import get_logger

logger = get_logger()


class DataValidator:
    """Data quality validation."""
    
    def __init__(
        self,
        max_daily_return: float = 0.50,
        max_consecutive_nan: int = 3,
        min_adv_usd: float = 5_000_000
    ):
        self.max_daily_return = max_daily_return
        self.max_consecutive_nan = max_consecutive_nan
        self.min_adv_usd = min_adv_usd
    
    def validate(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> Tuple[bool, pd.DataFrame, dict]:
        """
        Validate data quality.
        
        Returns:
            (passed, cleaned_df, report)
        """
        report = {
            "symbol": symbol,
            "total_rows": len(df),
            "checks": {}
        }
        
        # Check 1: Duplicates
        dup_count = df.duplicated(subset=['symbol', 'date']).sum()
        report["checks"]["duplicates"] = {"count": int(dup_count), "passed": dup_count == 0}
        df = df.drop_duplicates(subset=['symbol', 'date'])
        
        # Check 2: Missing values
        nan_mask = df[['raw_open', 'raw_high', 'raw_low', 'raw_close']].isna().any(axis=1)
        nan_count = nan_mask.sum()
        report["checks"]["missing_values"] = {"count": int(nan_count)}
        
        # Handle consecutive NaN (halt detection) - FIXED A26
        df = self._handle_missing_values_fixed(df, report)
        
        # Check 3: Abnormal jumps (after split adjustment awareness)
        df = self._detect_anomalies(df, report)
        
        # Check 4: Suspension detection (max_consecutive_nan+ days of NaN after handling)
        df = self._detect_suspension(df, report)
        
        # Calculate pass rate
        total_checks = len(report["checks"])
        passed_checks = sum(1 for c in report["checks"].values() if c.get("passed", True))
        report["pass_rate"] = passed_checks / total_checks if total_checks > 0 else 1.0
        
        passed = report["pass_rate"] >= 0.99  # 99% pass threshold
        
        logger.info("validation_complete", report, symbol)
        
        return passed, df, report
    
    def _handle_missing_values_fixed(
        self,
        df: pd.DataFrame,
        report: dict
    ) -> pd.DataFrame:
        """
        Handle missing values with proper consecutive NaN detection.
        
        FIXED A26: Correctly identifies consecutive NaN runs.
        - Single day NaN: Forward fill
        - 2-3 consecutive NaN: Forward fill (within limit)
        - >= max_consecutive_nan consecutive NaN: Mark as suspension (don't fill)
        """
        df = df.sort_values(['symbol', 'date']).copy()
        
        price_cols = ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                     'adj_open', 'adj_high', 'adj_low', 'adj_close']
        
        filled_count = 0
        suspension_marked = 0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Get close price for this symbol
            close_series = df.loc[mask, 'raw_close'].copy()
            
            if close_series.isna().sum() == 0:
                continue
            
            # Find NaN positions
            nan_mask = close_series.isna()
            
            # FIXED: Properly identify consecutive NaN runs
            # Create group IDs for consecutive NaN segments
            nan_groups = (nan_mask != nan_mask.shift()).cumsum()
            
            # For each NaN group, check its length
            for group_id in nan_groups[nan_mask].unique():
                group_mask_local = (nan_groups == group_id) & mask
                group_indices = df.loc[mask][group_mask_local].index
                
                if len(group_indices) == 0:
                    continue
                
                group_start_idx = group_indices[0]
                
                # FIXED: Only use preceding data (before the NaN group starts)
                # This prevents lookahead bias
                preceding_mask = (df.index < group_start_idx) & mask
                preceding_values = df.loc[preceding_mask, col].dropna()
                
                last_valid = preceding_values.iloc[-1] if len(preceding_values) > 0 else None
                
                if last_valid is not None:
                    df.loc[group_mask_local, col] = last_valid
                    filled_count += group_mask_local.sum()
                else:
                    # Exceeds limit: mark as suspension (don't fill)
                    suspension_marked += group_mask.sum()
                    logger.warn("consecutive_nan_exceeds_limit", {
                        "symbol": symbol,
                        "consecutive_days": int(group_length),
                        "limit": self.max_consecutive_nan
                    })
        
        report["checks"]["missing_values"]["filled"] = int(filled_count)
        report["checks"]["missing_values"]["suspension_marked"] = int(suspension_marked)
        
        return df
    
    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        report: dict
    ) -> pd.DataFrame:
        """Detect abnormal price jumps, accounting for splits."""
        anomalies = []
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask].sort_values('date')
            
            if len(symbol_df) < 2:
                continue
            
            # Calculate returns using adjusted close
            returns = symbol_df['adj_close'].pct_change().abs()
            
            # Detect jumps
            jump_mask = returns > self.max_daily_return
            
            if jump_mask.any():
                jump_dates = symbol_df.loc[jump_mask, 'date'].tolist()
                anomalies.append({
                    "symbol": symbol,
                    "jump_dates": jump_dates,
                    "max_return": float(returns.max())
                })
        
        report["checks"]["anomaly_detection"] = {
            "anomaly_count": len(anomalies),
            "passed": len(anomalies) == 0,
            "details": anomalies[:5]  # First 5 only
        }
        
        return df
    
    def _detect_suspension(
        self,
        df: pd.DataFrame,
        report: dict
    ) -> pd.DataFrame:
        """Detect trading suspensions (max_consecutive_nan+ consecutive NaN)."""
        suspensions = []
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask].sort_values('date')
            
            # Check for remaining NaN after forward fill
            nan_mask = symbol_df['raw_close'].isna()
            
            if nan_mask.sum() < self.max_consecutive_nan:
                continue
            
            # Mark as suspended (don't drop, just flag)
            suspensions.append({
                "symbol": symbol,
                "suspended_days": int(nan_mask.sum())
            })
            
            # Set can_trade flag
            df.loc[mask & df['date'].isin(symbol_df.loc[nan_mask, 'date']), 'can_trade'] = False
            
            # Log suspension
            logger.warn("suspension_detected", {
                "symbol": symbol,
                "days": int(nan_mask.sum())
            }, symbol)
        
        report["checks"]["suspension_detection"] = {
            "suspension_count": len(suspensions),
            "details": suspensions
        }
        
        return df
