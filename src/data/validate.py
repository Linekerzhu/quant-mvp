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
        
        # Handle consecutive NaN (halt detection) - FIXED ALL BUGS
        df = self._handle_missing_values_fixed(df, report)
        
        # Check 3: Abnormal jumps (after split adjustment awareness)
        df = self._detect_anomalies(df, report)
        
        # FIX B3: Check price positivity and OHLC consistency
        df = self._check_price_integrity(df, report)
        
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
        
        FIXED ALL BUGS:
        1. Restored for col in price_cols loop
        2. Restored group_length >= max_consecutive_nan check
        3. Fixed lookahead bias (only use preceding data)
        4. Fixed variable references in else branch
        
        Logic:
        - Single day NaN: Forward fill
        - 1-2 consecutive NaN: Forward fill (within limit)
        - >= max_consecutive_nan consecutive NaN: Mark as suspension (don't fill)
        """
        df = df.sort_values(['symbol', 'date']).copy()
        df = df.reset_index(drop=True)  # P2-C4: Ensure index aligns with temporal order
        
        price_cols = ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                     'adj_open', 'adj_high', 'adj_low', 'adj_close']
        
        filled_count = 0
        suspension_marked = 0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            close_series = df.loc[mask, 'raw_close'].copy()
            
            if close_series.isna().sum() == 0:
                continue
            
            nan_mask = close_series.isna()
            nan_groups = (nan_mask != nan_mask.shift()).cumsum()
            
            for group_id in nan_groups[nan_mask].unique():
                # Get boolean mask for this group within symbol
                group_in_symbol = nan_groups == group_id
                group_length = group_in_symbol.sum()
                
                # Get global indices for this symbol
                symbol_indices = df.loc[mask].index
                
                # Build global mask for this NaN group
                group_global_mask = pd.Series(False, index=df.index)
                group_global_mask.loc[symbol_indices[group_in_symbol]] = True
                
                if group_length >= self.max_consecutive_nan:
                    # >= threshold: Mark as suspension, DO NOT fill
                    suspension_marked += group_length
                    logger.warn("consecutive_nan_exceeds_limit", {
                        "symbol": symbol,
                        "consecutive_days": int(group_length),
                        "limit": self.max_consecutive_nan
                    })
                    continue
                
                # < threshold: Forward fill using preceding data (NO lookahead bias)
                group_start = symbol_indices[group_in_symbol][0]
                preceding = df.loc[(df.index < group_start) & mask]
                
                filled_any = False
                for col in price_cols:
                    if col not in df.columns:
                        continue
                    valid_preceding = preceding[col].dropna()
                    if len(valid_preceding) > 0:
                        df.loc[group_global_mask, col] = valid_preceding.iloc[-1]
                        filled_any = True
                
                # Count only once per row (not once per column)
                if filled_any:
                    filled_count += group_global_mask.sum()
        
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
            # P1-B1 (R20): ffill before pct_change to detect jumps after NaN gaps
            # pct_change(fill_method=None) returns NaN after gaps, missing extreme values
            returns = symbol_df['adj_close'].ffill().pct_change(fill_method=None).abs()
            
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

    def _check_price_integrity(
        self,
        df: pd.DataFrame,
        report: dict
    ) -> pd.DataFrame:
        """
        FIX B3: Check price positivity and OHLC consistency.
        
        Validates:
        - All prices > 0 (negative prices indicate data corruption)
        - High >= max(Open, Close) (High should be highest)
        - Low <= min(Open, Close) (Low should be lowest)
        - Volume >= 0
        """
        price_cols = ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                     'adj_open', 'adj_high', 'adj_low', 'adj_close']
        
        violations = []
        
        # Check 1: Price positivity
        for col in price_cols:
            negative_mask = df[col] <= 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                violations.append({
                    'check': 'price_positive',
                    'column': col,
                    'count': int(negative_count)
                })
                # Mark as invalid
                df.loc[negative_mask, 'can_trade'] = False
        
        # Check 2: OHLC consistency - High >= max(Open, Close)
        for prefix in ['raw_', 'adj_']:
            ohlc_mask = (
                (df[f'{prefix}high'] < df[f'{prefix}open']) |
                (df[f'{prefix}high'] < df[f'{prefix}close'])
            )
            ohlc_count = ohlc_mask.sum()
            if ohlc_count > 0:
                violations.append({
                    'check': 'ohlc_high_consistency',
                    'prefix': prefix,
                    'count': int(ohlc_count)
                })
                df.loc[ohlc_mask, 'can_trade'] = False
            
            # Check 3: OHLC consistency - Low <= min(Open, Close)
            ohlc_mask = (
                (df[f'{prefix}low'] > df[f'{prefix}open']) |
                (df[f'{prefix}low'] > df[f'{prefix}close'])
            )
            ohlc_count = ohlc_mask.sum()
            if ohlc_count > 0:
                violations.append({
                    'check': 'ohlc_low_consistency',
                    'prefix': prefix,
                    'count': int(ohlc_count)
                })
                df.loc[ohlc_mask, 'can_trade'] = False
        
        # Check 4: Volume non-negative
        volume_mask = df['volume'] < 0
        volume_count = volume_mask.sum()
        if volume_count > 0:
            violations.append({
                'check': 'volume_non_negative',
                'count': int(volume_count)
            })
            df.loc[volume_mask, 'can_trade'] = False
        
        report["checks"]["price_integrity"] = {
            "passed": len(violations) == 0,
            "violations": violations
        }
        
        if violations:
            logger.warn("price_integrity_violations", {
                "count": len(violations),
                "details": violations[:5]  # Log first 5
            })
        
        return df
