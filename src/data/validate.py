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
        
        # Handle consecutive NaN (halt detection)
        df = self._handle_missing_values(df, report)
        
        # Check 3: Abnormal jumps (after split adjustment awareness)
        df = self._detect_anomalies(df, report)
        
        # Check 4: Suspension detection (5+ days of NaN after handling)
        df = self._detect_suspension(df, report)
        
        # Calculate pass rate
        total_checks = len(report["checks"])
        passed_checks = sum(1 for c in report["checks"].values() if c.get("passed", True))
        report["pass_rate"] = passed_checks / total_checks if total_checks > 0 else 1.0
        
        passed = report["pass_rate"] >= 0.99  # 99% pass threshold
        
        logger.info("validation_complete", report, symbol)
        
        return passed, df, report
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        report: dict
    ) -> pd.DataFrame:
        """Handle missing values with forward fill for single days."""
        df = df.sort_values(['symbol', 'date'])
        
        # Count consecutive NaN per symbol
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask]
            
            # Find gaps
            nan_mask = symbol_df['raw_close'].isna()
            
            if nan_mask.sum() == 0:
                continue
            
            # Forward fill single days
            df.loc[mask, 'nan_group'] = nan_mask.astype(int).cumsum()
            
            for col in ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                       'adj_open', 'adj_high', 'adj_low', 'adj_close']:
                df.loc[mask, col] = df.loc[mask].groupby('nan_group')[col].ffill(limit=1)
        
        if 'nan_group' in df.columns:
            df = df.drop(columns=['nan_group'])
        
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
        """Detect trading suspensions (5+ consecutive NaN)."""
        suspensions = []
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df.loc[mask].sort_values('date')
            
            # Check for remaining NaN after forward fill
            nan_mask = symbol_df['raw_close'].isna()
            
            if nan_mask.sum() < 5:
                continue
            
            # Mark as suspended (don't drop, just flag)
            suspensions.append({
                "symbol": symbol,
                "suspended_days": int(nan_mask.sum())
            })
            
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
