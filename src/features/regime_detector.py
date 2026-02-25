"""
Regime Detector Module

Detects market regime based on volatility level and trend strength (ADX).
Output is used as a soft feature, not a hard risk control driver.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import yaml

from src.ops.event_logger import get_logger

logger = get_logger()


class RegimeDetector:
    """
    Market regime detector using volatility and trend strength.
    
    Outputs continuous regime scores as features, not discrete state switches.
    """
    
    def __init__(self, config_path: str = "config/event_protocol.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Volatility thresholds
        self.low_vol_threshold = 0.15  # 15% annualized
        self.high_vol_threshold = 0.25  # 25% annualized
        
        # ADX thresholds
        self.strong_trend_threshold = 25
        self.weak_trend_threshold = 15
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime and add regime features.
        
        Args:
            df: DataFrame with price data (should include market index like SPY)
            
        Returns:
            DataFrame with regime features added
        """
        df = df.copy()
        
        # Calculate volatility regime (continuous score)
        df = self._calc_volatility_regime(df)
        
        # Calculate trend regime (ADX-based)
        df = self._calc_trend_regime(df)
        
        # Combined regime classification (for reference, not hard control)
        df['regime_combined'] = df['regime_volatility'].astype(str) + '_' + df['regime_trend'].astype(str)
        
        logger.info("regime_detected", {
            "regimes": df['regime_combined'].value_counts().to_dict()
        })
        
        return df
    
    def _calc_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility regime.
        
        Regime levels:
        - low: rv_20d < 15%
        - medium: 15% <= rv_20d < 25%
        - high: rv_20d >= 25%
        """
        # Calculate realized volatility if not present
        if 'rv_20d' not in df.columns:
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                log_returns = np.log(df.loc[mask, 'adj_close'] / 
                                    df.loc[mask, 'adj_close'].shift(1))
                df.loc[mask, 'rv_20d'] = log_returns.rolling(20, min_periods=1).std() * np.sqrt(252)
        
        # Discretize into regimes
        conditions = [
            df['rv_20d'] < self.low_vol_threshold,
            (df['rv_20d'] >= self.low_vol_threshold) & (df['rv_20d'] < self.high_vol_threshold),
            df['rv_20d'] >= self.high_vol_threshold
        ]
        choices = ['low', 'medium', 'high']
        
        df['regime_volatility'] = np.select(conditions, choices, default='medium')
        
        # Add continuous score (0-1 scale)
        df['regime_vol_score'] = np.clip(
            (df['rv_20d'] - self.low_vol_threshold) / 
            (self.high_vol_threshold - self.low_vol_threshold),
            0, 1
        )
        
        return df
    
    def _calc_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend regime using ADX.
        
        Regime levels:
        - weak: ADX < 15
        - moderate: 15 <= ADX < 25
        - strong: ADX >= 25
        """
        # Calculate ADX for each symbol
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            adx = self._calc_adx(df[mask].copy())
            df.loc[mask, 'adx_14'] = adx
        
        # Discretize into regimes
        conditions = [
            df['adx_14'] < self.weak_trend_threshold,
            (df['adx_14'] >= self.weak_trend_threshold) & (df['adx_14'] < self.strong_trend_threshold),
            df['adx_14'] >= self.strong_trend_threshold
        ]
        choices = ['weak', 'moderate', 'strong']
        
        df['regime_trend'] = np.select(conditions, choices, default='moderate')
        
        # Add continuous score (0-1 scale)
        df['regime_trend_score'] = np.clip(
            (df['adx_14'] - self.weak_trend_threshold) / 
            (self.strong_trend_threshold - self.weak_trend_threshold),
            0, 1
        )
        
        return df
    
    def _calc_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        """
        # Calculate +DM and -DM
        df['plus_dm'] = df['adj_high'].diff()
        df['minus_dm'] = -df['adj_low'].diff()
        
        df['plus_dm'] = df['plus_dm'].where(
            (df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), 0
        )
        df['minus_dm'] = df['minus_dm'].where(
            (df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), 0
        )
        
        # Calculate TR
        df['tr1'] = df['adj_high'] - df['adj_low']
        df['tr2'] = np.abs(df['adj_high'] - df['adj_close'].shift(1))
        df['tr3'] = np.abs(df['adj_low'] - df['adj_close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Smooth TR and DM
        df['atr'] = df['tr'].rolling(window=window, min_periods=1).mean()
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=window, min_periods=1).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=window, min_periods=1).mean() / df['atr']
        
        # Calculate DX and ADX
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        # P2-C1 (R21): Replace inf with NaN (not 0) for proper NaN propagation
        # inf occurs when plus_di + minus_di = 0 (flat price), which is an edge case
        df['dx'] = df['dx'].replace([np.inf, -np.inf], np.nan)
        adx = df['dx'].rolling(window=window, min_periods=1).mean()
        
        # P2-3 Fix: Return NaN for initial values instead of filling with 20
        # This maintains consistency with Plan's NaN handling strategy
        return adx
    
    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of detected regimes."""
        return {
            'volatility_distribution': df['regime_volatility'].value_counts().to_dict(),
            'trend_distribution': df['regime_trend'].value_counts().to_dict(),
            'combined_distribution': df['regime_combined'].value_counts().to_dict(),
            'volatility_score_mean': float(df['regime_vol_score'].mean()),
            'trend_score_mean': float(df['regime_trend_score'].mean())
        }
