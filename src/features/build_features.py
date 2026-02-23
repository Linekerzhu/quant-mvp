"""
Feature Engineering Module

Multi-time-scale feature calculation with dummy noise injection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml

from src.ops.event_logger import get_logger

logger = get_logger()


class FeatureEngineer:
    """Multi-time-scale feature engineer."""
    
    # SECURITY: Do not log raw price samples or feature values in production
    REQUIRED_COLUMNS = ['symbol', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
    
    def __init__(self, config_path: str = "config/features.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.version = self.config['version']
        self.dummy_seed = 42  # For reproducibility
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Column 'date' must be datetime type")
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from price data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with features added
        """
        import time
        start_time = time.time()
        
        # Validate input
        self._validate_input(df)
        
        df = df.copy()
        
        # Ensure sorted by symbol and date
        df = df.sort_values(['symbol', 'date'])
        
        # Calculate features by category (using groupby for performance)
        df = self._calc_momentum_features_fast(df)
        df = self._calc_volatility_features_fast(df)
        df = self._calc_volume_features_fast(df)
        df = self._calc_mean_reversion_features_fast(df)
        
        # Inject dummy noise feature (Plan v4)
        df = self._inject_dummy_noise(df)
        
        # Handle NaN values in features
        feature_cols = self._get_feature_columns(df)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Add feature version
        df['feature_version'] = self.version
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("features_built", {
            "version": self.version,
            "n_features": len(feature_cols),
            "rows": len(df),
            "elapsed_ms": elapsed_ms
        })
        
        return df
    
    def _calc_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features (legacy loop-based, kept for reference)."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Log returns for different windows
            for window in [5, 10, 20, 60]:
                df.loc[mask, f'returns_{window}d'] = (
                    np.log(df.loc[mask, 'adj_close'] / 
                           df.loc[mask, 'adj_close'].shift(window))
                )
            
            # RSI (14)
            df.loc[mask, 'rsi_14'] = self._calc_rsi(
                df.loc[mask, 'adj_close'], window=14
            )
            
            # MACD
            df.loc[mask, 'macd_line'], df.loc[mask, 'macd_signal'] = self._calc_macd(
                df.loc[mask, 'adj_close']
            )
        
        return df
    
    def _calc_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Realized volatility (std of log returns)
            log_returns = np.log(df.loc[mask, 'adj_close'] / 
                                df.loc[mask, 'adj_close'].shift(1))
            
            for window in [5, 20, 60]:
                df.loc[mask, f'rv_{window}d'] = (
                    log_returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
                )
            
            # ATR (14)
            df.loc[mask, 'atr_14'] = self._calc_atr(df[mask].copy(), window=14)
        
        return df
    
    def _calc_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Relative volume vs 20-day average
            df.loc[mask, 'relative_volume_20d'] = (
                df.loc[mask, 'volume'] / 
                df.loc[mask, 'volume'].rolling(window=20, min_periods=1).mean()
            )
            
            # OBV (On-Balance Volume)
            df.loc[mask, 'obv'] = self._calc_obv(df[mask].copy())
        
        return df
    
    def _calc_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion features."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Price vs SMA z-score
            for window in [20, 60]:
                sma = df.loc[mask, 'adj_close'].rolling(window=window, min_periods=1).mean()
                std = df.loc[mask, 'adj_close'].rolling(window=window, min_periods=1).std()
                df.loc[mask, f'price_vs_sma{window}_zscore'] = (
                    (df.loc[mask, 'adj_close'] - sma) / std.replace(0, np.nan)
                )
            
            # Price vs EMA z-score
            for window in [20]:
                ema = df.loc[mask, 'adj_close'].ewm(span=window, min_periods=1).mean()
                std = df.loc[mask, 'adj_close'].ewm(span=window, min_periods=1).std()
                df.loc[mask, f'price_vs_ema{window}_zscore'] = (
                    (df.loc[mask, 'adj_close'] - ema) / std.replace(0, np.nan)
                )
        
        return df
    
    def _inject_dummy_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject dummy noise feature for overfitting detection (Plan v4).
        
        This feature should NOT be used in actual prediction,
        only as a sentinel to detect overfitting.
        """
        np.random.seed(self.dummy_seed)
        df['dummy_noise'] = np.random.normal(0, 1, size=len(df))
        
        logger.info("dummy_noise_injected", {
            "seed": self.dummy_seed,
            "mean": float(df['dummy_noise'].mean()),
            "std": float(df['dummy_noise'].std())
        })
        
        return df
    
    def _calc_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral for initial values
    
    def _calc_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD line and signal."""
        ema12 = prices.ewm(span=12, min_periods=1).mean()
        ema26 = prices.ewm(span=26, min_periods=1).mean()
        
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
        
        return macd_line, macd_signal
    
    def _calc_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['adj_high'] - df['adj_low']
        high_close = np.abs(df['adj_high'] - df['adj_close'].shift(1))
        low_close = np.abs(df['adj_low'] - df['adj_close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    def _calc_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = [0]
        
        for i in range(1, len(df)):
            if df['adj_close'].iloc[i] > df['adj_close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['adj_close'].iloc[i] < df['adj_close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=df.index)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata)."""
        exclude = ['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 'raw_close',
                   'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume',
                   'feature_version', 'detected_split', 'split_ratio', 'is_suspended',
                   'suspension_start', 'can_trade']
        
        return [col for col in df.columns if col not in exclude]
    
    def get_feature_metadata(self) -> Dict:
        """Get feature metadata for tracking."""
        return {
            'version': self.version,
            'feature_count': len(self.config['categories']),
            'categories': list(self.config['categories'].keys()),
            'dummy_feature': 'dummy_noise',
            'dummy_seed': self.dummy_seed
        }
    
    # === Optimized GroupBy Methods (10-100x faster) ===
    
    def _calc_momentum_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features using groupby (optimized)."""
        # Log returns
        for window in [5, 10, 20, 60]:
            df[f'returns_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: np.log(x / x.shift(window))
            )
        
        # RSI using groupby
        df['rsi_14'] = df.groupby('symbol')['adj_close'].transform(
            lambda x: self._calc_rsi(x, window=14)
        )
        
        # MACD using groupby
        macd_results = df.groupby('symbol')['adj_close'].apply(
            lambda x: self._calc_macd(x)
        )
        df['macd_line'] = np.nan
        df['macd_signal'] = np.nan
        for symbol, (macd, signal) in macd_results.items():
            mask = df['symbol'] == symbol
            df.loc[mask, 'macd_line'] = macd.values
            df.loc[mask, 'macd_signal'] = signal.values
        
        return df
    
    def _calc_volatility_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features using groupby (optimized)."""
        # Realized volatility
        for window in [5, 20, 60]:
            df[f'rv_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: (
                    np.log(x / x.shift(1))
                    .rolling(window=window, min_periods=1)
                    .std() * np.sqrt(252)
                )
            )
        
        # ATR using groupby
        df['atr_14'] = df.groupby('symbol').apply(
            lambda g: self._calc_atr(g.reset_index(drop=True), window=14)
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _calc_volume_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features using groupby (optimized)."""
        # Relative volume
        df['relative_volume_20d'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(window=20, min_periods=1).mean()
        )
        
        # OBV using groupby
        df['obv'] = df.groupby('symbol').apply(
            lambda g: self._calc_obv(g.reset_index(drop=True))
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _calc_mean_reversion_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion features using groupby (optimized)."""
        # SMA z-scores
        for window in [20, 60]:
            sma = df.groupby('symbol')['adj_close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            std = df.groupby('symbol')['adj_close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'price_vs_sma{window}_zscore'] = (df['adj_close'] - sma) / std.replace(0, np.nan)
        
        # EMA z-score
        ema = df.groupby('symbol')['adj_close'].transform(
            lambda x: x.ewm(span=20, min_periods=1).mean()
        )
        ema_std = df.groupby('symbol')['adj_close'].transform(
            lambda x: x.ewm(span=20, min_periods=1).std()
        )
        df['price_vs_ema20_zscore'] = (df['adj_close'] - ema) / ema_std.replace(0, np.nan)
        
        return df
