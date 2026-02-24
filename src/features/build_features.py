"""
Feature Engineering Module

Multi-time-scale feature calculation with dummy noise injection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml

from src.ops.event_logger import get_logger
from src.features.regime_detector import RegimeDetector

logger = get_logger()


class FeatureEngineer:
    """Multi-time-scale feature engineer."""
    
    # SECURITY: Do not log raw price samples or feature values in production
    REQUIRED_COLUMNS = ['symbol', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
    
    def __init__(self, config_path: str = "config/features.yaml", 
                 protocol_path: str = "config/event_protocol.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # FIX A1: Load ATR window from event_protocol.yaml for consistency
        with open(protocol_path, 'r') as f:
            protocol_config = yaml.safe_load(f)
        self.atr_window = protocol_config['triple_barrier']['atr']['window']
        
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
        
        # Check if source provides reliable adj OHLC (Patch 1 compliance)
        provides_adj_ohlc = df.get('source_provides_adj_ohlc', pd.Series([True] * len(df))).all()
        
        if not provides_adj_ohlc:
            logger.warn("feature_degradation_ohlc_disabled", {
                "reason": "backup_source_no_adj_ohlc",
                "disabled_features": [f"atr_{self.atr_window}", "rsi_14", "macd_line", "macd_signal", "pv_correlation_5d"],
                "retained_features": ["returns_*d", "rv_*d", "relative_volume_20d", "obv", "sma/ema_zscore", "market_breadth", "vix_change_5d", "pv_divergence_*"]
            })
        
        # Calculate features by category (using groupby for performance)
        df = self._calc_momentum_features_fast(df, provides_adj_ohlc)
        df = self._calc_volatility_features_fast(df, provides_adj_ohlc)
        df = self._calc_volume_features_fast(df)
        df = self._calc_mean_reversion_features_fast(df)
        df = self._calc_market_features(df)  # B14: VIX + market breadth
        df = self._calc_divergence_features(df, provides_adj_ohlc)  # B15: price-volume divergence
        
        # FIX B1: Add RegimeDetector features (regime_volatility, regime_trend)
        df = RegimeDetector().detect_regime(df)
        
        # Inject dummy noise feature (Plan v4)
        df = self._inject_dummy_noise(df)
        
        # Handle NaN values in features
        # Plan requirement: Mark rows with NaN features as invalid instead of filling with 0
        
        # FIX A2: Source-aware features_valid calculation
        # When source lacks AdjOHLC, only check features that don't depend on OHLC
        if not provides_adj_ohlc:
            # Backup source (Tiingo): only check AdjClose-based features
            feature_cols_to_check = [c for c in feature_cols 
                                     if c not in ['rsi_14', 'macd_line', 'macd_signal', 
                                                  'atr_20', 'pv_correlation_5d', 
                                                  'regime_trend', 'regime_trend_score', 'adx_14']]
            logger.info("features_valid_backup_source", {
                "checked_features": len(feature_cols_to_check),
                "excluded_ohlc_features": 7
            })
        else:
            feature_cols_to_check = feature_cols
        
        # FIX A2: Detect both NaN and inf as invalid (only for checked features)
        nan_mask = df[feature_cols_to_check].isna().any(axis=1)
        inf_mask = np.isinf(df[feature_cols_to_check]).any(axis=1)
        invalid_mask = nan_mask | inf_mask
        df['features_valid'] = ~invalid_mask
        
        # Log NaN statistics
        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()
        if nan_count > 0 or inf_count > 0:
            logger.warn("features_with_invalid_detected", {
                "nan_rows": int(nan_count),
                "inf_rows": int(inf_count),
                "total_rows": len(df),
                "invalid_pct": round(100 * (nan_count + inf_count) / len(df), 2)
            })
        
        # Keep NaN as NaN (don't fill with 0) - downstream will skip invalid rows
        # df[feature_cols] = df[feature_cols].fillna(0)  # REMOVED per Plan v4
        
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
            
            # FIX A1: Use ATR window from event_protocol.yaml, rename column to match
            df.loc[mask, f'atr_{self.atr_window}'] = self._calc_atr(df[mask].copy(), window=self.atr_window)
        
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
        
        F1 Fix: Row-level deterministic noise using hashlib + RandomState.
        This ensures same (symbol, date) always gets same noise regardless of
        universe changes or DataFrame ordering.
        
        This feature should NOT be used in actual prediction,
        only as a sentinel to detect overfitting.
        """
        import hashlib
        
        # F1: Use hashlib for stable hash + RandomState for proper normal distribution
        def generate_deterministic_noise(row):
            hash_input = f"{row['symbol']}_{row['date'].strftime('%Y-%m-%d')}_seed{self.dummy_seed}"
            # Use hashlib for stable, cross-platform hash
            h = int(hashlib.sha256(hash_input.encode()).hexdigest()[:16], 16)
            # Use RandomState for proper N(0,1) distribution
            rng = np.random.RandomState(h % (2**31))
            return rng.randn()
        
        df['dummy_noise'] = df.apply(generate_deterministic_noise, axis=1)
        
        logger.info("dummy_noise_injected", {
            "seed": self.dummy_seed,
            "method": "hashlib_sha256",
            "mean": float(df['dummy_noise'].mean()),
            "std": float(df['dummy_noise'].std())
        })
        
        return df
    
    def _calc_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.
        
        FIX A1: Handle loss=0 (all gains) -> RSI=100, not NaN.
        Standard RSI uses Wilder's smoothing; we use SMA for simplicity.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        # FIX A1 (R13): Handle edge cases for RSI extremes
        # When loss=0 (all gains), RSI should be 100 (max strength)
        # When gain=0 (all losses), RSI should be 0 (min strength)
        # When both=0 (warmup/constant price), keep NaN (not 0 or 100)
        rs = gain / loss.replace(0, np.nan)  # Avoid div-by-zero warning
        rsi = 100 - (100 / (1 + rs))
        
        # FIX A1 (R13): Only set RSI extremes when there's actual movement
        # NOT when both gain and loss are 0 (constant price or warmup)
        rsi = np.where((loss == 0) & (gain > 0), 100.0, rsi)  # Pure uptrend
        rsi = np.where((gain == 0) & (loss > 0), 0.0, rsi)    # Pure downtrend
        # Both=0: RSI stays NaN (from rs=NaN), features_valid will filter
        
        return rsi  # Keep NaN for initial values; features_valid handles this
    
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
        """
        Calculate On-Balance Volume (P2-5: Vectorized implementation).
        
        OBV = cumulative sum of signed volume
        where sign = +1 if close > close_prev, -1 if close < close_prev, 0 otherwise
        """
        # Vectorized calculation
        close_diff = df['adj_close'].diff()
        signed_volume = df['volume'] * np.sign(close_diff)
        obv = signed_volume.cumsum().fillna(0)
        
        return obv
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata).
        
        P0-2 Fix: Expanded exclude list to prevent label leakage and PIT info leakage.
        Uses defensive assertion to catch any label columns.
        """
        exclude = ['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 'raw_close',
                   'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume',
                   'feature_version', 'detected_split', 'split_ratio', 'is_suspended',
                   'suspension_start', 'can_trade',
                   # P0-2: Additional exclusions for PIT and label leak prevention
                   'ingestion_timestamp', 'source_provides_adj_ohlc',
                   'features_valid',
                   'label', 'label_barrier', 'label_return', 'label_holding_days', 
                   'event_valid', 'sample_weight',
                   # FIX A1: Exclude RegimeDetector string columns (use numeric scores instead)
                   'regime_volatility', 'regime_trend', 'regime_combined',
                   # FIX A1: Exclude old atr_N columns that don't match current config
                   'atr_14', 'atr_5', 'atr_10', 'atr_60']  # Only atr_{self.atr_window} is valid
        
        # Defensive assertion: detect any label-related columns
        label_cols = [col for col in df.columns if col.startswith('label')]
        if label_cols:
            raise ValueError(f"Label columns detected in feature extraction: {label_cols}. " 
                           "This indicates data leakage. Ensure labels are not passed to build_features().")
        
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
    
    def _calc_momentum_features_fast(self, df: pd.DataFrame, provides_adj_ohlc: bool = True) -> pd.DataFrame:
        """Calculate momentum features using groupby (optimized)."""
        # Log returns
        for window in [5, 10, 20, 60]:
            df[f'returns_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: np.log(x / x.shift(window))
            )
        
        if provides_adj_ohlc:
            # RSI using groupby (requires reliable OHLC)
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
        else:
            # Set OHLC-dependent features to NaN when source doesn't provide reliable data
            df['rsi_14'] = np.nan
            df['macd_line'] = np.nan
            df['macd_signal'] = np.nan
        
        return df
    
    def _calc_volatility_features_fast(self, df: pd.DataFrame, provides_adj_ohlc: bool = True) -> pd.DataFrame:
        """Calculate volatility features using groupby (optimized)."""
        # Realized volatility (only needs adj_close)
        for window in [5, 20, 60]:
            df[f'rv_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: (
                    np.log(x / x.shift(1))
                    .rolling(window=window, min_periods=1)
                    .std() * np.sqrt(252)
                )
            )
        
        if provides_adj_ohlc:
            # FIX A1: Use ATR window from event_protocol.yaml
            atr_parts = []
            for symbol, group in df.groupby('symbol'):
                atr = self._calc_atr(group.reset_index(drop=True), window=self.atr_window)
                atr.index = group.index
                atr_parts.append(atr)
            df[f'atr_{self.atr_window}'] = pd.concat(atr_parts)
        else:
            # ATR requires OHLC, set to NaN when unavailable
            df[f'atr_{self.atr_window}'] = np.nan
        
        return df
    
    def _calc_volume_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features using groupby (optimized)."""
        # Relative volume
        df['relative_volume_20d'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(window=20, min_periods=1).mean()
        )
        
        # OBV per symbol (explicit loop — groupby.apply unreliable across pandas versions)
        obv_parts = []
        for symbol, group in df.groupby('symbol'):
            obv = self._calc_obv(group.reset_index(drop=True))
            obv.index = group.index
            obv_parts.append(obv)
        df['obv'] = pd.concat(obv_parts)
        
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
    
    def _calc_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market-wide features (B14).
        
        These features require market-level data and are calculated
        once per date then broadcast to all symbols.
        
        Features:
        - vix_change_5d: VIX 5-day change rate (requires external VIX data)
        - market_breadth: Advance/Decline ratio
        """
        # Market breadth: proportion of stocks advancing vs declining
        # Calculate daily returns for each symbol
        df['daily_return'] = df.groupby('symbol')['adj_close'].pct_change()
        
        # For each date, calculate market breadth
        df['market_breadth'] = df.groupby('date')['daily_return'].transform(
            lambda x: (x > 0).sum() / max((x != 0).sum(), 1)  # Advancing / Total non-zero
        )
        
        # VIX change rate placeholder (would require VIX data fetch)
        # For now, use realized volatility of SPY-like proxy (median cross-sectional vol)
        # This is a simplified proxy - production should fetch actual VIX
        df['vix_proxy_5d'] = df.groupby('date')['daily_return'].transform(
            lambda x: x.std() * np.sqrt(252) if len(x) > 1 else 0.15
        )
        
        # 5-day change in vol proxy (fixed: calculate once per date, broadcast to all symbols)
        # First, get unique dates and their vix_proxy values
        date_vol_proxy = df.groupby('date')['vix_proxy_5d'].first().reset_index()
        date_vol_proxy['vix_change_5d'] = date_vol_proxy['vix_proxy_5d'].pct_change(periods=5)
        
        # FIX C1: Prevent duplicate columns on re-entry
        if 'vix_change_5d' in df.columns:
            df = df.drop(columns=['vix_change_5d'])
        
        # Merge back to df
        df = df.merge(date_vol_proxy[['date', 'vix_change_5d']], on='date', how='left')
        
        # Clean up temp column
        df = df.drop(columns=['daily_return', 'vix_proxy_5d'])
        
        logger.info("market_features_calculated", {
            "features": ["market_breadth", "vix_change_5d"]
        })
        
        return df
    
    def _calc_divergence_features(self, df: pd.DataFrame, provides_adj_ohlc: bool = True) -> pd.DataFrame:
        """
        Calculate price-volume divergence features (B15).
        
        Detects divergences between price action and volume:
        - Price rising but volume declining (weak trend)
        - Price falling but volume declining (weak decline)
        """
        # Price trend (5-day slope) - uses adj_close only
        df['price_trend_5d'] = df.groupby('symbol')['adj_close'].transform(
            lambda x: (x - x.shift(5)) / x.shift(5).replace(0, np.nan)
        )
        
        # Volume trend (5-day slope of relative volume)
        df['volume_trend_5d'] = df.groupby('symbol')['relative_volume_20d'].transform(
            lambda x: (x - x.shift(5)) / x.shift(5).replace(0, np.nan)
        )
        
        # Divergence: price up, volume down (weak bullish)
        df['pv_divergence_bull'] = ((df['price_trend_5d'] > 0) & 
                                     (df['volume_trend_5d'] < 0)).astype(int)
        
        # Divergence: price down, volume down (weak bearish)
        df['pv_divergence_bear'] = ((df['price_trend_5d'] < 0) & 
                                     (df['volume_trend_5d'] < 0)).astype(int)
        
        if provides_adj_ohlc:
            # Continuous divergence score (requires more reliable data)
            corr_parts = []
            for symbol, group in df.groupby('symbol'):
                corr = group['price_trend_5d'].rolling(5, min_periods=3).corr(
                    group['volume_trend_5d']
                )
                corr_parts.append(corr)
            df['pv_correlation_5d'] = pd.concat(corr_parts)
        else:
            # Disable correlation feature when OHLC unreliable
            df['pv_correlation_5d'] = np.nan
        
        # Clean up temp columns
        df = df.drop(columns=['price_trend_5d', 'volume_trend_5d'])
        
        logger.info("divergence_features_calculated", {
            "features": ["pv_divergence_bull", "pv_divergence_bear", "pv_correlation_5d"]
        })
        
        return df
