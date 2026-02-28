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
        
        # P1-A2 (R23): Build OHLC exemption list from YAML meta
        self.ohlc_exempt_features = self._build_ohlc_exempt_list()
    
    def _build_ohlc_exempt_list(self) -> List[str]:
        """
        P1-A2 (R23): Build list of features requiring OHLC from YAML.
        
        This replaces hardcoded exemption list with meta-driven approach.
        """
        exempt_features = []
        
        # Iterate through all categories
        for category_name, category_data in self.config.get('categories', {}).items():
            for feature in category_data.get('features', []):
                if feature.get('requires_ohlc', False):
                    exempt_features.append(feature['name'])
        
        # Also check dummy features
        for feature in self.config.get('dummy_features', []):
            if feature.get('requires_ohlc', False):
                exempt_features.append(feature['name'])
        
        # R29-A3: Hardcoded OHLC-dependent features not in YAML
        # These features are computed internally but not defined in config
        hardcoded_ohlc_features = [
            'atr_20',           # Average True Range (needs high/low)
            'regime_trend_score',  # Correlated with ADX (needs high/low)
        ]
        exempt_features.extend(hardcoded_ohlc_features)
        
        # OR4-P2-1 (R25): Remove feature names from logs (no info value, potential leak)
        logger.info("ohlc_exempt_features_built", {
            "count": len(exempt_features),
            "from_yaml": len(exempt_features) - len(hardcoded_ohlc_features),
            "hardcoded": len(hardcoded_ohlc_features)
        })
        
        return exempt_features
    
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
        
        # P0-A1 (R22): Market features MUST be calculated before split
        # These features require cross-symbol perspective and should NOT be in _build_features_inner
        df = self._calc_market_features(df)  # B14: VIX + market breadth
        
        # P0-A1 (R21): Per-symbol source detection instead of batch-level .all()
        # .all() causes ALL symbols to lose regime features if ANY symbol is backup source
        # Solution: Split into primary/backup batches, process separately, then merge
        
        has_source_flag = 'source_provides_adj_ohlc' in df.columns
        
        if has_source_flag:
            primary_mask = df['source_provides_adj_ohlc'] == True
            primary_df = df[primary_mask].copy()
            backup_df = df[~primary_mask].copy()
            
            if len(backup_df) > 0:
                logger.warn("mixed_source_batch_detected", {
                    "primary_symbols": primary_df['symbol'].nunique() if len(primary_df) > 0 else 0,
                    "backup_symbols": backup_df['symbol'].nunique(),
                    "total_rows": len(df)
                })
        else:
            # No source flag - assume all primary
            primary_df = df.copy()
            backup_df = pd.DataFrame()
        
        # Process primary batch (full features)
        if len(primary_df) > 0:
            primary_df = self._build_features_inner(primary_df, provides_adj_ohlc=True)
        
        # Process backup batch (degraded features)
        if len(backup_df) > 0:
            backup_df = self._build_features_inner(backup_df, provides_adj_ohlc=False)
        
        # Merge batches
        df = pd.concat([primary_df, backup_df]).sort_values(['symbol', 'date'])
        
        # Handle NaN values in features
        feature_cols = self._get_feature_columns(df)
        
        # Source-aware features_valid calculation
        if has_source_flag and len(backup_df) > 0:
            # Mixed batch: only check features for backup source
            backup_valid_mask = df['source_provides_adj_ohlc'] == False
            # P1-A2 (R23): Use meta-driven exemption list instead of hardcoded
            backup_feature_cols = [c for c in feature_cols 
                                   if c not in self.ohlc_exempt_features]
            
            # Primary source: check all features
            primary_valid_mask = df['source_provides_adj_ohlc'] == True
            
            # Calculate features_valid for each batch
            df_primary = df[primary_valid_mask].copy()
            df_backup = df[backup_valid_mask].copy()
            
            if len(df_primary) > 0:
                nan_mask_primary = df_primary[feature_cols].isna().any(axis=1)
                inf_mask_primary = np.isinf(df_primary[feature_cols]).any(axis=1)
                df_primary['features_valid'] = ~(nan_mask_primary | inf_mask_primary)
            
            if len(df_backup) > 0:
                nan_mask_backup = df_backup[backup_feature_cols].isna().any(axis=1)
                inf_mask_backup = np.isinf(df_backup[backup_feature_cols]).any(axis=1)
                df_backup['features_valid'] = ~(nan_mask_backup | inf_mask_backup)
            
            df = pd.concat([df_primary, df_backup]).sort_values(['symbol', 'date'])
        else:
            # Single source batch: original logic
            nan_mask = df[feature_cols].isna().any(axis=1)
            inf_mask = np.isinf(df[feature_cols]).any(axis=1)
            df['features_valid'] = ~(nan_mask | inf_mask)
        
        # Keep NaN as NaN (don't fill with 0)
        df['feature_version'] = self.version
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("features_built", {
            "version": self.version,
            "rows": len(df),
            "primary_rows": len(primary_df) if has_source_flag and len(primary_df) > 0 else len(df),
            "backup_rows": len(backup_df) if has_source_flag else 0,
            "elapsed_ms": elapsed_ms
        })
        
        return df
    
    def _build_features_inner(self, df: pd.DataFrame, provides_adj_ohlc: bool) -> pd.DataFrame:
        """
        Internal feature builder for a single source type.
        
        P0-A1 (R21): Extracted to support per-symbol source detection.
        """
        if not provides_adj_ohlc:
            logger.warn("feature_degradation_ohlc_disabled", {
                "reason": "backup_source_no_adj_ohlc",
                "symbols": df['symbol'].nunique()
            })
        
        # Calculate features by category (using groupby for performance)
        df = self._calc_momentum_features_fast(df, provides_adj_ohlc)
        df = self._calc_volatility_features_fast(df, provides_adj_ohlc)
        df = self._calc_volume_features_fast(df)
        df = self._calc_mean_reversion_features_fast(df)
        # P0-A1 (R22): Market features calculated BEFORE split (line 74), skip here
        df = self._calc_divergence_features(df, provides_adj_ohlc)  # B15: price-volume divergence
        
        # P2-C1: RegimeDetector only when adj OHLC available
        if provides_adj_ohlc:
            df = RegimeDetector().detect_regime(df)
        else:
            # Backup source fallback: use simpler regime estimation
            df['adx_14'] = np.nan
            df['regime_volatility'] = 'unknown'
            df['regime_trend'] = 'unknown'
            df['regime_combined'] = 'unknown_unknown'
            # Approximate regime from realized volatility
            if 'rv_20d' in df.columns:
                df['regime_vol_score'] = np.clip(
                    (df['rv_20d'] - 0.15) / (0.25 - 0.15), 0, 1
                )
            else:
                df['regime_vol_score'] = np.nan
            df['regime_trend_score'] = np.nan
        
        # Normalize dollar-scale features
        # P1-B2 (R22): Replace macd_signal_pct with macd_histogram_pct
        # signal_pct has r=0.964 with line_pct (highly redundant)
        # histogram_pct = (line - signal) / close has r=0.333 with line_pct
        df['macd_line_pct'] = df['macd_line'] / df['adj_close']
        df['macd_histogram_pct'] = (df['macd_line'] - df['macd_signal']) / df['adj_close']
        
        # Inject dummy noise feature
        df = self._inject_dummy_noise(df)
        
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
                # OR2-07 Fix: Use min_periods=window to avoid noisy estimates in warmup period
                df.loc[mask, f'rv_{window}d'] = (
                    log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
                )
            
            # FIX A1: Use ATR window from event_protocol.yaml, rename column to match
            df.loc[mask, f'atr_{self.atr_window}'] = self._calc_atr(df[mask].copy(), window=self.atr_window)
        
        return df
    
    # P2 (R24-A2a): Deleted dead function _calc_volume_features
    # This function was never called (replaced by _calc_volume_features_fast)
    # and contained old buggy relative_volume_20d implementation
    
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
        
        FIX A2 (R17): Restore per-row determinism across universe changes.
        R15 vectorization broke cross-universe consistency: adding 1 symbol
        changed ALL noise values because np.random.seed(array) uses all seeds.
        
        Solution: Vectorized hash calculation + per-row RandomState generation.
        Speed: ~0.02s for 1000 rows (100x faster than R14, 10x slower than R15).
        Determinism: Same (symbol, date) -> same noise regardless of universe.
        
        This feature should NOT be used in actual prediction,
        only as a sentinel to detect overfitting.
        """
        import hashlib
        
        # Vectorized seed generation using string hash
        seeds = (
            df['symbol'].astype(str) + '_' + 
            df['date'].dt.strftime('%Y-%m-%d') + 
            f'_seed{self.dummy_seed}'
        ).apply(lambda x: int(hashlib.sha256(x.encode()).hexdigest()[:16], 16) % (2**31))
        
        # FIX A2 (R17): Per-row RandomState for cross-universe determinism
        # Each (symbol, date) gets independent RNG state
        df['dummy_noise'] = np.array([
            np.random.RandomState(int(s)).randn() for s in seeds
        ])
        
        # P2-C2 (R21): Removed mean/std from logs (no information value)
        logger.info("dummy_noise_injected", {
            "seed": self.dummy_seed,
            "method": "per_row_randomstate",
            "row_count": len(df)
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
                   'atr_14', 'atr_5', 'atr_10', 'atr_60',
                   # P0-A1 (R19): Use _pct normalized versions instead of dollar-scale originals
                   'atr_20',  # R19-A1: Add atr_20 to exclude (was missing)
                   'macd_line', 'macd_signal',
                   # P1-B1: Remove high-redundancy features (|r| > 0.8)
                   'regime_vol_score',       # r=0.90 with rv_20d
                   'regime_trend_score',     # r=0.82 with adx_14
                   'price_vs_ema20_zscore',  # r=0.97 with price_vs_sma20_zscore
                   'obv']                    # absolute volume, 500x cross-symbol variance
        
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
        # Log returns (only needs adj_close)
        for window in [5, 10, 20, 60]:
            df[f'returns_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: np.log(x / x.shift(window))
            )
        
        # R29-A3: RSI/MACD only need adj_close, NOT OHLC
        # These should be calculated for all sources
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
    
    def _calc_volatility_features_fast(self, df: pd.DataFrame, provides_adj_ohlc: bool = True) -> pd.DataFrame:
        # OR2-07 Fix: 使用 min_periods=window 避免warmup期噪声
        for window in [5, 20, 60]:
            df[f'rv_{window}d'] = df.groupby('symbol')['adj_close'].transform(
                lambda x: (
                    np.log(x / x.shift(1))
                    .rolling(window=window, min_periods=window)
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
        # Relative volume - P2 (R23): Use shift(1) to exclude current day from mean
        # Prevents self-reference that compresses extreme volume signals by 31%
        df['relative_volume_20d'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.shift(1).rolling(window=20, min_periods=1).mean()
        )
        
        # P2 (R24-A2b): Deleted OBV calculation
        # OBV is excluded in _get_feature_columns and never used
        # Removing saves O(n×symbols) computation time
        
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
        
        P2 (R24-A1): Only use primary source rows for market_breadth calculation.
        This prevents backup source symbols (with potentially unreliable data)
        from contaminating the market-wide breadth signal.
        """
        # P2 (R24-A1): Use only primary source rows for market-level calculations
        if 'source_provides_adj_ohlc' in df.columns:
            df_primary_only = df[df['source_provides_adj_ohlc'] == True]
        else:
            df_primary_only = df
        
        # Handle edge case: no primary sources available
        if len(df_primary_only) == 0:
            # Fallback: use all data (including backup sources)
            df_primary_only = df
        
        # Market breadth: proportion of stocks advancing vs declining
        # Calculate daily returns for each symbol (primary source only)
        df_primary_only = df_primary_only.copy()
        df_primary_only['daily_return'] = df_primary_only.groupby('symbol')['adj_close'].transform(
            lambda x: x.pct_change(fill_method=None)
        )
        
        # For each date, calculate market breadth (from primary sources only)
        # P2 (R25-B1): Exclude NaN from denominator (first day pct_change = NaN)
        date_breadth = df_primary_only.groupby('date')['daily_return'].agg(
            lambda x: (x.notna() & (x > 0)).sum() / max((x.notna() & (x != 0)).sum(), 1)
        ).reset_index()
        date_breadth.columns = ['date', 'market_breadth']
        
        # P2 (R26-B1): Drop existing market_breadth before merge (prevent duplicate columns)
        if 'market_breadth' in df.columns:
            df = df.drop(columns=['market_breadth'])
        
        # Broadcast back to all rows (including backup sources)
        df = df.merge(date_breadth, on='date', how='left')
        
        # VIX change rate placeholder (would require VIX data fetch)
        # For now, use realized volatility of SPY-like proxy (median cross-sectional vol)
        # This is a simplified proxy - production should fetch actual VIX
        df_primary_only['vix_proxy_5d'] = df_primary_only.groupby('date')['daily_return'].transform(
            lambda x: x.std() * np.sqrt(252) if len(x) > 1 else 0.15
        )
        
        # 5-day change in vol proxy (fixed: calculate once per date, broadcast to all symbols)
        # First, get unique dates and their vix_proxy values
        date_vol_proxy = df_primary_only.groupby('date')['vix_proxy_5d'].first().reset_index()
        date_vol_proxy['vix_change_5d'] = date_vol_proxy['vix_proxy_5d'].pct_change(periods=5)
        date_vol_proxy['vix_change_5d'] = date_vol_proxy['vix_change_5d'].clip(-2.0, 2.0)  # P0-A3: Cap outliers
        
        # FIX C1: Prevent duplicate columns on re-entry
        if 'vix_change_5d' in df.columns:
            df = df.drop(columns=['vix_change_5d'])
        
        # Merge back to df
        df = df.merge(date_vol_proxy[['date', 'vix_change_5d']], on='date', how='left')
        
        logger.info("market_features_calculated", {
            "features": ["market_breadth", "vix_change_5d"],
            "primary_only": True
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
        
        # R29-A3: pv_correlation_5d only needs adj_close + volume, NOT OHLC
        # Calculate for all sources
        corr_parts = []
        for symbol, group in df.groupby('symbol'):
            corr = group['price_trend_5d'].rolling(5, min_periods=3).corr(
                group['volume_trend_5d']
            )
            corr_parts.append(corr)
        df['pv_correlation_5d'] = pd.concat(corr_parts)
        # P0-A2 (R19): Fill NaN from constant-volume edge cases
        # rolling.corr() returns NaN when variance=0; treat as neutral (0 correlation)
        df['pv_correlation_5d'] = df['pv_correlation_5d'].fillna(0.0)
        
        # Clean up temp columns
        df = df.drop(columns=['price_trend_5d', 'volume_trend_5d'])
        
        logger.info("divergence_features_calculated", {
            "features": ["pv_divergence_bull", "pv_divergence_bear", "pv_correlation_5d"]
        })
        
        return df
