"""
Kronos Feature Extractor: Batch prediction caching for LTR integration.

Converts Kronos single-stock veto Oracle into a batch feature provider.
Predictions are cached as Parquet for build_features.py consumption.

Usage (daily pre-market):
    from src.models.kronos_feature_extractor import KronosFeatureExtractor
    extractor = KronosFeatureExtractor(kronos_url="...")
    extractor.extract_features(features_df, output_path="data/cache/kronos/")
    
Then in build_features.py:
    features = KronosFeatureExtractor.merge_kronos_features(features, "data/cache/kronos/")

AUDIT: Kronos predictions use only T-day and historical data (the Oracle processes
OHLCV up to date T and predicts T+5). No look-ahead in features.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from src.ops.event_logger import get_logger

logger = get_logger()


class KronosFeatureExtractor:
    """
    Batch Kronos prediction extractor for LTR feature pipeline.
    
    Two modes:
    1. LIVE: Calls Kronos API per-symbol (for daily_job.py)
    2. BACKTEST: Uses cached predictions from Parquet
    
    Output features:
    - kronos_pred_return: Kronos 5-day predicted return
    - kronos_confidence: Confidence derived from prediction magnitude
    """
    
    def __init__(
        self,
        kronos_url: Optional[str] = None,
        cache_dir: str = 'data/cache/kronos',
        lookback: int = 400,
        pred_len: int = 5,
    ):
        self.kronos_url = kronos_url or os.environ.get('KRONOS_ENDPOINT', '')
        self.cache_dir = cache_dir
        self.lookback = lookback
        self.pred_len = pred_len
        self._client = None
    
    def _get_client(self):
        """Lazy-load Kronos API client."""
        if self._client is None:
            from src.models.expert_oracle import KronosOracleClient
            self._client = KronosOracleClient(self.kronos_url)
        return self._client
    
    def extract_features(
        self,
        df: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None,
        symbols: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract Kronos predictions for a set of symbols on a given date.
        
        For LIVE mode: calls Kronos API for each symbol.
        Results are cached to Parquet for future use.
        
        Args:
            df: Feature DataFrame with OHLCV history
            target_date: Date to predict for (default: latest date)
            symbols: Symbols to predict (default: all in df)
            output_path: Override cache path
            
        Returns:
            DataFrame with ['symbol', 'date', 'kronos_pred_return', 'kronos_confidence']
        """
        if target_date is None:
            target_date = df['date'].max()
        
        if symbols is None:
            symbols = df['symbol'].unique().tolist()
        
        client = self._get_client()
        results = []
        
        n_success = 0
        n_fail = 0
        
        for sym in symbols:
            try:
                response = client.request_veto(
                    symbol=sym,
                    target_weight=0.05,
                    features_df=df,
                    lookback=self.lookback,
                )
                
                pred_ret = float(response.get('predicted_return', 0.0))
                # Confidence: higher for stronger predictions
                confidence = min(abs(pred_ret) * 10, 1.0)
                
                results.append({
                    'symbol': sym,
                    'date': target_date,
                    'kronos_pred_return': pred_ret,
                    'kronos_confidence': confidence,
                    'kronos_action': response.get('action', 'approve'),
                })
                n_success += 1
                
            except Exception as e:
                logger.error("kronos_feature_extraction_failed", {
                    "symbol": sym, "error": str(e)
                })
                results.append({
                    'symbol': sym,
                    'date': target_date,
                    'kronos_pred_return': 0.0,
                    'kronos_confidence': 0.0,
                    'kronos_action': 'error',
                })
                n_fail += 1
        
        result_df = pd.DataFrame(results)
        
        # Save to cache
        save_path = output_path or os.path.join(
            self.cache_dir, f"kronos_{target_date.strftime('%Y%m%d')}.parquet"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_parquet(save_path, index=False)
        
        print(f"[Kronos] Extracted {n_success}/{len(symbols)} predictions "
              f"({n_fail} failed), saved to {save_path}")
        
        return result_df
    
    def extract_features_backtest(
        self,
        df: pd.DataFrame,
        correlation: float = 0.0,
        seed: int = 42,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate SAFE simulated Kronos predictions for backtesting.
        
        ⚠️  AUDIT C4: Previous version used shift(-5) future returns — GOD'S EYE LEAKAGE.
        This version uses ONLY historical volatility to generate plausible noise predictions.
        It does NOT correlate with future returns at all.
        
        Args:
            df: DataFrame with ['symbol', 'date', 'adj_close']
            correlation: MUST be 0.0 — any positive value raises error (was leakage source)
            seed: Random seed for reproducibility
            output_path: Where to save the combined Parquet
        """
        if correlation > 0:
            raise RuntimeError(
                "AUDIT C4: correlation > 0 uses future returns = LOOK-AHEAD BIAS. "
                "Synthetic predictions must use correlation=0 (pure noise based on historical vol)."
            )
        
        np.random.seed(seed)
        df_sorted = df.sort_values(['symbol', 'date']).copy()
        
        # Use HISTORICAL volatility (backward-looking only) to scale noise
        df_sorted['_hist_vol'] = df_sorted.groupby('symbol')['adj_close'].transform(
            lambda x: x.pct_change().rolling(20).std().fillna(0.02)
        )
        
        results = []
        for _, row in df_sorted.iterrows():
            hist_vol = float(row.get('_hist_vol', 0.02))
            # Pure noise scaled by historical vol — NO future information
            pred_ret = np.random.randn() * hist_vol * np.sqrt(5)
            pred_ret = np.clip(pred_ret, -0.5, 0.5)
            confidence = min(abs(pred_ret) * 10, 1.0)
            
            results.append({
                'symbol': row['symbol'],
                'date': pd.Timestamp(row['date']),
                'kronos_pred_return': float(pred_ret),
                'kronos_confidence': float(confidence),
            })
        
        result_df = pd.DataFrame(results)
        
        save_path = output_path or os.path.join(self.cache_dir, 'kronos_backtest.parquet')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_parquet(save_path, index=False)
        
        n_nonzero = (result_df['kronos_pred_return'].abs() > 0.001).sum()
        print(f"[Kronos] Generated {len(result_df)} SAFE backtest predictions (no future leakage), "
              f"{n_nonzero} active ({n_nonzero/len(result_df)*100:.1f}%)")
        
        return result_df
    
    @staticmethod
    def merge_kronos_features(
        df: pd.DataFrame,
        cache_path: str = 'data/cache/kronos/kronos_backtest.parquet',
    ) -> pd.DataFrame:
        """
        Merge Kronos predictions into feature DataFrame.
        
        No Kronos dependency at merge time (reads Parquet).
        
        Args:
            df: Feature DataFrame with ['symbol', 'date']
            cache_path: Path to Kronos Parquet cache
            
        Returns:
            DataFrame with kronos_pred_return and kronos_confidence added
        """
        if not os.path.exists(cache_path):
            # If cache dir, try to find latest file
            if os.path.isdir(cache_path):
                files = sorted([f for f in os.listdir(cache_path) if f.endswith('.parquet')])
                if files:
                    cache_path = os.path.join(cache_path, files[-1])
                else:
                    df['kronos_pred_return'] = 0.0
                    df['kronos_confidence'] = 0.0
                    print("[Kronos] No cache found, using zeros")
                    return df
            else:
                df['kronos_pred_return'] = 0.0
                df['kronos_confidence'] = 0.0
                print("[Kronos] No cache found, using zeros")
                return df
        
        kronos_df = pd.read_parquet(cache_path)
        kronos_df['date'] = pd.to_datetime(kronos_df['date'])
        
        n_before = len(df)
        merge_cols = ['kronos_pred_return', 'kronos_confidence']
        for c in merge_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        
        df = df.merge(
            kronos_df[['symbol', 'date'] + merge_cols],
            on=['symbol', 'date'], how='left'
        )
        
        df['kronos_pred_return'] = df['kronos_pred_return'].fillna(0.0)
        df['kronos_confidence'] = df['kronos_confidence'].fillna(0.0)
        
        assert len(df) == n_before, f"Kronos merge changed rows: {n_before} → {len(df)}"
        
        n_active = (df['kronos_pred_return'].abs() > 0.001).sum()
        print(f"[Kronos] Merged, {n_active}/{len(df)} rows have active predictions")
        
        return df
