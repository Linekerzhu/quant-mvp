import numpy as np
import pandas as pd

def add_advanced_features(df):
    """Add advanced technical and statistical features."""
    print("[V2.2] Adding advanced features...")
    df = df.copy()
    g = df.groupby('symbol')['adj_close']
    
    # 1. Volatility Regime (20d vs 60d)
    df['vol_regime'] = df['rv_20d'] / (df['rv_60d'] + 1e-8)
    
    # 2. Price Position in Recent Range (Stochastic-like)
    high_20 = df.groupby('symbol')['adj_high'].transform(lambda x: x.rolling(20).max())
    low_20 = df.groupby('symbol')['adj_low'].transform(lambda x: x.rolling(20).min())
    df['price_range_pos'] = (df['adj_close'] - low_20) / (high_20 - low_20 + 1e-8)
    df['price_range_pos'] = df['price_range_pos'].clip(0, 1)  # Prevent extreme values when high==low
    df.loc[high_20.isna(), 'price_range_pos'] = 0.5
    
    # 3. ATR normalized by price
    df['atr_pct'] = df['atr_20'] / df['adj_close']
    
    # 4. Moving Average Cross Strength
    sma_20 = g.transform(lambda x: x.rolling(20).mean())
    sma_60 = g.transform(lambda x: x.rolling(60).mean())
    df['sma_cross_strength'] = (sma_20 - sma_60) / (sma_60 + 1e-8)
    
    # 5. Volume Trend Acceleration
    vol_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    vol_60 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(60).mean())
    df['vol_trend_accel'] = vol_20 / (vol_60 + 1e-8)
    
    print(f"  Added 5 new base features")
    return df
