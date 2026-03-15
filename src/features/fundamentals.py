"""
Fundamental + Sentiment Factor Provider — v5.1

Downloads and caches fundamental data + analyst sentiment from yfinance.
Designed for monthly rebalancing.

Fundamental Factors:
  1. Earnings Yield = 1/PE  (cheapness)
  2. ROE = Return on Equity  (profitability)
  3. Profit Margin  (quality)
  4. Earnings Growth  (growth)
  5. Low Debt = -Debt/Equity  (safety)

Market Sentiment Factors:
  6. Analyst Upside = target price / current - 1  (Street consensus)
  7. Analyst Consensus = 5 - recommendation score  (conviction, higher=more bullish)
  8. Earnings Surprise = quarterly earnings growth  (beat/miss momentum)

Usage:
    from src.features.fundamentals import FundamentalProvider
    fp = FundamentalProvider()
    fund_df = fp.get_fundamentals(symbols, date="2026-03-14")
    # Returns DataFrame with columns: symbol, earnings_yield, roe, ...
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)


class FundamentalProvider:
    """Fetch and cache fundamental data for S&P500 stocks."""
    
    CACHE_DIR = os.path.join(PROJECT_ROOT, "data/cache/fundamentals")
    CACHE_TTL_DAYS = 7  # Refresh fundamentals weekly
    
    FACTOR_COLS = [
        # Fundamental quality
        "earnings_yield",     # 1/PE — cheapness
        "roe",                # Return on equity — profitability
        "profit_margin",      # Net margin — quality
        "earnings_growth",    # YoY earnings growth
        "revenue_growth",     # YoY revenue growth
        "low_debt",           # -Debt/Equity — financial health
        # Market sentiment
        "analyst_upside",     # Target price upside %
        "analyst_consensus",  # 4=strong buy, 0=sell
        "earnings_surprise",  # Recent quarter earnings growth
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or self.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_fundamentals(
        self, 
        symbols: List[str], 
        date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get fundamental factors for a list of symbols.
        Uses cached data if fresh enough, otherwise downloads from yfinance.
        
        Returns DataFrame with columns: symbol + FACTOR_COLS
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        cache_path = os.path.join(self.cache_dir, f"fundamentals_{date[:7]}.parquet")
        
        # Check cache
        if not force_refresh and os.path.exists(cache_path):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if cache_age.days < self.CACHE_TTL_DAYS:
                cached = pd.read_parquet(cache_path)
                # Check if we have most of the requested symbols
                cached_syms = set(cached["symbol"].tolist())
                missing = [s for s in symbols if s not in cached_syms]
                if len(missing) < len(symbols) * 0.1:  # < 10% missing
                    print(f"[FUND] Using cached fundamentals ({len(cached)} symbols, {cache_age.days}d old)")
                    return cached[cached["symbol"].isin(symbols)]
        
        # Download fresh data
        print(f"[FUND] Downloading fundamentals for {len(symbols)} symbols...")
        return self._download_fundamentals(symbols, cache_path)
    
    def _download_fundamentals(self, symbols: List[str], cache_path: str) -> pd.DataFrame:
        """Download fundamental data from yfinance in batches."""
        try:
            import yfinance as yf
        except ImportError:
            print("[FUND] yfinance not installed, returning empty")
            return pd.DataFrame(columns=["symbol"] + self.FACTOR_COLS)
        
        results = []
        failed = []
        
        for i, sym in enumerate(symbols):
            if i > 0 and i % 50 == 0:
                print(f"[FUND]   Progress: {i}/{len(symbols)}")
                time.sleep(0.5)  # Rate limiting
            
            try:
                t = yf.Ticker(sym)
                info = t.info
                
                # --- Fundamental metrics ---
                pe = info.get("trailingPE")
                fwd_pe = info.get("forwardPE")
                roe = info.get("returnOnEquity")
                profit_margin = info.get("profitMargins")
                debt_equity = info.get("debtToEquity")
                earnings_growth = info.get("earningsGrowth")
                revenue_growth = info.get("revenueGrowth")
                
                # Derived: Earnings Yield
                best_pe = fwd_pe if fwd_pe and fwd_pe > 0 else pe
                earnings_yield = 1.0 / best_pe if best_pe and best_pe > 0 else None
                
                # Derived: Low Debt (negate so higher = healthier)
                low_debt = -(debt_equity / 100.0) if debt_equity is not None else None
                
                # --- Market sentiment metrics ---
                # Analyst target price upside
                target_price = info.get("targetMeanPrice")
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                analyst_upside = None
                if target_price and current_price and current_price > 0:
                    analyst_upside = (target_price / current_price) - 1.0
                
                # Analyst consensus (1=StrongBuy → 5=Sell, invert to higher=better)
                rec_mean = info.get("recommendationMean")
                analyst_consensus = (5.0 - rec_mean) if rec_mean else None
                
                # Earnings surprise (latest quarter YoY growth)
                earnings_surprise = info.get("earningsQuarterlyGrowth")
                
                results.append({
                    "symbol": sym,
                    # Fundamental
                    "earnings_yield": earnings_yield,
                    "roe": roe,
                    "profit_margin": profit_margin,
                    "earnings_growth": earnings_growth,
                    "revenue_growth": revenue_growth,
                    "low_debt": low_debt,
                    # Sentiment
                    "analyst_upside": analyst_upside,
                    "analyst_consensus": analyst_consensus,
                    "earnings_surprise": earnings_surprise,
                    # Raw reference
                    "_pe": pe,
                    "_fwd_pe": fwd_pe,
                    "_debt_equity": debt_equity,
                    "_rec_mean": rec_mean,
                    "_target_price": target_price,
                    "_n_analysts": info.get("numberOfAnalystOpinions"),
                })
                
            except Exception as e:
                failed.append(sym)
                results.append({"symbol": sym})
        
        df = pd.DataFrame(results)
        
        # Save cache
        df.to_parquet(cache_path, index=False)
        
        n_valid = df[self.FACTOR_COLS].notna().all(axis=1).sum()
        print(f"[FUND] Downloaded {len(df)} symbols, {n_valid} fully valid, {len(failed)} failed")
        if failed:
            print(f"[FUND]   Failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        return df
    
    @staticmethod
    def compute_composite(
        df: pd.DataFrame,
        weights: dict = None,
    ) -> pd.Series:
        """
        Compute cross-sectional composite fundamental + sentiment score.
        
        Default weights (quality + market heat balanced):
          20% Earnings Yield  (cheapness)
          15% ROE  (profitability)
          10% Profit Margin  (quality)
          10% Earnings Growth  (growth)
           5% Low Debt  (safety)
          20% Analyst Upside  (Street conviction — what's being chased)
          10% Analyst Consensus  (buy/sell rating)
          10% Earnings Surprise  (beat/miss momentum)
        """
        if weights is None:
            weights = {
                # Fundamental quality (60%)
                "earnings_yield": 0.20,
                "roe": 0.15,
                "profit_margin": 0.10,
                "earnings_growth": 0.10,
                "low_debt": 0.05,
                # Market sentiment (40%)
                "analyst_upside": 0.20,
                "analyst_consensus": 0.10,
                "earnings_surprise": 0.10,
            }
        
        composite = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                ranked = df[col].rank(pct=True, na_option="bottom")
                composite += w * ranked
        
        return composite
