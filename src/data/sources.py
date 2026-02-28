"""
Data Sources Module

Contains data source implementations for dual-source ingestion:
- YFinanceSource: Primary source (yfinance)
- TiingoSource: Backup source (Tiingo API)

Author: 李得勤
Date: 2026-02-28
"""

import os
import time
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

try:
    import requests_cache
    REQUESTS_CACHE_AVAILABLE = True
except ImportError:
    REQUESTS_CACHE_AVAILABLE = False

from src.ops.event_logger import get_logger, EventLevel

logger = get_logger()


class DataSource:
    """Base class for data sources."""
    
    name: str = "base"
    provides_adj_ohlc: bool = True
    
    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for symbols. Returns None on failure."""
        raise NotImplementedError
    
    def should_failover(self) -> bool:
        """Check if should failover to backup."""
        return False


class YFinanceSource(DataSource):
    """yfinance data source (primary)."""
    
    name = "yfinance"
    provides_adj_ohlc = True
    
    def __init__(self):
        self.session = self._create_session()
        self.consecutive_failures = 0
        self.max_failures_before_failover = 3
    
    def _create_session(self, use_cache: bool = False):
        """
        Create session with retry logic.
        
        P1-3: Supports requests-cache for development/debugging.
        """
        # P1-3: Use CachedSession if available and enabled
        if use_cache and REQUESTS_CACHE_AVAILABLE:
            session = requests_cache.CachedSession(
                cache_name='data/cache/yfinance_cache',
                backend='sqlite',
                expire_after=3600  # 1 hour cache
            )
            logger.info("using_cached_session", {"cache": "sqlite"})
        else:
            session = requests.Session()
        
        # Exponential backoff retry (Plan v4 patch)
        retry = Retry(
            total=5,
            backoff_factor=1,  # 1, 2, 4, 8, 16 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_max=60  # Max 60 seconds (urllib3 2.x param name)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from yfinance with rate limiting.
        
        Fetches both adjusted and raw prices for proper corporate action detection.
        """
        all_data = []
        # FIX A1: Track current consecutive failures (reset on success)
        failures = 0
        
        for symbol in symbols:
            try:
                # Rate limiting: min 0.5s between requests (Plan v4 patch)
                time.sleep(0.5)
                
                ticker = yf.Ticker(symbol, session=self.session)
                
                # Fetch adjusted prices (backward-adjusted for splits/dividends)
                adj_hist = ticker.history(start=start, end=end, auto_adjust=True)
                
                if adj_hist.empty:
                    logger.warn("data_fetch_empty", {"symbol": symbol, "source": "yfinance", "type": "adj"})
                    failures += 1
                    continue
                
                time.sleep(0.5)  # Rate limiting between calls
                
                # Fetch raw prices (actual trade prices)
                raw_hist = ticker.history(start=start, end=end, auto_adjust=False)
                
                if raw_hist.empty:
                    logger.warn("data_fetch_empty", {"symbol": symbol, "source": "yfinance", "type": "raw"})
                    failures += 1
                    continue
                
                # FIX C2: Reset failure count on success
                failures = 0
                
                # Merge adjusted and raw data
                adj_hist = adj_hist.reset_index()
                raw_hist = raw_hist.reset_index()
                
                # Standardize column names
                adj_hist = adj_hist.rename(columns={
                    'Open': 'adj_open',
                    'High': 'adj_high',
                    'Low': 'adj_low',
                    'Close': 'adj_close',
                    'Volume': 'volume',
                    'Date': 'date'
                })
                
                raw_hist = raw_hist.rename(columns={
                    'Open': 'raw_open',
                    'High': 'raw_high',
                    'Low': 'raw_low',
                    'Close': 'raw_close',
                    'Date': 'date'
                })
                
                # Merge on date
                merged = pd.merge(
                    adj_hist[['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']],
                    raw_hist[['date', 'raw_open', 'raw_high', 'raw_low', 'raw_close']],
                    on='date',
                    how='inner'
                )
                
                merged['symbol'] = symbol
                
                # Mark that yfinance provides full adj OHLC
                merged['source_provides_adj_ohlc'] = True
                
                # A24: Add ingestion timestamp for PIT tracking
                merged['ingestion_timestamp'] = pd.Timestamp.now()
                
                all_data.append(merged[['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 
                                      'raw_close', 'adj_open', 'adj_high', 'adj_low', 
                                      'adj_close', 'volume', 'ingestion_timestamp', 'source_provides_adj_ohlc']])
                
                logger.info("data_fetch_success", {
                    "symbol": symbol, 
                    "rows": len(merged),
                    "date_range": f"{merged['date'].min()} to {merged['date'].max()}",
                    "source": "yfinance"
                }, symbol)
                
            except Exception as e:
                logger.error("data_fetch_error", {"symbol": symbol, "error": str(e), "source": "yfinance"})
                failures += 1
                continue
        
        # FIX A1: Use current consecutive failures (not max), reset to 0 on success above
        self.consecutive_failures = failures
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)
    
    def should_failover(self) -> bool:
        """Check if should failover to backup."""
        return self.consecutive_failures >= self.max_failures_before_failover


class TiingoSource(DataSource):
    """
    Tiingo API data source (backup).
    
    LIMITATION: Tiingo free tier does not provide adjusted OHLC.
    When using backup, features requiring AdjOHLC are degraded.
    """
    
    name = "tiingo"
    provides_adj_ohlc = False  # Critical limitation
    
    BASE_URL = "https://api.tiingo.com/tiingo/daily"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TIINGO_API_KEY')
        if not self.api_key:
            logger.warn("tiingo_api_key_not_set", {})
        
        self.session = self._create_session()
    
    def _create_session(self):
        """Create session with headers."""
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        })
        
        # Retry configuration
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        
        return session
    
    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Tiingo.
        
        Tiingo provides adjusted close but not adjusted OHLC.
        """
        if not self.api_key:
            logger.error("tiingo_no_api_key", {})
            return None
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Rate limiting
                time.sleep(1.0)  # Tiingo is stricter
                
                url = f"{self.BASE_URL}/{symbol}"
                params = {
                    'startDate': start,
                    'endDate': end,
                    'format': 'json'
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.warn("tiingo_no_data", {"symbol": symbol})
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Standardize column names
                df = df.rename(columns={
                    'date': 'date',
                    'open': 'raw_open',
                    'high': 'raw_high',
                    'low': 'raw_low',
                    'close': 'raw_close',
                    'volume': 'volume',
                    'adjClose': 'adj_close',
                    'adjVolume': 'adj_volume'
                })
                
                # Parse date
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                
                # Tiingo provides adjClose but not full adj OHLC (Patch 1 compliance)
                # DO NOT approximate adj_open/high/low - set to NaN to disable OHLC features
                df['adj_open'] = np.nan
                df['adj_high'] = np.nan
                df['adj_low'] = np.nan
                
                # Mark that this source doesn't provide reliable adj OHLC
                df['source_provides_adj_ohlc'] = False
                
                df['symbol'] = symbol
                
                # A24: Add ingestion timestamp for PIT tracking
                df['ingestion_timestamp'] = pd.Timestamp.now()
                
                # Select columns to match yfinance format
                df = df[['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 
                        'raw_close', 'adj_open', 'adj_high', 'adj_low', 
                        'adj_close', 'volume', 'ingestion_timestamp', 'source_provides_adj_ohlc']]
                
                all_data.append(df)
                
                logger.info("tiingo_fetch_success", {
                    "symbol": symbol,
                    "rows": len(df),
                    "source": "tiingo"
                }, symbol)
                
            except Exception as e:
                logger.error("tiingo_fetch_error", {"symbol": symbol, "error": str(e)})
                continue
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)
