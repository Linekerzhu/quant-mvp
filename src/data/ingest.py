"""
Data Ingestion Module

Dual-source data collection with automatic failover:
- Primary: yfinance
- Backup: Tiingo or Alpha Vantage
"""

import os
import time
from typing import Optional, List, Dict
import pandas as pd
import yfinance as yf
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.ops.event_logger import get_logger, EventLevel

logger = get_logger()


class DataSource:
    """Base class for data sources."""
    
    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for symbols. Returns None on failure."""
        raise NotImplementedError


class YFinanceSource(DataSource):
    """yfinance data source (primary)."""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self):
        """Create session with retry logic."""
        import requests
        session = requests.Session()
        
        # Exponential backoff retry (Plan v4 patch)
        retry = Retry(
            total=5,
            backoff_factor=1,  # 1, 2, 4, 8, 16 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            max_backoff=60  # Max 60 seconds
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
        
        for symbol in symbols:
            try:
                # Rate limiting: min 0.5s between requests (Plan v4 patch)
                time.sleep(0.5)
                
                ticker = yf.Ticker(symbol, session=self.session)
                
                # Fetch adjusted prices (backward-adjusted for splits/dividends)
                adj_hist = ticker.history(start=start, end=end, auto_adjust=True)
                
                if adj_hist.empty:
                    logger.warn("data_fetch_empty", {"symbol": symbol, "source": "yfinance", "type": "adj"})
                    continue
                
                time.sleep(0.5)  # Rate limiting between calls
                
                # Fetch raw prices (actual trade prices)
                raw_hist = ticker.history(start=start, end=end, auto_adjust=False)
                
                if raw_hist.empty:
                    logger.warn("data_fetch_empty", {"symbol": symbol, "source": "yfinance", "type": "raw"})
                    continue
                
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
                
                all_data.append(merged[['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 
                                      'raw_close', 'adj_open', 'adj_high', 'adj_low', 
                                      'adj_close', 'volume']])
                
                logger.info("data_fetch_success", {
                    "symbol": symbol, 
                    "rows": len(merged),
                    "date_range": f"{merged['date'].min()} to {merged['date'].max()}"
                }, symbol)
                
            except Exception as e:
                logger.error("data_fetch_error", {"symbol": symbol, "error": str(e), "source": "yfinance"})
                continue
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)


class BackupSource(DataSource):
    """Backup data source (Tiingo or Alpha Vantage)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TIINGO_API_KEY')
    
    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """Placeholder for backup source implementation."""
        # TODO: Implement Tiingo or Alpha Vantage fallback
        logger.warn("backup_source_not_implemented", {"symbols": symbols})
        return None


class DualSourceIngest:
    """Dual-source ingestion with automatic failover."""
    
    def __init__(self):
        self.primary = YFinanceSource()
        self.backup = BackupSource()
        self.failover_delay = 60  # Seconds before failover
    
    def ingest(
        self,
        symbols: List[str],
        start: str,
        end: str,
        use_backup: bool = False
    ) -> pd.DataFrame:
        """
        Ingest data with automatic failover.
        
        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            use_backup: Force use of backup source
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If both sources fail
        """
        source_name = "backup" if use_backup else "primary"
        source = self.backup if use_backup else self.primary
        
        logger.info("ingest_start", {
            "symbols_count": len(symbols),
            "start": start,
            "end": end,
            "source": source_name
        })
        
        data = source.fetch(symbols, start, end)
        
        if data is None and not use_backup:
            # Failover to backup
            logger.warn("primary_source_failed", {"failover_delay": self.failover_delay})
            time.sleep(self.failover_delay)
            return self.ingest(symbols, start, end, use_backup=True)
        
        if data is None:
            raise RuntimeError("Both primary and backup sources failed")
        
        logger.info("ingest_complete", {
            "rows": len(data),
            "symbols": data['symbol'].nunique(),
            "source": source_name
        })
        
        return data
