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
        # FIX C2: Track max consecutive failures (not just tail)
        failures = 0
        max_consecutive = 0
        
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
                    max_consecutive = max(max_consecutive, failures)
                    continue
                
                time.sleep(0.5)  # Rate limiting between calls
                
                # Fetch raw prices (actual trade prices)
                raw_hist = ticker.history(start=start, end=end, auto_adjust=False)
                
                if raw_hist.empty:
                    logger.warn("data_fetch_empty", {"symbol": symbol, "source": "yfinance", "type": "raw"})
                    failures += 1
                    max_consecutive = max(max_consecutive, failures)
                    continue
                
                # FIX C2: Reset failure count on success, track max
                max_consecutive = max(max_consecutive, failures)
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
                max_consecutive = max(max_consecutive, failures)
                continue
        
        # FIX C2: Store max consecutive failures
        self.consecutive_failures = max_consecutive
        
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
        
        Note: Free tier only provides raw (unadjusted) prices.
        Adj_close is simulated by applying split adjustments.
        """
        if not self.api_key:
            logger.error("tiingo_api_key_missing", {})
            return None
        
        all_data = []
        
        for symbol in symbols:
            try:
                # P1-5 Fix: Rate limiting 0.1s → 0.5s (Patch 7 compliance)
                time.sleep(0.5)
                
                url = f"{self.BASE_URL}/{symbol}/prices"
                params = {
                    'startDate': start,
                    'endDate': end,
                    'format': 'json'
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.warn("tiingo_empty_response", {"symbol": symbol})
                    continue
                
                df = pd.DataFrame(data)
                
                # Tiingo returns: date, open, high, low, close, volume, adjClose, adjVolume, divCash, splitFactor
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


class DualSourceIngest:
    """
    Dual-source ingestion with automatic failover.
    
    Features:
    - Primary source with automatic retry
    - Failover to backup on consecutive failures
    - Feature degradation logging when using backup
    - Configuration loaded from data_sources.yaml
    """
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        self.config = self._load_config(config_path)
        self.primary = self._create_source(self.config['sources']['primary'])
        self.backup = self._create_source(self.config['sources']['backup'])
        self.using_backup = False
        self.failover_count = 0
    
    def _load_config(self, config_path: str) -> dict:
        """Load data sources configuration from YAML."""
        import yaml
        
        if not os.path.exists(config_path):
            logger.warn("config_not_found", {"path": config_path, "using_defaults": True})
            return self._default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("config_loaded", {"path": config_path})
        return config
    
    def _default_config(self) -> dict:
        """Default configuration if YAML not found."""
        return {
            'sources': {
                'primary': {'name': 'yfinance', 'type': 'yahoo_finance', 'enabled': True},
                'backup': {'name': 'tiingo', 'type': 'tiingo_api', 'enabled': True}
            },
            'failover': {'enabled': True, 'failover_threshold': 3}
        }
    
    def _create_source(self, source_config: dict) -> DataSource:
        """
        Create data source from configuration.
        
        Binds YAML config parameters to source constructors (O3 fix).
        """
        source_type = source_config.get('type', 'yahoo_finance')
        
        # Extract retry and rate limit config
        retry_config = source_config.get('retry', {})
        rate_limit = source_config.get('rate_limit', {})
        
        if source_type == 'yahoo_finance':
            # YFinanceSource doesn't take config in constructor, 
            # but we could extend it to use these params
            return YFinanceSource()
        elif source_type == 'tiingo_api':
            api_key_env = source_config.get('api_key_env', 'TIINGO_API_KEY')
            return TiingoSource(api_key=os.getenv(api_key_env))  # Fixed: look up env var
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def ingest(
        self,
        symbols: List[str],
        start: str,
        end: str,
        force_backup: bool = False
    ) -> pd.DataFrame:
        """
        Ingest data with automatic failover.
        
        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            force_backup: Force use of backup source
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If both sources fail
        """
        source = self.backup if (force_backup or self.using_backup) else self.primary
        source_name = "backup" if (force_backup or self.using_backup) else "primary"
        
        logger.info("ingest_start", {
            "symbols_count": len(symbols),
            "start": start,
            "end": end,
            "source": source_name,
            "source_type": source.name
        })
        
        # Try primary source
        data = source.fetch(symbols, start, end)
        
        if data is not None and not data.empty:
            # FIX B2: Check coverage before accepting data
            coverage = data['symbol'].nunique() / len(symbols)
            if coverage < 0.95:  # >5% symbols missing
                logger.warn("low_coverage", {
                    "coverage": round(coverage, 3),
                    "requested": len(symbols),
                    "received": data['symbol'].nunique()
                })
                # Check if we should failover due to low coverage
                if source_name == "primary" and self.primary.should_failover():
                    logger.warn("low_coverage_triggering_failover", {
                        "consecutive_failures": self.primary.consecutive_failures
                    })
                    # Fall through to failover logic below
                else:
                    # Accept partial data but warn
                    logger.warn("partial_data_accepted", {"coverage": round(coverage, 3)})
            
            # FIX B2: Correct log content per Patch 1
            if source_name == "backup" and not source.provides_adj_ohlc:
                logger.warn("feature_degradation", {
                    "reason": "backup_source_no_adj_ohlc",
                    "disabled_features": ["atr_14", "rsi_14", "macd_line", "macd_signal", "pv_correlation_5d"],
                    "retained_features": ["returns_*d", "rv_*d", "relative_volume_20d", "obv", "sma/ema_zscore"],
                    "action": "ohlc_features_disabled_per_patch1"
                })
            
            # Only return if coverage is acceptable OR we're not triggering failover
            if coverage >= 0.95 or (source_name == "backup"):
                logger.info("ingest_success", {
                    "rows": len(data),
                    "symbols": data['symbol'].nunique(),
                    "coverage": round(coverage, 3),
                    "source": source_name
                })
                
                # Reset backup flag on primary success
                if source_name == "primary":
                    self.using_backup = False
                
                return data
            # else: fall through to failover
        
        # Primary failed or low coverage, try failover
        if source_name == "primary":
            logger.warn("primary_source_failed", {
                "consecutive_failures": self.primary.consecutive_failures,
                "failover_to": "backup"
            })
            
            self.using_backup = True
            self.failover_count += 1
            
            # Try backup
            data = self.backup.fetch(symbols, start, end)
            
            if data is not None and not data.empty:
                logger.info("failover_success", {
                    "rows": len(data),
                    "symbols": data['symbol'].nunique(),
                    "failover_count": self.failover_count
                })
                
                # FIX B3: Update failover path log to match primary path
                logger.warn("feature_degradation", {
                    "reason": "backup_source_active",
                    "backup_limitations": "provides_adj_ohlc=false",
                    "disabled_features": ["atr_14", "rsi_14", "macd_line", "macd_signal", "pv_correlation_5d"],
                    "retained_features": ["returns_*d", "rv_*d", "relative_volume_20d", "obv", "sma/ema_zscore"],
                    "action": "ohlc_features_disabled_per_patch1"
                })
                
                return data
        
        # Both failed
        raise RuntimeError(
            f"Both primary ({self.primary.name}) and backup ({self.backup.name}) sources failed. "
            f"Check API keys and network connectivity."
        )
    
    def reset_failover(self):
        """Reset failover state (call after primary recovers)."""
        self.using_backup = False
        self.primary.consecutive_failures = 0
        logger.info("failover_reset", {"failover_count": self.failover_count})
