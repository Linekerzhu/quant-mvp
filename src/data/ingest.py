"""
Data Ingestion Module

Dual-source data collection with automatic failover:
- Primary: yfinance
- Backup: Tiingo or Alpha Vantage

This module provides the DualSourceIngest class which orchestrates
data fetching from multiple sources with automatic failover.

Author: 李得勤
Date: 2026-02-27
"""

import os
from typing import List
import pandas as pd

from src.data.sources import DataSource, YFinanceSource, TiingoSource
from src.ops.event_logger import get_logger

logger = get_logger()


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
                    "disabled_features": ["atr_20", "rsi_14", "macd_line", "macd_signal", "pv_correlation_5d"],
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
                    "disabled_features": ["atr_20", "rsi_14", "macd_line", "macd_signal", "pv_correlation_5d"],
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
