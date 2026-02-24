"""
Data Integrity Module

Hash freezing and historical drift detection.
Implements Plan v4 patch requirements for RawClose + adj_factor.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.ops.event_logger import get_logger, EventLevel

logger = get_logger()


class IntegrityManager:
    """Manages data integrity through hashing and drift detection."""
    
    def __init__(
        self,
        hash_file: str = "data/processed/integrity_hashes.parquet",
        snapshot_dir: str = "data/snapshots"
    ):
        self.hash_file = Path(hash_file)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing hashes
        self.hashes = self._load_hashes()
    
    def _load_hashes(self) -> pd.DataFrame:
        """Load existing hash records."""
        if self.hash_file.exists():
            df = pd.read_parquet(self.hash_file)
            # FIX B1: Set index for O(1) lookup instead of O(n) filter
            if not df.empty:
                df = df.set_index(['symbol', 'date'])
            return df
        # Return empty DataFrame with MultiIndex structure
        return pd.DataFrame(
            columns=['adj_hash', 'raw_hash', 'adj_factor_hash', 'timestamp']
        ).set_index(pd.MultiIndex.from_arrays([[], []], names=['symbol', 'date']))
    
    def _compute_row_hash(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Compute content hashes for a data row.
        
        Returns:
            (adj_hash, raw_hash, adj_factor_hash)
        """
        # Adjusted price fields
        adj_data = {
            'adj_open': row.get('adj_open'),
            'adj_high': row.get('adj_high'),
            'adj_low': row.get('adj_low'),
            'adj_close': row.get('adj_close'),
            'volume': row.get('volume')
        }
        adj_hash = hashlib.sha256(
            json.dumps(adj_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        # Raw close
        raw_data = {'raw_close': row.get('raw_close')}
        raw_hash = hashlib.sha256(
            json.dumps(raw_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        # Adjustment factor (adj_close / raw_close)
        adj_close = row.get('adj_close')
        raw_close = row.get('raw_close')
        if adj_close is not None and raw_close is not None and raw_close != 0:
            adj_factor = adj_close / raw_close
        else:
            adj_factor = 1.0
        
        adj_factor_hash = hashlib.sha256(
            json.dumps({'adj_factor': round(adj_factor, 6)}, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return adj_hash, raw_hash, adj_factor_hash
    
    def freeze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and store hashes for new data.
        
        Args:
            df: DataFrame with symbol, date, price columns
            
        Returns:
            DataFrame with hash columns added
        """
        records = []
        
        for _, row in df.iterrows():
            adj_hash, raw_hash, adj_factor_hash = self._compute_row_hash(row)
            
            records.append({
                'symbol': row['symbol'],
                'date': pd.Timestamp(row['date']).strftime('%Y-%m-%d'),
                'adj_hash': adj_hash,
                'raw_hash': raw_hash,
                'adj_factor_hash': adj_factor_hash,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        new_hashes = pd.DataFrame(records)
        
        # Merge with existing hashes
        self.hashes = pd.concat([self.hashes, new_hashes], ignore_index=True)
        self.hashes = self.hashes.drop_duplicates(
            subset=['symbol', 'date'], keep='last'
        )
        
        # Save using WAP (O1 fix)
        from src.data.wap_utils import write_parquet_wap
        write_parquet_wap(self.hashes, self.hash_file)
        
        logger.info("data_frozen", {"records": len(new_hashes)})
        
        return df
    
    def detect_drift(
        self,
        df: pd.DataFrame,
        universe_size: int
    ) -> Tuple[bool, List[Dict]]:
        """
        Detect historical data drift by comparing with stored hashes.
        
        Args:
            df: New data to compare
            universe_size: Current universe size for adaptive threshold
            
        Returns:
            (should_freeze, drift_events)
        """
        if self.hashes.empty:
            logger.info("no_historical_hashes", {"action": "skip_drift_detection"})
            return False, []
        
        drift_events = []
        
        # Per-symbol-day drift tracking
        symbol_day_drifts = {}
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            date = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
            
            # Compute new hashes
            adj_hash, raw_hash, adj_factor_hash = self._compute_row_hash(row)
            
            # FIX B1: Use index lookup O(1) instead of filter O(n)
            try:
                stored = self.hashes.loc[(symbol, date)]
            except KeyError:
                continue  # New data point
            
            # Check for drift in any hash
            if (stored['adj_hash'] != adj_hash or
                stored['raw_hash'] != raw_hash or
                stored['adj_factor_hash'] != adj_factor_hash):
                
                drift_type = []
                if stored['adj_hash'] != adj_hash:
                    drift_type.append('adj')
                if stored['raw_hash'] != raw_hash:
                    drift_type.append('raw')
                if stored['adj_factor_hash'] != adj_factor_hash:
                    drift_type.append('adj_factor')
                
                event = {
                    'symbol': symbol,
                    'date': date,
                    'drift_type': drift_type,
                    'stored_at': stored['timestamp']
                }
                drift_events.append(event)
                
                # Track per-symbol consecutive days
                if symbol not in symbol_day_drifts:
                    symbol_day_drifts[symbol] = []
                symbol_day_drifts[symbol].append(date)
        
        # Determine severity (Plan v4 adaptive threshold)
        drift_threshold = max(10, int(0.01 * universe_size))
        
        # FIX B1: Separate raw vs adj drift - only raw drift should trigger freeze
        # adj drift is expected due to yfinance retroactive dividend adjustments
        raw_drift_events = [e for e in drift_events if 'raw' in e['drift_type']]
        adj_only_events = [e for e in drift_events if e['drift_type'] == ['adj'] or e['drift_type'] == ['adj', 'adj_factor']]
        
        if adj_only_events:
            logger.info("adj_recalc_expected", {
                "count": len(adj_only_events),
                "note": "Dividend adjustments are retroactively applied by yfinance"
            })
        
        # Use only raw drift for freeze decision
        should_freeze = False
        
        if len(drift_events) == 0:
            logger.info("drift_check_passed", {"drifts": 0})
        elif len(raw_drift_events) == 0 and len(adj_only_events) > 0:
            # Only adj drift - INFO only, no freeze
            logger.info("adj_only_drift_no_freeze", {
                "adj_drifts": len(adj_only_events)
            })
        elif len(raw_drift_events) == 1:
            # Single day raw drift - WARN only
            logger.warn("single_day_raw_drift", raw_drift_events[0], raw_drift_events[0]['symbol'])
        else:
            # FIX B1: Rebuild symbol_day_drifts from raw_drift_events only
            # (exclude adj-only events to prevent false positive freeze)
            raw_symbol_day_drifts = {}
            for e in raw_drift_events:
                raw_symbol_day_drifts.setdefault(e['symbol'], []).append(e['date'])
            
            # FIXED A25: Check for consecutive TRADING days per symbol (not calendar days)
            max_consecutive = 0
            for symbol, dates in raw_symbol_day_drifts.items():
                sorted_dates = sorted([pd.Timestamp(d) for d in dates])
                consecutive = 1
                for i in range(1, len(sorted_dates)):
                    d1 = sorted_dates[i-1]
                    d2 = sorted_dates[i]
                    # Use np.busday_count for trading day calculation
                    trading_days = np.busday_count(d1.strftime('%Y-%m-%d'), d2.strftime('%Y-%m-%d'))
                    if trading_days <= 1:  # Within 1 trading day (consecutive)
                        consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, consecutive)
                        consecutive = 1
                max_consecutive = max(max_consecutive, consecutive)
            
            # Check universe threshold - use raw_drift only
            unique_raw_symbols = len(set(e['symbol'] for e in raw_drift_events))
            
            if max_consecutive >= 5:
                # ERROR: Same symbol 5+ consecutive days
                logger.error("consecutive_raw_drift", {
                    "max_consecutive": max_consecutive,
                    "symbols": list(raw_symbol_day_drifts.keys())[:5]
                })
                should_freeze = True
            elif unique_raw_symbols >= drift_threshold:
                # ERROR: Universe threshold
                logger.error("universe_raw_drift_threshold", {
                    "drift_symbols": unique_raw_symbols,
                    "threshold": drift_threshold
                })
                should_freeze = True
            else:
                # Multiple single-day drifts - WARN
                logger.warn("multiple_single_day_raw_drifts", {
                    "count": len(raw_drift_events),
                    "symbols": unique_raw_symbols
                })
        
        return should_freeze, drift_events
    
    def create_snapshot(self, df: pd.DataFrame, version: str) -> Path:
        """
        Create a versioned data snapshot.
        
        Args:
            df: Data to snapshot
            version: Version string (e.g., '2024-06-15_v1')
            
        Returns:
            Path to snapshot file
        """
        snapshot_path = self.snapshot_dir / f"data_{version}.parquet"
        
        # P1-2 Fix: Use centralized WAP utility
        from src.data.wap_utils import write_parquet_wap
        write_parquet_wap(df, snapshot_path)
        
        logger.info("snapshot_created", {"version": version, "path": str(snapshot_path)})
        
        return snapshot_path
