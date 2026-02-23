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
            return pd.read_parquet(self.hash_file)
        return pd.DataFrame(columns=[
            'symbol', 'date', 'adj_hash', 'raw_hash', 'adj_factor_hash',
            'timestamp'
        ])
    
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
        
        # Save
        self.hashes.to_parquet(self.hash_file, index=False)
        
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
            
            # Check against stored
            stored = self.hashes[
                (self.hashes['symbol'] == symbol) &
                (self.hashes['date'] == date)
            ]
            
            if stored.empty:
                continue  # New data point
            
            stored = stored.iloc[0]
            
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
        
        should_freeze = False
        
        if len(drift_events) == 0:
            logger.info("drift_check_passed", {"drifts": 0})
        elif len(drift_events) == 1:
            # Single day drift - WARN only
            logger.warn("single_day_drift", drift_events[0], drift_events[0]['symbol'])
        else:
            # Check for consecutive days per symbol
            max_consecutive = 0
            for symbol, dates in symbol_day_drifts.items():
                sorted_dates = sorted(dates)
                consecutive = 1
                for i in range(1, len(sorted_dates)):
                    d1 = pd.Timestamp(sorted_dates[i-1])
                    d2 = pd.Timestamp(sorted_dates[i])
                    if (d2 - d1).days <= 3:  # Within 3 trading days
                        consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, consecutive)
                        consecutive = 1
                max_consecutive = max(max_consecutive, consecutive)
            
            # Check universe threshold
            unique_symbols = len(set(e['symbol'] for e in drift_events))
            
            if max_consecutive >= 5:
                # ERROR: Same symbol 5+ consecutive days
                logger.error("consecutive_drift", {
                    "max_consecutive": max_consecutive,
                    "symbols": list(symbol_day_drifts.keys())[:5]
                })
                should_freeze = True
            elif unique_symbols >= drift_threshold:
                # ERROR: Universe threshold
                logger.error("universe_drift_threshold", {
                    "drift_symbols": unique_symbols,
                    "threshold": drift_threshold
                })
                should_freeze = True
            else:
                # Multiple single-day drifts - WARN
                logger.warn("multiple_single_day_drifts", {
                    "count": len(drift_events),
                    "symbols": unique_symbols
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
        
        # Write-Audit-Publish pattern (Plan v4 patch)
        temp_path = snapshot_path.with_suffix('.tmp')
        
        # Write
        df.to_parquet(temp_path, index=False)
        
        # Audit
        try:
            audit_df = pd.read_parquet(temp_path)
            assert len(audit_df) == len(df), "Row count mismatch"
            assert list(audit_df.columns) == list(df.columns), "Column mismatch"
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            logger.error("snapshot_audit_failed", {"error": str(e)})
            raise
        
        # Publish (atomic rename)
        temp_path.rename(snapshot_path)
        
        logger.info("snapshot_created", {"version": version, "path": str(snapshot_path)})
        
        return snapshot_path
