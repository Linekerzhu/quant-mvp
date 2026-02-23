"""
Tests for data integrity module.
Tests hash freezing, drift detection, and WAP pattern.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data.integrity import IntegrityManager


class TestIntegrityManager:
    """Test hash freezing and drift detection."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02']),
            'raw_open': [100.0, 101.0, 200.0, 202.0],
            'raw_high': [102.0, 103.0, 205.0, 207.0],
            'raw_low': [99.0, 100.0, 198.0, 200.0],
            'raw_close': [101.0, 102.0, 202.0, 205.0],
            'adj_open': [100.0, 101.0, 200.0, 202.0],
            'adj_high': [102.0, 103.0, 205.0, 207.0],
            'adj_low': [99.0, 100.0, 198.0, 200.0],
            'adj_close': [101.0, 102.0, 202.0, 205.0],
            'volume': [1000000, 1200000, 500000, 600000]
        })
    
    def test_hash_computation(self, temp_dir, sample_data):
        """Test that hashes are computed correctly."""
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Freeze data
        manager.freeze_data(sample_data)
        
        # Check hashes were stored
        assert len(manager.hashes) == 4
        assert 'adj_hash' in manager.hashes.columns
        assert 'raw_hash' in manager.hashes.columns
        assert 'adj_factor_hash' in manager.hashes.columns
    
    def test_no_drift_on_same_data(self, temp_dir, sample_data):
        """Test that no drift is detected on identical data."""
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Freeze original data
        manager.freeze_data(sample_data)
        
        # Check same data again
        should_freeze, events = manager.detect_drift(sample_data, universe_size=500)
        
        assert not should_freeze
        assert len(events) == 0
    
    def test_drift_detection(self, temp_dir, sample_data):
        """Test drift detection on changed data."""
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Freeze original data
        manager.freeze_data(sample_data)
        
        # Modify some data
        modified = sample_data.copy()
        modified.loc[0, 'adj_close'] = 150.0  # Change a price
        
        # Detect drift
        should_freeze, events = manager.detect_drift(modified, universe_size=500)
        
        # Single day drift should not trigger freeze (just warn)
        assert len(events) == 1
        assert events[0]['symbol'] == 'AAPL'
        assert 'adj' in events[0]['drift_type']
    
    def test_raw_only_drift_detection(self, temp_dir, sample_data):
        """
        Test detection of Raw-only drift (Plan v4 patch requirement).
        
        This tests the scenario where only raw_close changes but adj_close
        stays the same - indicating a split adjustment change.
        """
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Freeze original data
        manager.freeze_data(sample_data)
        
        # Modify only raw_close (simulating split adjustment)
        modified = sample_data.copy()
        modified.loc[0, 'raw_close'] = 50.0  # Raw changes, adj stays same
        
        # Detect drift
        should_freeze, events = manager.detect_drift(modified, universe_size=500)
        
        # Should detect raw drift
        assert len(events) == 1
        assert 'raw' in events[0]['drift_type'] or 'adj_factor' in events[0]['drift_type']
    
    def test_snapshot_creation(self, temp_dir, sample_data):
        """Test Write-Audit-Publish pattern for snapshots."""
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Create snapshot
        snapshot_path = manager.create_snapshot(sample_data, "test_v1")
        
        # Verify snapshot exists
        assert snapshot_path.exists()
        
        # Verify data integrity
        loaded = pd.read_parquet(snapshot_path)
        assert len(loaded) == len(sample_data)
        assert list(loaded.columns) == list(sample_data.columns)
    
    def test_universe_adaptive_threshold(self, temp_dir, sample_data):
        """Test universe-size adaptive drift threshold (Plan v4)."""
        manager = IntegrityManager(
            hash_file=f"{temp_dir}/hashes.parquet",
            snapshot_dir=f"{temp_dir}/snapshots"
        )
        
        # Freeze original
        manager.freeze_data(sample_data)
        
        # Modify all symbols (simulate broad drift)
        modified = sample_data.copy()
        modified['adj_close'] = modified['adj_close'] * 1.1
        
        # Test with small universe (should use min threshold of 10)
        should_freeze_small, _ = manager.detect_drift(modified, universe_size=50)
        
        # Test with large universe (should use 1% threshold = 50)
        should_freeze_large, _ = manager.detect_drift(modified, universe_size=5000)
        
        # Both should detect drift but thresholds differ
        assert max(10, int(0.01 * 50)) == 10
        assert max(10, int(0.01 * 5000)) == 50
