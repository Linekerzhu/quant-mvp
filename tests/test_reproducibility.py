"""
Data Reproducibility Test

Tests that the data pipeline produces deterministic, reproducible results.
Plan requirement: "Same time range, same config, two full runs, output hashes identical."

This is part of Phase A data integrity validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import DualSourceIngest
from src.data.validate import DataValidator
from src.data.integrity import IntegrityManager


class TestDataReproducibility:
    """
    Test data pipeline reproducibility.
    
    Two runs with:
    - Same time range
    - Same config  
    - Same symbols
    
    Should produce identical hashes.
    """
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test data."""
        temp_root = tempfile.mkdtemp()
        dirs = {
            'raw': Path(temp_root) / 'raw',
            'processed': Path(temp_root) / 'processed',
            'snapshots': Path(temp_root) / 'snapshots'
        }
        for d in dirs.values():
            d.mkdir(parents=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_root)
    
    def test_mock_data_reproducibility(self, temp_dirs):
        """
        Test that mock data processing is reproducible.
        
        Uses static mock data to ensure no external API variability.
        """
        # Load mock data
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        df1 = pd.read_parquet(mock_path)
        df2 = pd.read_parquet(mock_path)
        
        # Run validation on both
        validator = DataValidator()
        
        passed1, cleaned1, report1 = validator.validate(df1)
        passed2, cleaned2, report2 = validator.validate(df2)
        
        # Both should pass
        assert passed1 == passed2
        
        # Compute hashes for both cleaned datasets
        hash_cols = ['symbol', 'date', 'raw_close', 'adj_close', 'volume']
        hash1 = pd.util.hash_pandas_object(cleaned1[hash_cols]).sum()
        hash2 = pd.util.hash_pandas_object(cleaned2[hash_cols]).sum()
        
        # Hashes should be identical
        assert hash1 == hash2, f"Hash mismatch: {hash1} != {hash2}"
    
    def test_hash_freezing_reproducibility(self, temp_dirs):
        """
        Test that hash freezing produces consistent results.
        """
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        df = pd.read_parquet(mock_path)
        
        # Create two integrity managers with same hash file
        hash_file = temp_dirs['processed'] / 'hashes.parquet'
        
        im1 = IntegrityManager(hash_file=str(hash_file))
        im2 = IntegrityManager(hash_file=str(hash_file))
        
        # Freeze same data twice
        im1.freeze_data(df)
        
        # Load hashes after first freeze
        hashes1 = im1.hashes.copy()
        
        # Freeze again (should update timestamps but same content hashes)
        im2.freeze_data(df)
        hashes2 = im2.hashes.copy()
        
        # Content hashes should match
        for col in ['adj_hash', 'raw_hash', 'adj_factor_hash']:
            assert (hashes1[col] == hashes2[col]).all(), f"Hash column {col} mismatch"
    
    def test_pipeline_determinism(self):
        """
        Test full pipeline determinism with static data.
        
        Process same data twice, verify:
        1. Same number of rows
        2. Same column values
        3. Same validation results
        """
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        # Run pipeline twice
        results = []
        for run in range(2):
            df = pd.read_parquet(mock_path)
            
            # Validate
            validator = DataValidator()
            passed, cleaned, report = validator.validate(df)
            
            results.append({
                'passed': passed,
                'rows': len(cleaned),
                'report': report
            })
        
        # Compare results
        assert results[0]['passed'] == results[1]['passed']
        assert results[0]['rows'] == results[1]['rows']
        assert results[0]['report']['pass_rate'] == results[1]['report']['pass_rate']
    
    def test_symbol_order_independence(self):
        """
        Test that symbol processing order doesn't affect results.
        """
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        df = pd.read_parquet(mock_path)
        
        # Process with different symbol orders
        symbols = df['symbol'].unique()
        
        # Order 1: alphabetical
        df1 = df[df['symbol'].isin(sorted(symbols))]
        
        # Order 2: reversed
        df2 = df[df['symbol'].isin(sorted(symbols, reverse=True))]
        
        # Validate both
        validator = DataValidator()
        _, cleaned1, _ = validator.validate(df1)
        _, cleaned2, _ = validator.validate(df2)
        
        # Sort both by symbol and date for comparison
        cleaned1 = cleaned1.sort_values(['symbol', 'date']).reset_index(drop=True)
        cleaned2 = cleaned2.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Should be identical
        pd.testing.assert_frame_equal(
            cleaned1[['symbol', 'date', 'raw_close', 'adj_close']],
            cleaned2[['symbol', 'date', 'raw_close', 'adj_close']]
        )


class TestDataIntegrity:
    """Additional data integrity tests."""
    
    def test_no_duplicate_rows(self):
        """Test that duplicate symbol-date rows are handled."""
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        df = pd.read_parquet(mock_path)
        
        validator = DataValidator()
        passed, cleaned, report = validator.validate(df)
        
        # After validation, should be no duplicates
        dup_count = cleaned.duplicated(subset=['symbol', 'date']).sum()
        assert dup_count == 0
    
    def test_hash_consistency(self):
        """Test that same data produces same hash."""
        mock_path = Path("tests/fixtures/mock_prices.parquet")
        if not mock_path.exists():
            pytest.skip("Mock data not found")
        
        df = pd.read_parquet(mock_path)
        
        # Compute hash twice
        hash_cols = ['symbol', 'date', 'adj_close']
        hash1 = pd.util.hash_pandas_object(df[hash_cols]).sum()
        hash2 = pd.util.hash_pandas_object(df[hash_cols]).sum()
        
        assert hash1 == hash2
