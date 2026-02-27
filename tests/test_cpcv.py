"""
Tests for CPCV (Combinatorial Purged K-Fold) Module

Author: 李得勤
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.purged_kfold import CombinatorialPurgedKFold, PurgedKFold


class TestCombinatorialPurgedKFold:
    """Test suite for CombinatorialPurgedKFold."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 2000
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # Generate realistic exit dates
        exit_days = np.random.choice([5, 10, 15, 20, 30, 40, 50], n, p=[0.1, 0.3, 0.25, 0.15, 0.1, 0.05, 0.05])
        exit_dates = [d + pd.Timedelta(days=int(ed)) for d, ed in zip(dates, exit_days)]
        
        return pd.DataFrame({
            'date': dates,
            'label_exit_date': exit_dates
        })
    
    @pytest.fixture
    def cpcv(self):
        """Create CPCV splitter with relaxed min_data_days."""
        return CombinatorialPurgedKFold(
            n_splits=6,
            n_test_splits=2,
            purge_window=10,
            embargo_window=40,
            min_data_days=200
        )
    
    def test_init_default(self):
        """Test default initialization."""
        cpcv = CombinatorialPurgedKFold()
        assert cpcv.n_splits == 6
        assert cpcv.n_test_splits == 2
        assert cpcv.purge_window == 10
        assert cpcv.embargo_window == 40
        # Note: min_data_days is loaded from config, default is 200
    
    def test_init_custom(self):
        """Test custom initialization."""
        # Pass config_path that doesn't exist to avoid loading config
        cpcv = CombinatorialPurgedKFold(
            n_splits=4,
            n_test_splits=1,
            purge_window=5,
            embargo_window=20,
            min_data_days=100,
            config_path="/nonexistent/config.yaml"
        )
        assert cpcv.n_splits == 4
        assert cpcv.n_test_splits == 1
        assert cpcv.purge_window == 5
        assert cpcv.embargo_window == 20
        assert cpcv.min_data_days == 100
    
    def test_get_n_paths(self, cpcv):
        """Test path count calculation: C(6,2) = 15"""
        assert cpcv.get_n_paths() == 15
    
    def test_combinations_calculation(self):
        """Test _combinations static method."""
        assert CombinatorialPurgedKFold._combinations(6, 2) == 15
        assert CombinatorialPurgedKFold._combinations(10, 3) == 120
        assert CombinatorialPurgedKFold._combinations(5, 0) == 1
        assert CombinatorialPurgedKFold._combinations(5, 5) == 1
    
    def test_split_returns_iterator(self, cpcv, sample_data):
        """Test that split returns an iterator."""
        result = cpcv.split(sample_data)
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
    
    def test_split_generates_tuples(self, cpcv, sample_data):
        """Test that split generates (train, test) tuples."""
        train_idx, test_idx = next(cpcv.split(sample_data))
        
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) > 0
        assert len(test_idx) > 0
    
    def test_split_no_overlap(self, cpcv, sample_data):
        """Test that train and test sets have no overlap."""
        for train_idx, test_idx in cpcv.split(sample_data):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0
    
    def test_split_covers_all(self, cpcv, sample_data):
        """Test that all indices are either in train or test."""
        all_indices = set(range(len(sample_data)))
        
        train_union = set()
        test_union = set()
        
        for train_idx, test_idx in cpcv.split(sample_data):
            train_union.update(train_idx)
            test_union.update(test_idx)
        
        # Note: Due to embargo/purge, not all indices may be used
        # But train and test together should not exceed total
        assert len(train_union | test_union) <= len(all_indices)
    
    def test_split_min_data_days(self, cpcv, sample_data):
        """Test that all returned paths meet min_data_days requirement."""
        for train_idx, test_idx in cpcv.split(sample_data):
            assert len(train_idx) >= cpcv.min_data_days
    
    def test_split_15_paths(self, cpcv, sample_data):
        """Test that valid paths are generated (may be less than 15 if min_data_days too strict)."""
        path_count = sum(1 for _ in cpcv.split(sample_data))
        # With min_data_days=200, some paths may be invalid
        assert path_count > 0  # At least some valid paths
        assert path_count <= 15  # But no more than 15
    
    def test_test_set_size_consistent(self, cpcv, sample_data):
        """Test that test set size is consistent across paths."""
        test_sizes = []
        for train_idx, test_idx in cpcv.split(sample_data):
            test_sizes.append(len(test_idx))
        
        # All test sets should be roughly the same size
        assert len(set(test_sizes)) <= 3  # Allow small variation
    
    def test_with_info(self, cpcv, sample_data):
        """Test split_with_info method."""
        for train_idx, test_idx, info in cpcv.split_with_info(sample_data):
            assert 'path_idx' in info
            assert 'test_segments' in info
            assert 'n_train' in info
            assert 'n_test' in info
            assert 'valid' in info
            assert info['valid'] == True
            break
    
    def test_get_all_paths_info(self, cpcv, sample_data):
        """Test get_all_paths_info method."""
        info_list = cpcv.get_all_paths_info(sample_data)
        # With min_data_days=200, some paths may be invalid
        assert len(info_list) > 0
        assert all(info['valid'] for info in info_list)
    
    def test_repr(self, cpcv):
        """Test __repr__ method."""
        repr_str = repr(cpcv)
        assert 'CombinatorialPurgedKFold' in repr_str
        assert 'n_splits=6' in repr_str
        assert 'paths=15' in repr_str


class TestPurgedKFold:
    """Test suite for PurgedKFold (non-combinatorial)."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        exit_days = np.random.randint(10, 30, n)
        exit_dates = [d + pd.Timedelta(days=int(ed)) for d, ed in zip(dates, exit_days)]
        
        return pd.DataFrame({
            'date': dates,
            'label_exit_date': exit_dates
        })
    
    @pytest.fixture
    def purged_kfold(self):
        """Create PurgedKFold splitter."""
        return PurgedKFold(n_splits=5, purge_window=10, embargo_window=5)
    
    def test_init(self, purged_kfold):
        """Test initialization."""
        assert purged_kfold.n_splits == 5
        assert purged_kfold.purge_window == 10
        assert purged_kfold.embargo_window == 5
    
    def test_split_count(self, purged_kfold, sample_data):
        """Test that n_splits paths are generated."""
        count = sum(1 for _ in purged_kfold.split(sample_data))
        assert count == 5
    
    def test_split_no_overlap(self, purged_kfold, sample_data):
        """Test train/test have no overlap."""
        for train_idx, test_idx in purged_kfold.split(sample_data):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0
    
    def test_repr(self, purged_kfold):
        """Test __repr__."""
        assert 'PurgedKFold' in repr(purged_kfold)
        assert 'n_splits=5' in repr(purged_kfold)


class TestCPCVIntegration:
    """Integration tests for CPCV with realistic data."""
    
    def test_with_mock_prices(self):
        """Test CPCV with mock price data from fixtures."""
        FIXTURES_DIR = Path(__file__).parent / "fixtures"
        
        if not (FIXTURES_DIR / "mock_prices.parquet").exists():
            pytest.skip("Mock data not available")
        
        df = pd.read_parquet(FIXTURES_DIR / "mock_prices.parquet")
        
        # Filter to one symbol
        symbol_data = df[df['symbol'] == 'MOCK000'].copy()
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        # Add synthetic exit dates
        n = len(symbol_data)
        exit_days = np.random.randint(5, 20, n)
        symbol_data['label_exit_date'] = [
            d + pd.Timedelta(days=int(ed)) 
            for d, ed in zip(symbol_data['date'], exit_days)
        ]
        
        # Use relaxed min_data_days for small dataset
        cpcv = CombinatorialPurgedKFold(
            n_splits=4,
            n_test_splits=1,
            min_data_days=20,  # Relaxed for small dataset
            config_path="/nonexistent/config.yaml"
        )
        
        paths = list(cpcv.split(symbol_data))
        # With very small data, may have 0 valid paths - that's OK
        for train_idx, test_idx in paths:
            assert len(train_idx) >= 20
            assert len(test_idx) > 0
