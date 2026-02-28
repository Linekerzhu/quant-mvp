"""
Combinatorial Purged K-Fold Cross-Validation

Implements AFML Ch7: Combinatorial Purged K-Fold Cross-Validation
for time series financial data.

Key features:
- Purge: Remove overlapping samples between train/test
- Embargo: Add buffer after test period
- Combinatorial: Generate multiple train/test paths

Author: 李得勤
Date: 2026-02-27
"""

import logging
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List
from itertools import combinations
from pandas.tseries.offsets import BDay  # H-02: 使用交易日而非日历日

from src.ops.event_logger import get_logger

logger = get_logger()


def _has_overlap(entry_date, exit_date, purge_start, purge_end):
    """
    检查样本持有期是否与 purge 窗口重叠
    
    Args:
        entry_date: 样本进入日期
        exit_date: 样本退出日期（Triple Barrier 实际退出日）
        purge_start: purge 窗口开始
        purge_end: purge 窗口结束
    
    Returns:
        bool: 是否有重叠
    """
    if pd.isna(entry_date) or pd.isna(exit_date):
        return False
    
    # 持有期与 purge 窗口有重叠的条件
    # 等价于: NOT (完全在purge前 OR 完全在purge后)
    # 即: exit_date >= purge_start AND entry_date <= purge_end
    return exit_date >= purge_start and entry_date <= purge_end


class CombinatorialPurgedKFold:
    """
    AFML Ch7: Combinatorial Purged K-Fold Cross-Validation
    
    Ensures zero information leakage between train and test sets
    for time series financial data.
    
    Parameters (from config/training.yaml):
        n_splits: 6          # Split timeline into 6 segments
        n_test_splits: 2     # Select 2 segments for test each time
        purge_window: 10     # Days (= max_holding_days)
        embargo_window: 40  # Days
        min_data_days: 630   # Minimum training days required
    
    Combinations: C(6,2) = 15 CPCV paths
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 10,
        embargo_window: int = 60,  # MEDIUM-03: 与config保持一致
        min_data_days: int = 200,
        config_path: str = None  # Deprecated, kept for backward compatibility
    ):
        """
        Initialize CPCV splitter.
        
        Args:
            n_splits: Number of time segments to split
            n_test_splits: Number of segments to use as test per split
            purge_window: Days to purge after test set (overlap prevention)
            embargo_window: Additional embargo days after test set
            min_data_days: Minimum training samples required per path
            config_path: Deprecated, ignored. Use from_config() instead.
        
        Note:
            所有参数通过构造函数注入，违反依赖倒置原则。
            调用方负责从配置文件加载参数后传入。
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.min_data_days = min_data_days
    
    @classmethod
    def from_config(cls, config: dict) -> 'CombinatorialPurgedKFold':
        """
        从配置字典创建实例（保持向后兼容）。
        
        Args:
            config: 包含 'validation.cpcv' 的配置字典
        
        Returns:
            CombinatorialPurgedKFold 实例
        """
        cpcv = config.get('validation', {}).get('cpcv', {})
        return cls(
            n_splits=cpcv.get('n_splits', 6),
            n_test_splits=cpcv.get('n_test_splits', 2),
            purge_window=cpcv.get('purge_window', 10),
            embargo_window=cpcv.get('embargo_window', 40),
            min_data_days=cpcv.get('min_data_days', 200)
        )
    
    def get_n_paths(self) -> int:
        """Return number of CPCV paths: C(n_splits, n_test_splits)"""
        return self._combinations(self.n_splits, self.n_test_splits)
    
    @staticmethod
    def _combinations(n: int, r: int) -> int:
        """Calculate C(n,r) = n! / (r! * (n-r)!)"""
        if r > n:
            return 0
        if r == 0 or r == n:
            return 1
        # Optimize for large numbers
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - r + i) // i
        return result
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        exit_date_col: str = 'label_exit_date'
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for CPCV.
        
        Args:
            df: DataFrame with date and label_exit_date columns
            date_col: Column name for trigger/entry dates
            exit_date_col: Column name for actual exit dates from Triple Barrier
            
        Yields:
            Tuple of (train_indices, test_indices) for each CPCV path
        """
        # Sort by date
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)
        
        # Calculate segment boundaries
        segment_size = n_samples // self.n_splits
        segments = []
        for i in range(self.n_splits):
            start = i * segment_size
            end = start + segment_size if i < self.n_splits - 1 else n_samples
            segments.append((start, end))
        
        # Generate all combinations of test segments
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_seg_indices in test_combinations:
            # Build test set indices
            test_indices = []
            for seg_idx in test_seg_indices:
                start, end = segments[seg_idx]
                test_indices.extend(range(start, end))
            
            # Get test date range
            test_dates = df.loc[test_indices, date_col]
            test_min_date = test_dates.min()
            test_max_date = test_dates.max()
            
            # Calculate purge range using actual exit dates
            # Purge: Any sample whose [entry, exit] overlaps with test period
            # H-02 Fix: 使用 BDay (交易日) 而非日历日
            purge_start = test_min_date - BDay(self.purge_window)
            purge_end = test_max_date + BDay(self.purge_window)
            
            # Calculate embargo range
            embargo_end = test_max_date + BDay(self.embargo_window)
            
            # Build train set with purging and embargo
            train_indices = []
            for idx in range(n_samples):
                if idx in test_indices:
                    continue
                
                row_date = df.loc[idx, date_col]
                
                # Skip if within embargo period
                if row_date <= embargo_end and row_date > test_max_date:
                    continue
                
                # Check purge overlap using both entry_date and exit_date
                if exit_date_col in df.columns:
                    entry_date = df.loc[idx, date_col]  # entry date
                    exit_date = df.loc[idx, exit_date_col]
                    
                    if _has_overlap(entry_date, exit_date, purge_start, purge_end):
                        continue  # 有重叠，跳过
                
                train_indices.append(idx)
            
            # Check minimum training size
            if len(train_indices) >= self.min_data_days:
                yield (np.array(train_indices), np.array(test_indices))
    
    def split_with_info(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        exit_date_col: str = 'label_exit_date'
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
        """
        Generate splits with metadata information.
        
        Yields:
            Tuple of (train_indices, test_indices, info_dict)
        """
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)
        
        segment_size = n_samples // self.n_splits
        segments = []
        for i in range(self.n_splits):
            start = i * segment_size
            end = start + segment_size if i < self.n_splits - 1 else n_samples
            segments.append((start, end))
        
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for path_idx, test_seg_indices in enumerate(test_combinations):
            test_indices = []
            for seg_idx in test_seg_indices:
                start, end = segments[seg_idx]
                test_indices.extend(range(start, end))
            
            test_dates = df.loc[test_indices, date_col]
            test_min_date = test_dates.min()
            test_max_date = test_dates.max()
            
            # OR2-05 Fix: 使用 BDay (交易日) 而非日历日
            purge_start = test_min_date - BDay(self.purge_window)
            purge_end = test_max_date + BDay(self.purge_window)
            embargo_end = test_max_date + BDay(self.embargo_window)
            
            train_indices = []
            for idx in range(n_samples):
                if idx in test_indices:
                    continue
                
                row_date = df.loc[idx, date_col]
                
                if row_date <= embargo_end and row_date > test_max_date:
                    continue
                
                # Check purge overlap using both entry_date and exit_date
                if exit_date_col in df.columns:
                    entry_date = df.loc[idx, date_col]  # entry date
                    exit_date = df.loc[idx, exit_date_col]
                    
                    if _has_overlap(entry_date, exit_date, purge_start, purge_end):
                        continue  # 有重叠，跳过
                
                train_indices.append(idx)
            
            info = {
                'path_idx': path_idx,
                'test_segments': test_seg_indices,
                'n_train': len(train_indices),
                'n_test': len(test_indices),
                'valid': len(train_indices) >= self.min_data_days
            }
            
            if len(train_indices) >= self.min_data_days:
                yield (np.array(train_indices), np.array(test_indices), info)
    
    def get_all_paths_info(self, df: pd.DataFrame) -> List[dict]:
        """
        Get information about all CPCV paths without yielding splits.
        
        Returns:
            List of path info dictionaries
        """
        info_list = []
        for _, _, info in self.split_with_info(df):
            info_list.append(info)
        return info_list
    
    def __repr__(self):
        return (
            f"CombinatorialPurgedKFold("
            f"n_splits={self.n_splits}, "
            f"n_test_splits={self.n_test_splits}, "
            f"purge={self.purge_window}d, "
            f"embargo={self.embargo_window}d, "
            f"paths={self.get_n_paths()})"
        )


class PurgedKFold:
    """
    Standard Purged K-Fold (non-combinatorial).
    
    Simpler version that splits into n_splits and iterates through them.
    Useful for quick validation before running full CPCV.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 10,
        embargo_window: int = 5
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        exit_date_col: str = 'label_exit_date'
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged K-Fold splits."""
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)
        
        segment_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Test: fold-th segment
            test_start = fold * segment_size
            test_end = (fold + 1) * segment_size if fold < self.n_splits - 1 else n_samples
            test_indices = np.arange(test_start, test_end)
            
            # Get test date range
            test_dates = df.loc[test_indices, date_col]
            test_min_date = test_dates.min()
            test_max_date = test_dates.max()
            
            # OR2-05 Fix: 使用 BDay (交易日)
            purge_end = test_max_date + BDay(self.purge_window)
            
            # Embargo range
            embargo_end = test_max_date + BDay(self.embargo_window)
            
            # Build train set
            train_indices = []
            for idx in range(n_samples):
                if idx in test_indices:
                    continue
                
                row_date = df.loc[idx, date_col]
                
                # Embargo check
                if row_date <= embargo_end and row_date > test_max_date:
                    continue
                
                # Purge check using both entry_date and exit_date
                if exit_date_col in df.columns:
                    entry_date = df.loc[idx, date_col]  # entry date
                    exit_date = df.loc[idx, exit_date_col]
                    
                    if pd.notna(exit_date) and pd.notna(entry_date):
                        # Check if holding period overlaps with [test_min_date, purge_end]
                        if _has_overlap(entry_date, exit_date, test_min_date, purge_end):
                            continue
                
                train_indices.append(idx)
            
            yield (np.array(train_indices), test_indices)
    
    def __repr__(self):
        return f"PurgedKFold(n_splits={self.n_splits}, purge={self.purge_window}d)"
