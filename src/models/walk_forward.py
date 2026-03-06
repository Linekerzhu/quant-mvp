import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from pandas.tseries.offsets import BDay

from src.models.validators import BaseValidator

class WalkForwardValidator(BaseValidator):
    """
    Time Series Walk-Forward Validator with Purging.
    
    Splits the data into expanding training windows and fixed sliding testing windows.
    Ensures that a `purge_window` is respected before the test set sequence avoids lookahead.
    
    This is used as a baseline comparator to CPCV.
    """
    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 20,
        embargo_window: int = 0, # Embargo is generally forward-looking, not needed for pure past-to-future walk-forward without future train data
        min_train_days: int = 100
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.min_train_days = min_train_days
        
    def get_n_splits(self) -> int:
        return self.n_splits
        
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(df[date_col]).dt.normalize()
        unique_dates = np.sort(dates.unique())
        total_days = len(unique_dates)
        
        remaining_days = total_days - self.min_train_days
        if remaining_days <= 0:
            raise ValueError(f"Not enough days ({total_days}) for min_train_days={self.min_train_days}")
            
        test_size = remaining_days // self.n_splits
        if test_size <= 0:
            raise ValueError(f"Calculated test_size={test_size} <= 0. Reduce n_splits or min_train_days.")
            
        for i in range(self.n_splits):
            test_start_idx = self.min_train_days + i * test_size
            test_end_idx = test_start_idx + test_size
            if i == self.n_splits - 1:
                test_end_idx = total_days
                
            test_start_date = pd.Timestamp(unique_dates[test_start_idx])
            test_end_date = pd.Timestamp(unique_dates[test_end_idx - 1])
            
            # In expanding walk-forward, we only use data prior to test_start_date.
            purge_cutoff = test_start_date - BDay(self.purge_window)
            
            train_mask = dates <= purge_cutoff
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)
            
            if not train_mask.any() or not test_mask.any():
                continue
                
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def __repr__(self):
        return f"WalkForwardValidator(n_splits={self.n_splits}, purge={self.purge_window})"
