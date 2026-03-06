import pytest
import pandas as pd
import numpy as np

from src.models.walk_forward import WalkForwardValidator

@pytest.fixture
def mock_df():
    dates = pd.date_range("2020-01-01", periods=150, freq='B')
    return pd.DataFrame({'date': dates, 'value': range(150)})

def test_walk_forward_splits_correctly(mock_df):
    validator = WalkForwardValidator(n_splits=5, purge_window=5, min_train_days=50)
    
    splits = list(validator.split(mock_df))
    assert len(splits) == 5
    
    train_0, test_0 = splits[0]
    train_dates = mock_df.iloc[train_0]['date']
    test_dates = mock_df.iloc[test_0]['date']
    
    assert len(train_dates) > 0
    assert len(test_dates) > 0
    
    # Train should end before test starts by at least the purge window
    train_max = train_dates.max()
    test_min = test_dates.min()
    
    assert train_max <= test_min - pd.tseries.offsets.BDay(5)
    
    # Check that in expanding window, later train sets overlap with earlier train sets
    train_1, test_1 = splits[1]
    assert len(train_1) > len(train_0) # expanding window
    
def test_walk_forward_insufficient_data():
    dates = pd.date_range("2020-01-01", periods=30, freq='B')
    df = pd.DataFrame({'date': dates})
    
    validator = WalkForwardValidator(n_splits=5, purge_window=5, min_train_days=50)
    with pytest.raises(ValueError):
        list(validator.split(df))
