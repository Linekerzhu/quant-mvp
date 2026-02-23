"""
Mock Data Generator for Quant MVP Tests

Generates synthetic price data with:
- Normal price movements
- Stock splits (raw vs adj divergence)
- Trading halts (consecutive NaN)
- Abnormal jumps (>50%)
- Delisted symbols

This is used for deterministic unit tests without network calls.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def generate_mock_prices(
    n_symbols: int = 10,
    n_days: int = 252,
    start_date: str = "2024-01-01"
) -> pd.DataFrame:
    """
    Generate mock price data with various scenarios.
    
    Returns DataFrame with columns:
    - symbol, date, raw_open, raw_high, raw_low, raw_close, 
    - adj_open, adj_high, adj_low, adj_close, volume
    """
    start = pd.Timestamp(start_date)
    dates = pd.date_range(start=start, periods=n_days, freq='B')  # Business days
    
    records = []
    
    for i in range(n_symbols):
        symbol = f"MOCK{i:03d}"
        
        # Base price
        base_price = np.random.uniform(50, 500)
        
        # Generate random walk
        returns = np.random.normal(0.0005, 0.02, n_days)  # Mean 0.05%, vol 2%
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add OHLC variation
        daily_vol = prices * 0.015  # 1.5% intraday vol
        
        for j, date in enumerate(dates):
            price = prices[j]
            vol = daily_vol[j]
            
            # Generate OHLC
            open_p = price * (1 + np.random.normal(0, 0.005))
            close_p = price
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.01)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.01)))
            
            volume = int(np.random.uniform(1e6, 50e6))
            
            records.append({
                'symbol': symbol,
                'date': date,
                'raw_open': open_p,
                'raw_high': high_p,
                'raw_low': low_p,
                'raw_close': close_p,
                'adj_open': open_p,  # Will be adjusted for splits
                'adj_high': high_p,
                'adj_low': low_p,
                'adj_close': close_p,
                'volume': volume
            })
    
    df = pd.DataFrame(records)
    
    # Inject scenarios
    # Note: 2024-06-15 is Saturday, use 2024-06-17 (Monday) instead
    df = _inject_split_scenario(df, symbol='MOCK000', split_date='2024-06-17', ratio=2.0)
    df = _inject_halt_scenario(df, symbol='MOCK001', halt_start='2024-03-15', n_days=5)
    df = _inject_jump_scenario(df, symbol='MOCK002', jump_date='2024-04-20', jump_pct=0.55)
    df = _inject_delist_scenario(df, symbol='MOCK003', delist_date='2024-08-15')
    
    return df


def _inject_split_scenario(
    df: pd.DataFrame,
    symbol: str,
    split_date: str,
    ratio: float = 2.0
) -> pd.DataFrame:
    """Inject a 2:1 stock split scenario.
    
    For backward-adjusted prices:
    - Pre-split: raw = base * ratio (higher historical raw), adj = base
    - Post-split: raw = base, adj = base
    
    This creates the divergence where pre-split raw/adj ~ ratio,
    and post-split raw/adj ~ 1.0
    """
    mask = df['symbol'] == symbol
    split_date_ts = pd.Timestamp(split_date)
    
    # Find the split date in the data
    split_mask = mask & (df['date'] == split_date_ts)
    
    if split_mask.sum() == 0:
        print(f"Warning: Split date {split_date} not found for {symbol}")
        return df
    
    # Get split index
    split_idx = df[split_mask].index[0]
    
    # Pre-split mask
    pre_split_mask = mask & (df['date'] < split_date_ts)
    
    # Save the base prices (these are the post-split/adjusted prices)
    base_open = df.loc[pre_split_mask, 'raw_open'].copy()
    base_high = df.loc[pre_split_mask, 'raw_high'].copy()
    base_low = df.loc[pre_split_mask, 'raw_low'].copy()
    base_close = df.loc[pre_split_mask, 'raw_close'].copy()
    base_volume = df.loc[pre_split_mask, 'volume'].copy()
    
    # Pre-split raw prices are higher (before the split)
    df.loc[pre_split_mask, 'raw_open'] = base_open * ratio
    df.loc[pre_split_mask, 'raw_high'] = base_high * ratio
    df.loc[pre_split_mask, 'raw_low'] = base_low * ratio
    df.loc[pre_split_mask, 'raw_close'] = base_close * ratio
    
    # Pre-split adj prices are the base (backward-adjusted)
    # These represent what the price would be if adjusted for the split
    df.loc[pre_split_mask, 'adj_open'] = base_open
    df.loc[pre_split_mask, 'adj_high'] = base_high
    df.loc[pre_split_mask, 'adj_low'] = base_low
    df.loc[pre_split_mask, 'adj_close'] = base_close
    
    # Pre-split volume is higher (more shares before split)
    df.loc[pre_split_mask, 'volume'] = base_volume * ratio
    
    return df


def _inject_halt_scenario(
    df: pd.DataFrame,
    symbol: str,
    halt_start: str,
    n_days: int = 5
) -> pd.DataFrame:
    """Inject a trading halt scenario (consecutive NaN)."""
    mask = df['symbol'] == symbol
    start_date = pd.Timestamp(halt_start)
    
    for i in range(n_days):
        date = start_date + timedelta(days=i)
        # Skip weekends
        while date.weekday() >= 5:
            date += timedelta(days=1)
        
        halt_mask = mask & (df['date'] == date)
        df.loc[halt_mask, ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                          'adj_open', 'adj_high', 'adj_low', 'adj_close']] = np.nan
    
    return df


def _inject_jump_scenario(
    df: pd.DataFrame,
    symbol: str,
    jump_date: str,
    jump_pct: float = 0.55
) -> pd.DataFrame:
    """Inject an abnormal price jump (>50%)."""
    mask = df['symbol'] == symbol
    date_mask = mask & (df['date'] == jump_date)
    
    if df[date_mask].empty:
        return df
    
    # Get previous close
    idx = df[date_mask].index[0]
    if idx > 0:
        prev_close = df.loc[idx - 1, 'raw_close']
        df.loc[idx, 'raw_open'] = prev_close * (1 + jump_pct)
        df.loc[idx, 'raw_close'] = prev_close * (1 + jump_pct * 0.9)
        df.loc[idx, 'raw_high'] = prev_close * (1 + jump_pct * 1.1)
        df.loc[idx, 'raw_low'] = prev_close * (1 + jump_pct * 0.8)
        
        # Adj prices same (not a split)
        df.loc[idx, 'adj_open'] = df.loc[idx, 'raw_open']
        df.loc[idx, 'adj_close'] = df.loc[idx, 'raw_close']
        df.loc[idx, 'adj_high'] = df.loc[idx, 'raw_high']
        df.loc[idx, 'adj_low'] = df.loc[idx, 'raw_low']
    
    return df


def _inject_delist_scenario(
    df: pd.DataFrame,
    symbol: str,
    delist_date: str
) -> pd.DataFrame:
    """Inject a delisting scenario (no data after delist date)."""
    mask = df['symbol'] == symbol
    delist = pd.Timestamp(delist_date)
    
    # Set all data after delist date to NaN
    post_delist = mask & (df['date'] > delist)
    df.loc[post_delist, ['raw_open', 'raw_high', 'raw_low', 'raw_close',
                         'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']] = np.nan
    
    return df


def generate_mock_corporate_actions() -> pd.DataFrame:
    """Generate mock corporate actions data."""
    records = [
        {
            'symbol': 'MOCK000',
            'event_type': 'split',
            'event_date': '2024-06-17',  # Monday (15th is Saturday)
            'ratio': 2.0,
            'details': '2:1 stock split'
        },
        {
            'symbol': 'MOCK003',
            'event_type': 'delist',
            'event_date': '2024-08-15',
            'reason': 'acquisition',
            'details': 'Acquired by parent company'
        }
    ]
    return pd.DataFrame(records)


if __name__ == '__main__':
    # Generate and save mock data
    prices = generate_mock_prices()
    actions = generate_mock_corporate_actions()
    
    print("Mock prices shape:", prices.shape)
    print("\nFirst few rows:")
    print(prices.head())
    print("\nCorporate actions:")
    print(actions)
    
    # Save to parquet for tests
    prices.to_parquet('tests/fixtures/mock_prices.parquet', index=False)
    actions.to_csv('tests/fixtures/mock_corporate_actions.csv', index=False)
    print("\nMock data saved to tests/fixtures/")
