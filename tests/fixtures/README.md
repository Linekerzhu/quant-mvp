# Test Fixtures

This directory contains static mock data for unit tests.

All tests must use these fixtures - **no network calls allowed** in unit tests.

## Files

- `mock_prices.parquet` - Synthetic OHLCV data with scenarios:
  - Normal price movements (MOCK004+)
  - Stock split (MOCK000, 2:1 on 2024-06-15)
  - Trading halt (MOCK001, 5-day halt)
  - Abnormal jump (MOCK002, +55% day)
  - Delisting (MOCK003, after 2024-08-15)
  
- `mock_corporate_actions.csv` - Corporate actions reference
  - Splits with ratios
  - Delistings with reasons

## Regenerating Data

```bash
python tests/fixtures/generate_mock_data.py
```

The random seed (42) ensures reproducibility.

## Adding New Scenarios

Edit `generate_mock_data.py` and add new injector functions:

```python
def _inject_my_scenario(df, symbol, ...):
    # Modify df in place
    return df
```

Then call it in `generate_mock_prices()`.
