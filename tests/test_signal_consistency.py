import pytest
import pandas as pd
from src.ops.signal_consistency import SignalConsistency

def test_signal_consistency():
    signals = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'TSLA'],
        'target_weight': [0.1, -0.05, 0.0]
    })
    
    orders = [
        {'symbol': 'AAPL', 'qty': 10, 'side': 'buy', 'price': 150.0},
        {'symbol': 'MSFT', 'qty': 2, 'side': 'buy', 'price': 300.0}, # Mismatch direction
        {'symbol': 'NVDA', 'qty': 1, 'side': 'buy', 'price': 800.0} # Ghost order
    ]
    
    metrics = SignalConsistency.verify(signals, orders)
    
    assert metrics['total_signals'] == 2 # TSLA skipped as 0
    assert metrics['total_orders'] == 3
    assert metrics['inconsistencies'] == 2
    
    types = [i['type'] for i in metrics['details']]
    assert 'direction_mismatch' in types # MSFT
    assert 'ghost_order' in types # NVDA
