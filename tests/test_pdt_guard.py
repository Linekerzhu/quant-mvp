import pytest
import pandas as pd

from src.risk.pdt_guard import PDTGuard

def test_pdt_guard_clean_account():
    guard = PDTGuard()
    trade_history = []
    
    compliant, details = guard.check_pdt_compliance(trade_history, 20000)
    assert compliant is True
    assert details['day_trade_count'] == 0
    assert details['remaining_trades'] == 3

def test_pdt_guard_account_over_25k():
    guard = PDTGuard()
    
    # Even with lots of day trades, an account over 25k should be fine
    trade_history = [
        {'date': pd.Timestamp.now(), 'symbol': 'AAPL', 'action': 'buy'},
        {'date': pd.Timestamp.now(), 'symbol': 'AAPL', 'action': 'sell'}
    ] * 5
    
    compliant, details = guard.check_pdt_compliance(trade_history, 30000)
    assert compliant is True
    assert details['pdt_applies'] is False

def test_pdt_guard_multiple_day_trades():
    guard = PDTGuard()
    now = pd.Timestamp.now()
    
    # 3 day trades (max allowed without triggering PDT is 3)
    trade_history = [
        {'date': now, 'symbol': 'AAPL', 'action': 'buy'},
        {'date': now, 'symbol': 'AAPL', 'action': 'sell'},
        {'date': now, 'symbol': 'MSFT', 'action': 'buy'},
        {'date': now, 'symbol': 'MSFT', 'action': 'sell'},
        {'date': now, 'symbol': 'TSLA', 'action': 'buy'},
        {'date': now, 'symbol': 'TSLA', 'action': 'sell'},
    ]
    
    compliant, details = guard.check_pdt_compliance(trade_history, 20000)
    assert compliant is True
    assert details['day_trade_count'] == 3
    assert details['remaining_trades'] == 0

    # 4th day trade will trigger non-compliant
    trade_history.extend([
        {'date': now, 'symbol': 'AMZN', 'action': 'buy'},
        {'date': now, 'symbol': 'AMZN', 'action': 'sell'},
    ])
    
    compliant2, details2 = guard.check_pdt_compliance(trade_history, 20000)
    assert compliant2 is False
    assert details2['day_trade_count'] == 4
    
def test_pdt_guard_ignores_old_trades():
    guard = PDTGuard(rolling_window_days=5)
    now = pd.Timestamp.now()
    old_date = now - pd.tseries.offsets.BDay(10)
    
    trade_history = [
        {'date': old_date, 'symbol': 'AAPL', 'action': 'buy'},
        {'date': old_date, 'symbol': 'AAPL', 'action': 'sell'},
        {'date': old_date, 'symbol': 'MSFT', 'action': 'buy'},
        {'date': old_date, 'symbol': 'MSFT', 'action': 'sell'},
        {'date': old_date, 'symbol': 'TSLA', 'action': 'buy'},
        {'date': old_date, 'symbol': 'TSLA', 'action': 'sell'},
        {'date': old_date, 'symbol': 'AMZN', 'action': 'buy'},
        {'date': old_date, 'symbol': 'AMZN', 'action': 'sell'},
    ]
    
    compliant, details = guard.check_pdt_compliance(trade_history, 20000)
    # Since these are older than 5 business days, they should not count
    assert compliant is True
    assert details['day_trade_count'] == 0
