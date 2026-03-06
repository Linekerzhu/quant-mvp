import pytest
import pandas as pd
import numpy as np

from src.risk.position_sizing import IndependentKellySizer
from src.risk.risk_engine import RiskEngine

@pytest.fixture
def mock_signals():
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'side': [1, -1, 1],
        'prob': [0.6, 0.55, 0.7],
        'avg_win': [0.04, 0.03, 0.05],
        'avg_loss': [0.03, 0.02, 0.04],
        'realized_vol': [0.2, 0.25, 0.15]
    })

def test_independent_kelly_formula():
    sizer = IndependentKellySizer()
    # f = p/a - q/b where a is loss, b is win
    # p=0.6, q=0.4, w=0.04, l=0.03 => f = 0.6/0.03 - 0.4/0.04 = 20 - 10 = 10.
    f = sizer.calculate_kelly_fraction(0.6, 0.04, 0.03)
    assert f == 10.0

    # Loss case
    f2 = sizer.calculate_kelly_fraction(0.4, 0.04, 0.03)
    # 0.4/0.03 - 0.6/0.04 = 13.333 - 15 < 0 -> 0 
    assert f2 == 0.0

def test_position_sizing_normalization(mock_signals):
    sizer = IndependentKellySizer()
    # Set max gross leverage lower to force normalization
    sizer.max_gross_leverage = 0.5
    
    positions = sizer.calculate_positions(mock_signals)
    
    # Check max gross leverage limit (should not exceed)
    assert positions['target_weight'].abs().sum() <= 0.5 + 1e-5
    
    # Check limit clipping
    for w in positions['target_weight']:
        assert abs(w) <= sizer.max_single

def test_drawdown_scaling(mock_signals):
    sizer = IndependentKellySizer()
    sizer.max_single = 1.0  # Prevent individual limits from hiding the effect
    pos_normal = sizer.calculate_positions(mock_signals, current_drawdown=0.0)
    pos_dd = sizer.calculate_positions(mock_signals, current_drawdown=0.08) # 8% DD is between 5% and 10%
    
    # The positions in DD should be smaller than normal
    w_normal = pos_normal['target_weight'].abs().sum()
    w_dd = pos_dd['target_weight'].abs().sum()
    assert w_dd < w_normal

def test_risk_engine_l2_limits():
    engine = RiskEngine()
    engine.single_stock_max = 0.10
    engine.industry_max = 0.30
    
    # Test single stock capping
    target_pos = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT'],
        'target_weight': [0.15, -0.05]
    })
    
    clipped = engine.validate_positions(target_pos)
    assert clipped.loc[0, 'target_weight'] == 0.10
    
    # Test sector capping
    target_pos_sector = pd.DataFrame({
        'symbol': ['A1', 'A2', 'A3', 'A4'],
        'target_weight': [0.10, 0.10, 0.10, 0.10]
    })
    sector_map = {'A1': 'Tech', 'A2': 'Tech', 'A3': 'Tech', 'A4': 'Tech'}
    
    clipped_sector = engine.validate_positions(target_pos_sector, sector_map)
    # Total exposure is 0.40, limit is 0.30. Should scale by 0.3/0.4 = 0.75
    for w in clipped_sector['target_weight']:
        assert abs(w - 0.075) < 1e-5

def test_risk_engine_l3_l4():
    engine = RiskEngine()
    
    # Normal day
    health = engine.check_portfolio_health(daily_pnl=0.005, current_drawdown=0.02, consecutive_loss_days=0)
    assert health['status'] == "NORMAL"
    
    # Kill switch due to DD
    health_dd = engine.check_portfolio_health(daily_pnl=-0.001, current_drawdown=0.15, consecutive_loss_days=1)
    assert health_dd['status'] == "KILL_SWITCH"
    assert engine.is_kill_switched
    
    # Daily loss limit
    engine = RiskEngine()
    health_loss = engine.check_portfolio_health(daily_pnl=-0.02, current_drawdown=0.05, consecutive_loss_days=1)
    assert health_loss['status'] == "HALT_TRADING_TODAY"
    assert health_loss['multiplier'] == 0.0
    
    # Consecutive losses
    health_cons = engine.check_portfolio_health(daily_pnl=-0.001, current_drawdown=0.05, consecutive_loss_days=3)
    assert health_cons['status'] == "REDUCE_RISK"
    assert health_cons['multiplier'] <= 0.5
