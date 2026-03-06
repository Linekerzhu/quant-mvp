import pytest
import pandas as pd
import numpy as np

from src.risk.position_sizing import IndependentKellySizer
from src.risk.risk_engine import RiskEngine

def test_extreme_stress_scenarios():
    """
    Simulate Black Swan / Extreme Market Events to ensure 
    the Risk Engine and Position Sizer degrade gracefully.
    """
    sizer = IndependentKellySizer()
    sizer.min_single = 0.0001 # Disable dust culling for precise numeric tests
    engine = RiskEngine()
    
    # ---------------------------------------------------------
    # Scenario A: VIX Spike / Flash Crash
    # Stock hits an extreme realized volatility (e.g., 200% annualized)
    # ---------------------------------------------------------
    extreme_vol_signals = pd.DataFrame({
        'symbol': ['SPY', 'QQQ'],
        'side': [1, -1],
        'prob': [0.55, 0.60],
        'avg_win': [0.03, 0.04],
        'avg_loss': [0.03, 0.04],
        'realized_vol': [2.00, 1.50] # Extreme volatility! Normal is 0.15~0.20
    })
    
    # Volatility bounds should drastically reduce the target weight
    # target_vol is 0.15. For SPY, multiplier is 0.15 / 2.0 = 0.075 (7.5% of normal position)
    pos = sizer.calculate_positions(extreme_vol_signals)
    
    spy_weight = pos.loc[pos['symbol'] == 'SPY', 'target_weight'].iloc[0]
    qqq_weight = pos.loc[pos['symbol'] == 'QQQ', 'target_weight'].iloc[0]
    
    # Assert severe reduction (should be less than 1%)
    assert abs(spy_weight) < 0.05
    assert abs(qqq_weight) < 0.05
    
    # ---------------------------------------------------------
    # Scenario B: Cascading Portfolio Drawdown (Approaching Kill Switch)
    # ---------------------------------------------------------
    # 8% Drawdown: Should trigger position floor (25% of allowed limits)
    pos_moderate_dd = sizer.calculate_positions(extreme_vol_signals, current_drawdown=0.08)
    
    spy_weight_dd = pos_moderate_dd.loc[pos_moderate_dd['symbol'] == 'SPY', 'target_weight'].iloc[0]
    assert abs(spy_weight_dd) < abs(spy_weight)

    # Risk Engine Check
    # At 8% DD, Auto-degrade should flag True
    health = engine.check_portfolio_health(daily_pnl=-0.05, current_drawdown=0.08, consecutive_loss_days=2)
    assert health['status'] == "HALT_TRADING_TODAY" # daily loss -5% > 1%
    assert health['is_auto_degraded'] is True
    assert health['is_kill_switched'] is False
    
    # ---------------------------------------------------------
    # Scenario C: Black Swan (Triggering Kill-Switch)
    # ---------------------------------------------------------
    # At 12% DD (e.g., flash crash wipes out large portion)
    health_kill = engine.check_portfolio_health(daily_pnl=-0.08, current_drawdown=0.13, consecutive_loss_days=1)
    
    assert health_kill['status'] == "KILL_SWITCH"
    assert health_kill['is_kill_switched'] is True
    assert health_kill['multiplier'] == 0.0
    
    # ---------------------------------------------------------
    # Scenario D: Extended Losing Streak (Bleeding Death)
    # ---------------------------------------------------------
    # Even if DD is small, 4 consecutive losing days triggers reduction
    health_bleed = engine.check_portfolio_health(daily_pnl=-0.005, current_drawdown=0.04, consecutive_loss_days=4)
    assert health_bleed['status'] == "REDUCE_RISK"
    assert health_bleed['multiplier'] <= 0.5
