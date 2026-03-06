import pytest
import os
import yaml
from unittest.mock import patch, MagicMock
from src.backtest.cost_calibration import CostCalibrator

@patch("src.backtest.cost_calibration.CostCalibrator._save_config")
def test_cost_calibration(mock_save):
    calibrator = CostCalibrator()
    
    # Mock config
    calibrator.config = {
        "spread_bps": {
            "by_adv_bucket": {
                "low": 2.0,
                "mid": 1.0,
                "high": 0.5
            }
        },
        "calibration": {
            "min_samples_for_update": 2,
            "alert_threshold_mult": 2.0,
            "max_param_change_pct": 100.0
        }
    }
    
    # Generate mock trades for 'high' bucket
    # Assuming mid price 100.0, we want to simulate 1.5 bps slippage
    # 1.5 bps of $100 = $0.015
    # Fill price = $100.015 (worse)
    trades = [
        {'symbol': 'AAPL', 'qty': 10, 'side': 'buy', 'submitted_mid_price': 100.0, 'fill_price': 100.015, 'adv_usd': 500_000_000},
        {'symbol': 'MSFT', 'qty': 20, 'side': 'buy', 'submitted_mid_price': 100.0, 'fill_price': 100.015, 'adv_usd': 300_000_000},
        {'symbol': 'TSLA', 'qty': 10, 'side': 'sell', 'submitted_mid_price': 100.0, 'fill_price': 99.985, 'adv_usd': 200_000_000}
    ]
    
    # Run calibration
    updates = calibrator.calibrate(trades)
    
    # Check that high bucket was updated from 0.5 to 1.0 (Clipped by max_change_pct = 100%)
    assert updates is True
    assert mock_save.called
    assert calibrator.config["spread_bps"]["by_adv_bucket"]["high"] == pytest.approx(1.0, 0.01)
