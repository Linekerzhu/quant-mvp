import pytest
from src.execution.slippage_model import SlippageModel

def test_futu_slippage_model():
    model = SlippageModel()
    
    # 1. Buy 100 shares @ $50 ($5000), default ADV
    res1 = model.estimate_cost(qty=100, price=50.0, side="buy")
    assert res1['commission'] >= 0.99 # min order
    assert res1['platform_fee'] >= 1.00 # min platform
    assert res1['sec_fee'] == 0.0 # SEC is sell only
    assert res1['taf_fee'] == 0.0 # TAF is sell only
    assert res1['finra_fee'] > 0.0
    
    # 2. Sell 100 shares @ $50 ($5000), low ADV
    res2 = model.estimate_cost(qty=100, price=50.0, side="sell", adv_usd=10_000_000)
    assert res2['sec_fee'] > 0.0
    assert res2['taf_fee'] > 0.0
    
    # spread bps should be 2.0 based on config for low bucket
    assert res2['slippage_cost'] > 0.0
    assert res2['fill_price'] < 50.0 # sell is worse than mid
