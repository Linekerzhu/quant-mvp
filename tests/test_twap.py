import time
from typing import Dict, List
from src.execution.twap_executor import TwapExecutor

class MockFutuExecutor:
    def __init__(self):
        self.submitted_orders = []
        
    def submit_orders(self, orders: List[Dict]):
        self.submitted_orders.extend(orders)
        print(f"Mock executed {len(orders)} orders.")
        for o in orders:
            print(f"  -> {o['symbol']} {o['side']} {o['qty']} shares @ ${o['price']} (Slice {o.get('twap_slice', '1/1')})")
        return [f"oid_{i}" for i in range(len(orders))]

def test_twap_slicing():
    print("--- Testing TWAP Executor Slicing Logic ---")
    mock_futu = MockFutuExecutor()
    
    # 30 minute window, max $10k per slice, min 10 shares
    twap = TwapExecutor(mock_futu, duration_minutes=30, max_slice_value=10000, min_slice_qty=10)
    
    # Large order: 1500 shares of AAPL @ $150 = $225,000 total value
    # Expected slices = math.ceil(225000 / 10000) = 23 slices
    # Let's say we have 600 shares of TSLA @ 200 = $120,000 total value -> 12 slices
    parent_orders = [
        {"symbol": "AAPL", "side": "buy", "qty": 1500, "price": 150.0},
        {"symbol": "TSLA", "side": "buy", "qty": 600, "price": 200.0},
        {"symbol": "PENNY", "side": "buy", "qty": 25, "price": 2.0} # $50 value. Should NOT be sliced because qty is too small (25 / 2 < 10?) Wait, 25/2 = 12.5 > 10. Could be 2 slices of 12 & 13.
    ]
    
    for order in parent_orders:
        slices = twap._calculate_slices(order)
        print(f"\nParent Order: {order['symbol']} {order['qty']} shares @ ${order['price']} (Total: ${order['qty']*order['price']:,.2f})")
        print(f"Generated {len(slices)} slices:")
        for s in slices[:3]: # print first 3
            slice_desc = s.get('twap_slice', '1/1')
            print(f"  {slice_desc}: {s['qty']} shares (${s['qty']*s['price']:,.2f})")
        if len(slices) > 3:
            print(f"  ... and {len(slices)-3} more slices of {slices[-1]['qty']} shares")
            
    print("\nTotal execution batches:")
    # We will patch time.sleep to not actually sleep for the test
    original_sleep = time.sleep
    time.sleep = lambda x: print(f"[System Wait] Pausing for {x:.1f} seconds...")
    
    twap.execute_twap_batch(parent_orders)
    
    time.sleep = original_sleep
    
    assert len(mock_futu.submitted_orders) == sum(len(twap._calculate_slices(o)) for o in parent_orders)
    print("\n✅ TWAP Simulation passed")

if __name__ == '__main__':
    test_twap_slicing()
