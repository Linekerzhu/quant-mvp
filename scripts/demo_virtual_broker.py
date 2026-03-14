import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.virtual_broker import VirtualBrokerClient

broker = VirtualBrokerClient()

# Print Initial
print("--- Initial State ---")
print(broker.get_account_summary())

# Test Auth Reject
print("\n--- Try reject (wrong password) ---")
broker.admin_deposit("wrong", 50000)
print(broker.get_account_summary())

# Test Auth Accept
print("\n--- Deposit 50K (right password) ---")
broker.admin_deposit("888888", 50000)
print(broker.get_account_summary())

# Inject AAPL for simulated past trades
print("\n--- Inject 100 shares of AAPL @ $150 ---")
broker.admin_inject_stock("888888", "AAPL", 100, 150.0)

prices = {"AAPL": 160.0}
print(broker.get_account_summary(current_prices=prices))
print("Holdings:")
print(broker.get_positions())

# Let's say model wants to sell 50 AAPL
print("\n--- Submit Order: Sell 50 AAPL @ Market($160) ---")
orders = [{"symbol": "AAPL", "qty": 50, "side": "sell", "order_type": "limit"}]
executed = broker.submit_orders(orders, prices)
print("Executed:", executed)
print(broker.get_account_summary(current_prices=prices))
print("Holdings:")
print(broker.get_positions())
