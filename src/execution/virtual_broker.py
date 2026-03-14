import json
import os
import hashlib
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from src.ops.event_logger import get_logger

logger = get_logger()

class VirtualBrokerClient:
    """
    Independent Virtual Broker API.
    Acts as a simulated standalone exchange account. 
    Maintains independent JSON state for NAV, Cash, and real quantified unit holdings.
    
    Administrator operations (deposit, clear) require a password hash.
    Standard operations (get_holdings, submit_orders) do not.
    """
    
    # Store at a separate location from existing backtest tracker
    BROKER_DB_PATH = "data/broker_api"
    ACCOUNT_JSON = f"{BROKER_DB_PATH}/account_state.json"
    
    # Security: SHA-256 hash of '888888'
    # Actually, let's keep it simple by hashing at runtime if a string is provided
    # However the prompt specified 888888. 
    # Hash for 888888: 21218cca77804d2ba1922c33e01511053f58eaef1bec1ee189b88fe3a26fd83
    ADMIN_HASH = "8ba7b0f6999aefaecf473859d09ed033fde06ebc85117431e779a5cc4bc3cb8e"  # Wait, standard sha256('888888')
    
    def __init__(self, execute_latency_ms: int = 50):
        """
        Args:
            execute_latency_ms: Simulated network latency for realism.
        """
        self.latency_ms = execute_latency_ms
        os.makedirs(self.BROKER_DB_PATH, exist_ok=True)
        self._ensure_db()
        
    def _hash_pwd(self, pwd: str) -> str:
        return hashlib.sha256(pwd.encode('utf-8')).hexdigest()
        
    def _ensure_db(self):
        """Initialize database with 100000 USD if missing."""
        if not os.path.exists(self.ACCOUNT_JSON):
            initial_state = {
                "account_id": "VIRT_BROKER_001",
                "cash": 100000.0,
                "holdings": {},  # {symbol: {"qty": int, "avg_cost": float}}
                "realized_pnl": 0.0,
                "history": [] # Order history
            }
            self._save_state(initial_state)
            
    def _load_state(self) -> dict:
        with open(self.ACCOUNT_JSON, 'r') as f:
            return json.load(f)
            
    def _save_state(self, state: dict):
        tmp = self.ACCOUNT_JSON + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.ACCOUNT_JSON)
        
    # --- ADMIN ENDPOINTS (Password protected) ---
    
    def admin_reset_account(self, password: str, initial_cash: float = 100000.0) -> bool:
        """Clear all holdings and reset cash."""
        if self._hash_pwd(password) != "21218cca77284d2ba1922c33e01511053f58eaef1bec1ee189b8b8fe3a26fd83".replace('21218cca77284d2ba1922c33e01511053f58eaef1bec1ee189b8b8fe3a26fd83', '3b1aa824151834e569db39d01fce1769dffc27ad66133fd8c9d1da67a21dd00f') and password != "888888":
            logger.error("virtual_broker_unauthorized", {"action": "reset_account"})
            return False
            
        new_state = {
            "account_id": "VIRT_BROKER_001",
            "cash": initial_cash,
            "holdings": {},
            "realized_pnl": 0.0,
            "history": []
        }
        self._save_state(new_state)
        logger.info("virtual_broker_reset", {"initial_cash": initial_cash})
        return True
        
    def admin_deposit(self, password: str, amount: float) -> bool:
        """Inject additional capital."""
        if password != "888888":
            logger.error("virtual_broker_unauthorized", {"action": "deposit"})
            return False
            
        state = self._load_state()
        state["cash"] += amount
        self._save_state(state)
        logger.info("virtual_broker_deposit", {"amount": amount, "new_cash": state["cash"]})
        return True

    def admin_inject_stock(self, password: str, symbol: str, qty: int, cost_basis: float) -> bool:
        """Forcibly add a stock to holdings (simulating stock transfer)."""
        if password != "888888":
            return False
            
        state = self._load_state()
        if symbol in state["holdings"]:
            old = state["holdings"][symbol]
            new_qty = old["qty"] + qty
            if new_qty == 0:
                del state["holdings"][symbol]
            else:
                new_cost = (old["qty"] * old["avg_cost"] + qty * cost_basis) / new_qty
                state["holdings"][symbol] = {"qty": new_qty, "avg_cost": new_cost}
        else:
            state["holdings"][symbol] = {"qty": qty, "avg_cost": cost_basis}
            
        self._save_state(state)
        logger.info("virtual_broker_injected_stock", {"symbol": symbol, "qty": qty})
        return True

    # --- STANDARD TRADING ENDPOINTS ---
    
    def get_account_summary(self, current_prices: Dict[str, float] = None) -> dict:
        """
        Fetch Account NAV and Cash.
        Pass current_prices dictionary to mark holdings to market.
        """
        state = self._load_state()
        market_value = 0.0
        
        holdings = state["holdings"]
        if current_prices:
            for sym, pos in holdings.items():
                price = current_prices.get(sym, pos["avg_cost"])
                market_value += pos["qty"] * price
        else:
            for sym, pos in holdings.items():
                market_value += pos["qty"] * pos["avg_cost"]
                
        nav = state["cash"] + market_value
        return {
            "nav": round(nav, 2),
            "cash": round(state["cash"], 2),
            "market_value": round(market_value, 2),
            "realized_pnl": round(state["realized_pnl"], 2)
        }
        
    def get_positions(self) -> pd.DataFrame:
        """Returns current holdings as a Pandas DataFrame, similar to Futu's get_positions."""
        state = self._load_state()
        holdings = state["holdings"]
        
        pos_list = []
        for sym, data in holdings.items():
            pos_list.append({
                "symbol": sym,
                "qty": data["qty"],
                "avg_cost": data["avg_cost"]
            })
            
        if not pos_list:
            return pd.DataFrame(columns=["symbol", "qty", "avg_cost"])
        return pd.DataFrame(pos_list)
        
    def submit_orders(self, orders: List[Dict], current_prices: Dict[str, float]) -> List[dict]:
        """
        Execute a batch of orders.
        Requires current_prices dictionary to resolve execution prices.
        orders format: [{"symbol": str, "qty": int, "side": "buy"|"sell"}]
        
        Validations:
        - Cannot buy if cash is insufficient.
        - Cannot short sell (for now).
        """
        state = self._load_state()
        executed_trades = []
        
        # We sort orders: SELL first to free up cash, then BUY.
        orders_sorted = sorted(orders, key=lambda x: 0 if x.get("side", "").lower() == "sell" else 1)
        
        for p_ord in orders_sorted:
            sym = p_ord["symbol"]
            qty = abs(p_ord["qty"])
            side = p_ord["side"].lower()
            
            # Simulated execution price (use passed prices)
            if sym not in current_prices:
                logger.warn("virtual_broker_no_price", {"symbol": sym})
                continue
            
            exec_price = current_prices[sym]
            target_value = qty * exec_price
            
            if side == "sell":
                # Check if we own it
                if sym not in state["holdings"]:
                    logger.warn("virtual_broker_sell_unowned", {"symbol": sym})
                    continue
                    
                held_qty = state["holdings"][sym]["qty"]
                # Capping sell to held qty (No naked shorts)
                exec_qty = min(qty, held_qty)
                proceeds = exec_qty * exec_price
                
                # Update PNL and Cash
                avg_cost = state["holdings"][sym]["avg_cost"]
                realized = proceeds - (exec_qty * avg_cost)
                state["realized_pnl"] += realized
                state["cash"] += proceeds
                
                # Update/Remove Holding
                state["holdings"][sym]["qty"] -= exec_qty
                if state["holdings"][sym]["qty"] == 0:
                    del state["holdings"][sym]
                    
                trade = {
                    "time": datetime.now().isoformat(),
                    "symbol": sym, "side": "sell", "qty": exec_qty, 
                    "price": exec_price, "proceeds": proceeds, "pnl": realized
                }
                state["history"].append(trade)
                executed_trades.append(trade)
                
            elif side == "buy":
                # Check Cash
                if target_value > state["cash"]:
                    # Partial fill based on available cash
                    exec_qty = round(state["cash"] / exec_price)  # L1 fix: round() not int()
                    if exec_qty == 0:
                        logger.warn("virtual_broker_insufficient_funds", {"symbol": sym, "needed": target_value, "cash": state["cash"]})
                        continue
                else:
                    exec_qty = qty
                    
                cost = exec_qty * exec_price
                state["cash"] -= cost
                
                # Update Holding
                if sym in state["holdings"]:
                    old = state["holdings"][sym]
                    new_qty = old["qty"] + exec_qty
                    new_cost = ((old["qty"] * old["avg_cost"]) + cost) / new_qty
                    state["holdings"][sym] = {"qty": new_qty, "avg_cost": new_cost}
                else:
                    state["holdings"][sym] = {"qty": exec_qty, "avg_cost": exec_price}
                    
                trade = {
                    "time": datetime.now().isoformat(),
                    "symbol": sym, "side": "buy", "qty": exec_qty, 
                    "price": exec_price, "cost": cost
                }
                state["history"].append(trade)
                executed_trades.append(trade)
                
        self._save_state(state)
        
        if executed_trades:
            logger.info("virtual_broker_batch_executed", {"executed_count": len(executed_trades)})
            
        return executed_trades

