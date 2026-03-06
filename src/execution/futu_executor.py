import futu as ft
from typing import Dict, List, Optional
import time
import pandas as pd

from src.ops.event_logger import get_logger

logger = get_logger()

class FutuExecutor:
    """
    Execution engine for Futu OpenAPI.
    Handles order routing, monitoring, and unlocks.
    """
    def __init__(self, host='127.0.0.1', port=11111, trd_env=ft.TrdEnv.SIMULATE, pwd_unlock: str = None):
        """
        Initialize the Futu executor.
        
        Args:
            host: OpenD address (default: 127.0.0.1 or host.docker.internal)
            port: OpenD port (default: 11111)
            trd_env: ft.TrdEnv.SIMULATE or ft.TrdEnv.REAL
            pwd_unlock: MD5 of trading password (only needed for REAL)
        """
        self.host = host
        self.port = port
        self.trd_env = trd_env
        self.pwd_unlock = pwd_unlock
        self.ctx = None
        self._market = ft.TrdMarket.US
        
        logger.info("futu_executor_init", {"host": host, "port": port, "env": "REAL" if trd_env == ft.TrdEnv.REAL else "SIMULATE"})

    def connect(self) -> bool:
        """Connect to OpenD and unlock trade if in REAL environment."""
        try:
            # Note: For US margin accounts we use OpenUSTradeContext
            self.ctx = ft.OpenUSTradeContext(host=self.host, port=self.port)
            
            if self.trd_env == ft.TrdEnv.REAL:
                if not self.pwd_unlock:
                    raise ValueError("pwd_unlock (MD5) is required for real trading")
                
                ret, data = self.ctx.unlock_trade(self.pwd_unlock)
                if ret != ft.RET_OK:
                    logger.error("futu_unlock_failed", {"msg": data})
                    return False
                    
                logger.info("futu_unlocked", {"status": "success"})
                
            return True
            
        except Exception as e:
            logger.error("futu_connection_error", {"error": str(e)})
            return False

    def get_account_value(self) -> float:
        """Get total asset value (USD)."""
        if not self.ctx:
            return 0.0
            
        ret, data = self.ctx.accinfo_query(trd_env=self.trd_env)
        if ret == ft.RET_OK:
            # Use 'total_assets'
            if not data.empty:
                val = data['total_assets'][0]
                return float(val)
        
        logger.error("futu_accinfo_error", {"msg": data if ret != ft.RET_OK else "Empty data"})
        return 0.0

    def get_positions(self) -> pd.DataFrame:
        """Get current positions."""
        if not self.ctx:
            return pd.DataFrame()
            
        ret, data = self.ctx.position_list_query(trd_env=self.trd_env)
        if ret == ft.RET_OK:
            return data
            
        logger.error("futu_position_error", {"msg": data})
        return pd.DataFrame()

    def submit_orders(self, orders: List[Dict]) -> List[str]:
        """
        Submit a batch of orders.
        Orders should have: symbol, qty, price (for limit), side ('buy'/'sell'), order_type ('limit'/'market')
        """
        if not self.ctx:
            return []
            
        order_ids = []
        for o in orders:
            # Format symbol from AAPL to US.AAPL
            code = f"US.{o['symbol']}"
            side = ft.TrdSide.BUY if o['side'].lower() == 'buy' else ft.TrdSide.SELL
            
            # Use normal LIMIT or MARKET order
            order_type = ft.OrderType.NORMAL if o.get('order_type', 'limit') == 'limit' else ft.OrderType.MARKET
            
            # Place order
            ret, data = self.ctx.place_order(
                price=float(o.get('price', 0.0)),
                qty=float(o['qty']),
                code=code,
                trd_side=side,
                order_type=order_type,
                trd_env=self.trd_env
            )
            
            if ret == ft.RET_OK:
                oid = data['order_id'][0]
                order_ids.append(oid)
                logger.info("futu_order_placed", {"symbol": o['symbol'], "order_id": oid, "qty": o['qty'], "side": o['side']})
            else:
                logger.error("futu_order_failed", {"symbol": o['symbol'], "msg": data})
                
            # Avoid hitting rate limits
            time.sleep(0.1)
            
        return order_ids

    def close(self):
        """Close context."""
        if self.ctx:
            self.ctx.close()
            self.ctx = None
