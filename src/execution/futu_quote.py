import futu as ft
from typing import Dict, List, Optional
import time

from src.ops.event_logger import get_logger

logger = get_logger()

class FutuQuote:
    """
    Real-time quote engine for Futu OpenAPI.
    Handles order book subscriptions for mid-price execution mapping.
    """
    def __init__(self, host='127.0.0.1', port=11111):
        self.host = host
        self.port = port
        self.ctx = None
        
        logger.info("futu_quote_init", {"host": host, "port": port})

    def connect(self) -> bool:
        """Connect to OpenD quote context."""
        try:
            self.ctx = ft.OpenQuoteContext(host=self.host, port=self.port)
            return True
        except Exception as e:
            logger.error("futu_quote_error", {"error": str(e)})
            return False

    def get_orderbook(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Temporarily subscribe to ORDER_BOOK, fetch best Bid/Ask, and unsubscribe.
        Returns map of {symbol: {'bid': val, 'ask': val, 'mid': val}}
        """
        if not self.ctx or not symbols:
            return {}
            
        # Format symbols from AAPL to US.AAPL
        codes = [f"US.{sym}" for sym in symbols]
        
        # Subscribe
        ret, data = self.ctx.subscribe(codes, [ft.SubType.ORDER_BOOK], is_first_push=False)
        if ret != ft.RET_OK:
            logger.error("futu_subscribe_failed", {"msg": data})
            return {}
            
        results = {}
        for sym, code in zip(symbols, codes):
            # Fetch
            ret, data = self.ctx.get_order_book(code)
            if ret == ft.RET_OK and not data.empty:
                # Find best bid/ask
                # data format has 'Bid' and 'Ask' Series containing lists of [price, qty, num]
                # Assuming index 0 is best
                try:
                    best_bid = data['Bid'][0][0][0] if len(data['Bid']) > 0 and len(data['Bid'][0]) > 0 else 0.0
                    best_ask = data['Ask'][0][0][0] if len(data['Ask']) > 0 and len(data['Ask'][0]) > 0 else 0.0
                    
                    if best_bid > 0 and best_ask > 0:
                        results[sym] = {
                            'bid': best_bid,
                            'ask': best_ask,
                            'mid': (best_bid + best_ask) / 2.0
                        }
                except Exception as e:
                    logger.warning("futu_parse_orderbook_failed", {"symbol": sym, "err": str(e)})
            else:
                logger.warning("futu_get_orderbook_failed", {"symbol": sym, "msg": data if ret != ft.RET_OK else "Empty data"})
                
            time.sleep(0.1) # Prevent limit rate
            
        # Unsubscribe to free quotas
        self.ctx.unsubscribe(codes, [ft.SubType.ORDER_BOOK])
        
        return results

    def close(self):
        """Close quote context."""
        if self.ctx:
            self.ctx.close()
            self.ctx = None
