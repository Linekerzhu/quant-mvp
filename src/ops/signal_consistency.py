import pandas as pd
from typing import List, Dict, Any
from src.ops.event_logger import get_logger

logger = get_logger()

class SignalConsistency:
    """
    Verifies that the generated research signals accurately translate
    to the submitted orders.
    """
    @staticmethod
    def verify(research_signals: pd.DataFrame, submitted_orders: List[Dict]) -> Dict[str, Any]:
        """
        Verify target weights against orders.
        
        Args:
            research_signals: DataFrame with 'symbol', 'target_weight'
            submitted_orders: List of dicts (symbol, qty, side, price)
            
        Returns:
            Dict with consistency metrics
        """
        # Filter non-zero target weights
        active_signals = research_signals[research_signals['target_weight'] != 0].copy()
        
        order_map = {o['symbol']: o for o in submitted_orders}
        
        inconsistencies = []
        
        for _, row in active_signals.iterrows():
            sym = row['symbol']
            weight = row['target_weight']
            expected_side = 'buy' if weight > 0 else 'sell'
            
            if sym not in order_map:
                inconsistencies.append({
                    'type': 'missing_order',
                    'symbol': sym,
                    'target_weight': float(weight)
                })
                continue
                
            order = order_map[sym]
            if order['side'].lower() != expected_side:
                inconsistencies.append({
                    'type': 'direction_mismatch',
                    'symbol': sym,
                    'expected': expected_side,
                    'actual': order['side']
                })
        
        # Check for ghost orders (orders submitted without a valid signal)
        signal_symbols = set(active_signals['symbol'])
        for sym in order_map.keys():
            if sym not in signal_symbols:
                inconsistencies.append({
                    'type': 'ghost_order',
                    'symbol': sym,
                    'order': order_map[sym]
                })
                
        metrics = {
            "total_signals": len(active_signals),
            "total_orders": len(submitted_orders),
            "inconsistencies": len(inconsistencies),
            "inconsistency_rate": len(inconsistencies) / max(1, len(active_signals)),
            "details": inconsistencies
        }
        
        if inconsistencies:
            logger.warn("signal_consistency_failed", metrics)
        else:
            logger.info("signal_consistency_passed", {"total": len(active_signals)})
            
        return metrics
