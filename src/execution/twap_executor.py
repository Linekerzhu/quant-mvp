"""
Time-Weighted Average Price (TWAP) Execution Algorithm
Phase L3: Micro-Structure Execution

Slices large orders into smaller chunks to reduce market impact (slippage).
Executes them at regular intervals over a specified time window.
"""

import time
import math
from typing import Dict, List
import pandas as pd
from datetime import datetime

from src.execution.futu_executor import FutuExecutor
from src.ops.event_logger import get_logger

logger = get_logger()

class TwapExecutor:
    """
    TWAP Algorithm Wrapper for FutuExecutor.
    Slices large quantity orders into N chunks and fires them 
    over a defined duration to minimize market impact.
    """
    
    def __init__(self, 
                 futu_executor: FutuExecutor, 
                 duration_minutes: int = 30,
                 max_slice_value: float = 10000.0,
                 min_slice_qty: int = 10):
        """
        Args:
            futu_executor: Initialized FutuExecutor instance
            duration_minutes: Total time window to push all slices (e.g., 30 mins)
            max_slice_value: Maximum USD value per slice to avoid impact ($10k default)
            min_slice_qty: Minimum share quantity per slice (10 shares)
        """
        self.executor = futu_executor
        self.duration_minutes = duration_minutes
        self.max_slice_value = max_slice_value
        self.min_slice_qty = min_slice_qty
        
    def _calculate_slices(self, order: Dict) -> List[Dict]:
        """
        Calculate the required slices for a single parent order.
        Returns a list of sub-orders.
        """
        sym = order['symbol']
        total_qty = float(order['qty'])
        side = order['side']
        
        # Need current price to calculate USD value of slice
        # In a real environment, we'd pull live quote. 
        # For simplicity, if price is in the order dict, use it. Otherwise default safely to 100 for slicing math.
        price = float(order.get('price', 100.0))
        total_value = total_qty * price
        
        slices = []
        
        # Determine number of slices required
        # Constraint 1: Value limit per slice
        num_slices_by_value = math.ceil(total_value / self.max_slice_value)
        
        # Constraint 2: Quantity limit per slice
        num_slices_by_qty = math.floor(total_qty / self.min_slice_qty)
        if num_slices_by_qty < 1: 
            num_slices_by_qty = 1 
            
        # We want the max slices that don't violate minimum quantity
        n_slices = min(num_slices_by_value, num_slices_by_qty)
        
        # If order is too small to slice, just execute as 1 chunk
        if n_slices <= 1:
            return [order]
            
        base_qty = math.floor(total_qty / n_slices)
        remainder = int(total_qty - (base_qty * n_slices))
        
        for i in range(n_slices):
            # Give remainder to the first slice
            slice_qty = base_qty + (remainder if i == 0 else 0)
            if slice_qty > 0:
                sub_order = order.copy()
                sub_order['qty'] = slice_qty
                sub_order['twap_slice'] = f"{i+1}/{n_slices}"
                slices.append(sub_order)
                
        logger.info("twap_sliced_order", {
            "symbol": sym, 
            "total_qty": total_qty, 
            "n_slices": len(slices),
            "slice_qty": base_qty
        })
        
        return slices
        
    def execute_twap_batch(self, parent_orders: List[Dict]):
        """
        Takes a list of parent orders (the target rebalancing deltas),
        slices them, and executes the slices asynchronously across the time window.
        
        Note: In a true production system, this sleep approach blocks the thread.
        A robust implementation would use async/await or APScheduler to map
        execution times. For Phase L MVP, we demonstrate the slicing logic.
        """
        if not parent_orders:
            return
            
        logger.info("twap_batch_started", {"n_orders": len(parent_orders), "duration_min": self.duration_minutes})
        
        # 1. Slice all parent orders
        all_slices = {} # key: slice_index, val: list of orders to execute at that index
        max_n_slices = 0
        
        for order in parent_orders:
            slices = self._calculate_slices(order)
            n = len(slices)
            max_n_slices = max(max_n_slices, n)
            
            for i, sub_order in enumerate(slices):
                if i not in all_slices:
                    all_slices[i] = []
                all_slices[i].append(sub_order)
                
        if max_n_slices == 0:
            return
            
        # 2. Calculate time interval between slices
        # If duration is 30 mins and we have 5 slices, interval is 6 mins wait.
        interval_seconds = (self.duration_minutes * 60) / max_n_slices
        
        # 3. Execute Over Time
        for i in range(max_n_slices):
            chunk_orders = all_slices.get(i, [])
            if chunk_orders:
                logger.info("twap_executing_chunk", {"chunk": i+1, "total_chunks": max_n_slices, "n_orders_in_chunk": len(chunk_orders)})
                # Submit via FutuExecutor
                self.executor.submit_orders(chunk_orders)
                
            # Sleep until next interval (if not the last one)
            if i < max_n_slices - 1:
                logger.info("twap_waiting", {"seconds": interval_seconds, "next_chunk": i+2})
                # In production, this would be an async sleep or scheduled task.
                time.sleep(interval_seconds)
                
        logger.info("twap_batch_completed", {"max_slices": max_n_slices})
