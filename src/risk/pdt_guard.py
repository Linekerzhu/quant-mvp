"""
PDT (Pattern Day Trader) Guard Module

Monitors and enforces PDT rules for accounts under $25k.
PDT Rule: 4+ day trades in 5 rolling business days triggers 90-day restriction.

Moved from universe.py (O5 fix) - PDT should be in risk module, not universe.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay  # FIX B5: Import business day offset

from src.ops.event_logger import get_logger

logger = get_logger()


class PDTGuard:
    """
    Pattern Day Trader rule enforcement.
    
    Tracks day trades and prevents PDT violation for sub-$25k accounts.
    """
    
    def __init__(self, max_day_trades: int = 3, rolling_window_days: int = 5):
        """
        Initialize PDT Guard.
        
        Args:
            max_day_trades: Maximum allowed day trades (default 3, PDT limit is 4)
            rolling_window_days: Rolling window for day trade counting (default 5)
        """
        self.max_day_trades = max_day_trades
        self.rolling_window_days = rolling_window_days
    
    def check_pdt_compliance(
        self,
        trade_history: List[Dict],
        account_value: float
    ) -> Tuple[bool, Dict]:
        """
        Check if account is compliant with PDT rules.
        
        Args:
            trade_history: List of trade records with 'date', 'symbol', 'action'
            account_value: Current account value in USD
            
        Returns:
            (is_compliant, details)
            is_compliant: True if trade is allowed
            details: Dict with day trade count, remaining trades, etc.
        """
        # PDT only applies to accounts under $25k
        if account_value >= 25000:
            return True, {'pdt_applies': False, 'account_value': account_value}
        
        # Count day trades in rolling window
        day_trade_count = self._count_day_trades(trade_history)
        
        # Check compliance
        is_compliant = day_trade_count < self.max_day_trades
        remaining = max(0, self.max_day_trades - day_trade_count)
        
        details = {
            'pdt_applies': True,
            'account_value': account_value,
            'day_trade_count': day_trade_count,
            'max_allowed': self.max_day_trades,
            'remaining_trades': remaining,
            'is_compliant': is_compliant
        }
        
        if not is_compliant:
            logger.error("pdt_violation_imminent", {
                "day_trades": day_trade_count,
                "max_allowed": self.max_day_trades,
                "account_value": account_value
            })
        
        return is_compliant, details
    
    def _count_day_trades(self, trade_history: List[Dict]) -> int:
        """
        Count day trades in rolling window.
        
        A day trade is buying and selling the same security on the same day.
        """
        if not trade_history:
            return 0
        
        # Filter to recent trades within window
        # FIX B5: Use business days (BDay) not calendar days for PDT calculation
        cutoff_date = pd.Timestamp.now() - BDay(self.rolling_window_days)
        
        recent_trades = [
            t for t in trade_history
            if pd.Timestamp(t.get('date', '1900-01-01')) >= cutoff_date
        ]
        
        # Group trades by symbol and date
        trades_by_symbol_date = {}
        for trade in recent_trades:
            key = (trade.get('symbol'), trade.get('date'))
            if key not in trades_by_symbol_date:
                trades_by_symbol_date[key] = []
            trades_by_symbol_date[key].append(trade.get('action'))
        
        # Count day trades (buy + sell same day, same symbol)
        day_trades = 0
        for (symbol, date), actions in trades_by_symbol_date.items():
            if 'buy' in actions and 'sell' in actions:
                day_trades += 1
        
        return day_trades
    
    def can_trade_symbol(
        self,
        symbol: str,
        trade_history: List[Dict],
        account_value: float
    ) -> bool:
        """
        Check if trading a specific symbol would violate PDT.
        
        This is a simplified check - assumes the trade would be a day trade.
        """
        is_compliant, details = self.check_pdt_compliance(trade_history, account_value)
        return is_compliant
    
    def get_pdt_status(self, trade_history: List[Dict], account_value: float) -> Dict:
        """
        Get full PDT status for display/monitoring.
        """
        is_compliant, details = self.check_pdt_compliance(trade_history, account_value)
        
        # Calculate days until reset
        if trade_history:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=self.rolling_window_days)
            oldest_trade = min(
                pd.Timestamp(t.get('date', pd.Timestamp.now()))
                for t in trade_history
            )
            days_until_reset = max(0, (oldest_trade + pd.Timedelta(days=self.rolling_window_days) - pd.Timestamp.now()).days)
        else:
            days_until_reset = 0
        
        details['days_until_reset'] = days_until_reset
        details['pdt_restricted'] = not is_compliant and account_value < 25000
        
        return details
