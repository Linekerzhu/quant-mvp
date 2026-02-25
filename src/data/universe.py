"""
Universe Management Module

Manages stock universe, filtering, and component changes.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

from src.ops.event_logger import get_logger

logger = get_logger()


class UniverseManager:
    """Manages the trading universe."""
    
    def __init__(self, config_path: str = "config/universe.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.min_adv = self.config['filters']['min_adv_usd']
        self.min_listing_years = self.config['filters']['min_listing_years']
        self.min_history_days = self.config['filters']['min_history_days']
        
        # Component change handling
        self.cold_start_days = self.config['component_changes']['new_component']['min_history_days']
        # P0 (R26-S1): Fix key name - YAML uses 'limit_slippage' not 'exit_limit_slippage'
        self.exit_slippage = self.config['component_changes']['removed_component']['limit_slippage']
    
    def get_sp500_tickers(self) -> List[str]:
        """Get current S&P 500 tickers."""
        try:
            # Try to fetch from Wikipedia (free source)
            import requests
            from bs4 import BeautifulSoup
            
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': 'constituents'})
            tickers = []
            
            for row in table.find_all('tr')[1:]:  # Skip header
                ticker = row.find_all('td')[0].text.strip()
                tickers.append(ticker.replace('.', '-'))  # BRK.B -> BRK-B
            
            return tickers
        except Exception as e:
            logger.error("sp500_fetch_failed", {"error": str(e)})
            # FIX B2: Use static fallback file instead of MOCK tickers
            fallback_path = Path("data/sp500_fallback.csv")
            if fallback_path.exists():
                logger.info("using_fallback_tickers", {"source": str(fallback_path)})
                fallback_df = pd.read_csv(fallback_path)
                return fallback_df['symbol'].tolist()
            else:
                logger.error("no_fallback_available")
                raise RuntimeError(
                    f"Failed to fetch S&P 500 tickers and no fallback file at {fallback_path}. "
                    "Please ensure data/sp500_fallback.csv exists or check network connectivity."
                )
    
    def filter_liquidity(
        self,
        df: pd.DataFrame,
        lookback_days: int = 20
    ) -> List[str]:
        """
        Filter symbols by liquidity (ADV > $5M).
        
        Args:
            df: Price data with volume
            lookback_days: Days to calculate ADV
            
        Returns:
            List of liquid symbols
        """
        # P0-A5: Guard clause for empty DataFrame
        if df.empty or 'date' not in df.columns:
            return []
        
        # P1 (R26-A1): Use BusinessDay instead of calendar days
        # 20 calendar days ≈ 15 trading days, causing most symbols to be rejected
        from pandas.tseries.offsets import BDay
        recent = df[df['date'] >= df['date'].max() - BDay(lookback_days)]
        
        liquid = []
        
        for symbol in recent['symbol'].unique():
            symbol_df = recent[recent['symbol'] == symbol]
            
            if len(symbol_df) < lookback_days * 0.8:  # Require 80% data coverage
                continue
            
            # Calculate ADV in USD
            symbol_df = symbol_df.copy()
            symbol_df['dollar_volume'] = symbol_df['raw_close'] * symbol_df['volume']
            adv = symbol_df['dollar_volume'].mean()
            
            if adv >= self.min_adv:
                liquid.append(symbol)
        
        logger.info("liquidity_filter", {
            "input_count": df['symbol'].nunique(),
            "output_count": len(liquid),
            "min_adv": self.min_adv
        })
        
        return liquid
    
    def check_history_length(
        self,
        df: pd.DataFrame,
        min_days: Optional[int] = None
    ) -> List[str]:
        """
        Check if symbols have sufficient history.
        
        Args:
            df: Price data
            min_days: Minimum history days (default from config)
            
        Returns:
            List of symbols with sufficient history
        """
        # P0-A5: Guard clause for empty DataFrame
        if df.empty or 'symbol' not in df.columns:
            return []
        
        min_days = min_days or self.min_history_days
        
        valid = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            valid_days = symbol_df['raw_close'].notna().sum()
            
            if valid_days >= min_days:
                valid.append(symbol)
        
        return valid
    
    def handle_component_change(
        self,
        new_components: List[str],
        removed_components: List[str],
        current_positions: Dict[str, float],
        df: pd.DataFrame,
        rebalance_date: Optional[str] = None,
        evidence_url: Optional[str] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Handle universe component changes.
        
        Args:
            new_components: List of new symbols added to universe
            removed_components: List of symbols removed from universe
            current_positions: Current portfolio positions
            df: Price data DataFrame
            rebalance_date: Rebalance effective date (ISO format)
            evidence_url: URL to evidence/source document (Patch 5)
            
        Returns:
            (new_universe, exit_orders)
        """
        exit_orders = []
        
        # Handle removed components
        for symbol in removed_components:
            if symbol in current_positions:
                # Get last price
                symbol_df = df[df['symbol'] == symbol]
                if not symbol_df.empty:
                    last_price = symbol_df['raw_close'].iloc[-1]
                    limit_price = last_price * (1 - self.exit_slippage)
                    
                    exit_orders.append({
                        'symbol': symbol,
                        'action': 'exit',
                        'reason': 'component_removed',
                        'order_type': 'limit',
                        'limit_price': limit_price,
                        'fallback': 'market_next_day',
                        # Patch 5 fields
                        'rebalance_date_source': rebalance_date,
                        'evidence': evidence_url
                    })
                    
                    logger.info("component_exit_order", {
                        "symbol": symbol,
                        "limit_price": limit_price,
                        "reason": "component_removed",
                        "rebalance_date": rebalance_date,
                        "evidence": evidence_url
                    }, symbol)
        
        # New components go into cold start (data collection only)
        for symbol in new_components:
            logger.info("component_cold_start", {
                "symbol": symbol,
                "min_days": self.cold_start_days,
                # Patch 5 fields
                "rebalance_date": rebalance_date,
                "evidence": evidence_url
            }, symbol)
        
        return exit_orders
    
    def build_universe(
        self,
        df: pd.DataFrame,
        current_universe: Optional[List[str]] = None
    ) -> Dict:
        """
        Build the complete trading universe.
        
        Returns:
            Universe info dict with:
            - symbols: List of active symbols
            - cold_start: List of symbols in cold start
            - metadata: Universe metadata
        """
        # Get base universe
        if current_universe is None:
            base_tickers = self.get_sp500_tickers()
        else:
            base_tickers = current_universe
        
        # P0-A5: Guard clause for empty DataFrame (first run scenario)
        if df.empty or 'symbol' not in df.columns:
            return {
                'symbols': sorted(base_tickers[:10]),
                'cold_start': sorted(base_tickers[:10]),
                'count': min(len(base_tickers), 10),
                'cold_start_count': min(len(base_tickers), 10),
                'metadata': {
                    'base_tickers': len(base_tickers),
                    'with_data': 0,
                    'liquid': 0,
                    'sufficient_history': 0,
                    'survivorship_bias_method': self.config['universe']['survivorship_bias']['method'],
                    'first_run': True
                }
            }
        
        # Filter to available data
        available = df[df['symbol'].isin(base_tickers)]['symbol'].unique().tolist()
        
        # Filter by liquidity
        liquid = self.filter_liquidity(df)
        
        # Filter by history length
        with_history = self.check_history_length(df)
        
        # Active universe (all criteria met)
        active = list(set(available) & set(liquid) & set(with_history))
        
        # Cold start (new components with data but insufficient history)
        cold_start = list(set(available) & set(liquid) - set(with_history))
        
        # Build info
        info = {
            'symbols': sorted(active),
            'cold_start': sorted(cold_start),
            'count': len(active),
            'cold_start_count': len(cold_start),
            'metadata': {
                'base_tickers': len(base_tickers),
                'with_data': len(available),
                'liquid': len(liquid),
                'sufficient_history': len(with_history),
                'survivorship_bias_method': self.config['universe']['survivorship_bias']['method']
            }
        }
        
        logger.info("universe_built", info['metadata'])
        
        return info
    
    # Note: PDT compliance checking moved to src/risk/pdt_guard.py (O5 fix)
    # Use PDTGuard class for PDT rule enforcement
