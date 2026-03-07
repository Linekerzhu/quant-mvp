"""
Virtual Portfolio Tracker

Maintains a simulated portfolio with:
- Position tracking (symbol, qty, avg_cost)
- Cash management with transaction friction
- Daily mark-to-market P&L
- Cumulative performance metrics

All state persisted to data/portfolio/state.json
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.ops.event_logger import get_logger

logger = get_logger()

PORTFOLIO_DIR = "data/portfolio"
STATE_FILE = f"{PORTFOLIO_DIR}/state.json"
TRADE_LOG = f"{PORTFOLIO_DIR}/trades.jsonl"


class PortfolioTracker:
    """Virtual portfolio tracker with friction modeling."""

    def __init__(self, initial_cash: float = 100000.0, friction: float = 0.002):
        """
        Args:
            initial_cash: Starting capital in USD
            friction: Per-trade friction (0.002 = 0.2% round-trip cost)
        """
        self.friction = friction
        self.initial_cash = initial_cash
        
        os.makedirs(PORTFOLIO_DIR, exist_ok=True)
        self.state = self._load_state(initial_cash)

    def _load_state(self, initial_cash: float) -> dict:
        """Load portfolio state from disk."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            logger.info("portfolio_loaded", {
                "cash": state['cash'],
                "positions": len(state.get('positions', {})),
                "inception": state.get('inception_date', 'unknown')
            })
            return state
        
        # Initialize new portfolio
        state = {
            "inception_date": datetime.now().strftime('%Y-%m-%d'),
            "initial_cash": initial_cash,
            "cash": initial_cash,
            "positions": {},  # {symbol: {qty, avg_cost, side}}
            "realized_pnl": 0.0,
            "total_friction_paid": 0.0,
            "trade_count": 0,
            "history": []  # Daily snapshots [{date, nav, cash, positions_value, ...}]
        }
        self._save_state(state)
        logger.info("portfolio_initialized", {"cash": initial_cash})
        return state

    def _save_state(self, state: dict):
        """Atomically save portfolio state."""
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)

    def _log_trade(self, trade: dict):
        """Append trade to log file."""
        with open(TRADE_LOG, 'a') as f:
            f.write(json.dumps(trade, default=str) + '\n')

    def execute_targets(self, targets: Dict[str, float], prices: Dict[str, float], 
                        trade_date: str) -> dict:
        """
        Execute target weights against virtual portfolio.
        
        Args:
            targets: {symbol: target_weight} - target portfolio weights
            prices: {symbol: current_price} - current market prices
            trade_date: Date string for logging
            
        Returns:
            Execution summary dict
        """
        state = self.state
        nav_before = self._calc_nav(prices)
        
        # Calculate target positions in shares
        target_positions = {}
        for sym, weight in targets.items():
            if abs(weight) < 0.005:  # Skip dust
                continue
            if sym not in prices or prices[sym] <= 0:
                continue
            target_usd = abs(weight) * nav_before
            target_qty = int(target_usd / prices[sym])
            if weight < 0:
                target_qty = -target_qty
            target_positions[sym] = target_qty

        # Determine trades needed
        current_positions = state['positions']
        trades_executed = []
        total_friction = 0.0

        # Close positions not in targets
        for sym in list(current_positions.keys()):
            if sym not in target_positions:
                pos = current_positions[sym]
                qty = pos['qty']
                price = prices.get(sym, pos['avg_cost'])
                
                # Close position
                proceeds = qty * price
                friction_cost = abs(proceeds) * self.friction
                total_friction += friction_cost
                
                # P&L
                cost_basis = qty * pos['avg_cost']
                realized = proceeds - cost_basis - friction_cost
                state['realized_pnl'] += realized
                state['cash'] += proceeds - friction_cost
                
                trade = {
                    'date': trade_date, 'symbol': sym, 'action': 'CLOSE',
                    'qty': -qty, 'price': price, 'friction': friction_cost,
                    'realized_pnl': realized
                }
                trades_executed.append(trade)
                self._log_trade(trade)
                del current_positions[sym]

        # Adjust existing positions and open new ones
        for sym, target_qty in target_positions.items():
            current_qty = current_positions.get(sym, {}).get('qty', 0)
            delta = target_qty - current_qty
            
            if abs(delta) < 1:  # No change needed
                continue
            
            # Skip short sells — this is a long-only virtual portfolio
            if target_qty < 0:
                logger.info("portfolio_skip_short", {"symbol": sym, "target_qty": target_qty})
                continue
                
            price = prices[sym]
            trade_value = abs(delta * price)
            friction_cost = trade_value * self.friction
            total_friction += friction_cost

            if delta > 0:  # Buy
                cost = delta * price + friction_cost
                if cost > state['cash']:
                    # Reduce qty to fit available cash
                    max_affordable = int((state['cash'] / (price * (1 + self.friction))))
                    delta = max_affordable
                    if delta <= 0:
                        continue
                    trade_value = delta * price
                    friction_cost = trade_value * self.friction
                    cost = trade_value + friction_cost
                
                state['cash'] -= cost
                
                # Update average cost (includes friction in cost basis)
                effective_price = cost / delta  # price + friction per share
                if sym in current_positions:
                    old_qty = current_positions[sym]['qty']
                    old_cost = current_positions[sym]['avg_cost']
                    new_qty = old_qty + delta
                    # Weighted average cost including friction
                    new_avg = (old_qty * old_cost + delta * effective_price) / new_qty
                    current_positions[sym] = {'qty': new_qty, 'avg_cost': new_avg}
                else:
                    current_positions[sym] = {'qty': delta, 'avg_cost': effective_price}
                    
            else:  # Sell (delta < 0, reducing existing position)
                sell_qty = abs(delta)
                
                # Can only sell what we actually hold
                if sym not in current_positions or current_positions[sym]['qty'] <= 0:
                    logger.warn("portfolio_sell_no_position", {"symbol": sym})
                    continue
                
                # Don't sell more than we hold
                sell_qty = min(sell_qty, current_positions[sym]['qty'])
                
                avg_cost = current_positions[sym]['avg_cost']
                gross_proceeds = sell_qty * price
                friction_cost = gross_proceeds * self.friction
                net_proceeds = gross_proceeds - friction_cost
                
                # Realized P&L = (sell_price - avg_cost) * qty - friction
                realized = sell_qty * (price - avg_cost) - friction_cost
                state['realized_pnl'] += realized
                state['cash'] += net_proceeds
                
                remaining = current_positions[sym]['qty'] - sell_qty
                if remaining <= 0:
                    del current_positions[sym]
                else:
                    current_positions[sym]['qty'] = remaining

            trade = {
                'date': trade_date, 'symbol': sym, 'action': 'BUY' if delta > 0 else 'SELL',
                'qty': delta, 'price': price, 'friction': friction_cost
            }
            trades_executed.append(trade)
            self._log_trade(trade)

        state['total_friction_paid'] += total_friction
        state['trade_count'] += len(trades_executed)
        state['positions'] = current_positions
        
        # Save daily snapshot
        nav_after = self._calc_nav(prices)
        daily_return = (nav_after / nav_before - 1) if nav_before > 0 else 0
        
        snapshot = {
            'date': trade_date,
            'nav': nav_after,
            'cash': state['cash'],
            'positions_value': nav_after - state['cash'],
            'positions_count': len(current_positions),
            'daily_return': daily_return,
            'cumulative_pnl': nav_after - state['initial_cash'],
            'cumulative_return': (nav_after / state['initial_cash'] - 1),
            'realized_pnl': state['realized_pnl'],
            'total_friction': state['total_friction_paid'],
            'trades_today': len(trades_executed)
        }
        state['history'].append(snapshot)
        
        self._save_state(state)
        
        logger.info("portfolio_executed", {
            "trades": len(trades_executed),
            "nav": nav_after,
            "daily_return": f"{daily_return:.4f}",
            "friction_today": total_friction
        })
        
        return {
            "trades": len(trades_executed),
            "nav": nav_after,
            "daily_return": daily_return,
            "total_friction": total_friction,
            "snapshot": snapshot
        }

    def _calc_nav(self, prices: Dict[str, float]) -> float:
        """Calculate current Net Asset Value."""
        positions_value = 0.0
        for sym, pos in self.state['positions'].items():
            price = prices.get(sym, pos['avg_cost'])
            positions_value += pos['qty'] * price
        return self.state['cash'] + positions_value

    def get_portfolio_report(self, prices: Dict[str, float]) -> str:
        """Generate portfolio summary for Telegram report."""
        state = self.state
        nav = self._calc_nav(prices)
        initial = state['initial_cash']
        cum_pnl = nav - initial
        cum_ret = (nav / initial - 1) * 100
        
        report = f"""💼 虚拟组合状态
━━━━━━━━━━━━━━━━
💰 净值: ${nav:,.0f} (初始 ${initial:,.0f})
📊 累计收益: ${cum_pnl:+,.0f} ({cum_ret:+.2f}%)
💵 可用现金: ${state['cash']:,.0f}
📈 已实现盈亏: ${state['realized_pnl']:+,.0f}
🔧 累计手续费: ${state['total_friction_paid']:,.0f}
📋 总交易次数: {state['trade_count']}
"""
        # Current holdings
        positions = state.get('positions', {})
        if positions:
            report += f"\n📦 当前持仓 ({len(positions)} 只):\n"
            holdings = []
            for sym, pos in positions.items():
                price = prices.get(sym, pos['avg_cost'])
                qty = pos['qty']
                mkt_val = qty * price
                cost_val = qty * pos['avg_cost']
                pnl = mkt_val - cost_val
                pnl_pct = (price / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] > 0 else 0
                holdings.append((sym, qty, price, mkt_val, pnl, pnl_pct))
            
            # Sort by absolute P&L
            holdings.sort(key=lambda x: abs(x[4]), reverse=True)
            for sym, qty, price, mkt_val, pnl, pnl_pct in holdings:
                icon = '🟢' if pnl >= 0 else '🔴'
                report += f"  {icon} {sym}: {qty}股 @${price:.1f} | ${pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
        else:
            report += "\n📦 当前无持仓\n"
            
        return report

    def get_performance_summary(self) -> dict:
        """Get performance metrics."""
        state = self.state
        history = state.get('history', [])
        if not history:
            return {}
        
        latest = history[-1]
        nav = latest['nav']
        initial = state['initial_cash']
        
        # Calculate max drawdown from history
        peak = initial
        max_dd = 0
        for snap in history:
            if snap['nav'] > peak:
                peak = snap['nav']
            dd = (peak - snap['nav']) / peak
            if dd > max_dd:
                max_dd = dd
        
        return {
            "nav": nav,
            "cumulative_pnl": nav - initial,
            "cumulative_return": (nav / initial - 1),
            "max_drawdown": max_dd,
            "total_trades": state['trade_count'],
            "total_friction": state['total_friction_paid'],
            "realized_pnl": state['realized_pnl'],
            "trading_days": len(history)
        }
