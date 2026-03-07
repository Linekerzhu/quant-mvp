#!/usr/bin/env python3
"""
Full Historical Backtest Engine

Replays the entire quant-mvp pipeline day-by-day over historical data:
  1. Generates SMA + Momentum signals per symbol per day
  2. Calculates win-rate and avg_win/avg_loss from past data
  3. Applies Fractional Kelly sizing
  4. Applies Risk Engine caps
  5. Executes against a virtual portfolio with friction
  6. Reports PnL, drawdown, Sharpe, and per-stock attribution

Usage:
    python backtest_full.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Project imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from src.signals.base_models import BaseModelSMA, BaseModelMomentum
from src.risk.position_sizing import IndependentKellySizer
from src.risk.risk_engine import RiskEngine

# ── Config ─────────────────────────────────────────────────
INITIAL_CASH   = 100_000.0
FRICTION_BPS   = 20          # 0.20% round-trip
MIN_HISTORY    = 80          # need 80 bars before trading a symbol
WARMUP_DAYS    = 80          # skip first N days globally
CONF_THRESHOLD = 0.52        # minimum probability to act
FWD_RETURN_DAYS = 5          # horizon for measuring signal quality
TOP_N_SYMBOLS  = 50          # limit traded universe per day


def load_features(path: str = "data/processed/features_2026-03-07.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure date is plain datetime (drop tz for easier comparison)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    return df


def generate_all_signals(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate SMA + Momentum signals for every symbol."""
    sma_model = BaseModelSMA(fast_window=20, slow_window=60)
    mom_model = BaseModelMomentum(window=20)

    all_sigs = []
    for sym in features_df['symbol'].unique():
        df_sym = features_df[features_df['symbol'] == sym].copy()
        if len(df_sym) < MIN_HISTORY:
            continue
        try:
            df_sma = sma_model.generate_signals(df_sym)
            df_sma['model'] = 'sma'
            all_sigs.append(df_sma)
        except Exception:
            pass
        try:
            df_mom = mom_model.generate_signals(df_sym)
            df_mom['model'] = 'momentum'
            all_sigs.append(df_mom)
        except Exception:
            pass

    if not all_sigs:
        raise RuntimeError("No signals generated")
    sig_df = pd.concat(all_sigs, ignore_index=True)
    return sig_df


def enrich_with_stats(sig_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Add prob, avg_win, avg_loss, realized_vol for each signal row."""
    # Pre-compute per-symbol forward returns
    fwd_rets = {}
    for sym in features_df['symbol'].unique():
        s = features_df[features_df['symbol'] == sym].sort_values('date')
        fwd = s['adj_close'].pct_change(FWD_RETURN_DAYS).shift(-FWD_RETURN_DAYS)
        fwd_rets[sym] = dict(zip(s['date'], fwd))

    results = []
    for sym in sig_df['symbol'].unique():
        sym_sigs = sig_df[sig_df['symbol'] == sym].sort_values('date')
        active = sym_sigs[sym_sigs['side'] != 0].copy()
        if active.empty:
            continue

        # Attach forward return
        active['fwd_ret'] = active['date'].map(fwd_rets.get(sym, {}))
        # Aligned return = fwd_ret * side
        active['aligned_ret'] = active['fwd_ret'] * active['side']

        # Rolling metrics (expanding window)
        active['prob'] = 0.50
        active['avg_win'] = 0.03
        active['avg_loss'] = 0.03

        for i in range(len(active)):
            past = active.iloc[:i]
            past_aligned = past['aligned_ret'].dropna()
            if len(past_aligned) > 10:
                wins = past_aligned[past_aligned > 0]
                losses = past_aligned[past_aligned < 0]
                p = len(wins) / max(len(past_aligned), 1)
                active.iloc[i, active.columns.get_loc('prob')] = np.clip(p, 0.01, 0.99)
                if len(wins) > 0:
                    active.iloc[i, active.columns.get_loc('avg_win')] = float(wins.mean())
                if len(losses) > 0:
                    active.iloc[i, active.columns.get_loc('avg_loss')] = float(losses.abs().mean())

        # Realized vol (20-day)
        if 'rv_20d' in active.columns:
            active['realized_vol'] = active['rv_20d'].fillna(0.20)
        else:
            active['realized_vol'] = 0.20

        results.append(active)

    return pd.concat(results, ignore_index=True)


def run_backtest(enriched: pd.DataFrame):
    """Day-by-day portfolio simulation."""
    sizer = IndependentKellySizer()
    risk = RiskEngine()

    dates = sorted(enriched['date'].unique())
    dates = dates[WARMUP_DAYS:]  # skip warmup

    cash = INITIAL_CASH
    positions = {}  # {sym: {'qty': int, 'avg_cost': float}}
    friction_rate = FRICTION_BPS / 10000.0

    history = []   # daily snapshots
    trade_log = []

    print(f"\n{'='*70}")
    print(f"  FULL HISTORICAL BACKTEST")
    print(f"  Period   : {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Capital  : ${INITIAL_CASH:,.0f}")
    print(f"  Universe : {enriched['symbol'].nunique()} symbols")
    print(f"  Trading days: {len(dates)}")
    print(f"{'='*70}\n")

    peak_nav = INITIAL_CASH
    max_dd = 0.0

    for day_idx, td in enumerate(dates):
        # Get today's signals
        day_sigs = enriched[
            (enriched['date'] == td) &
            (enriched['prob'] >= CONF_THRESHOLD)
        ].copy()

        # Merge duplicates (SMA + Momentum for same symbol)
        if day_sigs.empty:
            # Just mark-to-market
            pass
        else:
            # Kelly sizing
            try:
                positions_df = sizer.calculate_positions(day_sigs, current_drawdown=max_dd)
            except Exception:
                positions_df = pd.DataFrame({'symbol': [], 'target_weight': []})

            if not positions_df.empty:
                positions_df = positions_df.groupby('symbol', as_index=False)['target_weight'].sum()

                # Risk caps
                positions_df = risk.validate_positions(positions_df)

                # Remove dust
                positions_df = positions_df[positions_df['target_weight'].abs() >= 0.005]

                # Limit to top N by absolute weight
                positions_df = positions_df.reindex(
                    positions_df['target_weight'].abs().sort_values(ascending=False).index
                ).head(TOP_N_SYMBOLS)

        # Get prices for today
        prices = {}
        day_data = enriched[(enriched['date'] == td)].drop_duplicates('symbol')
        for _, row in day_data.iterrows():
            prices[row['symbol']] = float(row['adj_close'])

        # Also price existing positions using most recent data
        for sym in list(positions.keys()):
            if sym not in prices:
                # Use last known price
                pass

        # Calculate NAV before trading
        pos_val = sum(pos['qty'] * prices.get(s, pos['avg_cost']) for s, pos in positions.items())
        nav = cash + pos_val

        # Execute trades
        trades_today = 0
        friction_today = 0.0

        if not day_sigs.empty and not positions_df.empty:
            targets = dict(zip(positions_df['symbol'], positions_df['target_weight']))

            # Close positions not in targets
            for sym in list(positions.keys()):
                if sym not in targets:
                    pos = positions[sym]
                    price = prices.get(sym, pos['avg_cost'])
                    proceeds = pos['qty'] * price
                    fric = abs(proceeds) * friction_rate
                    cash += proceeds - fric
                    friction_today += fric
                    trades_today += 1
                    del positions[sym]

            # Open / adjust positions
            for sym, w in targets.items():
                if sym not in prices or prices[sym] <= 0:
                    continue
                price = prices[sym]
                target_usd = abs(w) * nav
                target_qty = int(target_usd / price)
                if w < 0:
                    continue  # long-only

                current_qty = positions.get(sym, {}).get('qty', 0)
                delta = target_qty - current_qty

                if abs(delta) < 1:
                    continue

                if delta > 0:  # buy
                    cost = delta * price * (1 + friction_rate)
                    if cost > cash:
                        delta = int(cash / (price * (1 + friction_rate)))
                        if delta <= 0:
                            continue
                        cost = delta * price * (1 + friction_rate)
                    fric = delta * price * friction_rate
                    cash -= cost
                    friction_today += fric
                    if sym in positions:
                        old = positions[sym]
                        new_qty = old['qty'] + delta
                        new_avg = (old['qty'] * old['avg_cost'] + delta * price) / new_qty
                        positions[sym] = {'qty': new_qty, 'avg_cost': new_avg}
                    else:
                        positions[sym] = {'qty': delta, 'avg_cost': price}
                    trades_today += 1

                elif delta < 0:  # sell (reduce)
                    sell_qty = min(abs(delta), positions.get(sym, {}).get('qty', 0))
                    if sell_qty <= 0:
                        continue
                    proceeds = sell_qty * price
                    fric = proceeds * friction_rate
                    cash += proceeds - fric
                    friction_today += fric
                    remaining = positions[sym]['qty'] - sell_qty
                    if remaining <= 0:
                        del positions[sym]
                    else:
                        positions[sym]['qty'] = remaining
                    trades_today += 1

        # End-of-day NAV
        pos_val = sum(pos['qty'] * prices.get(s, pos['avg_cost']) for s, pos in positions.items())
        nav_eod = cash + pos_val

        # Drawdown tracking
        if nav_eod > peak_nav:
            peak_nav = nav_eod
        dd = (peak_nav - nav_eod) / peak_nav
        if dd > max_dd:
            max_dd = dd

        daily_ret = (nav_eod / (history[-1]['nav'] if history else INITIAL_CASH)) - 1

        snap = {
            'date': td,
            'nav': nav_eod,
            'cash': cash,
            'positions_count': len(positions),
            'trades': trades_today,
            'friction': friction_today,
            'daily_return': daily_ret,
            'drawdown': dd,
            'max_dd': max_dd
        }
        history.append(snap)

        # Progress indicator every 20 days
        if day_idx % 20 == 0 or day_idx == len(dates) - 1:
            print(f"  [{td.strftime('%Y-%m-%d')}] NAV=${nav_eod:>10,.0f}  DD={dd:>6.2%}  Pos={len(positions):>3}  Trades={trades_today}")

    return history, positions, trade_log


def print_summary(history):
    """Print comprehensive performance report."""
    df = pd.DataFrame(history)
    
    nav_final = df['nav'].iloc[-1]
    nav_init = INITIAL_CASH
    total_ret = (nav_final / nav_init) - 1
    trading_days = len(df)
    
    # Annualized return
    years = trading_days / 252
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe ratio
    daily_rets = df['daily_return'].dropna()
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Sortino ratio
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_rets.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0
    
    # Max drawdown
    max_dd = df['max_dd'].max()
    
    # Calmar ratio
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    
    # Win rate (daily)
    win_days = (daily_rets > 0).sum()
    total_days = len(daily_rets)
    win_rate = win_days / total_days if total_days > 0 else 0
    
    # Total friction
    total_friction = df['friction'].sum()
    total_trades = df['trades'].sum()
    
    # Best / Worst day
    best_day = daily_rets.max()
    worst_day = daily_rets.min()

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Period          : {df['date'].iloc[0].strftime('%Y-%m-%d')} → {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading Days    : {trading_days}")
    print(f"  Initial Capital : ${nav_init:>12,.0f}")
    print(f"  Final NAV       : ${nav_final:>12,.0f}")
    print(f"  Total Return    : {total_ret:>+10.2%}")
    print(f"  Annualized Ret  : {ann_ret:>+10.2%}")
    print(f"{'─'*70}")
    print(f"  Sharpe Ratio    : {sharpe:>10.2f}")
    print(f"  Sortino Ratio   : {sortino:>10.2f}")
    print(f"  Calmar Ratio    : {calmar:>10.2f}")
    print(f"  Max Drawdown    : {max_dd:>10.2%}")
    print(f"{'─'*70}")
    print(f"  Daily Win Rate  : {win_rate:>10.2%}  ({win_days}/{total_days})")
    print(f"  Best Day        : {best_day:>+10.2%}")
    print(f"  Worst Day       : {worst_day:>+10.2%}")
    print(f"  Daily Vol (ann) : {daily_rets.std() * np.sqrt(252):>10.2%}")
    print(f"{'─'*70}")
    print(f"  Total Trades    : {int(total_trades):>10,}")
    print(f"  Total Friction  : ${total_friction:>10,.0f}")
    print(f"  Friction/NAV    : {total_friction/nav_init:>10.2%}")
    print(f"{'='*70}")
    
    # Monthly returns table
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['daily_return'].apply(lambda x: (1 + x).prod() - 1)
    print(f"\n  Monthly Returns:")
    print(f"  {'─'*40}")
    for m, r in monthly.items():
        bar = '█' * max(1, int(abs(r) * 500))
        color = '+' if r >= 0 else '-'
        print(f"  {m}  {r:>+7.2%}  {color}{bar}")

    # Save equity curve to CSV
    df[['date', 'nav', 'daily_return', 'drawdown', 'positions_count', 'trades']].to_csv(
        'reports/backtest_equity_curve.csv', index=False
    )
    print(f"\n  📊 Equity curve saved to reports/backtest_equity_curve.csv")
    
    return df


if __name__ == '__main__':
    print("Loading feature data...")
    features = load_features()
    
    print(f"Generating signals for {features['symbol'].nunique()} symbols...")
    signals = generate_all_signals(features)
    print(f"  Generated {len(signals)} signal rows")
    
    print("Enriching signals with rolling stats (prob, avg_win, avg_loss)...")
    enriched = enrich_with_stats(signals, features)
    print(f"  Enriched {len(enriched)} active signal rows")
    
    history, final_positions, trades = run_backtest(enriched)
    
    result_df = print_summary(history)
    
    # Print final positions
    if final_positions:
        print(f"\n  📦 Final Positions ({len(final_positions)}):")
        for sym, pos in sorted(final_positions.items(), key=lambda x: x[1]['qty'] * x[1]['avg_cost'], reverse=True)[:10]:
            print(f"    {sym:>6}: {pos['qty']:>5} shares @ ${pos['avg_cost']:.2f}")
