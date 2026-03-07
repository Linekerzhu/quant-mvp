#!/usr/bin/env python3
"""
Daily Job Runner with Telegram Notification (v2 — Mobile-Optimized)

Runs daily_job.py and sends a single, compact, actionable Telegram report.
"""

import os
import sys
import subprocess
import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Env & Telegram ──────────────────────────────────────

def load_env():
    env = {}
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    env[key] = val
    return env


def send_telegram(env, message, parse_mode="Markdown"):
    token = env.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = env.get('TELEGRAM_CHAT_ID', '')
    if not (token and chat_id):
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _split_message(message, max_len=4000)
    ok = True
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": chat_id, "text": chunk, "parse_mode": parse_mode
            }, timeout=10)
            if not resp.ok:
                print(f"Telegram fail: {resp.status_code}")
                ok = False
        except Exception as e:
            print(f"Telegram error: {e}")
            ok = False
    return ok


def _split_message(text, max_len=4000):
    if len(text) <= max_len:
        return [text]
    chunks, cur = [], ""
    for line in text.split('\n'):
        while len(line) > max_len:
            if cur: chunks.append(cur); cur = ""
            chunks.append(line[:max_len]); line = line[max_len:]
        if len(cur) + len(line) + 1 > max_len:
            if cur: chunks.append(cur)
            cur = line
        else:
            cur = cur + '\n' + line if cur else line
    if cur: chunks.append(cur)
    return chunks or [text]


# ── Data Readers ────────────────────────────────────────

def read_signals(trade_date):
    path = os.path.join(PROJECT_ROOT, f'data/processed/signals_{trade_date}.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df if len(df) > 0 else None
    return None


def read_targets(trade_date):
    path = os.path.join(PROJECT_ROOT, f'data/processed/targets_{trade_date}.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df if len(df) > 0 else None
    return None


def read_portfolio_summary():
    path = os.path.join(PROJECT_ROOT, 'data/portfolio/state.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        state = json.load(f)
    history = state.get('history', [])
    if not history:
        return None
    nav = history[-1]['nav']
    initial = state.get('initial_cash', 100000)
    peak = initial
    max_dd = 0
    for snap in history:
        if snap['nav'] > peak: peak = snap['nav']
        dd = (peak - snap['nav']) / peak
        if dd > max_dd: max_dd = dd
    return {
        'nav': nav, 'initial': initial,
        'cum_return': (nav / initial - 1) * 100,
        'max_dd': max_dd * 100,
        'trading_days': len(history),
        'total_trades': state.get('trade_count', 0),
        'total_friction': state.get('total_friction_paid', 0),
        'positions': len(state.get('positions', {})),
        'cash': state.get('cash', 0),
    }


def get_latest_prices(trade_date):
    """Get latest close prices from features file."""
    path = os.path.join(PROJECT_ROOT, f'data/processed/features_{trade_date}.parquet')
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    prices = {}
    for sym in df['symbol'].unique():
        sym_data = df[df['symbol'] == sym]
        if len(sym_data) > 0:
            for col in ['adj_close', 'raw_close', 'close']:
                if col in sym_data.columns:
                    val = sym_data[col].iloc[-1]
                    if pd.notna(val) and val > 0:
                        prices[sym] = float(val)
                        break
    return prices


def get_universe_count(trade_date):
    path = os.path.join(PROJECT_ROOT, f'data/processed/final_daily_{trade_date}.parquet')
    if os.path.exists(path):
        try: return pd.read_parquet(path)['symbol'].nunique()
        except: pass
    return None


# ── Report Builder (Mobile-Optimized) ───────────────────

def build_report(trade_date, elapsed, returncode, stderr):
    """Build compact, actionable Telegram report for mobile."""

    ok = returncode == 0
    L = []  # lines

    # ─ Header (compact) ─
    L.append(f"📊 *Quant MVP v6.0 日报*")
    L.append(f"📅 `{trade_date}` {'✅' if ok else '❌'} `{elapsed:.0f}s`")

    if not ok:
        L.append(f"\n⛔ *错误*:\n```\n{stderr[-300:]}\n```")
        return '\n'.join(L)

    universe = get_universe_count(trade_date)
    if universe:
        L.append(f"🌐 股票池 `{universe}` 只")

    signals_df = read_signals(trade_date)
    targets_df = read_targets(trade_date)
    prices = get_latest_prices(trade_date)
    perf = read_portfolio_summary()
    nav = perf['nav'] if perf else 100000

    # ─ Section 1: Dual Model Summary (compact) ─
    L.append("")
    L.append("*▸ 本地双模型*")
    if signals_df is not None and len(signals_df) > 0:
        buys = signals_df[signals_df['side'] > 0]
        sells = signals_df[signals_df['side'] < 0]
        avg_p = float(signals_df['prob'].mean()) * 100 if 'prob' in signals_df.columns else 0
        L.append(f"🟢买`{len(buys)}`个 🔴卖`{len(sells)}`个 置信`{avg_p:.0f}%`")

        # Top 5 buy signals
        if len(buys) > 0:
            top_buys = buys.sort_values('prob', ascending=False).head(5) if 'prob' in buys.columns else buys.head(5)
            buy_syms = ' '.join([f"`{r['symbol']}`" for _, r in top_buys.iterrows()])
            L.append(f"  TOP买: {buy_syms}")

        # Top 5 sell signals
        if len(sells) > 0:
            top_sells = sells.sort_values('prob', ascending=False).head(5) if 'prob' in sells.columns else sells.head(5)
            sell_syms = ' '.join([f"`{r['symbol']}`" for _, r in top_sells.iterrows()])
            L.append(f"  TOP卖: {sell_syms}")
    else:
        L.append("⚪ 无信号")

    # ─ Section 2: Oracle Verdict (compact) ─
    L.append("")
    L.append("*▸ Kronos 专家审查*")

    vetoed_syms = set()
    if targets_df is not None and 'oracle_action' in targets_df.columns:
        approved = targets_df[targets_df['oracle_action'] == 'approve']
        neutral = targets_df[targets_df['oracle_action'] == 'neutral']
        if signals_df is not None:
            signal_syms = set(signals_df['symbol'].unique())
            target_syms = set(targets_df['symbol'].unique())
            vetoed_syms = signal_syms - target_syms
        vetoed_count = len(vetoed_syms)

        L.append(f"✅`{len(approved)}` ⚖️`{len(neutral)}` 🚫`{vetoed_count}`")

        # Show vetoed symbols (ALL of them — this is critical info)
        if vetoed_count > 0:
            veto_list = ' '.join([f"`{s}`" for s in sorted(vetoed_syms)[:20]])
            L.append(f"否决: {veto_list}")
            if vetoed_count > 20:
                L.append(f"  +{vetoed_count - 20}只")
    elif targets_df is not None:
        L.append("ℹ️ Oracle未参与")
    else:
        L.append("⚪ 无目标需审查")

    # ─ Section 3: ACTIONABLE ORDERS (the most important part) ─
    L.append("")
    L.append("*▸ 📋 执行指令*")

    if targets_df is not None and len(targets_df) > 0 and 'target_weight' in targets_df.columns:
        # Filter out zero-weight targets — they are noise
        actionable = targets_df[targets_df['target_weight'].abs() >= 0.005].copy()
        actionable = actionable.sort_values('target_weight', key=abs, ascending=False)

        if len(actionable) > 0:
            buy_orders = actionable[actionable['target_weight'] > 0]
            sell_orders = actionable[actionable['target_weight'] < 0]
            total_w = actionable['target_weight'].sum() * 100

            L.append(f"买`{len(buy_orders)}`笔 卖`{len(sell_orders)}`笔 总仓`{total_w:.1f}%`")
            L.append("")

            # Buy orders with precise instructions
            if len(buy_orders) > 0:
                L.append("🟢 *买入:*")
                for _, r in buy_orders.iterrows():
                    sym = r['symbol']
                    w = r['target_weight'] * 100
                    price = prices.get(sym, 0)
                    oracle_ret = r.get('oracle_pred_ret', 0) * 100 if 'oracle_pred_ret' in r.index else 0

                    if price > 0:
                        qty = int(abs(r['target_weight']) * nav / price)
                        L.append(f"  `{sym}` {w:.1f}% ${price:.0f}×{qty}股")
                        if oracle_ret != 0:
                            L.append(f"    ↳Kronos `{oracle_ret:+.1f}%`")
                    else:
                        L.append(f"  `{sym}` {w:.1f}%")

            # Sell orders with precise instructions
            if len(sell_orders) > 0:
                L.append("🔴 *卖出:*")
                for _, r in sell_orders.iterrows():
                    sym = r['symbol']
                    w = r['target_weight'] * 100
                    price = prices.get(sym, 0)

                    if price > 0:
                        qty = int(abs(r['target_weight']) * nav / price)
                        L.append(f"  `{sym}` {w:.1f}% ${price:.0f}×{qty}股")
                    else:
                        L.append(f"  `{sym}` {w:.1f}%")
        else:
            L.append("⚪ 无有效仓位 (全部权重<0.5%)")

        # Count how many were filtered as zero-weight
        zero_w = len(targets_df) - len(actionable)
        if zero_w > 0:
            L.append(f"\n_({zero_w}只权重为0已过滤)_")
    else:
        L.append("⚪ 今日无交易")

    # ─ Section 4: Portfolio (one-line compact) ─
    L.append("")
    L.append("*▸ 虚拟组合*")
    if perf:
        ret_icon = "📈" if perf['cum_return'] >= 0 else "📉"
        L.append(f"💰`${perf['nav']:,.0f}` {ret_icon}`{perf['cum_return']:+.2f}%` 回撤`{perf['max_dd']:.1f}%`")
        L.append(f"持仓`{perf['positions']}`只 现金`${perf['cash']:,.0f}` 交易`{perf['total_trades']}`笔")
    else:
        L.append("ℹ️ 未初始化")

    # ─ Footer ─
    L.append("")
    L.append("_SMA+Momentum+Kronos | Quant MVP_")

    return '\n'.join(L)


# ── Main ────────────────────────────────────────────────

def main():
    env = load_env()
    start_time = datetime.now()
    trade_date = datetime.now().strftime('%Y-%m-%d')

    # Run daily_job.py using the same Python interpreter (venv-aware)
    result = subprocess.run(
        [sys.executable, 'src/ops/daily_job.py'],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env, 'PYTHONPATH': PROJECT_ROOT}
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    report = build_report(trade_date, elapsed, result.returncode, result.stderr)

    send_telegram(env, report)
    print(report)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
