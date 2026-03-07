#!/usr/bin/env python3
"""
Daily Job Runner with Rich Telegram Notification

Runs daily_job.py and sends a comprehensive multi-model decision report
to Telegram, highlighting local dual-model signals, expert Oracle opinions,
and final portfolio decisions.
"""

import os
import sys
import subprocess
import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Determine project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Environment ─────────────────────────────────────────────────────────

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
    """Send Telegram message with chunking for long content."""
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
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": parse_mode
            }, timeout=10)
            if not resp.ok:
                print(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
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
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(line[:max_len]); line = line[max_len:]
        if len(cur) + len(line) + 1 > max_len:
            if cur: chunks.append(cur)
            cur = line
        else:
            cur = cur + '\n' + line if cur else line
    if cur: chunks.append(cur)
    return chunks or [text]


# ─── Data Readers ────────────────────────────────────────────────────────

def read_signals(trade_date):
    """Read raw signals (before sizing) — shows both SMA and Momentum."""
    path = os.path.join(PROJECT_ROOT, f'data/processed/signals_{trade_date}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if len(df) > 0 else None


def read_targets(trade_date):
    """Read final targets (after Oracle veto) — includes oracle columns."""
    path = os.path.join(PROJECT_ROOT, f'data/processed/targets_{trade_date}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if len(df) > 0 else None


def read_portfolio_state():
    """Read virtual portfolio state."""
    path = os.path.join(PROJECT_ROOT, 'data/portfolio/state.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def read_portfolio_summary():
    """Read portfolio performance summary."""
    state = read_portfolio_state()
    if not state:
        return None
    history = state.get('history', [])
    if not history:
        return None

    nav = history[-1]['nav']
    initial = state.get('initial_cash', 100000)
    peak = initial
    max_dd = 0
    for snap in history:
        if snap['nav'] > peak:
            peak = snap['nav']
        dd = (peak - snap['nav']) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'nav': nav,
        'initial': initial,
        'cum_return': (nav / initial - 1) * 100,
        'max_dd': max_dd * 100,
        'trading_days': len(history),
        'total_trades': state.get('trade_count', 0),
        'total_friction': state.get('total_friction_paid', 0),
        'realized_pnl': state.get('realized_pnl', 0),
        'positions': len(state.get('positions', {})),
        'cash': state.get('cash', 0),
    }


def get_universe_count(trade_date):
    path = os.path.join(PROJECT_ROOT, f'data/processed/final_daily_{trade_date}.parquet')
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)['symbol'].nunique()
        except:
            pass
    return None


# ─── Report Builder ──────────────────────────────────────────────────────

def build_report(trade_date, elapsed, returncode, stderr):
    """Build a rich multi-model Telegram report."""

    status_icon = "✅" if returncode == 0 else "❌"
    status_text = "正常完成" if returncode == 0 else "执行失败"

    # ── Header ──
    lines = [
        f"{'━' * 28}",
        f"📊 *Quant MVP 每日决策报告*",
        f"{'━' * 28}",
        f"",
        f"📅 日期: `{trade_date}`",
        f"🔄 状态: {status_icon} {status_text}",
        f"⏱ 耗时: `{elapsed:.0f}s`",
    ]

    if returncode != 0:
        lines.append(f"\n⛔ *系统错误*:\n```\n{stderr[-400:]}\n```")
        return '\n'.join(lines)

    # ── Universe ──
    universe = get_universe_count(trade_date)
    if universe:
        lines.append(f"🌐 股票池: `{universe}` 只")

    # ── Signals (dual model breakdown) ──
    signals_df = read_signals(trade_date)
    targets_df = read_targets(trade_date)

    lines.append("")
    lines.append(f"{'─' * 28}")
    lines.append("🧠 *一、本地双模型信号*")
    lines.append(f"{'─' * 28}")

    if signals_df is not None and len(signals_df) > 0:
        buys = signals_df[signals_df['side'] > 0].copy()
        sells = signals_df[signals_df['side'] < 0].copy()
        total = len(signals_df)

        # Detect model source if available
        if 'model' in signals_df.columns:
            sma_count = len(signals_df[signals_df['model'] == 'sma'])
            mom_count = len(signals_df[signals_df['model'] == 'momentum'])
            lines.append(f"  📐 SMA 均线信号: `{sma_count}` 个")
            lines.append(f"  🚀 Momentum 动量信号: `{mom_count}` 个")
        else:
            lines.append(f"  合计信号: `{total}` 个 (SMA+Momentum)")

        avg_prob = float(signals_df['prob'].mean()) if 'prob' in signals_df.columns else 0
        lines.append(f"  📊 平均置信度: `{avg_prob*100:.1f}%`")
        lines.append("")

        # Top buys
        if len(buys) > 0:
            buys_sorted = buys.sort_values('prob', ascending=False) if 'prob' in buys.columns else buys
            lines.append(f"  🟢 *买入信号* ({len(buys)} 个):")
            for _, r in buys_sorted.head(8).iterrows():
                sym = r['symbol']
                prob = f"{r['prob']*100:.0f}%" if 'prob' in r.index else "–"
                model_tag = f"[{r['model'][:3].upper()}]" if 'model' in r.index else ""
                lines.append(f"    `{sym:6s}` 置信`{prob}` {model_tag}")
            if len(buys) > 8:
                lines.append(f"    _... +{len(buys)-8} 个_")

        # Top sells
        if len(sells) > 0:
            sells_sorted = sells.sort_values('prob', ascending=False) if 'prob' in sells.columns else sells
            lines.append(f"  🔴 *卖出信号* ({len(sells)} 个):")
            for _, r in sells_sorted.head(5).iterrows():
                sym = r['symbol']
                prob = f"{r['prob']*100:.0f}%" if 'prob' in r.index else "–"
                lines.append(f"    `{sym:6s}` 置信`{prob}`")
            if len(sells) > 5:
                lines.append(f"    _... +{len(sells)-5} 个_")
    else:
        lines.append("  ⚪ 今日双模型均无活跃信号")

    # ── Oracle Expert Review ──
    lines.append("")
    lines.append(f"{'─' * 28}")
    lines.append("🔮 *二、Kronos 外部专家判定*")
    lines.append(f"{'─' * 28}")

    if targets_df is not None and 'oracle_action' in targets_df.columns:
        approved = targets_df[targets_df['oracle_action'] == 'approve']
        neutral = targets_df[targets_df['oracle_action'] == 'neutral']
        # Vetoed symbols are NOT in targets_df (they were removed)
        total_reviewed = len(targets_df)

        # Count vetoed by comparing signals vs targets
        vetoed_count = 0
        if signals_df is not None:
            signal_syms = set(signals_df['symbol'].unique()) if len(signals_df) > 0 else set()
            target_syms = set(targets_df['symbol'].unique()) if len(targets_df) > 0 else set()
            vetoed_syms = signal_syms - target_syms
            vetoed_count = len(vetoed_syms)

        lines.append(f"  ✅ 批准: `{len(approved)}` 只")
        lines.append(f"  ⚖️ 中性: `{len(neutral)}` 只")
        lines.append(f"  🚫 否决: `{vetoed_count}` 只")
        lines.append("")

        # Show approved with Oracle prediction
        if len(approved) > 0:
            lines.append("  *批准清单 (含预测回报):*")
            for _, r in approved.head(10).iterrows():
                sym = r['symbol']
                w = r.get('target_weight', 0) * 100
                pred_ret = r.get('oracle_pred_ret', 0) * 100
                ret_icon = "📈" if pred_ret > 0 else "📉"
                lines.append(f"    {ret_icon} `{sym:6s}` 仓位`{w:.1f}%` Kronos预测`{pred_ret:+.1f}%`")

        # Show vetoed
        if vetoed_count > 0 and signals_df is not None:
            lines.append("")
            lines.append("  *否决清单 (被专家一票否决):*")
            for sym in sorted(vetoed_syms)[:8]:
                lines.append(f"    🚫 `{sym}` — 专家预测下跌风险过大")
            if vetoed_count > 8:
                lines.append(f"    _... +{vetoed_count-8} 只_")

    elif targets_df is not None and len(targets_df) > 0:
        lines.append("  ℹ️ 专家 Oracle 未配置/未参与今日决策")
        lines.append(f"  最终目标: `{len(targets_df)}` 只")
    else:
        lines.append("  ⚪ 无需专家审查 (无活跃目标)")

    # ── Final Targets ──
    lines.append("")
    lines.append(f"{'─' * 28}")
    lines.append("🎯 *三、最终交易目标*")
    lines.append(f"{'─' * 28}")

    if targets_df is not None and len(targets_df) > 0:
        total_weight = targets_df['target_weight'].sum() * 100 if 'target_weight' in targets_df.columns else 0
        lines.append(f"  交易标的: `{len(targets_df)}` 只")
        lines.append(f"  总仓位比例: `{total_weight:.1f}%`")
        lines.append("")

        targets_sorted = targets_df.sort_values('target_weight', ascending=False) if 'target_weight' in targets_df.columns else targets_df
        for _, r in targets_sorted.head(10).iterrows():
            sym = r['symbol']
            w = r.get('target_weight', 0) * 100
            abs_w = abs(w)
            bar_len = int(min(abs_w * 2, 10))
            if w >= 0:
                bar = '🟩' * bar_len + '⬜' * (5 - bar_len)
                lines.append(f"    🟢 `{sym:6s}` {bar} `{w:+.2f}%`")
            else:
                bar = '🟥' * bar_len + '⬜' * (5 - bar_len)
                lines.append(f"    🔴 `{sym:6s}` {bar} `{w:+.2f}%`")

        if len(targets_df) > 10:
            lines.append(f"    _... +{len(targets_df)-10} 只_")
    else:
        lines.append("  ⚪ 今日无交易 — 全员待命")

    # ── Portfolio Performance ──
    perf = read_portfolio_summary()
    lines.append("")
    lines.append(f"{'─' * 28}")
    lines.append("💼 *四、虚拟组合绩效*")
    lines.append(f"{'─' * 28}")

    if perf:
        ret_icon = "📈" if perf['cum_return'] >= 0 else "📉"
        lines.append(f"  💰 净值: `${perf['nav']:,.0f}`")
        lines.append(f"  {ret_icon} 累计回报: `{perf['cum_return']:+.2f}%`")
        lines.append(f"  📉 最大回撤: `{perf['max_dd']:.2f}%`")
        lines.append(f"  📊 交易天数: `{perf['trading_days']}` 天")
        lines.append(f"  🔄 累计交易: `{perf['total_trades']}` 笔")
        lines.append(f"  💸 累计摩擦: `${perf['total_friction']:.2f}`")
        lines.append(f"  📦 当前持仓: `{perf['positions']}` 只")
        lines.append(f"  💵 可用现金: `${perf['cash']:,.0f}`")
    else:
        lines.append("  ℹ️ 尚未初始化虚拟组合")

    # ── Footer ──
    lines.append("")
    lines.append(f"{'━' * 28}")
    lines.append(f"🤖 Quant MVP v6.0")
    lines.append(f"_SMA + Momentum + Kronos Oracle_")
    lines.append(f"{'━' * 28}")

    return '\n'.join(lines)


# ─── Main ────────────────────────────────────────────────────────────────

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

    # Build report
    report = build_report(trade_date, elapsed, result.returncode, result.stderr)

    # Send to Telegram
    send_telegram(env, report)

    # Also print to stdout for cron log
    print(report)

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
