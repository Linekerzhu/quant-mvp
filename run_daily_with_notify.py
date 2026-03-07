#!/usr/bin/env python3
"""
Daily Job Runner with LLM-Enhanced Telegram Notification

Runs daily_job.py, collects structured results, sends them to DeepSeek LLM
for professional report generation, then delivers to Telegram.
Falls back to template report if LLM unavailable.
"""

import os
import sys
import subprocess
import requests
import pandas as pd
import json
from datetime import datetime

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
                print(f"Telegram fail: {resp.status_code} {resp.text[:200]}")
                # If Markdown parse fails, retry without parse_mode
                if resp.status_code == 400 and "parse" in resp.text.lower():
                    requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=10)
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


# ── Collect Structured Data ─────────────────────────────

def collect_report_data(trade_date, elapsed, returncode, stderr):
    """Collect all pipeline data into a structured dict for LLM consumption."""

    data = {
        "date": trade_date,
        "status": "success" if returncode == 0 else "failed",
        "elapsed_seconds": round(elapsed),
        "error": stderr[-300:] if returncode != 0 else None,
    }

    if returncode != 0:
        return data

    data["universe_count"] = get_universe_count(trade_date)

    # Signals
    signals_df = read_signals(trade_date)
    if signals_df is not None and len(signals_df) > 0:
        buys = signals_df[signals_df['side'] > 0]
        sells = signals_df[signals_df['side'] < 0]
        avg_prob = float(signals_df['prob'].mean()) * 100 if 'prob' in signals_df.columns else 0

        top_buys = buys.sort_values('prob', ascending=False).head(5)['symbol'].tolist() if 'prob' in buys.columns and len(buys) > 0 else []
        top_sells = sells.sort_values('prob', ascending=False).head(5)['symbol'].tolist() if 'prob' in sells.columns and len(sells) > 0 else []

        data["signals"] = {
            "buy_count": len(buys),
            "sell_count": len(sells),
            "avg_confidence": round(avg_prob, 1),
            "top_buys": top_buys,
            "top_sells": top_sells,
        }
    else:
        data["signals"] = None

    # Targets & Oracle
    targets_df = read_targets(trade_date)
    prices = get_latest_prices(trade_date)
    perf = read_portfolio_summary()
    nav = perf['nav'] if perf else 100000

    if targets_df is not None and len(targets_df) > 0 and 'target_weight' in targets_df.columns:
        # Oracle stats
        has_oracle = 'oracle_action' in targets_df.columns
        oracle_info = None
        vetoed_syms = []
        if has_oracle:
            approved = targets_df[targets_df['oracle_action'] == 'approve']
            neutral = targets_df[targets_df['oracle_action'] == 'neutral']
            if signals_df is not None:
                signal_syms = set(signals_df['symbol'].unique())
                target_syms = set(targets_df['symbol'].unique())
                vetoed_syms = sorted(signal_syms - target_syms)
            oracle_info = {
                "approved": len(approved),
                "neutral": len(neutral),
                "vetoed": len(vetoed_syms),
                "vetoed_symbols": vetoed_syms,
            }

        # Actionable orders (filter zero-weight)
        actionable = targets_df[targets_df['target_weight'].abs() >= 0.005].copy()
        actionable = actionable.sort_values('target_weight', key=abs, ascending=False)
        zero_filtered = len(targets_df) - len(actionable)

        orders = []
        for _, r in actionable.iterrows():
            sym = r['symbol']
            w = float(r['target_weight'])
            price = prices.get(sym, 0)
            qty = int(abs(w) * nav / price) if price > 0 else 0
            oracle_pred = float(r.get('oracle_pred_ret', 0)) * 100 if has_oracle and 'oracle_pred_ret' in r.index else None

            orders.append({
                "symbol": sym,
                "direction": "BUY" if w > 0 else "SELL",
                "weight_pct": round(w * 100, 2),
                "price": round(price, 2) if price > 0 else None,
                "shares": qty,
                "kronos_pred_pct": round(oracle_pred, 1) if oracle_pred is not None else None,
            })

        data["oracle"] = oracle_info
        data["orders"] = orders
        data["zero_weight_filtered"] = zero_filtered
        data["total_weight_pct"] = round(actionable['target_weight'].sum() * 100, 1)
    else:
        data["oracle"] = None
        data["orders"] = []
        data["zero_weight_filtered"] = 0
        data["total_weight_pct"] = 0

    # Portfolio
    data["portfolio"] = perf

    return data


# ── DeepSeek LLM Report ────────────────────────────────

REPORT_PROMPT = """你是一个量化交易系统的日报生成器。请根据以下JSON数据，生成一份简洁、专业且适合在**Telegram手机端**阅读的每日交易报告。

规则：
1. 使用Telegram Markdown格式（*粗体*、`代码`、_斜体_）
2. 手机屏幕小，每行不超过35个字符，不要用长横线
3. 报告必须包含以下板块，用emoji标题分隔：
   - 📊 系统概况（一行：日期+状态+耗时）
   - 🧠 双模型信号（SMA+Momentum生成的买卖信号概况）
   - 🔮 Kronos专家审查（批准/否决数，列出所有被否决的股票代码）
   - 📋 执行指令（最重要板块！每笔订单必须写清：买/卖+股票代码+仓位比例+价格+股数；如果Kronos有预测，附上预测回报）
   - 💼 组合绩效（净值+回报+回撤+持仓数+现金）
4. 不要遗漏任何一笔订单！每一笔都要展示
5. 如果订单为空，写"今日无操作"
6. 末尾签名：_Quant MVP v6.0 | SMA+Momentum+Kronos_
7. 总长度控制在Telegram单条消息限制内（<4000字符）
8. 语言：中文为主，股票代码和数字用英文

JSON数据：
```json
{data_json}
```

直接输出报告内容，不要输出任何解释或开头语。"""


def generate_llm_report(env, report_data):
    """Call DeepSeek to generate a polished Telegram report."""
    api_key = env.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        print("[LLM] No DEEPSEEK_API_KEY, falling back to template")
        return None

    data_json = json.dumps(report_data, ensure_ascii=False, indent=2)
    prompt = REPORT_PROMPT.replace("{data_json}", data_json)

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result['choices'][0]['message']['content'].strip()

        # Strip markdown code fences if LLM wraps output
        if content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        print(f"[LLM] DeepSeek returned {len(content)} chars")
        return content

    except Exception as e:
        print(f"[LLM] DeepSeek failed: {e}")
        return None


# ── Fallback Template Report ────────────────────────────

def build_fallback_report(data):
    """Compact template report if LLM is unavailable."""
    L = []
    L.append(f"📊 *Quant MVP v6.0 日报*")
    L.append(f"📅 `{data['date']}` {'✅' if data['status']=='success' else '❌'} `{data['elapsed_seconds']}s`")

    if data['status'] != 'success':
        L.append(f"\n⛔ *错误*:\n```\n{data.get('error','')}\n```")
        return '\n'.join(L)

    uc = data.get('universe_count')
    if uc: L.append(f"🌐 股票池`{uc}`只")

    sig = data.get('signals')
    if sig:
        L.append(f"\n*▸ 双模型信号*")
        L.append(f"🟢买`{sig['buy_count']}`个 🔴卖`{sig['sell_count']}`个 置信`{sig['avg_confidence']:.0f}%`")

    orc = data.get('oracle')
    if orc:
        L.append(f"\n*▸ Kronos审查*")
        L.append(f"✅`{orc['approved']}` ⚖️`{orc['neutral']}` 🚫`{orc['vetoed']}`")
        if orc['vetoed_symbols']:
            L.append(f"否决: {' '.join('`'+s+'`' for s in orc['vetoed_symbols'][:20])}")

    orders = data.get('orders', [])
    L.append(f"\n*▸ 📋 执行指令*")
    if orders:
        buys = [o for o in orders if o['direction'] == 'BUY']
        sells = [o for o in orders if o['direction'] == 'SELL']
        L.append(f"买`{len(buys)}`笔 卖`{len(sells)}`笔 总仓`{data.get('total_weight_pct',0):.1f}%`")
        for o in orders:
            icon = "🟢" if o['direction'] == 'BUY' else "🔴"
            line = f"  {icon}`{o['symbol']}` {o['weight_pct']:+.1f}%"
            if o['price']: line += f" ${o['price']:.0f}×{o['shares']}股"
            if o.get('kronos_pred_pct') is not None: line += f" K`{o['kronos_pred_pct']:+.1f}%`"
            L.append(line)
        zf = data.get('zero_weight_filtered', 0)
        if zf > 0: L.append(f"_({zf}只权重0已过滤)_")
    else:
        L.append("⚪ 今日无操作")

    pf = data.get('portfolio')
    if pf:
        L.append(f"\n*▸ 组合*")
        L.append(f"💰`${pf['nav']:,.0f}` `{pf['cum_return']:+.2f}%` 回撤`{pf['max_dd']:.1f}%`")
        L.append(f"持仓`{pf['positions']}`只 现金`${pf['cash']:,.0f}`")

    L.append(f"\n_Quant MVP v6.0 | SMA+Momentum+Kronos_")
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

    # 1. Collect structured data
    report_data = collect_report_data(trade_date, elapsed, result.returncode, result.stderr)

    # 2. Try LLM-enhanced report
    report = generate_llm_report(env, report_data)

    # 3. Fall back to template if LLM fails
    if not report:
        report = build_fallback_report(report_data)
        print("[REPORT] Using fallback template")
    else:
        print("[REPORT] Using DeepSeek LLM report")

    # 4. Send to Telegram
    send_telegram(env, report)
    print(report)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
