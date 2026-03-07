#!/usr/bin/env python3
"""
Daily Job Runner with LLM-Enhanced Telegram Notification (v3 — Decision-Story)

Runs daily_job.py, collects each surviving stock's full decision path,
sends to DeepSeek LLM to narrate WHY each stock was chosen.
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
                # Retry without Markdown if parse error
                if resp.status_code == 400:
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


# ── Collect Decision-Story Data ─────────────────────────

def collect_report_data(trade_date, elapsed, returncode, stderr):
    """Collect data with full decision path for each surviving stock."""

    data = {
        "date": trade_date,
        "status": "success" if returncode == 0 else "failed",
        "elapsed_seconds": round(elapsed),
        "error": stderr[-300:] if returncode != 0 else None,
    }

    if returncode != 0:
        return data

    data["universe_count"] = get_universe_count(trade_date)

    signals_df = read_signals(trade_date)
    targets_df = read_targets(trade_date)
    prices = get_latest_prices(trade_date)
    perf = read_portfolio_summary()
    nav = perf['nav'] if perf else 100000

    # ── Pipeline funnel summary (just counts, no stock lists) ──
    funnel = {}
    if signals_df is not None and len(signals_df) > 0:
        total_sigs = len(signals_df)
        unique_syms = signals_df['symbol'].nunique()
        buy_sigs = len(signals_df[signals_df['side'] > 0])
        sell_sigs = len(signals_df[signals_df['side'] < 0])

        # Count conflicting symbols (both buy+sell from dual model)
        conflict_count = 0
        for sym in signals_df['symbol'].unique():
            sides = signals_df[signals_df['symbol'] == sym]['side'].unique()
            if 1 in sides and -1 in sides:
                conflict_count += 1

        funnel["total_signals"] = total_sigs
        funnel["unique_symbols_with_signals"] = unique_syms
        funnel["buy_signals"] = buy_sigs
        funnel["sell_signals"] = sell_sigs
        funnel["dual_model_conflicts"] = conflict_count

    oracle_vetoed = 0
    if targets_df is not None and 'oracle_action' in targets_df.columns:
        if signals_df is not None:
            signal_syms = set(signals_df['symbol'].unique())
            target_syms = set(targets_df['symbol'].unique())
            oracle_vetoed = len(signal_syms - target_syms)
    funnel["oracle_vetoed"] = oracle_vetoed

    zero_weight = 0
    actionable_count = 0
    if targets_df is not None and 'target_weight' in targets_df.columns:
        actionable_count = int((targets_df['target_weight'].abs() >= 0.005).sum())
        zero_weight = len(targets_df) - actionable_count
    funnel["zero_weight_filtered"] = zero_weight
    funnel["final_actionable"] = actionable_count

    data["funnel"] = funnel

    # ── Decision story per surviving stock ──
    orders = []
    if targets_df is not None and 'target_weight' in targets_df.columns:
        actionable = targets_df[targets_df['target_weight'].abs() >= 0.005].copy()
        actionable = actionable.sort_values('target_weight', key=abs, ascending=False)
        has_oracle = 'oracle_action' in targets_df.columns

        for _, r in actionable.iterrows():
            sym = r['symbol']
            w = float(r['target_weight'])
            price = prices.get(sym, 0)
            qty = int(abs(w) * nav / price) if price > 0 else 0

            # Trace back: what did the dual models say about this stock?
            signal_detail = None
            if signals_df is not None:
                sym_sigs = signals_df[signals_df['symbol'] == sym]
                if len(sym_sigs) > 0:
                    sides = sym_sigs['side'].tolist()
                    prob = float(sym_sigs['prob'].iloc[0]) if 'prob' in sym_sigs.columns else None
                    rv = float(sym_sigs['realized_vol'].iloc[0]) if 'realized_vol' in sym_sigs.columns else None
                    aw = float(sym_sigs['avg_win'].iloc[0]) if 'avg_win' in sym_sigs.columns else None
                    al = float(sym_sigs['avg_loss'].iloc[0]) if 'avg_loss' in sym_sigs.columns else None

                    if len(sides) == 2 and sides[0] == sides[1]:
                        model_agreement = "both_agree"
                    elif len(sides) == 2 and sides[0] != sides[1]:
                        model_agreement = "conflict"
                    else:
                        model_agreement = "single"

                    signal_detail = {
                        "confidence_pct": round(prob * 100, 1) if prob else None,
                        "volatility": round(rv, 4) if rv else None,
                        "avg_win_pct": round(aw * 100, 2) if aw else None,
                        "avg_loss_pct": round(al * 100, 2) if al else None,
                        "model_agreement": model_agreement,
                        "signal_direction": "buy" if sides[0] > 0 else "sell",
                    }

            # Oracle verdict for this stock
            oracle_detail = None
            if has_oracle:
                oracle_action = r.get('oracle_action', 'N/A')
                oracle_pred = float(r.get('oracle_pred_ret', 0)) * 100
                oracle_detail = {
                    "action": oracle_action,
                    "predicted_return_pct": round(oracle_pred, 1),
                }

            orders.append({
                "symbol": sym,
                "direction": "BUY" if w > 0 else "SELL",
                "weight_pct": round(w * 100, 2),
                "price_usd": round(price, 2) if price > 0 else None,
                "shares": qty,
                "signal": signal_detail,
                "oracle": oracle_detail,
            })

    data["orders"] = orders
    data["portfolio"] = perf

    return data


# ── DeepSeek LLM Report ────────────────────────────────

REPORT_PROMPT = """你是一个量化交易系统的日报生成器。基于下方JSON数据生成Telegram日报。

**核心原则：以最终入选的每只股票为叙事主线，讲清楚"它为什么被选中"。**

要求：
1. Telegram Markdown格式（*粗体*、`代码`、_斜体_）
2. 手机端友好（每行≤35字符，无长横线）
3. 结构：

📊 *系统概况* （一行：日期+状态+耗时）

🔻 *决策漏斗*
用一句话概括筛选过程：
"{universe}只股票→{signals}个信号→{conflicts}只双模型冲突抵消→{oracle_vetoed}只被Kronos否决→{zero_weight}只仓位过小→最终{final}只入选"

📋 *操作指令及选股理由*
这是报告最核心的部分。对每笔订单：
- 写出精确指令：方向+代码+价格+股数
- 用1-2句话解释选中原因，需包含：
  · 双模型是否一致看好/看空
  · 置信度和波动率特征
  · Kronos专家的态度和预测
  · 为什么它能通过仓位筛选（如低波动率等）
- 每笔指令之间空行分隔

如果没有操作，写"今日无操作——所有候选均未通过筛选"并简要说明原因。

💼 *组合状态* 净值+累计回报+回撤+现金（紧凑一行）

签名：_Quant MVP v6.0 | SMA+Momentum+Kronos_

注意：
- 不要罗列被淘汰的股票列表！只在漏斗概述中给出被淘汰的数量即可
- 重点是讲故事：为什么最终选中的这几只能突围
- 总字符数<3500

JSON数据：
```json
{data_json}
```

直接输出报告内容。"""


def generate_llm_report(env, report_data):
    """Call DeepSeek to generate a decision-story report."""
    api_key = env.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        return None

    data_json = json.dumps(report_data, ensure_ascii=False, indent=2)
    prompt = REPORT_PROMPT.replace("{data_json}", data_json)

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content'].strip()
        if content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        print(f"[LLM] DeepSeek returned {len(content)} chars")
        return content
    except Exception as e:
        print(f"[LLM] DeepSeek failed: {e}")
        return None


# ── Fallback Template ───────────────────────────────────

def build_fallback_report(data):
    L = []
    L.append(f"📊 *Quant MVP v6.0 日报*")
    L.append(f"📅 `{data['date']}` {'✅' if data['status']=='success' else '❌'} `{data['elapsed_seconds']}s`")

    if data['status'] != 'success':
        L.append(f"\n⛔ 错误:\n```\n{data.get('error','')}\n```")
        return '\n'.join(L)

    f = data.get('funnel', {})
    L.append(f"\n🔻 *决策漏斗*")
    L.append(f"{data.get('universe_count','?')}只→{f.get('total_signals',0)}信号→冲突{f.get('dual_model_conflicts',0)}→否决{f.get('oracle_vetoed',0)}→零仓{f.get('zero_weight_filtered',0)}→*{f.get('final_actionable',0)}只入选*")

    orders = data.get('orders', [])
    L.append(f"\n📋 *操作指令*")
    if orders:
        for o in orders:
            icon = "🟢买" if o['direction'] == 'BUY' else "🔴卖"
            L.append(f"\n{icon} `{o['symbol']}` ${o['price_usd']:.0f}×{o['shares']}股 ({o['weight_pct']:+.1f}%)")
            sig = o.get('signal', {}) or {}
            orc = o.get('oracle', {}) or {}
            details = []
            if sig.get('model_agreement'): details.append(f"双模型{'一致' if sig['model_agreement']=='both_agree' else '冲突'}")
            if sig.get('confidence_pct'): details.append(f"置信{sig['confidence_pct']:.0f}%")
            if sig.get('volatility'): details.append(f"波动率{sig['volatility']:.3f}")
            if orc.get('action'): details.append(f"Kronos:{orc['action']}({orc.get('predicted_return_pct',0):+.1f}%)")
            if details: L.append(f"  ↳{'|'.join(details)}")
    else:
        L.append("⚪ 今日无操作")

    pf = data.get('portfolio')
    if pf:
        L.append(f"\n💼 `${pf['nav']:,.0f}` `{pf['cum_return']:+.2f}%` 回撤`{pf['max_dd']:.1f}%` 现金`${pf['cash']:,.0f}`")

    L.append(f"\n_Quant MVP v6.0 | SMA+Momentum+Kronos_")
    return '\n'.join(L)


# ── Main ────────────────────────────────────────────────

def main():
    env = load_env()
    start_time = datetime.now()
    trade_date = datetime.now().strftime('%Y-%m-%d')

    result = subprocess.run(
        [sys.executable, 'src/ops/daily_job.py'],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={**os.environ, **env, 'PYTHONPATH': PROJECT_ROOT}
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    report_data = collect_report_data(trade_date, elapsed, result.returncode, result.stderr)

    report = generate_llm_report(env, report_data)
    if not report:
        report = build_fallback_report(report_data)
        print("[REPORT] Using fallback template")
    else:
        print("[REPORT] Using DeepSeek LLM report")

    send_telegram(env, report)
    print(report)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
