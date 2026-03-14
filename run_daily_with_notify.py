#!/usr/bin/env python3
"""
Daily Job Runner with LLM-Enhanced Telegram Notification (v5 — Factor Strategy)

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
import warnings

# Suppress urllib3 SSL warnings on Mac
warnings.filterwarnings("ignore", module="urllib3")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

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
    """Read portfolio state from VirtualBroker (primary) or old PortfolioTracker (fallback)."""
    # Primary: VirtualBroker independent account
    broker_path = os.path.join(PROJECT_ROOT, 'data/broker_api/account_state.json')
    # Fallback: old PortfolioTracker
    legacy_path = os.path.join(PROJECT_ROOT, 'data/portfolio/state.json')
    
    path = broker_path if os.path.exists(broker_path) else legacy_path
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        state = json.load(f)
    
    # VirtualBroker uses 'holdings', PortfolioTracker uses 'positions'
    holdings = state.get('holdings', state.get('positions', {}))
    cash = state.get('cash', 0)
    initial = state.get('initial_cash', 100000)
    realized_pnl = state.get('realized_pnl', 0)
    
    # Calculate NAV from holdings (mark-to-market at cost if no prices available yet)
    positions_value = sum(
        pos.get('qty', 0) * pos.get('avg_cost', 0) 
        for pos in holdings.values()
    )
    
    # Try to get NAV from history if available (more accurate)
    history = state.get('history', [])
    if history:
        nav = history[-1].get('nav', cash + positions_value)
    else:
        nav = cash + positions_value
    
    # Max drawdown
    peak = initial
    max_dd = 0
    for snap in history:
        snap_nav = snap.get('nav', initial)
        if snap_nav > peak: peak = snap_nav
        dd = (peak - snap_nav) / peak
        if dd > max_dd: max_dd = dd
    
    return {
        'nav': nav, 'initial': initial,
        'cum_return': (nav / initial - 1) * 100,
        'max_dd': max_dd * 100,
        'trading_days': len(history) if history else 0,
        'total_trades': state.get('trade_count', len(state.get('history', []))),
        'total_friction': state.get('total_friction_paid', 0),
        'realized_pnl': realized_pnl,
        'positions': len(holdings),
        'positions_detail': holdings,
        'cash': cash,
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
    funnel_path = os.path.join(PROJECT_ROOT, f'data/processed/funnel_stats_{trade_date}.json')
    if os.path.exists(funnel_path):
        try:
            with open(funnel_path, "r") as f:
                funnel = json.load(f)
        except:
            pass

    # Provide fallback metrics if not provided by JSON
    if "total_base_signals" not in funnel:
        funnel["total_base_signals"] = len(signals_df) if signals_df is not None else 0
    if "passed_meta_model" not in funnel:
        funnel["passed_meta_model"] = len(signals_df) if signals_df is not None else 0
    if "passed_consensus" not in funnel:
        funnel["passed_consensus"] = len(signals_df) if signals_df is not None else 0
    if "oracle_vetoed" not in funnel:
        funnel["oracle_vetoed"] = 0
    if "passed_sizing" not in funnel:
        funnel["passed_sizing"] = 0
        if targets_df is not None and 'target_weight' in targets_df.columns:
            funnel["passed_sizing"] = int((targets_df['target_weight'].abs() >= 0.005).sum())

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
            target_qty = int(abs(w) * nav / price) if price > 0 else 0
            
            existing_qty = 0
            if perf and 'positions_detail' in perf and sym in perf['positions_detail']:
                existing_qty = perf['positions_detail'][sym].get('qty', 0)
                
            delta_qty = target_qty - existing_qty
            if w < 0:
                delta_qty = -target_qty - existing_qty
                
            if delta_qty == 0:
                continue

            qty = abs(delta_qty)
            direction = "BUY" if delta_qty > 0 else "SELL"

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
                "direction": direction,
                "weight_pct": round(w * 100, 2),
                "price_usd": round(price, 2) if price > 0 else None,
                "shares": qty,
                "signal": signal_detail,
                "oracle": oracle_detail,
            })


        # Also need to check if there are sells from positions NOT in targets at all
        if perf and 'positions_detail' in perf:
            for sym, pos_data in perf['positions_detail'].items():
                if sym not in actionable['symbol'].values:
                    existing_qty = pos_data.get('qty', 0)
                    if existing_qty > 0:
                        price = prices.get(sym, 0)
                        
                        signal_detail = {"signal_direction": "close"}
                        oracle_detail = None
                        
                        orders.append({
                            "symbol": sym,
                            "direction": "SELL",
                            "weight_pct": 0.0,
                            "price_usd": round(price, 2) if price > 0 else None,
                            "shares": existing_qty,
                            "signal": signal_detail,
                            "oracle": oracle_detail,
                        })

    data["orders"] = orders
    data["portfolio"] = perf

    return data


# ── DeepSeek LLM Report ────────────────────────────────

REPORT_PROMPT = """你是一个量化交易系统的日报生成器。基于下方JSON数据生成Telegram日报。

**核心原则：以最终入选的每只股票为叙事主线，讲清楚"它为什么被选中"。**

本系统使用v5多因子策略：
- 70%动量(12-1月) + 15%质量(低波动) + 15%价值(相对SMA200)
- 截面排名选Top-15，月频再平衡
- 逆波动率加权 + 行业25%上限
- VIX崩盘保护 + ML择时

要求：
1. Telegram Markdown格式（*粗体*、`代码`、_斜体_）
2. 手机端友好（每行≤35字符，无长横线）
3. 结构：

📊 *系统概况* （一行：日期+状态+耗时）

🔻 *选股漏斗*
概括筛选过程：宇宙{universe}只→因子排名→Top-15→行业cap→最终{final}只

📋 *操作指令及选股理由*
对每笔订单：
- 写出精确指令：方向+代码+价格+股数
- 用1-2句话解释：动量强度、波动率特征、行业

💼 *持仓明细*
列出现有持仓。格式：`代码` 数量股 (成本价)

💰 *组合状态* 净值+累计回报+回撤+现金

签名：_Quant MVP v5 | 多因子月频+VIX保护_

注意：总字符数<3500

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
    L.append(f"📊 *Quant MVP v5 日报*")
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
    L.append(f"\n💼 *持仓明细*")
    if pf and pf.get('positions_detail'):
        for sym, detail in pf['positions_detail'].items():
            L.append(f"`{sym}` {detail.get('qty', 0)}股 (成本 ${detail.get('avg_cost', 0):.2f})")
    else:
        L.append("当前空仓")

    if pf:
        L.append(f"\n💰 `${pf['nav']:,.0f}` `{pf['cum_return']:+.2f}%` 回撤`{pf['max_dd']:.1f}%` 现金`${pf['cash']:,.0f}`")

    L.append(f"\n_Quant MVP v5 | 多因子月频+VIX保护_")
    return '\n'.join(L)


# ── Main ────────────────────────────────────────────────

def is_rebalance_day(trade_date: str) -> bool:
    """Check if today is a rebalance day (first trading day of the month)."""
    # Check if signals file was generated with actual trades
    targets_path = os.path.join(PROJECT_ROOT, f'data/processed/targets_{trade_date}.parquet')
    signals_path = os.path.join(PROJECT_ROOT, f'data/processed/signals_{trade_date}.parquet')
    
    if os.path.exists(targets_path):
        try:
            df = pd.read_parquet(targets_path)
            if len(df) > 0 and 'target_weight' in df.columns:
                return True
        except:
            pass
    
    # Fallback: check date — is it the first trading day of this month?
    from datetime import datetime as dt
    d = dt.strptime(trade_date, '%Y-%m-%d')
    if d.day <= 3:  # First 3 calendar days likely = first trading day
        return True
    
    return False


def build_brief_report(trade_date: str) -> str:
    """One-line daily brief for non-rebalance days."""
    pf = read_portfolio_summary()
    
    # Try to get VIX
    vix_str = ""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", period="1d", progress=False, auto_adjust=False)
        if len(vix) > 0:
            vix_val = float(vix.droplevel('Ticker', axis=1)['Close'].iloc[-1])
            if vix_val > 40:
                vix_str = f"🔴VIX={vix_val:.0f}"
            elif vix_val > 25:
                vix_str = f"🟡VIX={vix_val:.0f}"
            else:
                vix_str = f"VIX={vix_val:.0f}"
    except:
        pass
    
    # Kronos verdicts if any
    kronos_str = ""
    verdicts_path = os.path.join(PROJECT_ROOT, f'data/processed/kronos_verdicts_{trade_date}.json')
    if os.path.exists(verdicts_path):
        try:
            with open(verdicts_path) as f:
                verdicts = json.load(f)
            vetoed = [s for s, v in verdicts.items() if v.get('action') == 'veto']
            if vetoed:
                kronos_str = f" Kronos否决{len(vetoed)}只"
        except:
            pass
    
    if pf:
        # Daily change
        hist = pf.get('positions', 0)
        nav = pf['nav']
        change = pf['cum_return']
        dd = pf['max_dd']
        cash_pct = pf['cash'] / nav * 100 if nav > 0 else 0
        
        lines = [
            f"📈 `{trade_date}` NAV `${nav:,.0f}` `{change:+.1f}%`",
            f"持仓{pf['positions']}只 回撤{dd:.1f}% 现金{cash_pct:.0f}% {vix_str}{kronos_str}",
            f"_v5 多因子 | 下次再平衡: 下月初_",
        ]
    else:
        lines = [
            f"📈 `{trade_date}` 系统运行正常 ✅ {vix_str}",
            f"_v5 多因子 | 待首次再平衡_",
        ]
    
    return '\n'.join(lines)


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

    # Dual-mode reporting: full report on rebalance days, brief on regular days
    rebalance = is_rebalance_day(trade_date)
    
    if result.returncode != 0:
        # Error report — always send full details
        report_data = collect_report_data(trade_date, elapsed, result.returncode, result.stderr)
        report = build_fallback_report(report_data)
        print("[REPORT] Error report")
    elif rebalance:
        # 📋 Rebalance day — full LLM decision report
        report_data = collect_report_data(trade_date, elapsed, result.returncode, result.stderr)
        report = generate_llm_report(env, report_data)
        if not report:
            report = build_fallback_report(report_data)
            print("[REPORT] Rebalance day: fallback template")
        else:
            print("[REPORT] Rebalance day: DeepSeek LLM report")
    else:
        # 📈 Regular day — one-line brief
        report = build_brief_report(trade_date)
        print("[REPORT] Regular day: brief")

    send_telegram(env, report)
    print(report)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
