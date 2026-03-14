#!/usr/bin/env python3
"""
Quant-MVP Dashboard API v3 (FastAPI)
Professional-grade trading terminal API with decision chain,
enhanced indicators, funnel stats, and real-time portfolio data.
"""

import os
import json
import glob
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Quant-MVP Dashboard API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _latest(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(str(PROJECT_ROOT / pattern)))
    return Path(files[-1]) if files else None


def _s(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 2)


def _s4(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def _load_broker_state():
    """Load holdings from broker_api (primary) or portfolio state (fallback)."""
    broker_path = PROJECT_ROOT / "data/broker_api/account_state.json"
    legacy_path = PROJECT_ROOT / "data/portfolio/state.json"
    path = broker_path if broker_path.exists() else legacy_path
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_portfolio_history():
    """Load daily equity snapshots from portfolio/state.json (always has proper date/nav fields)."""
    path = PROJECT_ROOT / "data/portfolio/state.json"
    if not path.exists():
        return []
    with open(path) as f:
        state = json.load(f)
    history = state.get("history", [])
    # Only return entries that have a valid 'date' and 'nav' field
    return [h for h in history if h.get("date") and h.get("nav")]


def _load_trades():
    trade_log = PROJECT_ROOT / "data/portfolio/trades.jsonl"
    trades = []
    if trade_log.exists():
        with open(trade_log) as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except:
                    pass
    return trades


# ─── Endpoints ──────────────────────────────────────────

@app.get("/api/symbols")
def get_symbols():
    path = _latest("data/processed/features_*.parquet")
    if not path:
        return {"symbols": [], "date": None}
    df = pd.read_parquet(path, columns=["symbol"])
    symbols = sorted(df["symbol"].unique().tolist())
    date_str = path.stem.replace("features_", "")
    return {"symbols": symbols, "count": len(symbols), "date": date_str}


@app.get("/api/candles/{symbol}")
def get_candles(symbol: str, days: int = Query(default=120, le=400)):
    path = _latest("data/processed/features_*.parquet")
    if not path:
        raise HTTPException(404, "No features data")
    df = pd.read_parquet(path)
    sym_df = df[df["symbol"] == symbol].copy()
    if sym_df.empty:
        raise HTTPException(404, f"{symbol} not found")

    sym_df = sym_df.sort_values("date")
    close_col = "adj_close" if "adj_close" in sym_df.columns else "raw_close"

    # Compute overlays
    sym_df["sma_20"] = sym_df[close_col].rolling(20).mean()
    sym_df["sma_60"] = sym_df[close_col].rolling(60).mean()
    bb_m = sym_df[close_col].rolling(60).mean()
    bb_s = sym_df[close_col].rolling(60).std()
    sym_df["bb_upper"] = bb_m + 2.5 * bb_s
    sym_df["bb_lower"] = bb_m - 2.5 * bb_s

    # RSI
    delta = sym_df[close_col].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    sym_df["rsi_calc"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = sym_df[close_col].ewm(span=12).mean()
    ema26 = sym_df[close_col].ewm(span=26).mean()
    sym_df["macd_line_calc"] = ema12 - ema26
    sym_df["macd_signal_calc"] = sym_df["macd_line_calc"].ewm(span=9).mean()
    sym_df["macd_hist_calc"] = sym_df["macd_line_calc"] - sym_df["macd_signal_calc"]

    # Volume MA
    sym_df["vol_ma_20"] = sym_df["volume"].rolling(20).mean()

    sym_df = sym_df.tail(days)

    candles = []
    for _, r in sym_df.iterrows():
        ts = pd.Timestamp(r["date"])
        if ts.tzinfo:
            ts = ts.tz_localize(None)
        rec = {
            "time": ts.strftime("%Y-%m-%d"),
            "open": _s(r.get("adj_open", r.get("raw_open", 0))),
            "high": _s(r.get("adj_high", r.get("raw_high", 0))),
            "low": _s(r.get("adj_low", r.get("raw_low", 0))),
            "close": _s(r.get(close_col, 0)),
            "volume": int(r.get("volume", 0)),
        }
        if pd.notna(r["sma_20"]): rec["sma20"] = _s(r["sma_20"])
        if pd.notna(r["sma_60"]): rec["sma60"] = _s(r["sma_60"])
        if pd.notna(r["bb_upper"]): rec["bb_upper"] = _s(r["bb_upper"])
        if pd.notna(r["bb_lower"]): rec["bb_lower"] = _s(r["bb_lower"])
        # RSI
        if pd.notna(r.get("rsi_calc")):
            rec["rsi"] = _s(r["rsi_calc"])
        elif "rsi_14" in r.index and pd.notna(r["rsi_14"]):
            rec["rsi"] = _s(r["rsi_14"])
        # MACD
        if pd.notna(r.get("macd_line_calc")):
            rec["macd"] = _s4(r["macd_line_calc"])
            rec["macd_signal"] = _s4(r["macd_signal_calc"])
            rec["macd_hist"] = _s4(r["macd_hist_calc"])
        # ADX
        if "adx_14" in r.index and pd.notna(r["adx_14"]):
            rec["adx"] = _s(r["adx_14"])
        # Realized vol
        if "rv_20d" in r.index and pd.notna(r["rv_20d"]):
            rec["rv20"] = _s4(r["rv_20d"])
        # Volume MA
        if pd.notna(r.get("vol_ma_20")):
            rec["vol_ma"] = int(r["vol_ma_20"])
        candles.append(rec)

    # Attach trade markers from trades.jsonl
    all_trades = _load_trades()
    markers = [
        {
            "time": t["date"],
            "action": t.get("action", "BUY"),
            "qty": abs(t.get("qty", 0)),
            "price": round(t.get("price", 0), 2),
            "pnl": round(t.get("realized_pnl", 0), 2) if "realized_pnl" in t else None,
        }
        for t in all_trades if t.get("symbol") == symbol
    ]

    # Cost line from broker state
    cost_line = None
    state = _load_broker_state()
    if state:
        holdings = state.get("holdings", state.get("positions", {}))
        if symbol in holdings:
            cost_line = round(holdings[symbol].get("avg_cost", 0), 2)

    return {
        "symbol": symbol, "count": len(candles), "candles": candles,
        "trades": markers, "cost_line": cost_line,
    }


@app.get("/api/portfolio")
def get_portfolio():
    state = _load_broker_state()
    if not state:
        return {"error": "No portfolio data"}

    holdings = state.get("holdings", state.get("positions", {}))
    cash = state.get("cash", 0)
    initial = state.get("initial_cash", 100000)

    # Mark-to-market
    prices = {}
    feat_path = _latest("data/processed/features_*.parquet")
    if feat_path:
        feat_df = pd.read_parquet(feat_path)
        for sym in holdings:
            sd = feat_df[feat_df["symbol"] == sym]
            if not sd.empty:
                for col in ["adj_close", "raw_close"]:
                    if col in sd.columns:
                        v = sd[col].iloc[-1]
                        if pd.notna(v) and v > 0:
                            prices[sym] = float(v)
                            break

    # Days held calculation from trades
    all_trades = _load_trades()
    first_buy_date = {}
    for t in all_trades:
        sym = t.get("symbol", "")
        action = t.get("action", "")
        if action in ("BUY",) and sym not in first_buy_date:
            first_buy_date[sym] = t.get("date", "")
        elif action in ("CLOSE", "SELL"):
            first_buy_date.pop(sym, None)

    today_str = datetime.now().strftime("%Y-%m-%d")
    positions = []
    total_mv = 0
    for sym, pos in holdings.items():
        qty = pos.get("qty", 0)
        avg_cost = pos.get("avg_cost", 0)
        price = prices.get(sym, avg_cost)
        mv = qty * price
        pnl = mv - qty * avg_cost
        pnl_pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0

        # Days held
        days_held = 0
        if sym in first_buy_date and first_buy_date[sym]:
            try:
                buy_dt = datetime.strptime(first_buy_date[sym], "%Y-%m-%d")
                days_held = (datetime.now() - buy_dt).days
            except:
                pass

        total_mv += mv
        positions.append({
            "symbol": sym, "qty": qty,
            "avg_cost": round(avg_cost, 2), "price": round(price, 2),
            "market_value": round(mv, 2),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
        })

    nav = cash + total_mv
    for p in positions:
        p["weight"] = round(p["market_value"] / nav * 100, 1) if nav > 0 else 0

    # Use portfolio/state.json for daily equity history (broker_api history stores per-trade entries)
    history = _load_portfolio_history()
    peak = initial
    max_dd = 0
    equity = []
    for h in history:
        n = h.get("nav", initial)
        if n > peak: peak = n
        dd = (peak - n) / peak
        if dd > max_dd: max_dd = dd
        equity.append({"date": h.get("date", ""), "nav": round(n, 2),
                        "cash": round(h.get("cash", 0), 2),
                        "positions_value": round(h.get("positions_value", 0), 2),
                        "positions_count": h.get("positions_count", 0),
                        "trades": h.get("trades_today", 0),
                        "daily_return": round(h.get("daily_return", 0) * 100, 3),
                        "cum_pnl": round(h.get("cumulative_pnl", 0), 2),
                        "friction": round(h.get("total_friction", 0), 2)})

    return {
        "nav": round(nav, 2), "cash": round(cash, 2), "initial": initial,
        "cum_return": round((nav / initial - 1) * 100, 3),
        "realized_pnl": round(state.get("realized_pnl", 0), 2),
        "unrealized_pnl": round(sum(p["pnl"] for p in positions), 2),
        "total_friction": round(state.get("total_friction_paid", 0), 2),
        "trade_count": state.get("trade_count", 0),
        "max_drawdown": round(max_dd * 100, 2),
        "positions": positions,
        "invested_pct": round(total_mv / nav * 100, 1) if nav > 0 else 0,
        "cash_pct": round(cash / nav * 100, 1) if nav > 0 else 0,
        "equity_curve": equity,
    }


@app.get("/api/trades")
def get_trades(limit: int = Query(default=50, le=200)):
    all_trades = _load_trades()
    trades = [
        {
            "date": t.get("date", ""),
            "symbol": t.get("symbol", ""),
            "action": t.get("action", ""),
            "qty": abs(t.get("qty", 0)),
            "price": round(t.get("price", 0), 2),
            "friction": round(t.get("friction", 0), 2),
            "pnl": round(t.get("realized_pnl", 0), 2) if "realized_pnl" in t else None,
        }
        for t in all_trades
    ]
    return {"trades": list(reversed(trades[-limit:]))}


@app.get("/api/decision/{symbol}")
def get_decision(symbol: str):
    """Decision chain for a specific symbol — shows each model's vote + oracle + final weight."""
    # 1. Load all signal rows for this symbol
    sig_path = _latest("data/processed/signals_*.parquet")
    sig_date = sig_path.stem.replace("signals_", "") if sig_path else None

    models = {}
    consensus = {"vote": 0, "passed": False, "synth_side": 0}

    if sig_path and sig_path.exists():
        sig_df = pd.read_parquet(sig_path)
        sym_sigs = sig_df[sig_df["symbol"] == symbol]

        # Try to identify model rows (row0=SMA, row1=Momentum based on pipeline order)
        model_names = ["sma", "momentum"]
        rows = sym_sigs.reset_index(drop=True)
        for i, name in enumerate(model_names):
            if i < len(rows):
                r = rows.iloc[i]
                models[name] = {
                    "side": int(r.get("side", 0)),
                    "prob": _s4(r.get("prob")),
                    "realized_vol": _s4(r.get("realized_vol")),
                    "avg_win": _s4(r.get("avg_win")),
                    "avg_loss": _s4(r.get("avg_loss")),
                }
            else:
                models[name] = {"side": 0, "prob": None, "realized_vol": None}

        # Check mean_reversion from all signals (3 rows when MR is active)
        if len(rows) >= 3:
            r = rows.iloc[2]
            models["mean_reversion"] = {
                "side": int(r.get("side", 0)),
                "prob": _s4(r.get("prob")),
                "realized_vol": _s4(r.get("realized_vol")),
            }
        else:
            models["mean_reversion"] = {"side": 0, "prob": None}

        # Consensus (confidence-weighted voting)
        if not sym_sigs.empty:
            vote_sum = sum(
                int(r.get("side", 0)) * float(r.get("prob", 0.5))
                for _, r in sym_sigs.iterrows()
            )
            consensus = {
                "vote": round(vote_sum, 3),
                "passed": abs(vote_sum) >= 1.0,
                "synth_side": int(np.sign(vote_sum)) if abs(vote_sum) >= 1.0 else 0,
            }

    # 2. Oracle decision
    oracle = {"action": "N/A", "reason": "", "pred_ret": None}
    tgt_path = _latest("data/processed/targets_*.parquet")
    final_weight = 0.0
    if tgt_path and tgt_path.exists():
        tgt_df = pd.read_parquet(tgt_path)
        sym_tgt = tgt_df[tgt_df["symbol"] == symbol]
        if not sym_tgt.empty:
            r = sym_tgt.iloc[0]
            oracle = {
                "action": str(r.get("oracle_action", "N/A")),
                "reason": str(r.get("oracle_reason", "")),
                "pred_ret": _s4(r.get("oracle_pred_ret")),
            }
            final_weight = _s4(r.get("target_weight", 0)) or 0.0

    # 3. Key factors from latest features
    key_factors = {}
    feat_path = _latest("data/processed/features_*.parquet")
    if feat_path and feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
        sym_feat = feat_df[feat_df["symbol"] == symbol]
        if not sym_feat.empty:
            r = sym_feat.iloc[-1]
            factor_cols = [
                "rsi_14", "adx_14", "rv_20d", "rv_60d",
                "returns_5d", "returns_10d", "returns_20d", "returns_60d",
                "market_breadth", "vix_change_5d",
                "price_vs_sma20_zscore", "price_vs_sma60_zscore",
                "macd_line_pct", "macd_histogram_pct",
                "relative_volume_20d", "regime_combined",
            ]
            for col in factor_cols:
                if col in r.index and pd.notna(r[col]):
                    val = r[col]
                    if isinstance(val, str):
                        key_factors[col] = val
                    else:
                        key_factors[col] = round(float(val), 4)

    return {
        "symbol": symbol,
        "signal_date": sig_date,
        "models": models,
        "consensus": consensus,
        "oracle": oracle,
        "final_weight": final_weight,
        "key_factors": key_factors,
    }


@app.get("/api/funnel")
def get_funnel():
    """Decision funnel aggregated statistics."""
    # Try loading from saved funnel_stats JSON
    funnel_path = _latest("data/processed/funnel_stats_*.json")
    if funnel_path and funnel_path.exists():
        with open(funnel_path) as f:
            stats = json.load(f)
        date_str = funnel_path.stem.replace("funnel_stats_", "")
        stats["date"] = date_str
        return stats

    # Fallback: compute from signals + targets
    stats = {"date": None}
    sig_path = _latest("data/processed/signals_*.parquet")
    tgt_path = _latest("data/processed/targets_*.parquet")

    if sig_path:
        sig_df = pd.read_parquet(sig_path)
        stats["date"] = sig_path.stem.replace("signals_", "")
        stats["total_base_signals"] = len(sig_df)
        stats["unique_symbols_with_signals"] = int(sig_df["symbol"].nunique())
        # Estimate consensus: group by symbol, sum side*prob, count those >= 1.0
        if not sig_df.empty and "prob" in sig_df.columns:
            sig_df["vote"] = sig_df["side"] * sig_df["prob"]
            votes = sig_df.groupby("symbol")["vote"].sum()
            stats["passed_consensus"] = int((votes.abs() >= 1.0).sum())
            stats["dual_model_conflicts"] = int((votes.abs() < 1.0).sum())
        else:
            stats["passed_consensus"] = 0
            stats["dual_model_conflicts"] = 0

    if tgt_path:
        tgt_df = pd.read_parquet(tgt_path)
        if "oracle_action" in tgt_df.columns:
            stats["oracle_vetoed"] = int((tgt_df["oracle_action"] == "veto").sum())
        if "target_weight" in tgt_df.columns:
            stats["passed_sizing"] = int((tgt_df["target_weight"].abs() >= 0.005).sum())
            stats["zero_weight_filtered"] = int((tgt_df["target_weight"].abs() < 0.005).sum())
            stats["final_actionable"] = stats["passed_sizing"]

    # Universe count
    feat_path = _latest("data/processed/features_*.parquet")
    if feat_path:
        try:
            feat_df = pd.read_parquet(feat_path, columns=["symbol"])
            stats["universe_count"] = int(feat_df["symbol"].nunique())
        except:
            pass

    return stats


@app.get("/api/signals")
def get_signals():
    path = _latest("data/processed/signals_*.parquet")
    if not path:
        return {"signals": [], "date": None}
    df = pd.read_parquet(path)
    date_str = path.stem.replace("signals_", "")
    records = []
    for _, r in df.iterrows():
        rec = {"symbol": r["symbol"],
               "side": int(r.get("side", 0)),
               "prob": round(float(r.get("prob", 0)), 4)}
        if "realized_vol" in r.index: rec["vol"] = round(float(r["realized_vol"]), 4)
        records.append(rec)
    return {"date": date_str, "count": len(records), "signals": records}


@app.get("/api/targets")
def get_targets():
    path = _latest("data/processed/targets_*.parquet")
    if not path:
        return {"targets": [], "date": None}
    df = pd.read_parquet(path)
    date_str = path.stem.replace("targets_", "")
    records = []
    for _, r in df.iterrows():
        rec = {"symbol": r["symbol"],
               "weight": round(float(r.get("target_weight", 0)), 4)}
        if "oracle_action" in r.index: rec["oracle"] = r["oracle_action"]
        if "oracle_pred_ret" in r.index: rec["oracle_ret"] = round(float(r["oracle_pred_ret"]) * 100, 2)
        records.append(rec)
    return {"date": date_str, "count": len(records), "targets": records}


# ─── Static ─────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
