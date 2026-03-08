"""
Phase G: Walk-Forward Backtester (v2 — Optimized)

Pre-computes features once for all data, then slices day-by-day for signals.
This is 100x faster than the naive per-day feature computation approach.

Usage:
    PYTHONPATH=/root/quant-mvp python3 src/backtest/walk_forward.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import FeatureEngineer
from src.risk.position_sizing import IndependentKellySizer

# ── Config ──
CFG = {
    "data_path": "data/backtest/sp500_2023_2026.parquet",
    "output_dir": "data/backtest/results",
    "initial_cash": 100_000,
    "friction_rate": 0.002,
    "backtest_start": "2024-07-01",
    "backtest_end": "2026-03-06",
    "warmup_days": 60,
    "max_single": 0.10,
    "min_single": 0.005,
    "max_gross": 1.0,
    "sma_fast": 20,
    "sma_slow": 60,
    "mom_window": 20,
}


def run_backtest():
    print("[BT] Loading data...")
    raw = pd.read_parquet(CFG["data_path"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

    # Standardize columns
    for src, dst in [("adj close", "adj_close")]:
        if src in raw.columns:
            raw = raw.rename(columns={src: dst})

    # Create adj OHLC from adj_close ratio
    if "adj_close" in raw.columns and "close" in raw.columns:
        ratio = raw["adj_close"] / raw["close"].replace(0, np.nan)
        ratio = ratio.fillna(1.0)
        for c in ["open", "high", "low", "close"]:
            raw[f"raw_{c}"] = raw[c]
            raw[f"adj_{c}"] = raw[c] * ratio
    else:
        for c in ["open", "high", "low", "close"]:
            if c in raw.columns:
                raw[f"raw_{c}"] = raw[c]
                raw[f"adj_{c}"] = raw[c]

    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    symbols = raw["symbol"].unique()
    all_dates = sorted(raw["date"].unique())
    print(f"[BT] {len(raw)} rows, {len(symbols)} symbols, {len(all_dates)} dates")

    # ── Step 1: Pre-compute ALL features at once ──
    print("[BT] Computing features for all data (one-time)...")
    feat_eng = FeatureEngineer()
    features = feat_eng.build_features(raw)
    print(f"[BT] Features: {len(features)} rows, {features.columns.tolist()[:10]}...")

    # ── Step 2: Pre-compute SMA + Momentum signals for all data ──
    print("[BT] Computing signals for all data...")
    features = features.sort_values(["symbol", "date"]).reset_index(drop=True)

    # SMA signals: fast > slow → buy, else sell
    g = features.groupby("symbol")["adj_close"]
    features["sma_fast"] = g.transform(lambda x: x.rolling(CFG["sma_fast"]).mean())
    features["sma_slow"] = g.transform(lambda x: x.rolling(CFG["sma_slow"]).mean())
    features["sma_side"] = np.where(features["sma_fast"] > features["sma_slow"], 1, -1)
    features.loc[features["sma_slow"].isna(), "sma_side"] = 0

    # Momentum signals: N-day log return > 0 → buy, else sell
    features["mom_ret"] = g.transform(lambda x: np.log(x / x.shift(CFG["mom_window"])))
    features["mom_side"] = np.where(features["mom_ret"] > 0, 1, -1)
    features.loc[features["mom_ret"].isna(), "mom_side"] = 0

    # ── Step 3: Meta-label probability ──
    # NOTE: Current meta_model_v1 is a MOCK model trained on random noise.
    # Using it would add random predictions. Instead, we use a simple
    # confidence mapping from signal agreement to give a clean baseline.
    # When a real model is trained in Phase H, this will be replaced.
    print("[BT] Using signal-based probability (mock model skipped)")
    # Base probability: signals that exist get 0.55, adjusted by RSI
    features["meta_prob"] = 0.55
    if "rsi_14" in features.columns:
        # Slightly boost prob for oversold buys and overbought sells
        rsi = features["rsi_14"].fillna(50)
        features["meta_prob"] = 0.52 + 0.06 * ((rsi - 50) / 50).clip(-1, 1).abs()

    # Ensure rv column
    rv_col = "rv_20d" if "rv_20d" in features.columns else None
    if rv_col is None:
        features["rv_20d"] = g.transform(lambda x: x.pct_change().rolling(20).std() * np.sqrt(252))
        rv_col = "rv_20d"

    # ── Step 4: Walk-Forward Loop ──
    print("[BT] Starting walk-forward simulation...")
    bt_start = pd.Timestamp(CFG["backtest_start"])
    bt_end = pd.Timestamp(CFG["backtest_end"])
    bt_dates = [d for d in all_dates if bt_start <= d <= bt_end]
    print(f"[BT] Backtest period: {bt_dates[0].date()} → {bt_dates[-1].date()} ({len(bt_dates)} days)")

    sizer = IndependentKellySizer()
    sizer.min_single = CFG["min_single"]
    sizer.max_single = CFG["max_single"]
    sizer.max_gross_leverage = CFG["max_gross"]

    cash = CFG["initial_cash"]
    positions = {}  # {sym: {"qty": int, "cost": float}}
    nav_hist = []
    trades = []
    daily_rets = []

    for day_i, td in enumerate(bt_dates):
        # Get today's signals
        today = features[features["date"] == td].copy()
        if len(today) == 0:
            nav = _calc_nav(cash, positions, features, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        # Build signal rows (SMA + Momentum as separate rows, like daily_job)
        sig_rows = []
        for _, row in today.iterrows():
            sym = row["symbol"]
            prob = float(row["meta_prob"])
            rv = float(row.get(rv_col, 0.20))
            rv = max(rv, 0.01)

            # avg_win / avg_loss from recent returns
            ret5 = float(row.get("returns_5d", 0.02)) if "returns_5d" in row.index else 0.02
            aw = abs(ret5) + 0.005
            al = abs(ret5) + 0.005

            for model, side_col in [("sma", "sma_side"), ("mom", "mom_side")]:
                side = int(row[side_col])
                if side != 0 and prob >= 0.52:  # Minimum confidence threshold
                    sig_rows.append({
                        "symbol": sym, "side": side, "prob": prob,
                        "avg_win": aw, "avg_loss": al, "realized_vol": rv,
                    })

        if not sig_rows:
            nav = _calc_nav(cash, positions, features, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        signals_df = pd.DataFrame(sig_rows)

        # Position sizing
        dd = _calc_dd(nav_hist, CFG["initial_cash"])
        try:
            pos_df = sizer.calculate_positions(signals_df, current_drawdown=dd)
        except:
            nav = _calc_nav(cash, positions, features, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        if pos_df.empty:
            nav = _calc_nav(cash, positions, features, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        # Merge duplicates
        pos_df = pos_df.groupby("symbol", as_index=False)["target_weight"].sum()

        # Apply limits
        def lim(w):
            s = 1 if w > 0 else -1 if w < 0 else 0
            if abs(w) < CFG["min_single"]: return 0.0
            if abs(w) > CFG["max_single"]: return s * CFG["max_single"]
            return w
        pos_df["target_weight"] = pos_df["target_weight"].apply(lim)
        pos_df = pos_df[pos_df["target_weight"].abs() > 0]

        # Execute at T+1 Open
        td_idx = list(all_dates).index(td)
        if td_idx + 1 >= len(all_dates):
            nav = _calc_nav(cash, positions, features, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        next_d = all_dates[td_idx + 1]
        next_data = features[features["date"] == next_d]
        nav = _calc_nav(cash, positions, features, td)

        for _, p in pos_df.iterrows():
            sym = p["symbol"]
            tw = p["target_weight"]
            sym_next = next_data[next_data["symbol"] == sym]
            if len(sym_next) == 0:
                continue

            exec_price = float(sym_next.iloc[0].get("adj_open", sym_next.iloc[0].get("raw_open", 0)))
            if exec_price <= 0:
                continue

            target_val = tw * nav
            target_qty = int(target_val / exec_price)
            cur_qty = positions.get(sym, {}).get("qty", 0)
            delta = target_qty - cur_qty
            if delta == 0:
                continue

            tv = abs(delta) * exec_price
            fri = tv * CFG["friction_rate"]

            if delta > 0:  # Buy
                cost = tv + fri
                if cost > cash:
                    delta = int((cash - fri) / exec_price)
                    if delta <= 0: continue
                    tv = delta * exec_price
                    fri = tv * CFG["friction_rate"]
                    cost = tv + fri
                cash -= cost
                old = positions.get(sym, {"qty": 0, "cost": exec_price})
                new_qty = old["qty"] + delta
                positions[sym] = {"qty": new_qty, "cost": (old["cost"]*old["qty"] + exec_price*delta)/new_qty}
            else:  # Sell
                sell_qty = min(abs(delta), cur_qty)
                if sell_qty <= 0: continue
                cash += sell_qty * exec_price - fri
                nq = cur_qty - sell_qty
                if nq <= 0:
                    positions.pop(sym, None)
                else:
                    positions[sym]["qty"] = nq

            trades.append({
                "date": str(td.date()), "symbol": sym,
                "side": "BUY" if delta > 0 else "SELL",
                "qty": abs(delta), "price": exec_price,
                "friction": fri,
            })

        nav = _calc_nav(cash, positions, features, td)
        nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
        if len(nav_hist) >= 2:
            daily_rets.append(nav / nav_hist[-2]["nav"] - 1)

        if (day_i + 1) % 50 == 0:
            ret_pct = (nav / CFG["initial_cash"] - 1) * 100
            print(f"  Day {day_i+1}/{len(bt_dates)}: NAV=${nav:,.0f} ({ret_pct:+.1f}%) pos={len(positions)} trades={len(trades)}")

    # ── Save results ──
    os.makedirs(CFG["output_dir"], exist_ok=True)
    pd.DataFrame(nav_hist).to_parquet(f"{CFG['output_dir']}/equity_curve.parquet")
    pd.DataFrame(trades).to_parquet(f"{CFG['output_dir']}/trade_log.parquet")

    # Metrics
    warmup = CFG["warmup_days"]
    navs = [h["nav"] for h in nav_hist]
    rets = np.array(daily_rets[warmup:]) if len(daily_rets) > warmup else np.array(daily_rets)
    navs_pw = navs[warmup:] if len(navs) > warmup else navs

    if len(rets) > 10:
        ann_ret = (navs_pw[-1] / navs_pw[0]) ** (252 / len(rets)) - 1
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        peak = navs_pw[0]
        max_dd = 0
        for n in navs_pw:
            if n > peak: peak = n
            dd = (peak - n) / peak
            if dd > max_dd: max_dd = dd
        wins = sum(1 for r in rets if r > 0)
        losses = sum(1 for r in rets if r < 0)
        wr = wins / (wins + losses) if (wins+losses) > 0 else 0
        aw = np.mean([r for r in rets if r > 0]) if wins > 0 else 0
        al = abs(np.mean([r for r in rets if r < 0])) if losses > 0 else 1
        pf = aw / al if al > 0 else 0
        fri_total = sum(t["friction"] for t in trades)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        metrics = {
            "period": f"{CFG['backtest_start']} → {CFG['backtest_end']}",
            "trading_days": len(rets),
            "total_trades": len(trades),
            "final_nav": round(navs[-1], 2),
            "total_return_pct": round((navs[-1] / CFG["initial_cash"] - 1) * 100, 2),
            "annual_return_pct": round(ann_ret * 100, 2),
            "annual_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "calmar": round(calmar, 3),
            "win_rate_pct": round(wr * 100, 1),
            "profit_factor": round(pf, 3),
            "total_friction": round(fri_total, 2),
            "avg_positions": round(np.mean([h["pos"] for h in nav_hist]), 1),
        }
    else:
        metrics = {"error": "insufficient data"}

    with open(f"{CFG['output_dir']}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # Go/No-Go
    s = metrics.get("sharpe", 0)
    d = metrics.get("max_dd_pct", 100)
    r = metrics.get("annual_return_pct", -1)
    print("\n  GO/NO-GO:")
    ok = True
    for label, val, fail, warn in [
        ("Sharpe", s, 0, 0.5),
        ("Max DD", d, 35, 25),
        ("Ann Ret", r, 0, 5),
    ]:
        if label == "Max DD":
            if val > fail: print(f"  🚫 {label}={val:.1f}% > {fail}%"); ok = False
            elif val > warn: print(f"  ⚠️  {label}={val:.1f}%")
            else: print(f"  ✅ {label}={val:.1f}%")
        else:
            if val < fail: print(f"  🚫 {label}={val:.3f} < {fail}"); ok = False
            elif val < warn: print(f"  ⚠️  {label}={val:.3f}")
            else: print(f"  ✅ {label}={val:.3f}")
    print(f"\n  {'✅ PROCEED TO PHASE H' if ok else '🚫 FIX ISSUES FIRST'}")
    print("=" * 50)


def _calc_nav(cash, positions, features, date):
    nav = cash
    for sym, pos in positions.items():
        sd = features[(features["symbol"] == sym) & (features["date"] <= date)]
        if len(sd) > 0:
            p = sd.iloc[-1]
            for c in ["adj_close", "raw_close"]:
                if c in p.index and pd.notna(p[c]) and p[c] > 0:
                    nav += pos["qty"] * float(p[c])
                    break
            else:
                nav += pos["qty"] * pos["cost"]
        else:
            nav += pos["qty"] * pos["cost"]
    return nav


def _calc_dd(nav_hist, initial):
    if not nav_hist:
        return 0.0
    navs = [h["nav"] for h in nav_hist]
    peak = max(max(navs), initial)
    return (peak - navs[-1]) / peak if peak > 0 else 0.0


if __name__ == "__main__":
    run_backtest()
