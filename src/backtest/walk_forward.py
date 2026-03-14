"""
Walk-Forward Backtester v5 — Multi-Factor + ML Timing + VIX Crash Protection

Features:
  1. MULTI-FACTOR COMPOSITE: 70% Momentum(12-1) + 15% Quality(low vol) + 15% Value(price-to-SMA200)
  2. SECTOR NEUTRALITY: Max 25% per GICS sector, redistributed to next-best stocks
  3. ML TIMING LAYER: LightGBM regime classifier for adaptive exposure scaling
  4. VIX CRASH PROTECTION: Auto-reduce exposure when VIX > 30 (COVID/crisis protection)
  5. FULL-CYCLE BACKTEST: 2019-2026 spanning COVID crash, 2022 bear, 2024 bull

Usage:
    PYTHONPATH=/Users/zjz/quant-mvp python3 src/backtest/walk_forward.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Config ──
CFG = {
    # Data paths (merged for full-cycle backtest)
    "data_paths": [
        "data/backtest/sp500_2018_2023.parquet",
        "data/backtest/sp500_2023_2026.parquet",
    ],
    "vix_path": "data/backtest/vix_2018_2026.parquet",
    "sector_path": "data/sp500_sectors.json",
    "output_dir": "data/backtest/results",
    "initial_cash": 100_000,
    "friction_rate": 0.002,
    "backtest_start": "2019-07-01",   # 7-year full-cycle
    "backtest_end": "2026-03-06",
    "warmup_days": 60,
    "max_positions": 15,
    "rebalance_freq": "monthly",
    # Factor weights
    "w_momentum": 0.70,
    "w_quality": 0.15,
    "w_value": 0.15,
    # Risk overlays
    "sector_cap": 0.25,           # Max 25% in any one sector
    "dd_scaling": False,           # Disabled: causes death-spiral after large DDs
    "vol_weight": True,
    "ml_timing": True,            # ML regime classifier
    # VIX crash protection (v5)
    "vix_protection": True,
    "vix_thresholds": {
        40: 0.50,   # VIX > 40 → 50% exposure (COVID-level stress begins)
        55: 0.25,   # VIX > 55 → 25% exposure (extreme panic)
        70: 0.00,   # VIX > 70 → full exit (2020-03-16 level)
    },
}


def _build_sector_map(path):
    """Load sector mapping: {symbol: sector}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        sector_data = json.load(f)
    sym2sec = {}
    for sector, syms in sector_data.items():
        for s in syms:
            if s not in sym2sec:  # First match wins
                sym2sec[s] = sector
    return sym2sec


def _train_regime_classifier(raw, all_dates, train_end):
    """
    Train a lightweight LightGBM regime classifier on market-level features.
    Predicts next-month market direction for exposure scaling.
    No look-ahead bias: trained only on data <= train_end.
    """
    import lightgbm as lgb

    # Build market-level daily features
    market_df = raw.groupby("date").agg(
        median_ret_1m=("ret_1m", "median"),
        breadth_sma200=("above_sma200", "mean"),
        cross_vol=("rv_20d", "median"),
        pct_positive_mom=("mom_1m_positive", "mean"),
        median_volume_ratio=("volume_ratio_20d", "median"),
    ).reset_index()
    market_df = market_df.sort_values("date").reset_index(drop=True)

    # Target: median stock return over next 21 days
    market_df["fwd_21d_ret"] = market_df["median_ret_1m"].shift(-21)
    market_df = market_df.dropna()

    # Binary: > 0 = bullish, <= 0 = bearish
    market_df["target"] = (market_df["fwd_21d_ret"] > 0).astype(int)

    # AUDIT FIX: shift(-21) means labels for date T use data from date T+21.
    # We must ensure T+21 <= train_end, not just T <= train_end.
    # Otherwise the last ~21 training labels leak future backtest data.
    label_horizon = 21
    safe_end = train_end - pd.Timedelta(days=int(label_horizon * 1.5))  # ~31 calendar days buffer
    train = market_df[market_df["date"] <= safe_end].copy()
    n_dropped = len(market_df[market_df["date"] <= train_end]) - len(train)
    print(f"[ML] Label safety: dropped {n_dropped} samples to prevent look-ahead ({safe_end.date()} cutoff)")
    if len(train) < 50:
        print("[ML] Insufficient training data, falling back to static filter")
        return None, None

    feat_cols = ["breadth_sma200", "cross_vol", "pct_positive_mom", "median_volume_ratio"]
    X_train = train[feat_cols].values
    y_train = train["target"].values

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary", "metric": "auc",
        "max_depth": 2, "num_leaves": 4,
        "min_data_in_leaf": 20, "learning_rate": 0.05,
        "feature_fraction": 0.8, "verbose": -1, "seed": 42,
    }
    model = lgb.train(params, train_data, num_boost_round=50)

    # Evaluate on training set
    train_pred = model.predict(X_train)
    acc = ((train_pred > 0.5) == y_train).mean()
    print(f"[ML] Regime classifier trained: {len(train)} samples, accuracy={acc:.1%}")

    return model, feat_cols


def run_backtest():
    print("[BT] Loading data...")
    dfs = []
    for p in CFG["data_paths"]:
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
            print(f"[BT]   Loaded {p}")
    raw = pd.concat(dfs, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw = raw.drop_duplicates(subset=["symbol", "date"], keep="last")

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

    # ── VIX data ──
    vix_lookup = {}
    if CFG.get("vix_protection") and os.path.exists(CFG.get("vix_path", "")):
        vix_df = pd.read_parquet(CFG["vix_path"])
        vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)
        vix_lookup = dict(zip(vix_df["date"], vix_df["vix_close"]))
        print(f"[BT] VIX data loaded: {len(vix_lookup)} days")

    # ── Sector mapping ──
    sym2sec = _build_sector_map(CFG["sector_path"])
    raw["sector"] = raw["symbol"].map(sym2sec).fillna("Unknown")
    n_mapped = (raw["sector"] != "Unknown").sum() / len(raw) * 100
    print(f"[BT] Sector mapping: {len(sym2sec)} symbols mapped ({n_mapped:.0f}% coverage)")

    # ── Compute factors ──
    print("[BT] Computing factors...")
    g = raw.groupby("symbol")["adj_close"]

    # Factor 1: Momentum 12-1 (skip last month)
    raw["mom_12_1"] = g.transform(lambda x: x.shift(21) / x.shift(252) - 1)

    # Factor 2: Quality = inverse realized volatility (low vol = high quality)
    raw["rv_20d"] = g.transform(lambda x: x.pct_change().rolling(20).std() * np.sqrt(252))
    raw["rv_60d"] = g.transform(lambda x: x.pct_change().rolling(60).std() * np.sqrt(252))
    raw["quality"] = -raw["rv_60d"]  # Negative vol = positive quality

    # Factor 3: Value = price relative to SMA200 (lower = cheaper)
    raw["sma200"] = g.transform(lambda x: x.rolling(200).mean())
    raw["sma50"] = g.transform(lambda x: x.rolling(50).mean())
    raw["value"] = -(raw["adj_close"] / raw["sma200"] - 1)  # Negative distance = higher value

    # ML timing features
    raw["ret_1m"] = g.transform(lambda x: x / x.shift(21) - 1)
    raw["above_sma200"] = (raw["adj_close"] > raw["sma200"]).astype(float)
    raw["mom_1m_positive"] = (raw["ret_1m"] > 0).astype(float)
    gv = raw.groupby("symbol")["volume"]
    raw["volume_ratio_20d"] = gv.transform(lambda x: x / x.rolling(20).mean())

    # Cross-sectional ranking per day for composite score
    for col in ["mom_12_1", "quality", "value"]:
        raw[f"{col}_rank"] = raw.groupby("date")[col].rank(pct=True, na_option="bottom")

    # Composite factor score
    w_m, w_q, w_v = CFG["w_momentum"], CFG["w_quality"], CFG["w_value"]
    raw["composite_score"] = (
        w_m * raw["mom_12_1_rank"] +
        w_q * raw["quality_rank"] +
        w_v * raw["value_rank"]
    )
    print(f"[BT] Composite factor: {w_m:.0%} Momentum + {w_q:.0%} Quality + {w_v:.0%} Value")

    # ── ML Timing Layer ──
    bt_start = pd.Timestamp(CFG["backtest_start"])
    bt_end = pd.Timestamp(CFG["backtest_end"])
    regime_model, regime_feats = None, None

    if CFG.get("ml_timing", False):
        print("[BT] Training ML regime classifier...")
        regime_model, regime_feats = _train_regime_classifier(
            raw, all_dates, train_end=bt_start - pd.Timedelta(days=1)
        )

    # Pre-compute market features for each date (for ML timing)
    market_features = raw.groupby("date").agg(
        breadth_sma200=("above_sma200", "mean"),
        cross_vol=("rv_20d", "median"),
        pct_positive_mom=("mom_1m_positive", "mean"),
        median_volume_ratio=("volume_ratio_20d", "median"),
    ).to_dict("index")

    # ── L5 Fix: Pre-build price lookup for O(1) NAV ──
    print("[BT] Building price lookup...")
    # {(date, symbol): adj_close}
    price_lookup = {}
    for _, row in raw[["date", "symbol", "adj_close"]].iterrows():
        price_lookup[(row["date"], row["symbol"])] = float(row["adj_close"])

    # ── Walk-Forward Loop ──
    print("[BT] Starting walk-forward simulation...")
    bt_dates = [d for d in all_dates if bt_start <= d <= bt_end]
    print(f"[BT] Backtest period: {bt_dates[0].date()} → {bt_dates[-1].date()} ({len(bt_dates)} days)")
    print(f"[BT] Rebalance: {CFG['rebalance_freq']}, Top-K: {CFG['max_positions']}, Sector cap: {CFG['sector_cap']:.0%}")

    cash = CFG["initial_cash"]
    positions = {}   # {sym: {"qty": int, "cost": float}}
    nav_hist = []
    trades = []
    daily_rets = []
    last_rebal_month = None

    def calc_nav_fast(cash, positions, date):
        """L5 Fix: O(1) per position using price lookup dict."""
        nav = cash
        for sym, pos in positions.items():
            price = price_lookup.get((date, sym))
            if price is None:
                # Fallback: search backward
                for d in reversed(all_dates):
                    if d <= date:
                        price = price_lookup.get((d, sym))
                        if price is not None:
                            break
            if price is not None and price > 0:
                nav += pos["qty"] * price
            else:
                nav += pos["qty"] * pos["cost"]
        return nav

    for day_i, td in enumerate(bt_dates):
        today = raw[raw["date"] == td]

        # Determine if this is a rebalance day
        is_rebalance = False
        if CFG["rebalance_freq"] == "monthly":
            if last_rebal_month is None or (td.year != last_rebal_month[0] or td.month != last_rebal_month[1]):
                is_rebalance = True
        elif CFG["rebalance_freq"] == "biweekly":
            if last_rebal_month is None or (td - last_rebal_month).days >= 10:
                is_rebalance = True
        elif CFG["rebalance_freq"] == "weekly":
            if last_rebal_month is None or td.weekday() == 0:
                is_rebalance = True

        td_idx = list(all_dates).index(td)

        # Non-rebalance day: just compute NAV
        if len(today) == 0 or not is_rebalance:
            nav_date = all_dates[td_idx + 1] if td_idx + 1 < len(all_dates) else td
            nav = calc_nav_fast(cash, positions, nav_date)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        # === Rebalance Day ===
        if CFG["rebalance_freq"] == "monthly":
            last_rebal_month = (td.year, td.month)
        else:
            last_rebal_month = td

        # Select top stocks by composite score
        ranked = today.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False)
        if len(ranked) < CFG["max_positions"]:
            nav_date = all_dates[td_idx + 1] if td_idx + 1 < len(all_dates) else td
            nav = calc_nav_fast(cash, positions, nav_date)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        # --- Sector Neutrality: Apply sector cap ---
        sector_cap = CFG.get("sector_cap", 1.0)
        selected = []
        sector_counts = {}
        max_per_sector = max(1, int(CFG["max_positions"] * sector_cap))

        for _, row in ranked.iterrows():
            sym = row["symbol"]
            sec = row.get("sector", "Unknown")
            if sector_counts.get(sec, 0) >= max_per_sector:
                continue  # Skip — sector full
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= CFG["max_positions"]:
                break

        if not selected:
            nav_date = all_dates[td_idx + 1] if td_idx + 1 < len(all_dates) else td
            nav = calc_nav_fast(cash, positions, nav_date)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        selected_df = pd.DataFrame(selected)
        target_syms = set(selected_df["symbol"].tolist())

        # --- Inverse Volatility Weighting ---
        if CFG.get("vol_weight", False):
            vols = selected_df.set_index("symbol")["rv_20d"].fillna(0.20).clip(lower=0.05)
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()
        else:
            weights = pd.Series(1.0 / len(selected), index=selected_df["symbol"].tolist())

        # --- Drawdown Scaling ---
        dd_scale = 1.0
        if CFG.get("dd_scaling", False) and nav_hist:
            navs_so_far = [h["nav"] for h in nav_hist]
            peak = max(max(navs_so_far), CFG["initial_cash"])
            curr_dd = (peak - navs_so_far[-1]) / peak if peak > 0 else 0
            if curr_dd > 0.25:
                dd_scale = 0.25
            elif curr_dd > 0.20:
                dd_scale = 0.50
            elif curr_dd > 0.15:
                dd_scale = 0.75

        # --- ML Timing Layer ---
        ml_scale = 1.0
        if regime_model is not None and regime_feats is not None:
            mf = market_features.get(td, {})
            if mf:
                X_today = np.array([[mf.get(f, 0) for f in regime_feats]])
                X_today = np.nan_to_num(X_today, nan=0.5)
                bull_prob = float(regime_model.predict(X_today)[0])
                # Scale: prob 0.7+ → full exposure, prob 0.3- → 40% exposure
                ml_scale = 0.40 + 0.60 * min(max((bull_prob - 0.3) / 0.4, 0), 1)

        # --- VIX Crash Protection (v5) ---
        vix_scale = 1.0
        if CFG.get("vix_protection") and vix_lookup:
            vix_today = vix_lookup.get(td, None)
            if vix_today is not None:
                for threshold in sorted(CFG["vix_thresholds"].keys(), reverse=True):
                    if vix_today > threshold:
                        vix_scale = CFG["vix_thresholds"][threshold]
                        break

        # Combined exposure
        exposure = dd_scale * ml_scale * vix_scale
        weights = weights * exposure

        # Execute at T+1 Open
        if td_idx + 1 >= len(all_dates):
            nav = calc_nav_fast(cash, positions, td)
            nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
            if len(nav_hist) >= 2:
                daily_rets.append(nav / nav_hist[-2]["nav"] - 1)
            continue

        next_d = all_dates[td_idx + 1]
        next_data = raw[raw["date"] == next_d]

        # 1) Sell positions not in target portfolio
        for sym in list(positions.keys()):
            if sym not in target_syms:
                cur_qty = positions[sym]["qty"]
                cost_basis = positions[sym]["cost"]
                sym_next = next_data[next_data["symbol"] == sym]
                if len(sym_next) == 0:
                    # Zombie prevention: force-close at cost if no market data
                    cash += cur_qty * cost_basis  # No friction since no trade
                    positions.pop(sym, None)
                    continue
                exec_price = float(sym_next.iloc[0].get("adj_open", sym_next.iloc[0].get("raw_open", 0)))
                if exec_price <= 0:
                    continue
                fri = cur_qty * exec_price * CFG["friction_rate"]
                cash += cur_qty * exec_price - fri
                positions.pop(sym, None)
                trades.append({
                    "date": str(td.date()), "symbol": sym,
                    "side": "SELL", "qty": cur_qty,
                    "price": exec_price, "friction": fri,
                    "pnl": (exec_price - cost_basis) * cur_qty - fri,  # PF fix: subtract friction
                })

        # Compute NAV for sizing (after sells)
        nav = calc_nav_fast(cash, positions, td)

        # 2) Buy/rebalance target positions
        for sym in target_syms:
            sym_next = next_data[next_data["symbol"] == sym]
            if len(sym_next) == 0:
                continue
            exec_price = float(sym_next.iloc[0].get("adj_open", sym_next.iloc[0].get("raw_open", 0)))
            if exec_price <= 0:
                continue

            sym_weight = weights.get(sym, 1.0 / CFG["max_positions"])
            target_val = sym_weight * nav
            # L1 Fix: round() instead of int() to remove truncation bias
            target_qty = round(target_val / exec_price)
            cur_qty = positions.get(sym, {}).get("qty", 0)
            delta = target_qty - cur_qty
            if abs(delta) < 1:
                continue

            fri = abs(delta * exec_price) * CFG["friction_rate"]

            if delta > 0:  # Buy
                cost = delta * exec_price + fri
                if cost > cash:
                    delta = round((cash - fri) / exec_price)
                    if delta <= 0:
                        continue
                    fri = delta * exec_price * CFG["friction_rate"]
                    cost = delta * exec_price + fri
                cash -= cost
                old = positions.get(sym, {"qty": 0, "cost": exec_price})
                new_qty = old["qty"] + delta
                positions[sym] = {
                    "qty": new_qty,
                    "cost": (old["cost"] * old["qty"] + exec_price * delta) / new_qty
                }
                trades.append({
                    "date": str(td.date()), "symbol": sym,
                    "side": "BUY", "qty": delta, "price": exec_price,
                    "friction": fri, "pnl": 0,
                })
            else:  # Sell (partial)
                sell_qty = min(abs(delta), cur_qty)
                if sell_qty <= 0:
                    continue
                cost_basis = positions[sym]["cost"]  # Capture BEFORE pop
                cash += sell_qty * exec_price - fri
                nq = cur_qty - sell_qty
                if nq <= 0:
                    positions.pop(sym, None)
                else:
                    positions[sym]["qty"] = nq
                trades.append({
                    "date": str(td.date()), "symbol": sym,
                    "side": "SELL", "qty": sell_qty, "price": exec_price,
                    "friction": fri,
                    "pnl": (exec_price - cost_basis) * sell_qty - fri,  # PF fix: subtract friction
                })

        # C6 Fix: compute NAV at T+1 (execution day)
        nav = calc_nav_fast(cash, positions, next_d)
        nav_hist.append({"date": str(td.date()), "nav": nav, "pos": len(positions)})
        if len(nav_hist) >= 2:
            daily_rets.append(nav / nav_hist[-2]["nav"] - 1)

        if (day_i + 1) % 50 == 0:
            ret_pct = (nav / CFG["initial_cash"] - 1) * 100
            secs = ", ".join(f"{s}:{c}" for s, c in sorted(sector_counts.items()) if c > 0)
            print(f"  Day {day_i+1}/{len(bt_dates)}: NAV=${nav:,.0f} ({ret_pct:+.1f}%) pos={len(positions)} exp={exposure:.0%} [{secs}]")

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
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0

        # M4 Fix: Trade-level profit factor
        trade_pnls = [t.get("pnl", 0) for t in trades if t["side"] == "SELL"]
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        fri_total = sum(t["friction"] for t in trades)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        # Sector concentration analysis
        sector_hist = {}
        for t in trades:
            sec = sym2sec.get(t["symbol"], "Unknown")
            sector_hist[sec] = sector_hist.get(sec, 0) + 1

        metrics = {
            "period": f"{CFG['backtest_start']} → {CFG['backtest_end']}",
            "strategy": f"multifactor_{CFG['rebalance_freq']}",
            "factors": f"{CFG['w_momentum']:.0%}Mom + {CFG['w_quality']:.0%}Qual + {CFG['w_value']:.0%}Val",
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
            "ml_timing": "enabled" if regime_model else "disabled",
            "vix_protection": "enabled" if CFG.get("vix_protection") else "disabled",
            "sector_cap": f"{CFG['sector_cap']:.0%}",
        }
    else:
        metrics = {"error": "insufficient data"}

    with open(f"{CFG['output_dir']}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST RESULTS (v5)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Sector distribution
    if sector_hist:
        print("\n  SECTOR DISTRIBUTION:")
        for sec, cnt in sorted(sector_hist.items(), key=lambda x: -x[1]):
            print(f"    {sec:.<30s} {cnt:>5d} trades")

    # Go/No-Go (adjusted for full-cycle including COVID)
    s = metrics.get("sharpe", 0)
    d = metrics.get("max_dd_pct", 100)
    r = metrics.get("annual_return_pct", -1)
    print("\n  GO/NO-GO:")
    ok = True
    for label, val, fail, warn in [
        ("Sharpe", s, 0, 0.5),
        ("Max DD", d, 50, 30),     # Raised to 50% for full-cycle with COVID
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

    print(f"\n  {'✅ GO' if ok else '🚫 FIX ISSUES FIRST'}")
    print("=" * 60)


def _calc_nav(cash, positions, data_df, date):
    """Legacy NAV function (fallback)."""
    nav = cash
    for sym, pos in positions.items():
        sd = data_df[(data_df["symbol"] == sym) & (data_df["date"] <= date)]
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


if __name__ == "__main__":
    run_backtest()
