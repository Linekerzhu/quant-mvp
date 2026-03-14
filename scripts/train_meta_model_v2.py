#!/usr/bin/env python3
"""
⚠️  DEPRECATED — LTR meta model is no longer used by v4 multi-factor strategy.
Old model artifacts moved to models/_legacy/. DO NOT RUN unless reverting to LTR.

Phase H: Train Real Meta-Model v2

End-to-end pipeline:
1. Load 3-year historical data (2023-2025)
2. Build features (20+ features)
3. Generate Triple Barrier labels
4. Generate SMA + Momentum base signals
5. Train LightGBM meta-model via CPCV
6. Save model + metadata
7. Re-run Phase G backtest with real model
8. Compare results

Usage:
    PYTHONPATH=/root/quant-mvp python3 -u scripts/train_meta_model_v2.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import FeatureEngineer
from src.labels.triple_barrier import TripleBarrierLabeler
from src.labels.sample_weights import SampleWeightCalculator
from src.signals.base_models import BaseModelSMA, BaseModelMomentum
from src.models.meta_trainer import MetaTrainer
from src.models.model_io import ModelBundleManager

OUTPUT_DIR = "data/backtest/training"

# Features to use (all available from build_features.py)
FEATURE_COLS = [
    # Momentum (multi-horizon)
    "returns_5d", "returns_10d", "returns_20d", "returns_60d",
    # Volatility
    "rv_5d", "rv_20d", "rv_60d", "atr_20",
    # Mean reversion
    "rsi_14", "price_vs_sma20_zscore", "price_vs_sma60_zscore",
    # Trend
    "macd_line_pct", "macd_histogram_pct", "adx_14",
    # Volume
    "relative_volume_20d",
    # Market-wide
    "market_breadth", "vix_change_5d",
    # Regime (numeric scores, not string labels)
    "regime_trend_score", "regime_vol_score",
    # Price-volume divergence
    "pv_divergence_bull", "pv_divergence_bear",
    # Note: fracdiff is auto-added by MetaTrainer
]


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load data ──
    print("[H] Step 1: Loading 3-year historical data...")
    raw = pd.read_parquet("data/backtest/sp500_2023_2026.parquet")
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

    # Standardize columns
    for src, dst in [("adj close", "adj_close")]:
        if src in raw.columns:
            raw = raw.rename(columns={src: dst})

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
    print(f"[H]   {len(raw)} rows, {raw['symbol'].nunique()} symbols")
    dates = sorted(raw["date"].unique())
    print(f"[H]   Date range: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")

    # ── Step 2: Build features ──
    print("[H] Step 2: Building features (one-time)...")
    feat_eng = FeatureEngineer()
    features = feat_eng.build_features(raw)
    print(f"[H]   Features built: {len(features)} rows")
    
    # Check available FEATURE_COLS
    available = [f for f in FEATURE_COLS if f in features.columns]
    missing = [f for f in FEATURE_COLS if f not in features.columns]
    print(f"[H]   Available features: {len(available)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"[H]   Missing features (will skip): {missing}")
    FEATURE_COLS_FINAL = available

    # ── Step 3: Generate Triple Barrier labels ──
    print("[H] Step 3: Generating Triple Barrier labels...")
    labeler = TripleBarrierLabeler()

    # Use training window: 2023-07 to 2025-06 (2 years, leave rest for OOS)
    train_start = pd.Timestamp("2023-07-01")
    train_end = pd.Timestamp("2025-06-30")
    train_data = features[(features["date"] >= train_start) & (features["date"] <= train_end)].copy()
    print(f"[H]   Training window: {train_start.date()} → {train_end.date()}")
    
    symbols = train_data['symbol'].unique()
    n_symbols = len(symbols)
    print(f"[H]   Training data: {len(train_data)} rows, {n_symbols} symbols")

    # Ensure columns expected by TripleBarrierLabeler exist
    if "can_trade" not in train_data.columns:
        train_data["can_trade"] = True
    if "is_suspended" not in train_data.columns:
        train_data["is_suspended"] = False
    if "features_valid" not in train_data.columns:
        train_data["features_valid"] = True

    train_data = train_data.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    t_label = time.time()
    
    # Process symbol by symbol so we can see progress
    labeled_chunks = []
    logger_msg = f"[H]   >> This takes time, processing {n_symbols} symbols <<\n"
    print(logger_msg)
    
    import sys
    for i, sym in enumerate(symbols):
        mask = train_data['symbol'] == sym
        chunk = train_data[mask]
        labeled_chunk = labeler.label_events(chunk)
        labeled_chunks.append(labeled_chunk)
        
        # Print progress every 20 symbols without newline
        if (i+1) % 20 == 0 or (i+1) == n_symbols:
            sys.stdout.write(f"\r[H]     Labeling progress: {i+1}/{n_symbols} symbols ({(i+1)/n_symbols*100:.1f}%)")
            sys.stdout.flush()
            
    print(f"\n[H]   Labeling completed in {time.time() - t_label:.1f} seconds")
    labeled = pd.concat(labeled_chunks, ignore_index=True)
    
    # Check label distribution
    if "label" in labeled.columns:
        dist = labeled["label"].value_counts()
        print(f"[H]   Label distribution:")
        for k, v in sorted(dist.items()):
            pct = v / len(labeled) * 100
            label_name = {1: "Profit", -1: "Loss", 0: "Time"}.get(k, str(k))
            print(f"[H]     {label_name} ({k}): {v} ({pct:.1f}%)")

    valid_events = labeled[labeled.get("event_valid", True) == True]
    print(f"[H]   Valid events: {len(valid_events)}")

    # ── Step 4: Calculate sample weights ──
    print("[H] Step 4: Calculating sample weights...")
    weight_calc = SampleWeightCalculator()
    labeled = weight_calc.calculate_weights(labeled)
    if "sample_weight" in labeled.columns:
        sw = labeled["sample_weight"]
        print(f"[H]   Weight stats: mean={sw.mean():.3f}, std={sw.std():.3f}, "
              f"min={sw.min():.3f}, max={sw.max():.3f}")

    # ── Step 5: Train meta-model ──
    print("[H] Step 5: Training Meta-Model v2 via CPCV...")
    print(f"[H]   Using {len(FEATURE_COLS_FINAL)} features: {FEATURE_COLS_FINAL}")
    
    # Use SMA as primary base model (consistent with daily_job)
    base_model = BaseModelSMA(fast_window=20, slow_window=60)
    
    trainer = MetaTrainer()
    try:
        results = trainer.train(
            df=labeled,
            base_model=base_model,
            features=FEATURE_COLS_FINAL,
            price_col="adj_close"
        )
        
        print("\n[H] ══════════════════════════════════")
        print("[H]   META-MODEL v2 TRAINING RESULTS")
        print("[H] ══════════════════════════════════")
        print(f"[H]   Training samples: {results.get('n_training_samples', 'N/A')}")
        print(f"[H]   Mean AUC:  {results.get('mean_auc', 0):.4f} ± {results.get('std_auc', 0):.4f}")
        print(f"[H]   Min AUC:   {results.get('min_auc', 0):.4f}")
        print(f"[H]   Max AUC:   {results.get('max_auc', 0):.4f}")
        print(f"[H]   Mean Acc:  {results.get('mean_accuracy', 0):.4f}")
        print(f"[H]   PBO:       {results.get('pbo', 'N/A')}")
        print(f"[H]   Features:  {results.get('feature_list', [])}")
        print("[H] ══════════════════════════════════")

        # Save training report
        report = {
            "training_window": f"{train_start.date()} → {train_end.date()}",
            "n_training_samples": results.get("n_training_samples"),
            "n_features": len(results.get("feature_list", [])),
            "feature_list": results.get("feature_list", []),
            "mean_auc": results.get("mean_auc"),
            "std_auc": results.get("std_auc"),
            "min_auc": results.get("min_auc"),
            "max_auc": results.get("max_auc"),
            "mean_accuracy": results.get("mean_accuracy"),
            "pbo": results.get("pbo"),
            "pbo_status": results.get("pbo_status"),
            "n_paths": results.get("n_paths"),
            "training_time_sec": round(time.time() - t0, 1),
        }
        
        with open(f"{OUTPUT_DIR}/training_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[H] Training report saved to {OUTPUT_DIR}/training_report.json")
        print(f"[H] Model saved to models/meta_model_v1.txt (overwritten)")
        print(f"[H] Total time: {time.time() - t0:.0f}s")
        
        # Go/No-Go
        auc = results.get("mean_auc", 0)
        pbo = results.get("pbo", 1.0)
        print("\n[H] GO/NO-GO:")
        ok = True
        if auc < 0.55:
            print(f"[H] 🚫 Mean AUC = {auc:.4f} < 0.55")
            ok = False
        else:
            print(f"[H] ✅ Mean AUC = {auc:.4f}")
        if isinstance(pbo, (int, float)) and pbo > 0.40:
            print(f"[H] 🚫 PBO = {pbo:.2f} > 0.40")
            ok = False
        elif isinstance(pbo, (int, float)):
            print(f"[H] ✅ PBO = {pbo:.2f}")
        print(f"\n[H] {'✅ MODEL v2 READY' if ok else '🚫 MODEL NEEDS WORK'}")
        
    except Exception as e:
        print(f"\n[H] ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error report
        with open(f"{OUTPUT_DIR}/training_error.json", "w") as f:
            json.dump({
                "error": str(e),
                "training_time_sec": round(time.time() - t0, 1),
            }, f, indent=2)


if __name__ == "__main__":
    main()
