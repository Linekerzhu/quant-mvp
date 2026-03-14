#!/usr/bin/env python3
# ⚠️  DEPRECATED — LTR meta model no longer used by v4 strategy. See models/_legacy/
# Phase H: Direct training (bypass complex CPCV for initial v2 model)
# We train on 2023-01 to 2025-01, validate on 2025-01 to 2026-03
import pandas as pd, numpy as np, lightgbm as lgb, json, time, sys, warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import FeatureEngineer
from src.labels.triple_barrier import TripleBarrierLabeler
from src.signals.base_models import BaseModelSMA
from src.models.model_io import ModelBundleManager
from src.features.fracdiff import find_min_d_stationary, fracdiff_fixed_window

FEATURES = [
    "returns_5d","returns_10d","returns_20d","returns_60d",
    "rv_5d","rv_20d","rv_60d","atr_20","rsi_14",
    "price_vs_sma20_zscore","price_vs_sma60_zscore",
    "macd_line_pct","macd_histogram_pct","adx_14",
    "relative_volume_20d","market_breadth","vix_change_5d",
    "regime_trend_score","regime_vol_score",
    "pv_divergence_bull","pv_divergence_bear",
]

t0 = time.time()
os.makedirs("data/backtest/training", exist_ok=True)
print("[H] Loading data...")
raw = pd.read_parquet("data/backtest/sp500_2023_2026.parquet")
raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

if "adj_close" in raw.columns and "close" in raw.columns:
    ratio = raw["adj_close"] / raw["close"].replace(0, np.nan)
    ratio = ratio.fillna(1.0)
    for c in ["open", "high", "low", "close"]:
        raw["raw_" + c] = raw[c]
        raw["adj_" + c] = raw[c] * ratio

raw = raw.sort_values(["symbol","date"]).reset_index(drop=True)
print(f"[H] {len(raw)} rows, {raw['symbol'].nunique()} symbols")

print("[H] Building features...")
fe = FeatureEngineer()
features = fe.build_features(raw)

# Split into train and val periods
train_data = features[(features["date"] >= "2023-07-01") & (features["date"] <= "2025-01-31")].copy()
val_data = features[(features["date"] >= "2025-02-01") & (features["date"] <= "2026-03-06")].copy()

for df in [train_data, val_data]:
    df["can_trade"] = True
    df["is_suspended"] = False
    df["features_valid"] = True

train_data = train_data.sort_values(["symbol","date"]).reset_index(drop=True)
val_data = val_data.sort_values(["symbol","date"]).reset_index(drop=True)

print(f"[H] Train: {len(train_data)} rows, Val: {len(val_data)} rows")

def label_with_progress(data, name):
    labeler = TripleBarrierLabeler()
    symbols = data['symbol'].unique()
    n_symbols = len(symbols)
    labeled_chunks = []
    print(f"\n[H] Labeling {name} ({n_symbols} symbols)...")
    for i, sym in enumerate(symbols):
        chunk = data[data['symbol'] == sym]
        labeled_chunks.append(labeler.label_events(chunk))
        if (i+1) % 20 == 0 or (i+1) == n_symbols:
            sys.stdout.write(f"\r[H]   Progress: {i+1}/{n_symbols} ({(i+1)/n_symbols*100:.1f}%)")
            sys.stdout.flush()
    print()
    return pd.concat(labeled_chunks, ignore_index=True)

train_labeled = label_with_progress(train_data, "train")
val_labeled = label_with_progress(val_data, "validation")

# Filter: keep only events with labels
train_events = train_labeled[train_labeled["label"].notna() & (train_labeled["label"] != 0)].copy()
val_events = val_labeled[val_labeled["label"].notna() & (val_labeled["label"] != 0)].copy()

print(f"[H] Train events (no neutral): {len(train_events)}")
print(f"[H] Val events (no neutral): {len(val_events)}")

# Generate base signals
print("[H] Generating SMA signals...")
sma = BaseModelSMA()
for df_name, df in [("train", train_events), ("val", val_events)]:
    sides = []
    for sym in df["symbol"].unique():
        sym_all = features[features["symbol"] == sym].copy()
        if len(sym_all) < 61:
            continue
        sym_signals = sma.generate_signals(sym_all)
        sym_events = df[df["symbol"] == sym]
        merged = sym_events.merge(sym_signals[["date","side"]], on="date", how="left", suffixes=("","_sma"))
        sides.append(merged)
    if sides:
        result = pd.concat(sides)
        if "side_sma" in result.columns:
            df.loc[result.index, "side"] = result["side_sma"].values
        elif "side" not in df.columns:
            df["side"] = result["side"].values

# Filter side != 0
if "side" in train_events.columns:
    train_events = train_events[train_events["side"] != 0]
if "side" in val_events.columns:
    val_events = val_events[val_events["side"] != 0]

# Create meta-labels
train_events["meta_label"] = (train_events["label"] == 1).astype(int)
val_events["meta_label"] = (val_events["label"] == 1).astype(int)

print(f"[H] After side filter - Train: {len(train_events)}, Val: {len(val_events)}")
print(f"[H] Train meta_label distribution: {train_events['meta_label'].value_counts().to_dict()}")
print(f"[H] Val meta_label distribution: {val_events['meta_label'].value_counts().to_dict()}")

# Prepare features (add fracdiff)
print("[H] Calculating fracdiff (this takes a moment)...")
per_d = {}
for df in [train_events, val_events]:
    df["fracdiff"] = 0.0
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        prices = df.loc[mask, "adj_close"].copy()
        prices = np.log(prices.clip(lower=0.01))
        if len(prices) < 100:
            continue
        try:
            d = find_min_d_stationary(prices, threshold=0.05)
            fd = fracdiff_fixed_window(prices, d, 100)
            df.loc[mask, "fracdiff"] = fd.values
            per_d[sym] = float(d)
        except:
            pass

ALL_FEATURES = FEATURES + ["fracdiff"]
avail = [f for f in ALL_FEATURES if f in train_events.columns]
print(f"[H] Features available: {len(avail)}/{len(ALL_FEATURES)}")

X_train = train_events[avail].fillna(0).values
y_train = train_events["meta_label"].values
X_val = val_events[avail].fillna(0).values
y_val = val_events["meta_label"].values

print(f"[H] X_train: {X_train.shape}, X_val: {X_val.shape}")

# Train LightGBM
ds_train = lgb.Dataset(X_train, label=y_train)
ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

params = {
    "objective": "binary", "metric": "auc",
    "max_depth": 4, "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 5,
    "learning_rate": 0.05, "lambda_l1": 1.0,
    "verbose": -1, "n_jobs": -1
}

print("[H] Training LightGBM...")
model = lgb.train(
    params, ds_train, num_boost_round=500,
    valid_sets=[ds_val], valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

print(f"\n[H] ══════════════════════════════════")
print(f"[H]   Best iteration: {model.best_iteration}")
print(f"[H]   Best Val AUC: {model.best_score['val']['auc']:.4f}")

# Feature importance
imp = model.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": avail, "gain": imp}).sort_values("gain", ascending=False)
print("\n[H]   Top 10 features:")
for _, row in imp_df.head(10).iterrows():
    print(f"[H]     {row['feature']}: {row['gain']:.1f}")
print(f"[H] ══════════════════════════════════")

# Save model
mgr = ModelBundleManager()
metadata = {"feature_list": avail, "optimal_d": per_d}

# First delete any existing model to be safe
try:
    os.remove("models/meta_model_v1.txt")
except: pass

mgr.save_bundle(model, metadata)
print(f"\n[H] Model saved to models/meta_model_v1.txt! Features: {len(avail)}")
print(f"[H] Total time: {time.time()-t0:.0f}s")
