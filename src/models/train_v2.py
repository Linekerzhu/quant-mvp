"""
Meta Model V2.1 Trainer — Optimized
====================================
Key improvements over V2:
 1. Interaction features (momentum × volatility, trend × mean-reversion)
 2. Pruned noise features (only keep top contributors + interactions) 
 3. Probability calibration via Isotonic Regression
 4. Side-aware meta-labeling (label = 1 if base model direction was correct)
 5. Two base model variants: SMA-based and Momentum-based labels
"""
import os, json, sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.build_features import FeatureEngineer

# ── Configuration ──
DATA_PATH = "data/backtest/sp500_2023_2026.parquet"
MODEL_OUT = "models/meta_model_v2.txt"
META_OUT  = "models/meta_model_v2_metadata.json"
CALIB_OUT = "models/meta_model_v2_calibrator.json"

TRAIN_END = "2025-06-30"
VAL_START = "2025-07-01"

# Core features from V2.1
CORE_FEATURES = [
    'returns_60d', 'vix_change_5d', 'market_breadth', 'rv_20d',
    'price_vs_sma60_zscore', 'macd_line_pct', 'fracdiff', 'returns_10d',
    'returns_20d', 'adx_14', 'rsi_14', 'relative_volume_20d',
    
    # New V2.2 Base Features
    'vol_regime', 'price_range_pos', 'atr_pct', 
    'sma_cross_strength', 'vol_trend_accel'
]

# V2.1 Interactions + V2.2 New Interactions
INTERACTION_FEATURES = [
    # Proven V2.1 features
    'mom60_x_rv20', 'mom60_x_breadth', 'zscore60_x_adx', 
    'vix_x_rv20', 'rsi_x_zscore60', 'returns_10d_x_returns_60d', 
    'volume_x_macd',
    
    # New V2.2 Interaction features
    'trend_conviction',       # Trend strength × Price momentum
    'volatility_shock',       # VIX spike × Stock vol regime
    'reversal_setup',         # Oversold stochastic × Lower band touch
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "min_data_in_leaf": 150,  # Increased to prevent overfitting on more features
    "feature_fraction": 0.5,  # Subsample features more aggressively
    "bagging_fraction": 0.7,
    "bagging_freq": 2,
    "learning_rate": 0.005,
    "lambda_l1": 0.8,         # More L1 to automatically prune weak new features
    "lambda_l2": 1.2,
    "verbose": -1,
}
N_ESTIMATORS = 2000
EARLY_STOPPING = 100

FORWARD_DAYS = 5
PROFIT_THRESHOLD = 0.005  # Require at least 0.5% return (cover friction)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V2.1 + V2.2 non-linear interaction features."""
    df = df.copy()
    
    # V2.1 interactions
    df['mom60_x_rv20'] = df['returns_60d'] / (df['rv_20d'] + 1e-8)
    df['mom60_x_breadth'] = df['returns_60d'] * df['market_breadth']
    df['zscore60_x_adx'] = df['price_vs_sma60_zscore'] * (50 - df['adx_14'].clip(0, 50)) / 50
    df['vix_x_rv20'] = df['vix_change_5d'] * df['rv_20d']
    df['rsi_x_zscore60'] = (df['rsi_14'] - 50) / 50 * df['price_vs_sma60_zscore']
    df['returns_10d_x_returns_60d'] = np.sign(df['returns_10d']) * np.sign(df['returns_60d'])
    df['volume_x_macd'] = df['relative_volume_20d'] * df['macd_line_pct']
    
    # V2.2 interactions
    df['trend_conviction'] = df['sma_cross_strength'] * df['adx_14']
    df['volatility_shock'] = np.where(df['vix_change_5d'] > 0.1, df['vix_change_5d'] * df['vol_regime'], 0)
    df['reversal_setup'] = ((0.5 - df['price_range_pos']) * 2) * df['zscore60_x_adx']
    
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create meta-labels: was the SMA+Mom signal's direction correct AND profitable enough?"""
    print("[V2.2] Creating strict meta-labels (require >0.5% return)...")
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")["adj_close"]
    df["fwd_ret"] = g.transform(lambda x: x.shift(-FORWARD_DAYS) / x - 1)
    
    # SMA signal
    sma_fast = g.transform(lambda x: x.rolling(20).mean())
    sma_slow = g.transform(lambda x: x.rolling(60).mean())
    df["sma_side"] = np.where(sma_fast > sma_slow, 1, -1)
    df.loc[sma_slow.isna(), "sma_side"] = 0
    
    # Momentum signal
    mom_ret = g.transform(lambda x: np.log(x / x.shift(20)))
    df["mom_side"] = np.where(mom_ret > 0, 1, -1)
    df.loc[mom_ret.isna(), "mom_side"] = 0
    
    # Only keep unanimous signals to define cleanly separated base decisions
    df["combined_side"] = np.where((df["sma_side"] != 0) & (df["sma_side"] == df["mom_side"]), df["sma_side"], 0)
    
    # NEW V2.2: Label = 1 IF AND ONLY IF (direction is correct AND profit > absolute threshold)
    # This teaches the model to ignore marginal trades that get eaten by slippage
    df["meta_label"] = (df["combined_side"] * df["fwd_ret"] > PROFIT_THRESHOLD).astype(int)
    
    df = df.dropna(subset=["fwd_ret"])
    df = df[df["combined_side"] != 0].copy()
    
    return df



def calibrate_probabilities(y_val, raw_probs):
    """Fit isotonic regression to spread probability distribution."""
    print("[V2.1] Calibrating probabilities...")
    ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    ir.fit(raw_probs, y_val)
    calibrated = ir.predict(raw_probs)
    
    print(f"  Raw probs:  min={raw_probs.min():.4f}, max={raw_probs.max():.4f}, "
          f"std={raw_probs.std():.4f}")
    print(f"  Calibrated: min={calibrated.min():.4f}, max={calibrated.max():.4f}, "
          f"std={calibrated.std():.4f}")
    
    return ir, calibrated


def train_v2_1():
    print("=" * 60)
    print("  META MODEL V2.1 TRAINING (Optimized)")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[V2.1] Step 1: Loading data...")
    raw = pd.read_parquet(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    for src, dst in [("adj close", "adj_close")]:
        if src in raw.columns:
            raw = raw.rename(columns={src: dst})
    if "adj_close" in raw.columns and "close" in raw.columns:
        ratio = raw["adj_close"] / raw["close"].replace(0, np.nan)
        ratio = ratio.fillna(1.0)
        for c in ["open", "high", "low", "close"]:
            raw[f"raw_{c}"] = raw[c]
            raw[f"adj_{c}"] = raw[c] * ratio
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"[V2.1] Loaded: {len(raw)} rows, {raw['symbol'].nunique()} symbols")
    
    # Step 2: Build features
    print("\n[V2.2] Step 2: Building features...")
    feat_eng = FeatureEngineer()
    df = feat_eng.build_features(raw)
    
    # Step 2.5: Add advanced features
    try:
        from src.features.extra_features import add_advanced_features
        df = add_advanced_features(df)
    except ImportError:
        print("[V2.2] Warning: Could not import add_advanced_features")
    
    # Step 3: Add fracdiff
    if "fracdiff" not in df.columns or df["fracdiff"].isna().all():
        print("[V2.1] Step 3: Computing fracdiff...")
        try:
            from src.features.fracdiff import find_min_d_stationary, fracdiff_fixed_window
            df["fracdiff"] = 0.0
            for sym in df["symbol"].unique():
                mask = df["symbol"] == sym
                prices = np.log(df.loc[mask, "adj_close"])
                if len(prices) < 100:
                    continue
                try:
                    d = find_min_d_stationary(prices, threshold=0.05)
                    fd = fracdiff_fixed_window(prices, d, 100)
                    df.loc[mask, "fracdiff"] = fd.values
                except:
                    pass
        except ImportError:
            df["fracdiff"] = 0.0
    
    # Step 4: Add interaction features
    print("[V2.1] Step 4: Adding interaction features...")
    df = add_interaction_features(df)
    
    # Step 5: Create labels
    df = create_labels(df)
    
    # Final feature list
    FEATURE_LIST = CORE_FEATURES + INTERACTION_FEATURES
    
    # Fill missing
    for f in FEATURE_LIST:
        if f not in df.columns:
            print(f"  Warning: Missing feature '{f}', filling with 0")
            df[f] = 0.0
    df[FEATURE_LIST] = df[FEATURE_LIST].replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"[V2.1] After labeling: {len(df)} samples, {len(FEATURE_LIST)} features")
    print(f"[V2.1] Label balance: {df['meta_label'].value_counts().to_dict()}")
    
    # Step 6: Chronological split
    print(f"\n[V2.1] Step 6: Splitting (train <= {TRAIN_END}, val >= {VAL_START})...")
    train_df = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    val_df = df[df["date"] >= pd.Timestamp(VAL_START)]
    
    print(f"  Train: {len(train_df)} ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df)} ({val_df['date'].min().date()} → {val_df['date'].max().date()})")
    
    X_train = train_df[FEATURE_LIST].values
    y_train = train_df["meta_label"].values
    X_val = val_df[FEATURE_LIST].values
    y_val = val_df["meta_label"].values
    
    # Step 7: Train
    print("\n[V2.1] Step 7: Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_LIST)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_LIST, reference=dtrain)
    
    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(100)],
    )
    
    # Step 8: Evaluate raw model
    print("\n[V2.1] Step 8: Evaluation...")
    raw_train_preds = model.predict(X_train)
    raw_val_preds = model.predict(X_val)
    
    train_auc = roc_auc_score(y_train, raw_train_preds)
    val_auc = roc_auc_score(y_val, raw_val_preds)
    
    print(f"\n  Raw Model Results:")
    print(f"  Train AUC:      {train_auc:.4f}")
    print(f"  Val AUC:        {val_auc:.4f}")
    print(f"  Overfitting gap: {train_auc - val_auc:.4f}")
    print(f"  Num trees:      {model.num_trees()}")
    print(f"  Best iteration: {model.best_iteration}")
    
    # Step 9: Calibrate probabilities
    ir, cal_val_preds = calibrate_probabilities(y_val, raw_val_preds)
    cal_train_preds = ir.predict(raw_train_preds)
    
    cal_val_auc = roc_auc_score(y_val, cal_val_preds)
    cal_train_acc = accuracy_score(y_train, (cal_train_preds > 0.5).astype(int))
    cal_val_acc = accuracy_score(y_val, (cal_val_preds > 0.5).astype(int))
    
    print(f"\n  Calibrated Results:")
    print(f"  Val AUC (calibrated): {cal_val_auc:.4f}")
    print(f"  Train Accuracy: {cal_train_acc:.4f}")
    print(f"  Val Accuracy:   {cal_val_acc:.4f}")
    
    # Threshold analysis
    print(f"\n  Calibrated threshold analysis:")
    for t in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.60]:
        pct = (cal_val_preds >= t).sum() / len(cal_val_preds) * 100
        if (cal_val_preds >= t).sum() > 0:
            prec = y_val[cal_val_preds >= t].mean()
        else:
            prec = 0
        print(f"    prob >= {t:.2f}: {(cal_val_preds >= t).sum():6d} ({pct:5.1f}%) precision={prec:.3f}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_LIST, importance), key=lambda x: -x[1])
    print(f"\n  Top 15 Features:")
    for i, (f, imp) in enumerate(feat_imp[:15]):
        print(f"    {i+1:2d}. {f:35s} {imp:8.1f}")
    
    # Step 10: Save
    print(f"\n[V2.1] Step 10: Saving...")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save_model(MODEL_OUT)
    
    # Save calibrator (isotonic regression boundaries)
    calib_data = {
        "X_thresholds": ir.X_thresholds_.tolist() if hasattr(ir, 'X_thresholds_') else [],
        "y_thresholds": ir.y_thresholds_.tolist() if hasattr(ir, 'y_thresholds_') else [],
    }
    with open(CALIB_OUT, "w") as f:
        json.dump(calib_data, f)
    
    metadata = {
        "version": "v2.1",
        "feature_list": FEATURE_LIST,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "calibrated_val_auc": float(cal_val_auc),
        "train_accuracy": float(cal_train_acc),
        "val_accuracy": float(cal_val_acc),
        "num_trees": model.num_trees(),
        "best_iteration": model.best_iteration,
        "train_period": f"{train_df['date'].min().date()} → {train_df['date'].max().date()}",
        "val_period": f"{val_df['date'].min().date()} → {val_df['date'].max().date()}",
        "feature_importance": {f: float(imp) for f, imp in feat_imp},
        "lgb_params": LGB_PARAMS,
        "has_calibrator": True,
        "calibrator_path": CALIB_OUT,
        "interaction_features": INTERACTION_FEATURES,
        "improvements": [
            "interaction_features",
            "probability_calibration",
            "combined_side_labels",
            "pruned_noise_features",
            "slower_learning_rate",
            "l2_regularization",
        ],
    }
    
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  QUALITY GATE")
    print(f"{'='*60}")
    if val_auc > 0.55:
        print(f"  ✅ Val AUC {val_auc:.4f} > 0.55 → STRONG PASS")
    elif val_auc > 0.54:
        print(f"  ✅ Val AUC {val_auc:.4f} > 0.54 → PASS")
    elif val_auc > 0.52:
        print(f"  ⚠️  Val AUC {val_auc:.4f} > 0.52 → marginal")
    else:
        print(f"  ❌ Val AUC {val_auc:.4f} ≤ 0.52 → no edge")
    print(f"{'='*60}")
    
    return metadata


if __name__ == "__main__":
    train_v2_1()
