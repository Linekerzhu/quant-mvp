#!/usr/bin/env python3
"""
⚠️  DEPRECATED — LTR meta model is no longer used by v4 multi-factor strategy.
Old model artifacts moved to models/_legacy/. DO NOT RUN unless reverting to LTR.

Online AutoML Incremental Retraining (Phase L4)

Supports incremental learning (refitting) for the cross-sectional ranking LambdaRank 
LightGBM model (LTR). 
Since labels (future returns) take N days to resolve, on day T, we can only
incrementally train on the feature snapshots from day T-N.

This allows the model to continuously adapt to market regime shifts (Concept Drift) 
without needing a full 5-year historical rebuild every week.
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Adjust text paths for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ops.event_logger import get_logger
from src.models.model_io import ModelBundleManager
from src.models.meta_trainer import MetaTrainer

logger = get_logger()

def load_data_for_date(date_str: str) -> pd.DataFrame:
    """Load daily processed feature parquet for a specific date."""
    path = f"data/processed/features_{date_str}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

def run_incremental_training(trade_date: str, fwd_days: int = 5, force_target_date: str = None):
    """
    Run the incremental learning step.
    
    Args:
        trade_date: Current trade date 'YYYY-MM-DD'
        fwd_days: Forward return horizon for labels
        force_target_date: Override the T-N calculation for testing
    """
    logger.info("automl_retrain_started", {"trade_date": trade_date})
    
    # 1. Determine the target training date (T - fwd_days)
    if force_target_date:
        train_target_date = force_target_date
    else:
        curr_date = pd.to_datetime(trade_date)
        from pandas.tseries.offsets import BDay
        train_target_date = (curr_date - BDay(fwd_days)).strftime('%Y-%m-%d')
    
    logger.info("automl_target_resolved", {"trade_date": trade_date, "train_target_date": train_target_date})
    
    # 2. Load the old features from train_target_date
    df_features_full = load_data_for_date(train_target_date)
    if df_features_full.empty:
        logger.error("automl_no_features", {"date": train_target_date})
        return False
        
    df_features = df_features_full.copy()
    if 'date' in df_features.columns:
        if pd.api.types.is_datetime64_any_dtype(df_features['date']):
            df_features = df_features[df_features['date'].dt.strftime('%Y-%m-%d') == train_target_date]
        else:
            df_features = df_features[df_features['date'] == train_target_date]
            
    if len(df_features) == 0:
        logger.error("automl_features_empty_after_filter", {"date": train_target_date})
        return False
        
    # We also need the future prices up to trade_date to compute the `fwd_return` label.
    # In a true data lake, we'd query the DB. Let's load the latest raw daily data
    # (assuming it goes up to trade_date)
    try:
        from pandas.tseries.offsets import BDay
        import glob
        next_day = (pd.to_datetime(trade_date) + BDay(1)).strftime('%Y-%m-%d')
        path = f"data/raw/daily_{next_day}.parquet"
        if not os.path.exists(path):
            # R1 FIX: Dynamic fallback to the latest available raw file
            all_raw = sorted(glob.glob("data/raw/daily_*.parquet"))
            if not all_raw:
                logger.error("automl_no_raw_files_found")
                return False
            path = all_raw[-1]
            logger.info("automl_fallback_latest_raw", {"path": path})
            
        df_raw = pd.read_parquet(path)
    except Exception as e:
        logger.error("automl_no_raw_prices", {"error": str(e)})
        return False
        
    # 3. Compute Labels
    trainer = MetaTrainer()
    
    # df_features has 'adj_close' on T-5. We need 'adj_close' on T to calculate return.
    # We fetch T's closing prices and merge.
    df_t = df_raw.copy()
    if 'date' in df_t.columns:
        if pd.api.types.is_datetime64_any_dtype(df_t['date']):
            df_t = df_t[df_t['date'].dt.strftime('%Y-%m-%d') == trade_date]
        else:
            df_t = df_t[df_t['date'] == trade_date]
    
    df_t = df_t[['symbol', 'adj_close']].rename(columns={'adj_close': 'future_close'})
    df_features = df_features.merge(df_t, on='symbol', how='inner')
    
    if len(df_features) < 100:
        logger.warn("automl_insufficient_samples", {"count": len(df_features)})
        return False
        
    # Compute fwd_return manually instead of _build_ltr_labels which uses shift
    df_features['fwd_return'] = df_features['future_close'] / df_features['adj_close'] - 1.0
    
    # Rank to get quintiles [0, 1, 2, 3, 4]
    n_bins = trainer.config.get('ltr', {}).get('label_settings', {}).get('n_quantile_bins', 5)
    df_features['ltr_label'] = pd.qcut(df_features['fwd_return'], q=n_bins, labels=False, duplicates='drop')
    df_features['ltr_label'] = df_features['ltr_label'].fillna(n_bins // 2).astype(int)
    
    # Set query group (just 1 group because it's exactly 1 day)
    df_features['query_group'] = 1
    
    # 4. Load Existing Model
    mgr = ModelBundleManager()
    model = mgr.load_model()
    metadata = mgr.load_metadata()
    
    if model is None:
        logger.error("automl_no_existing_model")
        return False
        
    feature_list = metadata.get("feature_list", [])
    if not feature_list:
        logger.error("automl_no_feature_list_in_meta")
        return False
        
    # Ensure all features exist
    for f in feature_list:
        if f not in df_features.columns:
            # Maybe it's fracdiff, let's just 0-fill for online robustness if missing
            df_features[f] = 0.0
            
    X_train = df_features[feature_list]
    y_train = df_features['ltr_label']
    
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train, group=[len(X_train)])
    
    # 5. Incremental Training (Refit)
    # W3 FIX: Use LTR-specific params from config, not binary OR5 params
    ltr_config = trainer.config.get('ltr', {})
    ltr_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': ltr_config.get('eval_at', [5, 10]),
        'max_depth': ltr_config.get('max_depth', 3),
        'num_leaves': ltr_config.get('num_leaves', 7),
        'min_data_in_leaf': ltr_config.get('min_data_in_leaf', 100),
        'feature_fraction': ltr_config.get('feature_fraction', 0.5),
        'bagging_fraction': ltr_config.get('bagging_fraction', 0.7),
        'bagging_freq': ltr_config.get('bagging_freq', 5),
        'learning_rate': ltr_config.get('learning_rate', 0.01),
        'verbose': -1,
        'seed': 42,
    }
    
    logger.info("automl_refitting_model", {"samples": len(X_train)})
    
    start_time = datetime.now()
    new_model = lgb.train(
        ltr_params,
        train_data,
        num_boost_round=10, # Just 10 trees increment
        init_model=model,
        keep_training_booster=True
    )
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("automl_refit_complete", {"duration_seconds": duration})
    
    # 6. Save Bundle
    mgr.save_bundle(new_model, metadata)
    logger.info("automl_bundle_saved")
    
    print(f"✅ AutoML Incremental Update Complete for target date {train_target_date} in {duration:.2f}s")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="Current trade date YYYY-MM-DD")
    parser.add_argument("--fwd-days", type=int, default=5, help="Forward return days")
    parser.add_argument("--target-date", type=str, default=None, help="Override T-N rule, force train on specific feature date")
    args = parser.parse_args()
    
    run_incremental_training(args.date, args.fwd_days, force_target_date=args.target_date)
