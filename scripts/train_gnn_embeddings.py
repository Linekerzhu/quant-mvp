#!/usr/bin/env python3
"""
GNN Offline Training Cron Job

Runs weekly (Sunday night) to retrain stock embeddings using latest data.
Output: data/cache/gnn_embeddings.parquet (consumed by daily_job.py)

Usage (crontab):
    0 22 * * 0 cd /root/quant-mvp && python3 scripts/train_gnn_embeddings.py >> logs/gnn_cron.log 2>&1

AUDIT I-A3: This script is the ONLY place where PyTorch is imported in production.
The daily pipeline only reads the output Parquet file.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import json


def main():
    print(f"[GNN-Cron] Starting at {datetime.now().isoformat()}")
    
    # Load latest data
    data_path = 'data/backtest/sp500_2023_2026.parquet'
    if not os.path.exists(data_path):
        # Try daily data
        import glob
        daily_files = sorted(glob.glob('data/raw/daily_*.parquet'))
        if daily_files:
            data_path = daily_files[-1]
        else:
            print("[GNN-Cron] ERROR: No data found")
            sys.exit(1)
    
    raw = pd.read_parquet(data_path)
    raw['date'] = pd.to_datetime(raw['date']).dt.tz_localize(None)
    print(f"[GNN-Cron] Loaded {len(raw)} rows, {raw['symbol'].nunique()} symbols")
    
    # Build features
    from src.features.build_features import FeatureEngineer
    feat_eng = FeatureEngineer()
    features = feat_eng.build_features(raw)
    
    # Train GAT v2
    from src.models.embedding_model import GNNEmbeddingTrainer
    trainer = GNNEmbeddingTrainer(
        embed_dim=16,
        hidden_dim=32,
        epochs=100,
        graph_window=60,
        corr_threshold=0.3,
        lr=0.003,
        n_pairs=50,
    )
    
    results = trainer.train(features, train_ratio=0.85)
    print(f"[GNN-Cron] Training complete: best_epoch={results['best_epoch']}, "
          f"rank_corr={results.get('final_val_rank_corr', 'N/A')}")
    
    # Generate embeddings for all dates
    output_path = 'data/cache/gnn_embeddings.parquet'
    emb_df = trainer.generate_embeddings(features, output_path=output_path)
    
    # Save training metadata
    meta = {
        'trained_at': datetime.now().isoformat(),
        'data_path': data_path,
        'n_symbols': int(raw['symbol'].nunique()),
        'n_dates': int(raw['date'].nunique()),
        'n_embeddings': len(emb_df),
        'embed_dim': trainer.embed_dim,
        'architecture': 'GAT-v2',
        'results': {k: float(v) if isinstance(v, (int, float)) else v 
                    for k, v in results.items()},
    }
    
    meta_path = 'data/cache/gnn_training_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"[GNN-Cron] Done. {len(emb_df)} embeddings → {output_path}")
    print(f"[GNN-Cron] Metadata → {meta_path}")


if __name__ == '__main__':
    main()
