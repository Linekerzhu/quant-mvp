import pandas as pd
import numpy as np
import lightgbm as lgb
from src.models.model_io import ModelBundleManager
import os

def build_mock_model():
    print("Building a mock LightGBM Meta-Model for Phase E sandbox execution...")
    # 1. Dummy data
    X = pd.DataFrame({
        'volume_7d': np.random.randn(100),
        'ret_1d': np.random.randn(100),
        'fracdiff': np.random.randn(100)
    })
    y = np.random.randint(0, 2, 100)
    
    # 2. Train trivial model
    train_data = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'verbose': -1}
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # 3. Save it
    mgr = ModelBundleManager()
    
    meta = {
        "feature_list": ['volume_7d', 'ret_1d', 'fracdiff'],
        "optimal_d": {
            "US.AAPL": 0.45,
            "US.MSFT": 0.50,
            "US.SPY": 0.40
        }
    }
    
    mgr.save_bundle(model, meta)
    print("✅ Mock meta-model bundle successfully saved to models/. Ready for daily_job.py!")

if __name__ == "__main__":
    build_mock_model()
