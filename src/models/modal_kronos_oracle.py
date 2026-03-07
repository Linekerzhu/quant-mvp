import modal
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional
import os

app = modal.App("kronos-expert-oracle")

# The image for the serverless container
# We install transformers, torch, and pandas
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch>=2.0.0", "fastapi[standard]")
    .run_commands(
        "git clone https://github.com/shiyu-coder/Kronos.git /root/Kronos",
        "cd /root/Kronos && pip install -r requirements.txt"
    )
)

fastapi_app = FastAPI()

class PredictionRequest(BaseModel):
    symbol: str
    target_weight: float
    ohlcv_records: list # completely generic list to avoid Modal/FastAPI coercion
    pred_len: int = 5

@app.cls(gpu="T4", image=image, min_containers=0, scaledown_window=900)
class KronosService:
    @modal.enter()
    def setup(self):
        print("Initializing Kronos Model...")
        import sys
        
        # Add the cloned Kronos repo to sys.path so we can import their code
        if "/root/Kronos" not in sys.path:
            sys.path.append("/root/Kronos")
            
        from model import Kronos, KronosTokenizer, KronosPredictor
        
        try:
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            
            # Use cuda if available
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Loading Predictor on {device}")
            
            self.predictor = KronosPredictor(self.model, self.tokenizer, device=device, max_context=512)
        except Exception as e:
            print(f"Warning: model loading failed: {e}")
            self.predictor = None

    @modal.method()
    def predict(self, req: PredictionRequest) -> Dict[str, Any]:
        if getattr(self, 'predictor', None) is None:
            return {"status": "error", "message": "Predictor failed to initialize during enter()"}
            
        print(f"Received request for {req.symbol} with {len(req.ohlcv_records)} records.")
        if len(req.ohlcv_records) < 50:
             return {"symbol": req.symbol, "status": "error", "message": "need at least 50 history bars"}
             
        try:
            # 1. Prepare Data for predictor
            df = pd.DataFrame(req.ohlcv_records)
            # Ensure proper columns exist. KronosPredictor needs: open, high, low, close
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    return {"status": "error", "message": f"Missing column {col}"}
                    
            # Ensure timestamps are parsed — strip timezone to avoid .dt accessor errors
            df['timestamps'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            
            # Optional volume/amount
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            if 'amount' not in df.columns:
                df['amount'] = 0.0
                
            x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = df['timestamps']
            
            # We want to predict pred_len units ahead. Let's create dummy future timestamps (business days approx)
            last_time = x_timestamp.iloc[-1]
            future_daterange = pd.bdate_range(start=last_time + pd.Timedelta(days=1), periods=req.pred_len)
            y_timestamp = pd.Series(future_daterange)
            
            # Generate predictions via Monte Carlo sampling (sample_count=20).
            # Kronos internally generates `sample_count` independent stochastic trajectories
            # and averages them (np.mean over axis=1) to reduce variance.
            # With sample_count=1, AAPL showed 9.3% return swing between consecutive calls.
            # sample_count=20 stabilizes predictions at modest GPU cost (~2x latency).
            pred_df = self.predictor.predict(x_df, x_timestamp, y_timestamp, req.pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=20, verbose=False)
            
            # Predictor returns pred_df with future OHLC
            future_close = float(pred_df['close'].iloc[-1])
            last_close = float(x_df['close'].iloc[-1])
            predicted_return = (future_close / last_close) - 1.0
            
            action = "approve"
            if predicted_return < -0.015:
                action = "veto"
            elif predicted_return < 0.0:
                 action = "neutral"
                 
            return {
                "symbol": req.symbol,
                "status": "success",
                "predicted_return": round(predicted_return, 4),
                "action": action,
                "reason": f"Expected 5-day return: {predicted_return*100:.2f}%"
            }
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return {"symbol": req.symbol, "status": "error", "message": str(e)}

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def oracle_webhook(req: PredictionRequest):
    svc = KronosService()
    result = svc.predict.remote(req)
    return result
