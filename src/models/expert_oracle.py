import requests
from typing import Dict, Any, List
import pandas as pd
from src.ops.event_logger import get_logger

logger = get_logger()

class KronosOracleClient:
    """
    Client to interact with the Serverless Kronos model deployed on Modal.
    This acts as the L5 Veto Oracle.
    """
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def request_veto(self, symbol: str, target_weight: float, features_df: pd.DataFrame, lookback: int = 400) -> Dict[str, Any]:
        """
        Sends recent OHLCV data to the Oracle and gets a veto decision.
        Returns dict with 'action' (approve/veto/neutral) and 'predicted_return'.
        """
        if not self.endpoint_url:
            logger.warn("kronos_oracle_skipped_no_url", {"symbol": symbol})
            return {"action": "approve", "predicted_return": 0.0, "reason": "No endpoint configured"}

        try:
            # Extract last `lookback` records
            hist = features_df[features_df['symbol'] == symbol].tail(lookback).copy()
            if len(hist) < 50:
                logger.warn("kronos_insufficient_history", {"symbol": symbol, "rows": len(hist)})
                return {"action": "approve", "predicted_return": 0.0, "reason": "insufficient_history"}

            # Format to basic OHLCV dicts, drop NaNs
            hist = hist.ffill().fillna(0.0)
            
            ohlcv_records = []
            for _, row in hist.iterrows():
                # Strip timezone from timestamp for clean Modal parsing
                ts = pd.Timestamp(row['date'])
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                date_str = ts.strftime('%Y-%m-%d')
                # Map column names: features use adj_* or raw_* prefixes
                o = float(row.get('adj_open', row.get('raw_open', row.get('open', 0.0))))
                h = float(row.get('adj_high', row.get('raw_high', row.get('high', 0.0))))
                l = float(row.get('adj_low', row.get('raw_low', row.get('low', 0.0))))
                c = float(row.get('adj_close', row.get('raw_close', row.get('close', 0.0))))
                v = float(row.get('volume', 0.0))
                ohlcv_records.append({
                    'open': o, 'high': h, 'low': l, 'close': c,
                    'volume': v, 'timestamp': date_str
                })

            payload = {
                "symbol": symbol,
                "target_weight": target_weight,
                "ohlcv_records": ohlcv_records,
                "pred_len": 5
            }

            resp = requests.post(self.endpoint_url, json=payload, timeout=120) # High timeout for cold starts
            resp.raise_for_status()
            
            data = resp.json()
            
            # Phase I3: Dynamic veto threshold
            # Approximate VIX by taking median 20d annualized realized vol of universe * 100
            if 'rv_20d' in features_df.columns:
                vix_proxy = features_df['rv_20d'].median() * 100
            else:
                vix_proxy = 18.0  # Default to normal market
                
            if vix_proxy > 25:
                threshold = -0.03
            elif vix_proxy > 18:
                threshold = -0.01
            else:
                threshold = -0.005
                
            pred_ret = data.get("predicted_return", 0.0)
            
            # Override action based on dynamic threshold
            if pred_ret < threshold:
                data["action"] = "veto"
            elif pred_ret < 0.0:
                data["action"] = "neutral"
            else:
                data["action"] = "approve"
                
            data["reason"] += f" (Threshold: {threshold*100:.1f}%, VIX proxy: {vix_proxy:.1f})"
            
            return dict(data)
            
        except Exception as e:
            logger.error("kronos_oracle_failed", {"symbol": symbol, "error": str(e)})
            print(f"[ORACLE ERROR] {symbol}: {e}")
            if 'resp' in locals():
                print(f"[ORACLE RESPONSE HEADERS/BODY] {resp.text}")
            # Fail-open: If the oracle is offline, we trust our Meta Model
            return {"action": "approve", "predicted_return": 0.0, "error": str(e)}
