import os
import yaml
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from src.ops.event_logger import get_logger

logger = get_logger()

class CostCalibrator:
    """
    Weekly calibration loop for the cost model.
    Compares the assumed mid-price at order submission with the actual fill price,
    calculates the observed transaction cost (spread + impact), and updates the config.
    """
    def __init__(self, config_path: str = "config/training.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            self.config_path = Path(__file__).parent.parent.parent / config_path
            
        with open(self.config_path, 'r') as f:
            self.full_config = yaml.safe_load(f)
            
        self.config = self.full_config.get("cost_model", {})
        
    def _get_adv_bucket(self, adv_usd: float) -> str:
        if adv_usd < 20_000_000:
            return "low"
        elif adv_usd < 100_000_000:
            return "mid"
        else:
            return "high"
            
    def calibrate(self, recent_trades: List[Dict[str, Any]]) -> bool:
        """
        Run calibration based on recent trades.
        Each trade dict should have: 
        'symbol', 'qty', 'side', 'submitted_mid_price', 'fill_price', 'adv_usd'
        """
        if not recent_trades:
            logger.info("cost_calib_skipped", {"reason": "no_recent_trades"})
            return False
            
        df = pd.DataFrame(recent_trades)
        
        # Calculate observed slippage in BPS (basis points)
        # Bps formula: |fill_price - mid_price| / mid_price * 10000
        df['slippage_bps'] = (abs(df['fill_price'] - df['submitted_mid_price']) / df['submitted_mid_price']) * 10000.0
        
        # Determine ADV bucket
        df['adv_bucket'] = df['adv_usd'].apply(self._get_adv_bucket)
        
        calib_settings = self.config.get("calibration", {})
        min_samples = calib_settings.get("min_samples_for_update", 20)
        alert_mult = calib_settings.get("alert_threshold_mult", 2.0)
        max_change_pct = calib_settings.get("max_param_change_pct", 100.0) / 100.0
        
        updates_made = False
        
        for bucket in ['low', 'mid', 'high']:
            bucket_df = df[df['adv_bucket'] == bucket]
            if len(bucket_df) < min_samples:
                logger.info("cost_calib_insufficient_samples", {"bucket": bucket, "samples": len(bucket_df), "min_req": min_samples})
                continue
                
            observed_median_bps = bucket_df['slippage_bps'].median()
            current_bps = self.config["spread_bps"]["by_adv_bucket"][bucket]
            
            logger.info("cost_calib_bucket_stats", {
                "bucket": bucket,
                "samples": len(bucket_df),
                "observed_median_bps": float(observed_median_bps),
                "current_assumed_bps": float(current_bps)
            })
            
            # Check alert threshold
            if observed_median_bps > current_bps * alert_mult:
                logger.warn("cost_calib_high_slippage_alert", {
                    "bucket": bucket,
                    "observed": float(observed_median_bps),
                    "assumed": float(current_bps)
                })
                
            # Calculate new value with max change constraint
            max_allowed = current_bps * (1.0 + max_change_pct)
            min_allowed = current_bps * (1.0 - max_change_pct)
            new_bps = np.clip(observed_median_bps, min_allowed, max_allowed)
            
            if abs(new_bps - current_bps) > 1e-4: # if changed
                logger.info("cost_calib_updating_param", {
                    "bucket": bucket, 
                    "old_bps": float(current_bps), 
                    "new_bps": float(new_bps)
                })
                self.config["spread_bps"]["by_adv_bucket"][bucket] = float(new_bps)
                updates_made = True
                
        if updates_made:
            self._save_config()
            
        return updates_made
        
    def _save_config(self):
        # WAP pattern for YAML
        temp_path = str(self.config_path) + ".tmp"
        with open(temp_path, 'w') as f:
            yaml.safe_dump(self.full_config, f, sort_keys=False)
            
        os.replace(temp_path, self.config_path)
        logger.info("cost_calib_config_saved", {})
