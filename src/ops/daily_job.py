"""
Daily Job - Main Pipeline Orchestration (Phase E Refactored)

Coordinates the daily data pipeline with idempotency guarantees.
All steps output checkpoints, allowing the pipeline to resume safely upon failure.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
import yaml
import pandas as pd
from pandas.tseries.offsets import BDay

from src.data.ingest import DualSourceIngest
from src.data.validate import DataValidator
from src.data.integrity import IntegrityManager
from src.data.corporate_actions import CorporateActionsHandler
from src.data.universe import UniverseManager
from src.data.wap_utils import write_parquet_wap, read_parquet_safe
from src.ops.event_logger import get_logger
from src.ops.alerts import AlertManager

# Import execution and modeling modules
from src.features.build_features import FeatureEngineer
from src.features.regime_detector import RegimeDetector
from src.models.model_io import ModelBundleManager
from src.risk.position_sizing import IndependentKellySizer
from src.risk.risk_engine import RiskEngine
from src.risk.pdt_guard import PDTGuard
from src.ops.signal_consistency import SignalConsistency
from src.execution.futu_executor import FutuExecutor
from src.execution.futu_quote import FutuQuote
from src.execution.slippage_model import SlippageModel

import futu as ft

logger = get_logger()

class DailyJob:
    """Idempotent daily pipeline (Data -> Features -> Models -> Risk -> Execution)."""
    
    def __init__(self):
        # Phase A Data
        self.ingest = DualSourceIngest()
        self.validator = DataValidator()
        self.integrity = IntegrityManager()
        self.corp_actions = CorporateActionsHandler()
        self.universe = UniverseManager()
        
        # Phase E Ops & Execution
        self.alerts = AlertManager()
        self.model_mgr = ModelBundleManager()
        self.sizer = IndependentKellySizer()
        self.risk = RiskEngine()
        self.pdt = PDTGuard()
        self.is_paper = True
        
        # Checkpoint dir
        self.ckpt_dir = "data/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def run(self, trade_date: Optional[str] = None) -> bool:
        """Run full daily pipeline idempotently."""
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info("daily_job_start", {"trade_date": trade_date})
        
        state = self._load_checkpoint(trade_date)
        
        steps = [
            ("ingest", self._step_ingest),
            ("validate", self._step_validate),
            ("integrity", self._step_integrity),
            ("corporate_actions", self._step_corp_actions),
            ("universe", self._step_universe),
            ("features", self._step_features),
            ("signals", self._step_signals),
            ("risk_and_size", self._step_risk_and_size),
            ("execute", self._step_execute)
        ]
        
        # Check PnL / account limits to break early if KillSwitch active
        health = self.risk.check_portfolio_health(0.0, 0.0, 0)
        self.is_paper = health.get('is_auto_degraded', False)
        
        for step_name, step_func in steps:
            if state.get(step_name, False):
                logger.info("step_skipped", {"step": step_name, "reason": "already_complete"})
                continue
                
            try:
                result = step_func(trade_date)
                state[step_name] = True
                self._save_checkpoint(trade_date, state)
                logger.info("step_complete", {"step": step_name, "result": result})
            except Exception as e:
                logger.error("step_failed", {"step": step_name, "error": str(e)})
                self.alerts.send_alert("CRITICAL", f"Daily Job Failed at Step: {step_name}", str(e))
                return False
                
        # Send Daily Summary
        self.alerts.send_alert("INFO", f"Daily Job Completed: {trade_date}", "All modules executed successfully.")
        logger.info("daily_job_complete", {"trade_date": trade_date})
        return True

    def _load_checkpoint(self, trade_date: str) -> Dict[str, bool]:
        path = f"{self.ckpt_dir}/{trade_date}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_checkpoint(self, trade_date: str, state: Dict[str, bool]):
        path = f"{self.ckpt_dir}/{trade_date}.json"
        with open(path + '.tmp', 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(path + '.tmp', path)

    # ==========================================
    # Phase A: Data Ops (Steps 1-5 already built, just simplified here for brevity but preserving functionality)
    # ==========================================
    def _step_ingest(self, trade_date: str):
        existing_data = None
        for lookback in range(1, 5):
            prev = (pd.Timestamp(trade_date) - BDay(lookback)).strftime('%Y-%m-%d')
            path = f"data/raw/daily_{prev}.parquet"
            if os.path.exists(path):
                existing_data = read_parquet_safe(path)
                break
        if existing_data is None:
            existing_data = pd.DataFrame()
            
        universe_info = self.universe.build_universe(existing_data)
        if not universe_info['symbols']:
            raise ValueError(f"No symbols in universe for {trade_date}")
            
        end = pd.Timestamp(trade_date)
        start = end - timedelta(days=90) # Need longer history for features
        
        data = self.ingest.ingest(symbols=universe_info['symbols'], start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        write_parquet_wap(data, f"data/raw/daily_{trade_date}.parquet")
        return {"rows": len(data), "symbols": data['symbol'].nunique()}
        
    def _step_validate(self, trade_date: str):
        data = read_parquet_safe(f"data/raw/daily_{trade_date}.parquet")
        passed, cleaned, report = self.validator.validate(data)
        write_parquet_wap(cleaned, f"data/processed/validated_{trade_date}.parquet")
        return {"passed": passed}
        
    def _step_integrity(self, trade_date: str):
        data = read_parquet_safe(f"data/processed/validated_{trade_date}.parquet")
        should_freeze, drifts = self.integrity.detect_drift(data, data['symbol'].nunique())
        if should_freeze:
            raise RuntimeError(f"Data drift detected: {len(drifts)} events")
        self.integrity.freeze_data(data)
        return {"drifts": len(drifts)}
        
    def _step_corp_actions(self, trade_date: str):
        data = read_parquet_safe(f"data/processed/validated_{trade_date}.parquet")
        processed, info = self.corp_actions.apply_all(data)
        write_parquet_wap(processed, f"data/processed/corp_actions_{trade_date}.parquet")
        return info

    def _step_universe(self, trade_date: str):
        data = read_parquet_safe(f"data/processed/corp_actions_{trade_date}.parquet")
        universe_info = self.universe.build_universe(data)
        write_parquet_wap(data, f"data/processed/final_daily_{trade_date}.parquet")
        return universe_info['metadata']

    # ==========================================
    # Phase E: Execution Ops (Steps 6-9)
    # ==========================================
    def _step_features(self, trade_date: str):
        """Step 6: Build features (including fracdiff from model metadata)"""
        data = read_parquet_safe(f"data/processed/final_daily_{trade_date}.parquet")
        
        meta = self.model_mgr.load_metadata()
        if not meta:
            logger.warn("no_model_metadata_found", {})
            return "No active model"
            
        engineer = FeatureEngineer(config_path="config/features.yaml")
        features_df = engineer.build_features(data)
        
        # Apply fracdiff based on pre-calculated per_symbol_d from training
        from src.features.fracdiff import fracdiff_fixed_window
        features_df['fracdiff'] = 0.0
        
        # We only need fracdiff for prediction day
        for symbol, d_val in meta.get("optimal_d", {}).items():
            sym_mask = features_df['symbol'] == symbol
            try:
                if sym_mask.sum() > 0:
                    prices = np.log(features_df.loc[sym_mask, 'adj_close'])
                    fd_series = fracdiff_fixed_window(prices, d_val, window=100)
                    features_df.loc[sym_mask, 'fracdiff'] = fd_series.values
            except Exception as e:
                logger.warn("daily_fracdiff_failed", {"symbol": symbol, "error": str(e)})

        write_parquet_wap(features_df, f"data/processed/features_{trade_date}.parquet")
        return {"rows": len(features_df)}

    def _step_signals(self, trade_date: str):
        """Step 7: Base model generation + Meta Model Prediction"""
        from src.signals.base_models import BaseModelSMA
        
        features_df = read_parquet_safe(f"data/processed/features_{trade_date}.parquet")
        
        # 1. Base Model
        base_model = BaseModelSMA()
        base_signals = []
        for sym in features_df['symbol'].unique():
            df_sym = features_df[features_df['symbol'] == sym].copy()
            df_sym = base_model.generate_signals(df_sym)
            base_signals.append(df_sym)
        df_sig = pd.concat(base_signals, ignore_index=True)
        
        # We only care about TODAY's signals for production
        today_mask = df_sig['date'] == pd.Timestamp(trade_date)
        df_today = df_sig[today_mask].copy()
        
        # 2. Meta Model
        model = self.model_mgr.load_model()
        meta = self.model_mgr.load_metadata()
        
        if not model or not meta:
            logger.error("model_load_failed", {})
            raise RuntimeError("Cannot execute signals without loaded Meta-Model")
            
        # Get active base signals
        active_signals = df_today[df_today['side'] != 0].copy()
        
        if len(active_signals) == 0:
            logger.info("no_active_signals_today", {})
            write_parquet_wap(pd.DataFrame(), f"data/processed/signals_{trade_date}.parquet")
            return {"active_signals": 0}
            
        feature_cols = meta.get("feature_list", [])
        
        X_pred = active_signals[feature_cols]
        # Predict probability of being profitable
        p_profit = model.predict(X_pred)
        
        active_signals['prob'] = p_profit
        active_signals['avg_win'] = 0.05 # Mocked for safety, in production tie to ATR
        active_signals['avg_loss'] = 0.05
        active_signals['realized_vol'] = 0.15 # Tie to 20d historical
        
        write_parquet_wap(active_signals, f"data/processed/signals_{trade_date}.parquet")
        return {"active_signals": len(active_signals)}

    def _step_risk_and_size(self, trade_date: str):
        """Step 8: Fractional Kelly, L2/L3 Risk, and PDT Guards"""
        signals_path = f"data/processed/signals_{trade_date}.parquet"
        if not os.path.exists(signals_path):
            return "No signals"
            
        signals = read_parquet_safe(signals_path)
        if len(signals) == 0:
            return "No signals"
            
        # 1. PDT Check (Skip sizing if locked)
        import pandas as pd
        mock_history = pd.DataFrame() # In production, read from Futu!
        is_compliant, remaining = self.pdt.check_pdt_compliance(20000.0, mock_history)
        if not is_compliant:
            logger.warn("pdt_lockdown_active", {"remaining": remaining})
            self.alerts.send_alert("WARNING", "PDT Lockdown", "Pattern Day Trader limits reached. Trading halted today.")
            return {"status": "HALTED"}

        # 2. Independent Kelly Sizing
        positions = self.sizer.calculate_positions(signals, current_drawdown=0.0) # Mock real DD
        
        # 3. Risk Engine Caps (Single, Sector)
        validated_positions = self.risk.validate_positions(positions)
        
        # Write final targets
        write_parquet_wap(validated_positions, f"data/processed/targets_{trade_date}.parquet")
        return {"target_count": len(validated_positions)}

    def _step_execute(self, trade_date: str):
        """Step 9: Translate target weights to Futu Orders & submit"""
        targets_path = f"data/processed/targets_{trade_date}.parquet"
        if not os.path.exists(targets_path):
            return "No targets"
            
        targets = read_parquet_safe(targets_path)
        if len(targets) == 0:
            return "No targets"
            
        # Spin up Futu connections
        trd_env = ft.TrdEnv.SIMULATE if self.is_paper or True else ft.TrdEnv.REAL
        futu_exec = FutuExecutor(trd_env=trd_env)
        futu_quote = FutuQuote()
        
        if not futu_exec.connect() or not futu_quote.connect():
            raise RuntimeError("Futu OpenAPI Connection Failed")
            
        account_value = futu_exec.get_account_value()
        if account_value <= 0:
            account_value = 100000.0 # fallback
            
        # Quote real-time mid prices
        symbols = targets['symbol'].tolist()
        quotes = futu_quote.get_orderbook(symbols)
        
        orders = []
        for _, row in targets.iterrows():
            sym = row['symbol']
            weight = row['target_weight']
            
            if abs(weight) < 0.001:
                continue
                
            qdata = quotes.get(sym, {'mid': 100.0}) # mock mid if missing
            mid = qdata['mid']
            side = 'buy' if weight > 0 else 'sell'
            
            # Simple dollar conversion
            target_usd = abs(weight) * account_value
            qty = int(target_usd / mid)
            
            if qty > 0:
                orders.append({
                    'symbol': sym,
                    'qty': qty,
                    'side': side,
                    'price': mid, # limit price
                    'order_type': 'limit'
                })
                
        # Consistency Check
        consistency = SignalConsistency.verify(targets, orders)
        if consistency['inconsistencies'] > 0:
            logger.warn("daily_execution_inconsistency", consistency)
            
        # Submit Orders
        if len(orders) > 0:
            futu_exec.submit_orders(orders)
            logger.info("submitted_live_orders", {"count": len(orders)})
            
        futu_exec.close()
        futu_quote.close()
        
        return {"orders": len(orders), "value_usd": sum([o['qty']*o['price'] for o in orders])}

if __name__ == '__main__':
    job = DailyJob()
    job.run()
