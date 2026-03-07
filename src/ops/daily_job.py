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
from src.ops.portfolio_tracker import PortfolioTracker

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
        self.portfolio = PortfolioTracker(initial_cash=100000.0, friction=0.002)
        self.is_paper = True
        
        # Checkpoint dir
        self.ckpt_dir = "data/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def run(self, trade_date: Optional[str] = None, account_value: float = 100000.0) -> bool:
        """Run full daily pipeline idempotently.
        
        Args:
            trade_date: Override date (defaults to today)
            account_value: Assumed portfolio value for position sizing (USD)
        """
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')
        
        self.account_value = account_value
            
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
            ("execute", self._step_execute),
            ("virtual_portfolio", self._step_virtual_portfolio)
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
                state[step_name] = result if result is not None else True
                self._save_checkpoint(trade_date, state)
                logger.info("step_complete", {"step": step_name, "result": result})
            except Exception as e:
                logger.error("step_failed", {"step": step_name, "error": str(e)})
                self.alerts.send_alert("CRITICAL", f"🚨 运行异常: {step_name}", f"步骤 {step_name} 出现错误: {str(e)}")
                return False
                
        # Build Chinese Telegram Report with portfolio status
        report = self._build_telegram_report(trade_date, state, steps)
        self.alerts.send_alert("INFO", f"📊 跑盘完成 ({trade_date})", report)
        logger.info("daily_job_complete", {"trade_date": trade_date})
        return True
    
    def _get_latest_prices(self, trade_date: str) -> dict:
        """Get latest prices from features file."""
        prices = {}
        features_path = f"data/processed/features_{trade_date}.parquet"
        if os.path.exists(features_path):
            feat_df = read_parquet_safe(features_path)
            for sym in feat_df['symbol'].unique():
                sym_data = feat_df[feat_df['symbol'] == sym]
                if len(sym_data) > 0:
                    prices[sym] = float(sym_data['adj_close'].iloc[-1])
        return prices
    
    def _build_telegram_report(self, trade_date: str, state: dict, steps: list) -> str:
        """Build structured Chinese Telegram report with actionable trade instructions."""
        ingest = state.get('ingest', {})
        syms_count = ingest.get('symbols', '未知') if isinstance(ingest, dict) else '未知'
        
        signals = state.get('signals', {})
        active_sigs = signals.get('active_signals', 0) if isinstance(signals, dict) else 0
        
        risk_info = state.get('risk_and_size', {})
        target_count = risk_info.get('target_count', 0) if isinstance(risk_info, dict) else 0
        
        # Section 1: System status
        report = f"""📊 Quant-MVP 每日分析报告
📅 分析周期: {trade_date}

✅ 系统状态: 全部 {len(steps)} 个模块运行正常
📡 扫描: S&P500+NASDAQ100+DJIA
🔬 标的: {syms_count} 只 | 信号: {active_sigs} 个
"""
        # Section 2: Actionable trade instructions
        targets_path = f"data/processed/targets_{trade_date}.parquet"
        prices = self._get_latest_prices(trade_date)
        
        if os.path.exists(targets_path):
            targets = read_parquet_safe(targets_path)
            if len(targets) > 0 and 'target_weight' in targets.columns:
                targets = targets[targets['target_weight'].abs() >= 0.005].copy()
                targets = targets.sort_values('target_weight', key=abs, ascending=False)
                
                if len(targets) > 0:
                    report += "\n🎯 今日操作指令:\n"
                    buy_count = int((targets['target_weight'] > 0).sum())
                    sell_count = int((targets['target_weight'] < 0).sum())
                    
                    # Calculate NAV once (constant across all target rows)
                    nav = self.portfolio.state['cash'] + sum(
                        pos['qty'] * prices.get(s, pos['avg_cost'])
                        for s, pos in self.portfolio.state['positions'].items()
                    )
                    
                    for i, (_, row) in enumerate(targets.head(10).iterrows()):
                        sym = row['symbol']
                        w = row['target_weight']
                        price = prices.get(sym, 0.0)
                        icon = '📈' if w > 0 else '📉'
                        direction = '买' if w > 0 else '卖'
                        
                        if price > 0:
                            qty = int(abs(w) * nav / price)
                            report += f"{icon}{direction} {sym} {abs(w)*100:.1f}% ${price:.0f}×{qty}股\n"
                        else:
                            report += f"{icon}{direction} {sym} {abs(w)*100:.1f}%\n"
                    
                    if len(targets) > 10:
                        report += f"  +{len(targets)-10}个...\n"
                    report += f"买{buy_count}笔 卖{sell_count}笔\n"
                else:
                    report += "\n📭 今日无操作建议\n"
            else:
                report += "\n📭 今日无操作建议\n"
        else:
            report += "\n📭 今日无操作建议\n"
        
        # Section 3: Portfolio status
        report += "\n" + self.portfolio.get_portfolio_report(prices)
        
        report += "\n⚠️ 以上为模型建议+虚拟盘跟踪结果"
        return report

    def _load_checkpoint(self, trade_date: str) -> Dict[str, Any]:
        path = f"{self.ckpt_dir}/{trade_date}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_checkpoint(self, trade_date: str, state: Dict[str, Any]):
        path = f"{self.ckpt_dir}/{trade_date}.json"
        with open(path + '.tmp', 'w') as f:
            # P0 (BugFix): Convert numpy types (int64/float64/etc) using default=str
            json.dump(state, f, indent=2, default=str)
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
            
        end_date = pd.Timestamp(trade_date)
        start = end_date - pd.Timedelta(days=252)  # P0 Fix: 252 calendar days ≈ 180 trading days
        
        # yfinance end date is exclusive, so we add 1 day to fetch today's data (if run after market close)
        yf_end = end_date + pd.Timedelta(days=1)
        
        data = self.ingest.ingest(symbols=universe_info['symbols'], start=start.strftime('%Y-%m-%d'), end=yf_end.strftime('%Y-%m-%d'))
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
        """Step 7: Base model generation + Meta Model Prediction
        
        P0 Fix: Use the LATEST available date in the data instead of trade_date,
        because yfinance returns data up to the last completed trading day.
        Also calculate real avg_win/avg_loss from historical returns.
        """
        from src.signals.base_models import BaseModelSMA, BaseModelMomentum
        
        features_df = read_parquet_safe(f"data/processed/features_{trade_date}.parquet")
        
        # 1. Base Models (use both SMA and Momentum for more signal coverage)
        base_signals = []
        for sym in features_df['symbol'].unique():
            df_sym = features_df[features_df['symbol'] == sym].copy()
            
            # SMA crossover model
            df_sma = BaseModelSMA().generate_signals(df_sym)
            base_signals.append(df_sma)
            
            # Momentum breakout model
            df_mom = BaseModelMomentum().generate_signals(df_sym)
            base_signals.append(df_mom)
        df_sig = pd.concat(base_signals, ignore_index=True)
        
        # P0 Fix: Use the LATEST available date, not trade_date
        # yfinance only returns data up to the last closed trading day
        latest_date = df_sig['date'].max()
        logger.info("signal_date_selection", {
            "trade_date": trade_date,
            "latest_data_date": str(latest_date),
            "total_dates": df_sig['date'].nunique()
        })
        
        today_mask = df_sig['date'] == latest_date
        df_today = df_sig[today_mask].copy()
        
        # 2. Meta Model (try to load, fall back to probability-based ranking)
        model = self.model_mgr.load_model()
        meta = self.model_mgr.load_metadata()
        
        # Get active base signals (side != 0)
        active_signals = df_today[df_today['side'] != 0].copy()
        
        if len(active_signals) == 0:
            logger.info("no_active_signals_today", {"latest_date": str(latest_date)})
            write_parquet_wap(pd.DataFrame(), f"data/processed/signals_{trade_date}.parquet")
            return {"active_signals": 0}
        
        # 3. Calculate real win rate and avg_win/avg_loss from historical returns
        for idx, row in active_signals.iterrows():
            sym = row['symbol']
            sym_hist = df_sig[(df_sig['symbol'] == sym) & (df_sig['side'] != 0)].copy()
            
            if len(sym_hist) > 10:
                # Calculate forward 5-day returns for historical signals
                sym_all = features_df[features_df['symbol'] == sym].sort_values('date')
                fwd_returns = sym_all['adj_close'].pct_change(5).shift(-5)
                sym_hist_with_ret = sym_hist.merge(
                    pd.DataFrame({'date': sym_all['date'], 'fwd_ret': fwd_returns}),
                    on='date', how='left'
                )
                aligned = (sym_hist_with_ret['fwd_ret'] * sym_hist_with_ret['side']).dropna()
                wins = aligned[aligned > 0]
                losses = aligned[aligned < 0]
                
                prob = len(wins) / max(len(aligned), 1)
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.03
                avg_loss = float(losses.abs().mean()) if len(losses) > 0 else 0.03
            else:
                prob = 0.50
                avg_win = 0.03
                avg_loss = 0.03
            
            active_signals.at[idx, 'prob'] = np.clip(prob, 0.01, 0.99)
            active_signals.at[idx, 'avg_win'] = max(avg_win, 0.001)
            active_signals.at[idx, 'avg_loss'] = max(avg_loss, 0.001)
        
        # 4. Use real realized_vol from features if available
        if 'rv_20d' in active_signals.columns:
            active_signals['realized_vol'] = active_signals['rv_20d'].fillna(0.20)
        else:
            active_signals['realized_vol'] = 0.20
        
        # 5. If Meta Model is available, override probability with model prediction
        if model and meta:
            feature_cols = meta.get("feature_list", [])
            available_cols = [c for c in feature_cols if c in active_signals.columns]
            if len(available_cols) == len(feature_cols):
                try:
                    X_pred = active_signals[feature_cols]
                    p_profit = model.predict(X_pred)
                    active_signals['prob'] = np.clip(p_profit, 0.01, 0.99)
                    logger.info("meta_model_applied", {"count": len(active_signals)})
                except Exception as e:
                    logger.warn("meta_model_predict_failed", {"error": str(e), "using": "historical_prob"})
        else:
            logger.warn("no_meta_model", {"using": "historical_win_rate_as_prob"})
        
        # 6. Filter: only keep signals with prob > confidence_threshold
        conf_threshold = 0.52  # Slightly above random
        active_signals = active_signals[active_signals['prob'] >= conf_threshold].copy()
        
        logger.info("signals_generated", {
            "active_signals": len(active_signals),
            "latest_date": str(latest_date),
            "avg_prob": float(active_signals['prob'].mean()) if len(active_signals) > 0 else 0
        })
        
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
        mock_history = []  # In production, read from actual trade logs
        account_val = getattr(self, 'account_value', 100000.0)
        is_compliant, remaining = self.pdt.check_pdt_compliance(mock_history, account_val)
        if not is_compliant:
            logger.warn("pdt_lockdown_active", {"remaining": remaining})
            self.alerts.send_alert("WARNING", "PDT Lockdown", "Pattern Day Trader limits reached. Trading halted today.")
            return {"status": "HALTED"}

        # 2. Independent Kelly Sizing
        positions = self.sizer.calculate_positions(signals, current_drawdown=0.0) # Mock real DD
        
        # Merge duplicate symbols (e.g. from both SMA and Momentum models) by summing their weights
        if not positions.empty:
            positions = positions.groupby('symbol', as_index=False)['target_weight'].sum()
        
        # 3. Risk Engine Caps (Single, Sector)
        validated_positions = self.risk.validate_positions(positions)
        
        # Write final targets
        write_parquet_wap(validated_positions, f"data/processed/targets_{trade_date}.parquet")
        return {"target_count": len(validated_positions)}

    def _step_virtual_portfolio(self, trade_date: str):
        """Step 9a: Execute targets against virtual portfolio (with 0.2% friction)."""
        targets_path = f"data/processed/targets_{trade_date}.parquet"
        if not os.path.exists(targets_path):
            return {"status": "no_targets"}
            
        targets_df = read_parquet_safe(targets_path)
        if len(targets_df) == 0 or 'target_weight' not in targets_df.columns:
            return {"status": "no_targets"}
        
        # Get prices
        prices = self._get_latest_prices(trade_date)
        if not prices:
            return {"status": "no_prices"}
        
        # Build targets dict {symbol: weight}
        targets_dict = {}
        for _, row in targets_df.iterrows():
            sym = row['symbol']
            w = float(row['target_weight'])
            if abs(w) >= 0.005 and sym in prices:
                targets_dict[sym] = w
        
        if not targets_dict:
            return {"status": "no_valid_targets"}
        
        # Execute against virtual portfolio
        result = self.portfolio.execute_targets(targets_dict, prices, trade_date)
        
        snap = result.get('snapshot', {})
        return {
            "trades": result['trades'],
            "nav": round(snap.get('nav', 0), 2),
            "cumulative_pnl": round(snap.get('cumulative_pnl', 0), 2),
            "cumulative_return": f"{snap.get('cumulative_return', 0)*100:.2f}%",
            "friction_today": round(result['total_friction'], 2)
        }

    def _step_execute(self, trade_date: str):
        """Step 9: Translate target weights to Futu Orders & submit.
        
        P0 Fix: Quick socket probe before Futu SDK initialization to avoid
        60+ second retry loop when no gateway is running.
        """
        targets_path = f"data/processed/targets_{trade_date}.parquet"
        if not os.path.exists(targets_path):
            return {"status": "no_targets", "orders": 0, "value_usd": 0}
            
        targets = read_parquet_safe(targets_path)
        # Filter out dust positions for cleaner logging
        targets = targets[targets['target_weight'].abs() >= 0.005].copy()
        if len(targets) == 0:
            return {"status": "no_targets", "orders": 0, "value_usd": 0}
            
        logger.info("daily_targets_generated", {
            "count": len(targets),
            "buy": int((targets['target_weight'] > 0).sum()),
            "sell": int((targets['target_weight'] < 0).sum())
        })
        
        # Quick socket probe: is Futu OpenD running?
        import socket
        gateway_reachable = False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', 11111))
            gateway_reachable = (result == 0)
            sock.close()
        except Exception:
            pass
        
        if not gateway_reachable:
            logger.info("futu_gateway_unreachable", {
                "note": "No OpenD detected. Targets saved to parquet for manual execution.",
                "targets_count": len(targets)
            })
            return {
                "status": "targets_generated_but_not_executed",
                "orders": len(targets),
                "value_usd": 0
            }
        
        # Gateway is reachable - attempt live execution
        try:
            trd_env = ft.TrdEnv.SIMULATE if self.is_paper or True else ft.TrdEnv.REAL
            futu_exec = FutuExecutor(trd_env=trd_env)
            futu_quote = FutuQuote()
            
            if not futu_exec.connect() or not futu_quote.connect():
                return {"status": "targets_generated_but_not_executed", "orders": len(targets), "value_usd": 0}
                
            account_value = futu_exec.get_account_value()
            if account_value <= 0:
                account_value = getattr(self, 'account_value', 100000.0)
                
            symbols = targets['symbol'].tolist()
            quotes = futu_quote.get_orderbook(symbols)
            
            orders = []
            for _, row in targets.iterrows():
                sym = row['symbol']
                weight = row['target_weight']
                
                if abs(weight) < 0.001:
                    continue
                    
                qdata = quotes.get(sym, {'mid': 100.0})
                mid = qdata['mid']
                side = 'buy' if weight > 0 else 'sell'
                target_usd = abs(weight) * account_value
                qty = int(target_usd / mid)
                
                if qty > 0:
                    orders.append({
                        'symbol': sym, 'qty': qty, 'side': side,
                        'price': mid, 'order_type': 'limit'
                    })
                    
            if len(orders) > 0:
                futu_exec.submit_orders(orders)
                logger.info("submitted_live_orders", {"count": len(orders)})
                
            futu_exec.close()
            futu_quote.close()
            
            return {"status": "executed", "orders": len(orders), "value_usd": sum(o['qty']*o['price'] for o in orders)}

        except Exception as e:
            logger.error("futu_execution_failed", {"error": str(e)})
            return {"status": "targets_generated_but_not_executed", "orders": len(targets), "value_usd": 0}

if __name__ == "__main__":
    job = DailyJob()
    job.run()

