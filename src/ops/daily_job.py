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
from dotenv import load_dotenv
import warnings

# Suppress urllib3 SSL warnings on Mac
warnings.filterwarnings("ignore", module="urllib3")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Ensure cron jobs load .env variables securely
load_dotenv()

from src.data.ingest import DualSourceIngest
from src.data.validate import DataValidator
from src.data.integrity import IntegrityManager
from src.data.corporate_actions import CorporateActionsHandler
from src.data.universe import UniverseManager
from src.data.wap_utils import write_parquet_wap, read_parquet_safe
from src.ops.event_logger import get_logger
from src.ops.alerts import AlertManager
from src.ops.portfolio_tracker import PortfolioTracker

# v4 strategy modules
from src.features.build_features import FeatureEngineer
from src.risk.risk_engine import RiskEngine
from src.risk.pdt_guard import PDTGuard

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
        self.risk = RiskEngine()
        self.pdt = PDTGuard()
        self.portfolio = PortfolioTracker(initial_cash=100000.0, friction=0.002)
        self.account_value: float = 100000.0
        self.is_paper = True
        
        # Checkpoint dir
        self.ckpt_dir = "data/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.funnel_stats = {}
    
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
            ("virtual_portfolio", self._step_virtual_portfolio),
            ("automl_retrain", self._step_automl_retrain)
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
                
        # Save overarching funnel metrics for the notifier
        try:
            with open(f"data/processed/funnel_stats_{trade_date}.json", "w") as f:
                json.dump(self.funnel_stats, f, indent=2)
        except Exception as e:
            logger.error("failed_to_save_funnel_stats", {"error": str(e)})
            
        # Note: Rich Telegram report is now sent by run_daily_with_notify.py
        # to avoid duplicate messages. Only error alerts (line 118) are sent from here.
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
                            qty = round(abs(w) * nav / price)  # L1 fix: round() not int()
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
            logger.warn("drift_freeze_ignored", {"events": len(drifts)})
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
        """Step 6: Build features (v4: core technical indicators only)
        
        v4 strategy computes its own factors (momentum, quality, value) directly
        in _step_signals from adj_close. This step provides the base technical
        features needed for that computation.
        """
        data = read_parquet_safe(f"data/processed/final_daily_{trade_date}.parquet")
        
        engineer = FeatureEngineer(config_path="config/features.yaml")
        features_df = engineer.build_features(data)
        
        # Add advanced technical features (vol_regime, price_range_pos, etc.)
        try:
            from src.features.extra_features import add_advanced_features
            features_df = add_advanced_features(features_df)
        except ImportError:
            pass

        write_parquet_wap(features_df, f"data/processed/features_{trade_date}.parquet")
        
        logger.info("features_built_v4", {
            "rows": len(features_df),
            "cols": len(features_df.columns),
            "symbols": features_df['symbol'].nunique(),
        })
        
        return {"rows": len(features_df)}

    def _step_signals(self, trade_date: str):
        """Step 7: Multi-Factor Signal Generation (v4)
        
        Matches walk_forward.py v4:
        - Momentum 12-1 + Quality (low vol) + Value (price/SMA200)
        - Cross-sectional ranking → composite score  
        - No LTR model, no SMA/Mom voting
        """
        features_df = read_parquet_safe(f"data/processed/features_{trade_date}.parquet")
        if features_df.empty:
            return {"active_signals": 0}

        # Use the LATEST available date
        latest_date = features_df['date'].max()
        logger.info("signal_date_selection", {
            "trade_date": trade_date,
            "latest_data_date": str(latest_date),
        })

        # Monthly rebalance check: only generate signals on first trading day of month
        import json as _json
        rebal_state_path = os.path.join(self.ckpt_dir, "last_rebalance.json")
        should_rebalance = True
        if os.path.exists(rebal_state_path):
            with open(rebal_state_path) as f:
                last_rebal = _json.load(f)
            td = pd.Timestamp(trade_date)
            lr = last_rebal.get("year_month", "")
            if lr == f"{td.year}-{td.month:02d}":
                should_rebalance = False
                logger.info("monthly_rebalance_skip", {"trade_date": trade_date, "last": lr})
        
        if not should_rebalance:
            # No rebalance — carry forward existing targets
            write_parquet_wap(pd.DataFrame(), f"data/processed/signals_{trade_date}.parquet")
            return {"active_signals": 0, "reason": "not_rebalance_day"}

        # Save rebalance state
        td = pd.Timestamp(trade_date)
        with open(rebal_state_path, "w") as f:
            _json.dump({"year_month": f"{td.year}-{td.month:02d}", "date": trade_date}, f)

        # Compute factors on the latest date's cross-section
        g = features_df.groupby("symbol")["adj_close"]

        # Technical Factor 1: Momentum 12-1 (skip last month)
        features_df["mom_12_1"] = g.transform(lambda x: x.shift(21) / x.shift(252) - 1)

        # Technical Factor 2: Low Volatility (quality proxy)
        features_df["rv_60d"] = g.transform(lambda x: x.pct_change().rolling(60).std() * np.sqrt(252))
        features_df["low_vol"] = -features_df["rv_60d"]

        # Realized vol for inv-vol weighting
        features_df["rv_20d"] = g.transform(lambda x: x.pct_change().rolling(20).std() * np.sqrt(252))
        rv_map = features_df[features_df["date"] == latest_date].set_index("symbol")["rv_20d"].to_dict()

        # Cross-sectional ranking on latest date
        today = features_df[features_df["date"] == latest_date].copy()
        today["mom_12_1_rank"] = today["mom_12_1"].rank(pct=True, na_option="bottom")
        today["low_vol_rank"] = today["low_vol"].rank(pct=True, na_option="bottom")

        # --- Fundamental Factors (v5.1) ---
        # Download/cache fundamental data: PE, ROE, margins, growth, debt
        try:
            from src.features.fundamentals import FundamentalProvider
            fp = FundamentalProvider()
            all_syms = today["symbol"].unique().tolist()
            fund_df = fp.get_fundamentals(all_syms, date=trade_date)
            
            # Compute fundamental composite score
            fund_composite = fp.compute_composite(fund_df)
            fund_df["fundamental_score"] = fund_composite
            
            # Merge into today
            today = today.merge(
                fund_df[["symbol", "fundamental_score", "earnings_yield", "roe", "profit_margin", "earnings_growth"]],
                on="symbol", how="left"
            )
            today["fundamental_rank"] = today["fundamental_score"].rank(pct=True, na_option="bottom")
            has_fundamentals = True
            
            n_with_fund = today["fundamental_score"].notna().sum()
            logger.info("fundamentals_loaded", {
                "total": len(all_syms),
                "with_data": int(n_with_fund),
                "coverage": f"{n_with_fund/len(all_syms)*100:.0f}%",
            })
        except Exception as e:
            logger.warn("fundamentals_failed", {"error": str(e)})
            today["fundamental_rank"] = 0.5  # Neutral if unavailable
            has_fundamentals = False

        # Composite score: Technical + Fundamental
        # 40% Momentum + 25% Fundamental + 20% Low Volatility + 15% reserved for growth
        if has_fundamentals:
            today["composite_score"] = (
                0.40 * today["mom_12_1_rank"] +
                0.25 * today["fundamental_rank"] +
                0.20 * today["low_vol_rank"] +
                0.15 * today.get("earnings_growth", pd.Series(0.5, index=today.index)).rank(pct=True, na_option="bottom")
            )
            factor_desc = "40%Mom + 25%Fundamental + 20%LowVol + 15%Growth"
        else:
            # Fallback to price-only if fundamentals unavailable
            today["composite_score"] = (
                0.70 * today["mom_12_1_rank"] +
                0.30 * today["low_vol_rank"]
            )
            factor_desc = "70%Mom + 30%LowVol (no fundamentals)"
        
        logger.info("composite_factors", {"formula": factor_desc})

        # Select top-K with sector cap
        ranked = today.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False)

        max_positions = 15
        sector_cap_pct = 0.25
        max_per_sector = max(1, int(max_positions * sector_cap_pct))

        # Load sector map
        sector_path = "data/sp500_sectors.json"
        sym2sec = {}
        if os.path.exists(sector_path):
            with open(sector_path) as f:
                sector_data = _json.load(f)
            for sector, syms in sector_data.items():
                for s in syms:
                    if s not in sym2sec:
                        sym2sec[s] = sector

        selected = []
        sector_counts = {}
        for _, row in ranked.iterrows():
            sym = row["symbol"]
            sec = sym2sec.get(sym, "Unknown")
            if sector_counts.get(sec, 0) >= max_per_sector:
                continue
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= max_positions:
                break

        if not selected:
            write_parquet_wap(pd.DataFrame(), f"data/processed/signals_{trade_date}.parquet")
            return {"active_signals": 0}

        # --- Kronos Veto Layer (v5) ---
        # For each candidate, ask Kronos Oracle if the stock should be held.
        # Veto'd stocks are replaced by next-best candidates from ranked list.
        kronos_url = os.environ.get("KRONOS_ENDPOINT_URL", "")
        # Also try loading from .env
        if not kronos_url:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
            if os.path.exists(env_path):
                with open(env_path) as ef:
                    for line in ef:
                        line = line.strip()
                        if line.startswith("KRONOS_ENDPOINT_URL="):
                            kronos_url = line.split("=", 1)[1]
                            break

        kronos_verdicts = {}
        vetoed_symbols = set()
        n_kronos_calls = 0

        if kronos_url:
            try:
                from src.models.expert_oracle import KronosOracleClient
                kronos_client = KronosOracleClient(kronos_url)
                
                logger.info("kronos_veto_start", {"candidates": len(selected), "url": kronos_url[:50]})
                
                for row in selected:
                    sym = row["symbol"]
                    try:
                        verdict = kronos_client.request_veto(
                            symbol=sym,
                            target_weight=0.05,
                            features_df=features_df,
                            lookback=400,
                        )
                        kronos_verdicts[sym] = verdict
                        n_kronos_calls += 1
                        
                        action = verdict.get("action", "approve")
                        pred_ret = verdict.get("predicted_return", 0.0)
                        
                        if action == "veto":
                            vetoed_symbols.add(sym)
                            logger.info("kronos_veto", {
                                "symbol": sym,
                                "pred_return": round(pred_ret, 4),
                                "reason": verdict.get("reason", ""),
                            })
                    except Exception as e:
                        logger.warn("kronos_call_failed", {"symbol": sym, "error": str(e)})
                        # Fail-open: keep the candidate
                
                logger.info("kronos_veto_done", {
                    "calls": n_kronos_calls,
                    "vetoed": len(vetoed_symbols),
                    "vetoed_symbols": list(vetoed_symbols),
                })
            except ImportError:
                logger.warn("kronos_import_failed", {"msg": "expert_oracle not available"})
        else:
            logger.info("kronos_skipped", {"reason": "no endpoint configured"})

        # Replace vetoed stocks with next-best candidates
        if vetoed_symbols:
            approved = [r for r in selected if r["symbol"] not in vetoed_symbols]
            
            # Collect symbols already selected (approved + vetoed)
            all_selected_syms = {r["symbol"] for r in selected}
            
            # Fill slots from ranked list (skipping already-selected symbols)
            for _, row in ranked.iterrows():
                if len(approved) >= max_positions:
                    break
                sym = row["symbol"]
                if sym in all_selected_syms:
                    continue
                sec = sym2sec.get(sym, "Unknown")
                # Respect sector cap
                approved_sec_count = {}
                for r in approved:
                    s = sym2sec.get(r["symbol"], "Unknown")
                    approved_sec_count[s] = approved_sec_count.get(s, 0) + 1
                if approved_sec_count.get(sec, 0) >= max_per_sector:
                    continue
                
                # Optionally check replacement with Kronos too
                if kronos_url and sym not in kronos_verdicts:
                    try:
                        v = kronos_client.request_veto(sym, 0.05, features_df, 400)
                        kronos_verdicts[sym] = v
                        if v.get("action") == "veto":
                            all_selected_syms.add(sym)
                            continue
                    except:
                        pass
                
                approved.append(row)
                all_selected_syms.add(sym)
            
            selected = approved

        # Save Kronos verdicts for analysis/reporting
        if kronos_verdicts:
            import json as _json3
            verdicts_path = f"data/processed/kronos_verdicts_{trade_date}.json"
            with open(verdicts_path, "w") as vf:
                _json3.dump({
                    sym: {"action": v.get("action"), "pred_return": v.get("predicted_return", 0), 
                          "reason": v.get("reason", "")}
                    for sym, v in kronos_verdicts.items()
                }, vf, indent=2)

        selected_df = pd.DataFrame(selected)

        # Inverse volatility weighting
        vols = selected_df["symbol"].map(rv_map).fillna(0.20).clip(lower=0.05)
        inv_vol = 1.0 / vols
        selected_df["target_weight"] = (inv_vol / inv_vol.sum()).values
        selected_df["side"] = 1  # All long

        # Save
        output_cols = ["symbol", "side", "target_weight", "composite_score", "mom_12_1"]
        output_df = selected_df[[c for c in output_cols if c in selected_df.columns]].copy()
        write_parquet_wap(output_df, f"data/processed/signals_{trade_date}.parquet")

        self.funnel_stats["total_base_signals"] = len(ranked)
        self.funnel_stats["passed_meta_model"] = len(selected)
        self.funnel_stats["kronos_vetoed"] = len(vetoed_symbols)
        self.funnel_stats["kronos_calls"] = n_kronos_calls
        self.funnel_stats["unique_symbols_with_signals"] = len(selected)

        logger.info("signals_generated_v5", {
            "active_signals": len(selected),
            "sectors": len(set(sym2sec.get(r["symbol"], "?") for r in selected)),
            "avg_weight": float(output_df['target_weight'].mean()),
            "kronos_vetoed": len(vetoed_symbols),
        })

        return {"active_signals": len(selected), "kronos_vetoed": len(vetoed_symbols)}

    def _step_risk_and_size(self, trade_date: str):
        """Step 8: Portfolio Construction (v4 — matches walk_forward.py)
        
        Uses pre-computed weights from _step_signals (inv-vol + sector cap).
        No Kelly, no HRP — those were part of the old LTR strategy.
        """
        signals_path = f"data/processed/signals_{trade_date}.parquet"
        if not os.path.exists(signals_path):
            return "No signals"

        signals = read_parquet_safe(signals_path)
        if len(signals) == 0 or 'target_weight' not in signals.columns:
            return {"target_count": 0, "reason": "no_signals_or_not_rebalance_day"}

        # 1. PDT Check
        mock_history = []
        try:
            if os.path.exists("data/portfolio/trades.jsonl"):
                with open("data/portfolio/trades.jsonl", "r") as f:
                    import json as _json
                    mock_history = [_json.load(line) for line in f]
        except Exception as e:
            logger.error("pdt_history_load_failed", {"error": str(e)})

        account_val = getattr(self, 'account_value', 100000.0)
        is_compliant, remaining = self.pdt.check_pdt_compliance(mock_history, account_val)
        if not is_compliant:
            logger.warn("pdt_lockdown_active", {"remaining": remaining})
            return {"status": "HALTED"}

        # 2. Weights already computed by _step_signals (inv-vol + sector cap)
        # Also pass sector_map to risk engine for secondary industry cap enforcement
        import json as _json2
        sym2sec_risk = {}
        sector_path = "data/sp500_sectors.json"
        if os.path.exists(sector_path):
            with open(sector_path) as f:
                sector_data = _json2.load(f)
            for sector, syms in sector_data.items():
                for s in syms:
                    if s not in sym2sec_risk:
                        sym2sec_risk[s] = sector
        positions = signals.copy()
        validated_positions = self.risk.validate_positions(positions, sector_map=sym2sec_risk)

        self.funnel_stats["passed_sizing"] = len(validated_positions)
        self.funnel_stats["passed_consensus"] = len(validated_positions)

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
        """Step 9: Execute target weights via VirtualBrokerClient.
        
        Uses the independent Virtual Broker to simulate real trading:
        - Delta reconciliation (only trades the difference)
        - Real cash management and P&L tracking
        - Persisted state in data/broker_api/account_state.json
        """
        targets_path = f"data/processed/targets_{trade_date}.parquet"
        if not os.path.exists(targets_path):
            return {"status": "no_targets", "orders": 0, "value_usd": 0}
            
        targets = read_parquet_safe(targets_path)
        targets = targets[targets['target_weight'].abs() >= 0.005].copy()
        if len(targets) == 0:
            return {"status": "no_targets", "orders": 0, "value_usd": 0}
            
        logger.info("daily_targets_generated", {
            "count": len(targets),
            "buy": int((targets['target_weight'] > 0).sum()),
            "sell": int((targets['target_weight'] < 0).sum())
        })
        
        try:
            from src.execution.virtual_broker import VirtualBrokerClient
            vbroker = VirtualBrokerClient()
            
            prices = self._get_latest_prices(trade_date)
            if not prices:
                logger.warn("no_prices_for_execution", {"date": trade_date})
                return {"status": "no_prices", "orders": 0, "value_usd": 0}
            
            account_summary = vbroker.get_account_summary(current_prices=prices)
            account_value = account_summary.get("nav", 100000.0)
            if account_value <= 0:
                account_value = getattr(self, 'account_value', 100000.0)
                
            # Extract current broker holdings
            current_positions_df = vbroker.get_positions()
            actual_holdings = {}
            if not current_positions_df.empty:
                for _, row in current_positions_df.iterrows():
                    actual_holdings[row['symbol']] = round(row['qty'])  # L1 fix
                    
            # Calculate target quantities from weights
            target_holdings = {}
            for _, row in targets.iterrows():
                sym = row['symbol']
                weight = row['target_weight']
                if abs(weight) < 0.001 or sym not in prices:
                    continue
                target_usd = abs(weight) * account_value
                qty = round(target_usd / prices[sym])  # L1 fix: round() not int()
                if qty > 0:
                    target_holdings[sym] = qty
                    
            # Delta reconciliation: generate orders
            orders = []
            
            # Sell positions no longer in targets (full close)
            for sym, actual_qty in actual_holdings.items():
                t_qty = target_holdings.get(sym, 0)
                delta = t_qty - actual_qty
                if delta < 0:
                    orders.append({
                        "symbol": sym, "qty": abs(delta), "side": "sell",
                    })
                    
            # Buy new or increase existing positions
            for sym, t_qty in target_holdings.items():
                actual_qty = actual_holdings.get(sym, 0)
                delta = t_qty - actual_qty
                if delta > 0:
                    orders.append({
                        "symbol": sym, "qty": delta, "side": "buy", "price": prices[sym]
                    })
                elif delta < 0:
                    # W2 FIX: Also generate sell orders for TWAP slicing
                    orders.append({
                        "symbol": sym, "qty": abs(delta), "side": "sell", "price": prices[sym]
                    })
            if len(orders) > 0:
                # Phase L3: Use TWAP Slicing for execution to simulate minimal market impact
                from src.execution.twap_executor import TwapExecutor
                
                # We need a small wrapper since TwapExecutor expects FutuExecutor but we can duck-type it 
                # to work with VirtualBrokerClient for the simulation backtest
                class VirtualTwapWrapper:
                    def submit_orders(self, chunk_orders):
                        return vbroker.submit_orders(chunk_orders, current_prices=prices)
                        
                # 5 minute window for simulation, max $10k per slice
                twap = TwapExecutor(VirtualTwapWrapper(), duration_minutes=5, max_slice_value=10000)
                
                # In simulation we won't actually sleep 5 minutes, we'll fast forward
                import time
                original_sleep = time.sleep
                time.sleep = lambda x: logger.debug("twap_sim_sleep", {"seconds": x})
                
                # R2 FIX: Use try/finally to guarantee sleep restoration
                try:
                    twap.execute_twap_batch(orders)
                finally:
                    time.sleep = original_sleep
                
                logger.info("virtual_broker_twap_executed", {
                    "parent_orders": len(orders),
                    "date": trade_date
                })
            else:
                logger.info("virtual_broker_no_delta", {"date": trade_date})
            
            final_summary = vbroker.get_account_summary(current_prices=prices)
            logger.info("virtual_broker_post_execution", {
                "nav": final_summary["nav"],
                "cash": final_summary["cash"],
                "market_value": final_summary["market_value"],
                "realized_pnl": final_summary["realized_pnl"]
            })
            
            return {
                "status": "executed",
                "orders": len(orders),
                "nav": final_summary["nav"],
                "cash": final_summary["cash"],
                "realized_pnl": final_summary["realized_pnl"]
            }
            
        except Exception as e:
            logger.error("virtual_broker_execution_failed", {"error": str(e)})
            return {"status": "execution_failed", "orders": 0, "error": str(e)}


    def _step_automl_retrain(self, trade_date: str) -> dict:
        """Phase L4: Disabled — LTR model no longer used.
        
        The v4 strategy uses price-derived factors (momentum, quality, value)
        which don't require model training. The LTR meta model has been
        proven to have zero alpha (Sharpe -2.579 post-audit).
        """
        logger.info("step_automl_retrain_disabled", {
            "date": trade_date,
            "reason": "v4_strategy_uses_factors_not_ml_model"
        })
        return {"status": "disabled", "reason": "v4_factor_strategy"}

if __name__ == "__main__":
    job = DailyJob()
    job.run()

