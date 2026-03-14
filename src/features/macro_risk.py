"""
Macro Risk Monitor — 宏观经济风险评估

Aggregates macro indicators to produce a risk score (0-100) that determines
overall equity exposure. Higher score = more risk = lower exposure.

Indicators:
  1. Yield Curve (10Y-3M spread): Inversion = recession signal
  2. Credit Stress (HYG drawdown): Distressed = corporate risk
  3. Dollar Strength (DXY trend): Strong dollar = EM/export pressure  
  4. Oil Shock (CL spike): Sudden > 30% = inflation/stagflation risk
  5. Gold Signal (GC trend): Safe-haven rush = risk-off
  6. VIX Level: Already used separately, but included in composite
  7. Rate Environment (TNX level + trend): Rising rates = tighter conditions

Usage:
    from src.features.macro_risk import MacroRiskMonitor
    mrm = MacroRiskMonitor()
    score, details = mrm.get_risk_score()
    exposure_scale = mrm.score_to_exposure(score)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)


class MacroRiskMonitor:
    """
    Computes a macro risk score [0-100] from freely available market data.
    
    Score interpretation:
      0-20:  Low risk → 100% exposure
      20-40: Moderate → 90% exposure  
      40-60: Elevated → 70% exposure
      60-80: High → 50% exposure
      80-100: Crisis → 25% exposure
    """
    
    CACHE_DIR = os.path.join(PROJECT_ROOT, "data/cache/macro")
    
    INDICATORS = {
        "^TNX":     "10y_yield",     # 10-year Treasury yield
        "^IRX":     "3m_yield",      # 3-month T-bill yield
        "DX-Y.NYB": "dxy",           # Dollar Index
        "GC=F":     "gold",          # Gold futures
        "CL=F":     "oil",           # Crude oil
        "HYG":      "hyg",           # High yield bond ETF
        "TLT":      "tlt",           # Long Treasury ETF
        "^VIX":     "vix",           # Volatility index
    }
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        os.makedirs(self.CACHE_DIR, exist_ok=True)
    
    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """Download macro indicator data from yfinance."""
        import yfinance as yf
        
        data = {}
        for sym, name in self.INDICATORS.items():
            try:
                df = yf.download(sym, period="1y", progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel('Ticker', axis=1)
                df = df[['Close']].rename(columns={'Close': name})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data[name] = df
            except Exception as e:
                print(f"[MACRO] Failed to download {sym}: {e}")
        
        return data
    
    def get_risk_score(self) -> Tuple[float, Dict[str, dict]]:
        """
        Calculate composite macro risk score.
        
        Returns:
            (score: float 0-100, details: dict of sub-scores)
        """
        data = self._download_data()
        details = {}
        sub_scores = []
        
        # 1. Yield Curve Signal (10Y - 3M spread)
        # Inverted = recession warning (15% weight)
        if "10y_yield" in data and "3m_yield" in data:
            y10 = data["10y_yield"].iloc[-1].values[0]
            y3m = data["3m_yield"].iloc[-1].values[0]
            spread = y10 - y3m
            
            if spread < -0.5:
                yc_score = 100  # Deeply inverted
            elif spread < 0:
                yc_score = 70   # Mildly inverted
            elif spread < 0.5:
                yc_score = 40   # Flat (cautious)
            elif spread < 1.5:
                yc_score = 15   # Normal
            else:
                yc_score = 5    # Steep (growth expected)
            
            details["yield_curve"] = {
                "spread": round(spread, 2),
                "score": yc_score,
                "signal": "inverted" if spread < 0 else "normal",
            }
            sub_scores.append(("yield_curve", yc_score, 0.20))
        
        # 2. Credit Stress (HYG drawdown from 52-week high)
        # HYG falling = corporate stress (15% weight)
        if "hyg" in data:
            hyg = data["hyg"]["hyg"]
            hyg_peak = hyg.rolling(252, min_periods=20).max().iloc[-1]
            hyg_dd = (hyg_peak - hyg.iloc[-1]) / hyg_peak * 100
            
            if hyg_dd > 10:
                credit_score = 100
            elif hyg_dd > 5:
                credit_score = 70
            elif hyg_dd > 3:
                credit_score = 40
            else:
                credit_score = 10
            
            details["credit_stress"] = {
                "hyg_drawdown_pct": round(float(hyg_dd), 1),
                "score": credit_score,
            }
            sub_scores.append(("credit_stress", credit_score, 0.15))
        
        # 3. Dollar Strength (DXY 3-month change)
        # Rapidly strengthening dollar = risk off (10% weight)
        if "dxy" in data:
            dxy = data["dxy"]["dxy"]
            if len(dxy) > 63:
                dxy_chg = (dxy.iloc[-1] / dxy.iloc[-63] - 1) * 100
            else:
                dxy_chg = 0
            
            if dxy_chg > 8:
                dxy_score = 80
            elif dxy_chg > 4:
                dxy_score = 50
            elif dxy_chg > 0:
                dxy_score = 20
            else:
                dxy_score = 10  # Weakening dollar = risk on
            
            details["dollar"] = {
                "dxy_3m_change_pct": round(float(dxy_chg), 1),
                "score": dxy_score,
            }
            sub_scores.append(("dollar", dxy_score, 0.10))
        
        # 4. Oil Shock (3-month price change)
        # Sudden spike > 30% = inflation/supply shock (10% weight)
        if "oil" in data:
            oil = data["oil"]["oil"]
            if len(oil) > 63:
                oil_chg = (oil.iloc[-1] / oil.iloc[-63] - 1) * 100
            else:
                oil_chg = 0
            
            if oil_chg > 40:
                oil_score = 90
            elif oil_chg > 20:
                oil_score = 60
            elif oil_chg > 10:
                oil_score = 30
            elif oil_chg < -30:
                oil_score = 50  # Crash = demand destruction signal
            else:
                oil_score = 10
            
            details["oil"] = {
                "oil_3m_change_pct": round(float(oil_chg), 1),
                "score": oil_score,
            }
            sub_scores.append(("oil", oil_score, 0.10))
        
        # 5. Gold Signal (3-month change)
        # Rapidly rising gold = flight to safety (10% weight)
        if "gold" in data:
            gold = data["gold"]["gold"]
            if len(gold) > 63:
                gold_chg = (gold.iloc[-1] / gold.iloc[-63] - 1) * 100
            else:
                gold_chg = 0
            
            if gold_chg > 15:
                gold_score = 60
            elif gold_chg > 8:
                gold_score = 35
            else:
                gold_score = 10
            
            details["gold"] = {
                "gold_3m_change_pct": round(float(gold_chg), 1),
                "score": gold_score,
            }
            sub_scores.append(("gold", gold_score, 0.10))
        
        # 6. VIX Level (20% weight — most responsive)
        if "vix" in data:
            vix = float(data["vix"]["vix"].iloc[-1])
            
            if vix > 50:
                vix_score = 100
            elif vix > 35:
                vix_score = 80
            elif vix > 25:
                vix_score = 50
            elif vix > 18:
                vix_score = 20
            else:
                vix_score = 5
            
            details["vix"] = {
                "level": round(vix, 1),
                "score": vix_score,
            }
            sub_scores.append(("vix", vix_score, 0.20))
        
        # 7. Rate Environment (10Y level + trend)
        # High and rising rates = tighter conditions (5% weight)
        if "10y_yield" in data:
            tnx = data["10y_yield"]["10y_yield"]
            y10_now = float(tnx.iloc[-1])
            if len(tnx) > 63:
                y10_3m_ago = float(tnx.iloc[-63])
                rate_rising = y10_now > y10_3m_ago
            else:
                rate_rising = False
            
            if y10_now > 5.0 and rate_rising:
                rate_score = 80
            elif y10_now > 4.5:
                rate_score = 50
            elif y10_now > 3.5:
                rate_score = 25
            else:
                rate_score = 10
            
            details["rates"] = {
                "10y_yield": round(y10_now, 2),
                "rising": rate_rising,
                "score": rate_score,
            }
            sub_scores.append(("rates", rate_score, 0.05))
        
        # Composite weighted score
        if sub_scores:
            total_weight = sum(w for _, _, w in sub_scores)
            composite = sum(s * w for _, s, w in sub_scores) / total_weight
        else:
            composite = 50  # No data = moderate risk
        
        composite = round(min(max(composite, 0), 100), 1)
        
        return composite, details
    
    @staticmethod
    def score_to_exposure(score: float) -> float:
        """
        Convert macro risk score to equity exposure multiplier.
        
        0-20:  1.00 (full exposure)
        20-40: 0.90  
        40-60: 0.70
        60-80: 0.50
        80-100: 0.25
        """
        if score <= 20:
            return 1.00
        elif score <= 40:
            return 0.90
        elif score <= 60:
            return 0.70
        elif score <= 80:
            return 0.50
        else:
            return 0.25
    
    def print_dashboard(self):
        """Print a human-readable macro risk dashboard."""
        score, details = self.get_risk_score()
        exposure = self.score_to_exposure(score)
        
        print("\n" + "=" * 50)
        print("  📊 MACRO RISK DASHBOARD")
        print("=" * 50)
        
        for name, d in details.items():
            s = d.get("score", 0)
            icon = "🟢" if s < 30 else "🟡" if s < 60 else "🔴"
            
            if name == "yield_curve":
                print(f"  {icon} Yield Curve: {d['spread']:+.2f}% ({d['signal']})")
            elif name == "credit_stress":
                print(f"  {icon} Credit Stress: HYG DD {d['hyg_drawdown_pct']:.1f}%")
            elif name == "dollar":
                print(f"  {icon} Dollar (DXY): {d['dxy_3m_change_pct']:+.1f}% (3m)")
            elif name == "oil":
                print(f"  {icon} Oil: {d['oil_3m_change_pct']:+.1f}% (3m)")
            elif name == "gold":
                print(f"  {icon} Gold: {d['gold_3m_change_pct']:+.1f}% (3m)")
            elif name == "vix":
                print(f"  {icon} VIX: {d['level']:.0f}")
            elif name == "rates":
                print(f"  {icon} 10Y Rate: {d['10y_yield']:.2f}% {'↑' if d['rising'] else '↓'}")
        
        risk_label = (
            "低风险" if score < 20 else
            "温和" if score < 40 else
            "偏高" if score < 60 else
            "高风险" if score < 80 else
            "危机"
        )
        
        print(f"\n  综合风险: {score:.0f}/100 ({risk_label})")
        print(f"  建议仓位: {exposure:.0%}")
        print("=" * 50)
