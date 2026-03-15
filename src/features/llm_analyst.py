"""
LLM Analyst — DeepSeek 定性研判层

On rebalance day, sends candidate stocks + their fundamentals to DeepSeek LLM
for qualitative analysis. The LLM acts as a "qualitative risk expert" that can
flag risks invisible to quantitative factors:
  - Lawsuits, regulatory investigations
  - Management changes, accounting scandals
  - Geopolitical exposure (sanctions, tariffs)
  - Thematic alignment (AI, green energy, defense, etc.)
  - Earnings quality concerns

Output: Per-stock sentiment score (-1 to +1) and brief rationale.

Usage:
    from src.features.llm_analyst import LLMAnalyst
    analyst = LLMAnalyst(api_key="sk-...")
    results = analyst.analyze_candidates(candidates_df, macro_context)
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)


ANALYSIS_PROMPT = """你是一个资深美股投研分析师。现在是{date}。

我的量化系统通过技术面+基本面筛选出了以下{n}只候选股票，准备持有一个月。
请你从定性角度逐一研判，找出量化因子无法捕捉的风险和机会。

候选股票及其数据：
{stocks_table}

当前宏观环境：
{macro_context}

请对每只股票给出：
1. sentiment: 情绪评分 (-1.0到+1.0, 负数=看空/有风险, 正数=看好)
2. flag: 风险标签 (safe/caution/danger)  
3. themes: 关联的市场主题 (如: AI, 新能源, 国防, 医药创新 等)
4. reason: 一句话理由 (≤30字)

重点关注：
- 近期是否有诉讼、监管调查、管理层丑闻
- 是否处于当前市场追捧的热门赛道
- 地缘政治风险（中国制裁、关税）
- 盈利质量是否可持续
- 行业竞争格局变化

请严格按以下JSON格式输出，不要输出其他内容：
```json
{{
  "market_view": "对当前市场的一句话判断",
  "stocks": {{
    "AAPL": {{"sentiment": 0.5, "flag": "safe", "themes": ["AI", "消费电子"], "reason": "..."}},
    ...
  }}
}}
```"""


class LLMAnalyst:
    """Uses DeepSeek LLM for qualitative stock analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            # Try loading from .env
            env_path = os.path.join(PROJECT_ROOT, ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.strip().startswith("DEEPSEEK_API_KEY="):
                            self.api_key = line.strip().split("=", 1)[1]
                            break
    
    def analyze_candidates(
        self,
        candidates: pd.DataFrame,
        macro_context: str = "",
        date: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        Analyze candidate stocks using DeepSeek LLM.
        
        Args:
            candidates: DataFrame with columns: symbol, earnings_yield, roe, 
                       analyst_upside, analyst_consensus, earnings_growth, etc.
            macro_context: Text description of current macro environment
            date: Current date string
            
        Returns:
            Dict[symbol] -> {sentiment, flag, themes, reason}
        """
        if not self.api_key:
            print("[LLM] No DeepSeek API key, skipping analysis")
            return {}
        
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        # Build stocks table for prompt
        def _safe(val, default=0):
            """Convert NaN/None to default. Python's `or` doesn't catch NaN."""
            if val is None:
                return default
            try:
                import math
                if math.isnan(val):
                    return default
            except (TypeError, ValueError):
                pass
            return val
        
        table_lines = []
        for _, r in candidates.iterrows():
            sym = r.get("symbol", "?")
            ey = _safe(r.get("earnings_yield"), 0)
            roe = _safe(r.get("roe"), 0)
            margin = _safe(r.get("profit_margin"), 0)
            upside = _safe(r.get("analyst_upside"), 0)
            consensus = _safe(r.get("analyst_consensus"), 0)
            eg = _safe(r.get("earnings_growth"), 0)
            es = _safe(r.get("earnings_surprise"), 0)
            mom = _safe(r.get("mom_12_1"), 0)
            score = _safe(r.get("composite_score"), 0)
            
            rating_map = {4: "强买", 3: "买入", 2: "持有", 1: "卖出"}
            rating = rating_map.get(round(consensus), f"{consensus:.1f}")
            
            table_lines.append(
                f"  {sym:6s} | PE收益率{ey:.3f} | ROE{roe:.1%} | 利润率{margin:.1%} | "
                f"分析师{rating}(upside{upside:+.0%}) | "
                f"盈利增长{eg:+.0%} | 12月动量{mom:+.0%} | 综合分{score:.3f}"
            )
        
        stocks_table = "\n".join(table_lines)
        
        prompt = ANALYSIS_PROMPT.format(
            date=date,
            n=len(candidates),
            stocks_table=stocks_table,
            macro_context=macro_context or "无特别宏观信息",
        )
        
        # Call DeepSeek
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 2000,
                },
                timeout=30,
            )
            resp.raise_for_status()
            
            content = resp.json()["choices"][0]["message"]["content"].strip()
            
            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            
            stocks = result.get("stocks", {})
            market_view = result.get("market_view", "")
            
            print(f"[LLM] DeepSeek analyzed {len(stocks)} stocks")
            if market_view:
                print(f"[LLM] Market view: {market_view}")
            
            # Validate and normalize sentiments
            for sym in list(stocks.keys()):
                s = stocks[sym]
                s["sentiment"] = max(-1.0, min(1.0, float(s.get("sentiment", 0))))
                s["flag"] = s.get("flag", "safe")
                s["themes"] = s.get("themes", [])
                s["reason"] = s.get("reason", "")[:100]
            
            # Save analysis
            output = {
                "date": date,
                "market_view": market_view,
                "stocks": stocks,
            }
            save_path = os.path.join(PROJECT_ROOT, f"data/processed/llm_analysis_{date}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            return stocks
            
        except Exception as e:
            print(f"[LLM] DeepSeek analysis failed: {e}")
            return {}
    
    @staticmethod
    def apply_sentiment(
        candidates: pd.DataFrame,
        llm_results: Dict[str, dict],
        weight: float = 0.10,
    ) -> pd.DataFrame:
        """
        Adjust composite scores based on LLM sentiment.
        
        - Danger flag → reduce score by 20%
        - Negative sentiment → reduce proportionally  
        - Positive sentiment → boost proportionally
        - weight controls how much LLM affects final score
        """
        if not llm_results:
            return candidates
        
        df = candidates.copy()
        
        for idx, row in df.iterrows():
            sym = row["symbol"]
            if sym in llm_results:
                analysis = llm_results[sym]
                sentiment = analysis.get("sentiment", 0)
                flag = analysis.get("flag", "safe")
                
                # Apply sentiment adjustment
                adjustment = 1.0 + (sentiment * weight)
                
                # Danger flag: additional penalty (conservative — LLM may hallucinate)
                if flag == "danger":
                    adjustment *= 0.90  # -10% (not -20%, to limit hallucination damage)
                elif flag == "caution":
                    adjustment *= 0.95  # -5%
                
                if "composite_score" in df.columns:
                    df.at[idx, "composite_score"] *= adjustment
        
        return df
