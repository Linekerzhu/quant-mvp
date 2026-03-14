"""
Phase K: Sentiment Oracle for Dual-Oracle Architecture

Provides FinBERT-based sentiment analysis as a feature input to the LTR model.
Designed for offline batch processing with results cached as Parquet.

AUDIT K-A1: Sentiment is computed from T-day pre-market news only.
No T+1 or later news is used.

AUDIT K-A2: LLM hallucination guard — applies noise threshold filtering
to prevent extreme sentiment from neutral news.
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from src.ops.event_logger import get_logger

logger = get_logger()


class SentimentOracle:
    """
    Dual-Oracle sentiment feature generator.
    
    Architecture:
        1. Fetches news headlines for target symbols
        2. Computes sentiment score using FinBERT or simple rules-based fallback
        3. Applies noise filtering (K-A2)
        4. Exports as Parquet for build_features.py consumption
    
    In production, this runs as a daily pre-market batch job.
    The LTR model uses sentiment_score as an input feature (not a veto).
    """
    
    def __init__(
        self,
        noise_threshold: float = 0.3,
        cache_path: str = 'data/cache/sentiment_scores.parquet',
    ):
        """
        Args:
            noise_threshold: Minimum |sentiment_score| to be considered signal.
                            Below this threshold, sentiment is set to 0.0
                            (AUDIT K-A2: prevents LLM hallucination on neutral news)
            cache_path: Where to cache computed sentiment scores
        """
        self.noise_threshold = noise_threshold
        self.cache_path = cache_path
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy-load FinBERT model (only when needed)."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            logger.info(f"[Sentiment] FinBERT loaded: {model_name}")
        except ImportError:
            logger.info("[Sentiment] transformers not available, using rules-based fallback")
            self._model = "fallback"
    
    def score_text(self, text: str) -> float:
        """
        Score a single text for financial sentiment.
        
        Returns:
            sentiment_score ∈ [-1.0, +1.0]
            -1.0 = strongly negative
            +1.0 = strongly positive
            0.0 = neutral (or below noise threshold)
        """
        if self._model is None:
            self._load_model()
        
        if self._model == "fallback":
            return self._rules_based_score(text)
        
        import torch
        
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        
        # FinBERT output: [positive, negative, neutral]
        pos, neg, neu = probs[0], probs[1], probs[2]
        score = pos - neg  # [-1, +1]
        
        # AUDIT K-A2: Noise threshold filtering
        if abs(score) < self.noise_threshold:
            score = 0.0
        
        return float(score)
    
    @staticmethod
    def _rules_based_score(text: str) -> float:
        """
        Simple rules-based fallback when FinBERT is unavailable.
        Uses keyword matching on common financial sentiment words.
        """
        text_lower = text.lower()
        
        positive_words = [
            'beat', 'exceed', 'strong', 'growth', 'surge', 'rally',
            'upgrade', 'buy', 'outperform', 'bullish', 'positive',
            'record', 'high', 'rise', 'gain', 'profit', 'dividend',
        ]
        negative_words = [
            'miss', 'decline', 'weak', 'loss', 'plunge', 'crash',
            'downgrade', 'sell', 'underperform', 'bearish', 'negative',
            'risk', 'warning', 'cut', 'layoff', 'investigation',
        ]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        score = (pos_count - neg_count) / total
        return np.clip(score, -1.0, 1.0)
    
    def compute_batch_scores(
        self,
        news_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute sentiment scores for a batch of news items.
        
        AUDIT K-A1: news_df must have 'date' column representing when the news
        was published. Only pre-market news (before 9:30 AM) for date T is allowed.
        
        Args:
            news_df: DataFrame with columns ['symbol', 'date', 'headline']
            
        Returns:
            DataFrame with columns ['symbol', 'date', 'sentiment_score', 'n_headlines']
        """
        if news_df.empty:
            return pd.DataFrame(columns=['symbol', 'date', 'sentiment_score', 'n_headlines'])
        
        results = []
        
        for (symbol, date), group in news_df.groupby(['symbol', 'date']):
            scores = []
            for _, row in group.iterrows():
                headline = str(row.get('headline', ''))
                if headline and len(headline) > 5:
                    score = self.score_text(headline)
                    scores.append(score)
            
            if scores:
                # Aggregate: mean of non-zero scores, or 0 if all are noise
                non_zero = [s for s in scores if abs(s) >= self.noise_threshold]
                avg_score = np.mean(non_zero) if non_zero else 0.0
            else:
                avg_score = 0.0
            
            results.append({
                'symbol': symbol,
                'date': pd.Timestamp(date),
                'sentiment_score': float(avg_score),
                'n_headlines': len(scores),
            })
        
        return pd.DataFrame(results)
    
    def generate_synthetic_sentiment(
        self,
        df: pd.DataFrame,
        correlation_with_returns: float = 0.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate SAFE synthetic sentiment scores for backtesting.
        
        ⚠️  AUDIT C5: Previous version used shift(-5) future returns — GOD'S EYE LEAKAGE.
        This version uses ONLY historical momentum as a sentiment proxy.
        
        Args:
            df: DataFrame with ['symbol', 'date', 'adj_close']
            correlation_with_returns: MUST be 0.0 — positive raises error
            seed: Random seed
        """
        if correlation_with_returns > 0:
            raise RuntimeError(
                "AUDIT C5: correlation_with_returns > 0 uses future returns = LOOK-AHEAD BIAS. "
                "Use correlation_with_returns=0 for safe synthetic data."
            )
        
        np.random.seed(seed)
        df = df.copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Use PAST momentum as sentiment proxy (no future data)
        df['_past_ret'] = df.groupby('symbol')['adj_close'].transform(
            lambda x: x.pct_change(5).fillna(0)
        )
        
        results = []
        for (symbol, date), group in df.groupby(['symbol', 'date']):
            past_ret = group['_past_ret'].iloc[0]
            # Pure noise with slight past-momentum signal
            noise = np.random.randn() * 0.3
            sentiment = 0.1 * np.sign(past_ret) * min(abs(past_ret) * 5, 1.0) + 0.9 * noise
            sentiment = np.clip(sentiment, -1.0, 1.0)
            
            if abs(sentiment) < self.noise_threshold:
                sentiment = 0.0
            
            results.append({
                'symbol': symbol,
                'date': pd.Timestamp(date),
                'sentiment_score': float(sentiment),
            })
        
        sent_df = pd.DataFrame(results)
        
        # Save cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        sent_df.to_parquet(self.cache_path, index=False)
        
        n_nonzero = (sent_df['sentiment_score'] != 0).sum()
        print(f"[Sentiment] Generated {len(sent_df)} SAFE synthetic scores (no future leakage), "
              f"{n_nonzero} non-zero ({n_nonzero/len(sent_df)*100:.1f}%)")
        
        return sent_df
    
    @staticmethod
    def merge_sentiment(
        df: pd.DataFrame,
        sentiment_path: str = 'data/cache/sentiment_scores.parquet',
    ) -> pd.DataFrame:
        """
        Merge cached sentiment scores into feature DataFrame.
        
        AUDIT K-A1: This reads pre-computed sentiment. The computation
        step (compute_batch_scores or generate_synthetic) must ensure
        only pre-market news is used.
        
        Args:
            df: Feature DataFrame with ['symbol', 'date']
            sentiment_path: Path to sentiment Parquet
            
        Returns:
            DataFrame with 'sentiment_score' column added
        """
        if not os.path.exists(sentiment_path):
            df['sentiment_score'] = 0.0
            return df
        
        sent_df = pd.read_parquet(sentiment_path)
        sent_df['date'] = pd.to_datetime(sent_df['date'])
        
        n_before = len(df)
        df = df.merge(
            sent_df[['symbol', 'date', 'sentiment_score']],
            on=['symbol', 'date'], how='left'
        )
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        
        assert len(df) == n_before, (
            f"Sentiment merge changed row count: {n_before} → {len(df)}"
        )
        
        n_nonzero = (df['sentiment_score'] != 0).sum()
        print(f"[Sentiment] Merged, {n_nonzero}/{len(df)} rows have non-zero sentiment")
        
        return df
