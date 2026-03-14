"""
Hierarchical Risk Parity (HRP) Optimizer
Phase L2: Portfolio Variance Optimization

Distributes capital such that highly correlated stock clusters 
contribute equally to portfolio risk, avoiding Beta concentration.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional
import warnings

from src.ops.event_logger import get_logger

logger = get_logger()

class HierarchicalRiskParity:
    """
    Implements Hierarchical Risk Parity to adjust position weights.
    Takes directional conviction weights (from LTR/RL/Kelly) and scales them
    inversely to their cluster risk contribution.
    """
    
    def __init__(self, history_window: int = 120, max_weight: float = 0.15):
        self.history_window = history_window
        self.max_weight = max_weight
        
    def _correl_dist(self, corr: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance matrix from correlation matrix."""
        # Distance metric: d = sqrt(0.5 * (1 - r))
        # Clip to avoid float precision issues leading to negative sqrt
        dist = np.sqrt(np.clip((1 - corr) / 2., 0.0, 1.0))
        return dist
        
    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Sort clustered items by distance."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # Make space
            df0 = sort_ix[sort_ix >= num_items]  # Find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # Item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])  # Item 2
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])  # Re-index
            
        return sort_ix.tolist()
        
    def _get_cluster_var(self, cov: pd.DataFrame, c_items: List[int]) -> float:
        """Calculate variance of a cluster."""
        cov_slice = cov.iloc[c_items, c_items]
        # Inverse variance weights for cluster elements
        ivp = 1. / np.diag(cov_slice)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return c_var
        
    def _get_rec_bipart(self, cov: pd.DataFrame, sort_ix: List[int]) -> pd.Series:
        """Compute HRP weights via recursive bisection."""
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]  # initial all items in one cluster
        
        while len(c_items) > 0:
            c_items_new = []
            for i in range(0, len(c_items)):
                c = c_items[i]
                if len(c) == 1:
                    continue
                    
                # Split in half
                half = int(len(c) / 2)
                c_1 = c[:half]
                c_2 = c[half:]
                c_items_new.append(c_1)
                c_items_new.append(c_2)
                
                # Cluster variances
                c_var1 = self._get_cluster_var(cov, c_1)
                c_var2 = self._get_cluster_var(cov, c_2)
                
                # Allocation factor
                alpha = 1 - c_var1 / (c_var1 + c_var2)
                
                w[c_1] *= alpha
                w[c_2] *= (1 - alpha)
                
            c_items = c_items_new
            
        return w
        
    def optimize(self, 
                 raw_weights: Dict[str, float], 
                 returns_history: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize weights using HRP.
        
        Args:
            raw_weights: Dict mapping symbol -> target_weight (from base sizer)
                         Preserves direction (sign) and magnitude conviction.
            returns_history: DataFrame with columns = symbols, rows = dates (daily returns)
                             Must contain at least self.history_window rows.
                             
        Returns:
            Dict mapping symbol -> optimized target_weight
        """
        if not raw_weights:
            return {}
            
        symbols = list(raw_weights.keys())
        if len(symbols) <= 2:
            # HRP needs at least 3 items to cluster effectively.
            # If <=2, just cap max weight and return
            return {s: np.sign(w) * min(abs(w), self.max_weight) 
                    for s, w in raw_weights.items()}
            
        # 1. Filter history for target symbols
        avail_syms = [s for s in symbols if s in returns_history.columns]
        missing = set(symbols) - set(avail_syms)
        
        if missing:
            logger.warn("hrp_missing_history", {"symbols": list(missing)})
            
        if len(avail_syms) <= 2:
            logger.warn("hrp_fallback_insufficient_data", {"count": len(avail_syms)})
            return raw_weights
            
        # Use recent history (tail)
        hist_df = returns_history[avail_syms].tail(self.history_window).copy()
        
        # Guard against zero-variance columns (flatlines)
        hist_vars = hist_df.var()
        flat_syms = hist_vars[hist_vars < 1e-8].index.tolist()
        if flat_syms:
            logger.warn("hrp_flatline_symbols", {"symbols": flat_syms})
            # W1 FIX: Add tiny noise with fixed seed for reproducibility
            rng = np.random.RandomState(42)
            for fs in flat_syms:
                hist_df[fs] += rng.normal(0, 1e-6, len(hist_df))
                
        # 2. Compute Covariance and Correlation
        cov = hist_df.cov()
        corr = hist_df.corr()
        
        # Check for NaNs (can happen if array is constant but missed by var check)
        if corr.isna().values.any():
            corr = corr.fillna(0.0)
            cov = cov.fillna(1e-6)
            
        # 3. Distance Matrix & Linkage
        dist = self._correl_dist(corr)
        
        # Scipy linkage expects condensed distance matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fill diag with 0 before squareform
            np.fill_diagonal(dist.values, 0.0)
            condensed_dist = squareform(dist)
            link = linkage(condensed_dist, method='single')
            
        # 4. Quasi-Diagonalization (Sort indices)
        sort_ix = self._get_quasi_diag(link)
        
        # 5. Recursive Bisection (HRP Weights)
        # HRP gives us non-negative weights that sum to 1
        hrp_weights = self._get_rec_bipart(cov, sort_ix)
        
        # Map back to symbols
        sym_indices = {i: sym for i, sym in enumerate(avail_syms)}
        hrp_alloc = {sym_indices[i]: weight for i, weight in hrp_weights.items()}
        
        # 6. Re-integrate Conviction & Direction
        # HRP weights represent optimal risk allocation (always positive, sum=1)
        # Raw weights represent conviction and direction.
        # We blend them: Final = Direction * sqrt(Abs(Raw)) * HRP_Weight
        # Normalizing to match original target gross exposure.
        
        final_weights = {}
        target_gross = sum(abs(w) for w in raw_weights.values())
        
        blended = {}
        for sym in avail_syms:
            raw = raw_weights[sym]
            direction = np.sign(raw)
            # Square root of raw conviction to suppress extreme raw signals
            conviction = np.sqrt(abs(raw)) 
            risk_w = hrp_alloc[sym]
            
            blended[sym] = direction * conviction * risk_w
            
        # For missing symbols (no history), give them average HRP weight
        avg_hrp = 1.0 / len(avail_syms) if avail_syms else 0
        for sym in missing:
            raw = raw_weights[sym]
            direction = np.sign(raw)
            conviction = np.sqrt(abs(raw))
            blended[sym] = direction * conviction * avg_hrp
            
        # Normalize to target gross exposure
        current_gross = sum(abs(w) for w in blended.values())
        if current_gross > 0:
            scale = target_gross / current_gross
            for sym in symbols:
                w = blended[sym] * scale
                # Apply single position cap
                sign = 1 if w > 0 else -1 if w < 0 else 0
                final_weights[sym] = sign * min(abs(w), self.max_weight)
        else:
            final_weights = {sym: 0.0 for sym in symbols}
            
        logger.info("hrp_optimization_complete", {
            "n_symbols": len(symbols),
            "target_gross": round(target_gross, 3),
            "actual_gross": round(sum(abs(w) for w in final_weights.values()), 3)
        })
        
        return final_weights
