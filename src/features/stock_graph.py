"""
Phase I: Dynamic Stock Correlation Graph Builder

Constructs undirected correlation-based graphs for GNN embedding training.
Graphs are built from rolling return correlations with time-respecting windows.

AUDIT I-A1: Graph construction uses ONLY historical data (T-window to T-1). 
No T-day or future data is included.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from src.ops.event_logger import get_logger

logger = get_logger()


class StockGraphBuilder:
    """
    Build dynamic stock correlation graphs from rolling return data.
    
    For each date T, constructs an adjacency matrix from the rolling 
    correlation of log returns over the window [T-window, T-1].
    
    Parameters:
        window: Rolling window size in trading days (default 60)
        corr_threshold: Minimum absolute correlation to retain an edge (default 0.5)
        min_stocks: Minimum stocks needed to build a graph (default 10)
    """
    
    def __init__(
        self,
        window: int = 60,
        corr_threshold: float = 0.5,
        min_stocks: int = 10,
    ):
        self.window = window
        self.corr_threshold = corr_threshold
        self.min_stocks = min_stocks
    
    def build_graph(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """
        Build a correlation-based adjacency matrix for a target date.
        
        AUDIT I-A1: Uses returns from (target_date - window) to (target_date - 1 day).
        target_date is NOT included in the correlation window.
        
        Args:
            df: DataFrame with ['symbol', 'date', 'adj_close'] columns
            target_date: The date for which to build the graph
            
        Returns:
            Tuple of:
                - edge_index: (2, n_edges) numpy array of directed edge pairs
                - symbols: List of symbol names in node order
                - corr_matrix: (n_symbols, n_symbols) correlation matrix
        """
        # AUDIT I-A1: Strict cutoff at target_date (exclusive)
        hist = df[df['date'] < target_date].copy()
        
        if hist.empty:
            return np.empty((2, 0), dtype=int), [], np.empty((0, 0))
        
        # Keep only the most recent `window` trading days
        all_dates = sorted(hist['date'].unique())
        if len(all_dates) < self.window:
            recent_dates = all_dates
        else:
            recent_dates = all_dates[-self.window:]
        
        hist = hist[hist['date'].isin(recent_dates)]
        
        # Pivot to get returns matrix: (dates x symbols)
        price_pivot = hist.pivot_table(
            index='date', columns='symbol', values='adj_close'
        )
        
        # Drop symbols with too many missing prices (< 80% coverage)
        min_obs = int(len(recent_dates) * 0.8)
        valid_cols = price_pivot.columns[price_pivot.notna().sum() >= min_obs]
        price_pivot = price_pivot[valid_cols]
        
        if len(valid_cols) < self.min_stocks:
            return np.empty((2, 0), dtype=int), [], np.empty((0, 0))
        
        # Compute log returns
        log_returns = np.log(price_pivot / price_pivot.shift(1)).dropna()
        
        if len(log_returns) < 20:
            return np.empty((2, 0), dtype=int), list(valid_cols), np.eye(len(valid_cols))
        
        # Pearson correlation matrix
        corr_matrix = log_returns.corr().values
        symbols = list(valid_cols)
        n = len(symbols)
        
        # Threshold edges
        edge_src, edge_dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) >= self.corr_threshold:
                    # Undirected: add both directions
                    edge_src.extend([i, j])
                    edge_dst.extend([j, i])
        
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        
        return edge_index, symbols, corr_matrix
    
    def build_node_features(
        self,
        df: pd.DataFrame,
        symbols: list,
        target_date: pd.Timestamp,
        feature_cols: Optional[list] = None,
    ) -> np.ndarray:
        """
        Build node feature matrix for GNN input.
        
        Uses the features from the target_date for each symbol.
        Missing symbols get zero-filled features.
        
        Args:
            df: DataFrame with features
            symbols: List of symbols in graph node order
            target_date: Target date
            feature_cols: Feature columns to use (if None, uses defaults)
            
        Returns:
            (n_symbols, n_features) numpy array
        """
        if feature_cols is None:
            feature_cols = [
                'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d',
                'rv_5d', 'rv_20d', 'rv_60d',
                'rsi_14', 'adx_14',
                'macd_histogram_pct',
                'price_vs_sma20_zscore',
                'market_breadth',
            ]
        
        today = df[df['date'] == target_date]
        
        n_features = len(feature_cols)
        node_features = np.zeros((len(symbols), n_features))
        
        for i, sym in enumerate(symbols):
            sym_row = today[today['symbol'] == sym]
            if len(sym_row) > 0:
                for j, col in enumerate(feature_cols):
                    if col in sym_row.columns:
                        val = sym_row[col].iloc[0]
                        if np.isfinite(val):
                            node_features[i, j] = val
        
        return node_features
