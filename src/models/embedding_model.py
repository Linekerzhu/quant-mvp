"""
Phase I (v2): GNN Stock Embedding Model — GAT + Ranking Loss

MAJOR UPGRADES from v1:
1. GAT (Graph Attention Network) replaces GraphSAGE → learns neighbor importance
2. Pairwise BPR ranking loss → directly optimizes for cross-sectional ordering
3. 3-layer architecture with BatchNorm → deeper feature extraction
4. Residual connections → stable training for deeper networks
5. Longer default correlation window (60d) → denser, more meaningful graphs

AUDIT I-A3: Offline only. Production reads Parquet embeddings (no torch dependency).
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GATEmbedder(nn.Module if HAS_TORCH else object):
    """
    3-layer GAT with residual connections and batch normalization.
    
    Architecture:
        Input → GATConv(heads=4) → BN → ELU → Residual
              → GATConv(heads=4) → BN → ELU → Residual  
              → GATConv(heads=1) → Embedding(16)
        Embedding → Linear(1) → ranking score
    
    Key improvements over GraphSAGE v1:
    - Attention mechanism learns which correlated stocks matter most
    - Multi-head attention captures different types of relationships
    - Batch normalization stabilizes training across different graph sizes
    - Residual connections prevent gradient vanishing in deeper networks
    """
    
    def __init__(
        self, 
        in_features: int = 12, 
        hidden_dim: int = 32, 
        embed_dim: int = 16, 
        heads: int = 4,
        dropout: float = 0.2,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch and PyTorch Geometric required")
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Layer 1: Multi-head attention
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.proj1 = nn.Linear(in_features, hidden_dim * heads)  # residual projection
        
        # Layer 2: Multi-head attention
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        # residual: identity (same dim)
        
        # Layer 3: Single-head → embedding dimension
        self.conv3 = GATConv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.proj3 = nn.Linear(hidden_dim * heads, embed_dim)  # residual projection
        
        # Ranking head
        self.rank_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """Forward pass returning embeddings (16-dim)."""
        # Layer 1 with residual
        residual = self.proj1(x)
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h + residual)
        h = self.dropout(h)
        
        # Layer 2 with residual  
        residual = h
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h + residual)
        h = self.dropout(h)
        
        # Layer 3 → embedding
        residual = self.proj3(h)
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = h + residual  # no activation on final embedding
        
        return h
    
    def rank_score(self, x, edge_index):
        """Forward pass returning ranking scores for pairwise loss."""
        h = self.forward(x, edge_index)
        return self.rank_head(h).squeeze(-1)


class GNNEmbeddingTrainer:
    """
    Offline GNN training with pairwise ranking loss (BPR).
    
    v2 improvements:
    - Pairwise BPR loss: for each (stock_i, stock_j) where ret_i > ret_j,
      maximize score_i - score_j. This directly learns cross-sectional ranking.
    - GAT architecture with attention heads
    - Longer correlation window (60d default)
    
    AUDIT I-A1: Training uses T+5 returns as LABELS only. No look-ahead in features.
    AUDIT I-A3: Exports Parquet for production (no torch in production).
    """
    
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        lr: float = 0.003,
        epochs: int = 100,
        graph_window: int = 60,
        corr_threshold: float = 0.3,
        fwd_days: int = 5,
        n_pairs: int = 50,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GNN training")
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.graph_window = graph_window
        self.corr_threshold = corr_threshold
        self.fwd_days = fwd_days
        self.n_pairs = n_pairs  # pairs per graph for BPR loss
        
        self.model = None
        self.symbol_list = None
    
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ) -> Dict:
        """
        Train GAT embeddings with pairwise ranking loss.
        """
        from src.features.stock_graph import StockGraphBuilder
        
        if feature_cols is None:
            feature_cols = [
                'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d',
                'rv_5d', 'rv_20d', 'rv_60d',
                'rsi_14', 'adx_14',
                'macd_histogram_pct',
                'price_vs_sma20_zscore',
                'market_breadth',
            ]
        
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < 3:
            raise ValueError(f"Too few features available: {available_cols}")
        feature_cols = available_cols
        
        n_features = len(feature_cols)
        graph_builder = StockGraphBuilder(
            window=self.graph_window,
            corr_threshold=self.corr_threshold,
        )
        
        all_dates = sorted(df['date'].unique())
        valid_dates = all_dates[self.graph_window:-self.fwd_days]
        
        if len(valid_dates) < 20:
            raise ValueError(f"Insufficient dates for GNN training: {len(valid_dates)}")
        
        n_train = int(len(valid_dates) * train_ratio)
        train_dates = valid_dates[:n_train]
        val_dates = valid_dates[n_train:]
        
        print(f"[GNN-v2] Training dates: {len(train_dates)}, Val dates: {len(val_dates)}")
        print(f"[GNN-v2] Features: {n_features}, Embed dim: {self.embed_dim}, Architecture: GAT")
        
        # Build graph snapshots with continuous return labels (not binary)
        print("[GNN-v2] Building graph snapshots...")
        train_graphs = self._build_graph_snapshots(df, graph_builder, train_dates, feature_cols)
        val_graphs = self._build_graph_snapshots(df, graph_builder, val_dates, feature_cols)
        
        print(f"[GNN-v2] Built {len(train_graphs)} train + {len(val_graphs)} val graphs")
        
        if not train_graphs:
            raise ValueError("No valid training graphs could be built")
        
        # Initialize GAT model
        self.model = GATEmbedder(
            in_features=n_features,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 20
        no_improve = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            n_graphs = 0
            
            for graph_data in train_graphs:
                optimizer.zero_grad()
                
                scores = self.model.rank_score(graph_data.x, graph_data.edge_index)
                loss = self._bpr_loss(scores, graph_data.y)
                
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                n_graphs += 1
            
            train_loss /= max(n_graphs, 1)
            scheduler.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_rank_corr = 0
            n_val = 0
            
            with torch.no_grad():
                for graph_data in val_graphs:
                    scores = self.model.rank_score(graph_data.x, graph_data.edge_index)
                    loss = self._bpr_loss(scores, graph_data.y)
                    val_loss += loss.item()
                    
                    # Rank correlation as metric
                    from scipy.stats import spearmanr
                    s = scores.numpy()
                    y = graph_data.y.numpy()
                    if len(np.unique(y)) > 1:
                        rc, _ = spearmanr(s, y)
                        if not np.isnan(rc):
                            val_rank_corr += rc
                    n_val += 1
            
            val_loss /= max(n_val, 1)
            val_rank_corr /= max(n_val, 1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} rank_corr={val_rank_corr:.4f}")
            
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} (best={best_epoch+1})")
                break
        
        self.model.load_state_dict(best_state)
        
        return {
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'final_val_rank_corr': val_rank_corr,
            'n_train_graphs': len(train_graphs),
            'n_val_graphs': len(val_graphs),
            'embed_dim': self.embed_dim,
        }
    
    def _bpr_loss(self, scores: 'torch.Tensor', returns: 'torch.Tensor') -> 'torch.Tensor':
        """
        Bayesian Personalized Ranking (BPR) pairwise loss.
        
        For each pair (i, j) where return_i > return_j,
        loss = -log(sigmoid(score_i - score_j))
        
        This directly teaches the embedding to produce higher scores
        for stocks with higher returns — exactly what LTR needs.
        """
        n = len(scores)
        if n < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample pairs
        n_pairs = min(self.n_pairs, n * (n - 1) // 2)
        
        # Get indices sorted by returns (descending)
        sorted_idx = torch.argsort(returns, descending=True)
        
        total_loss = torch.tensor(0.0)
        count = 0
        
        for k in range(min(n_pairs, n - 1)):
            i = sorted_idx[k]      # higher return
            j = sorted_idx[-(k+1)] # lower return
            
            if returns[i] > returns[j]:
                diff = scores[i] - scores[j]
                total_loss = total_loss + F.softplus(-diff)
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return total_loss / count
    
    def generate_embeddings(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate embeddings for all dates and export as Parquet."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        from src.features.stock_graph import StockGraphBuilder
        
        if feature_cols is None:
            feature_cols = [
                'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d',
                'rv_5d', 'rv_20d', 'rv_60d',
                'rsi_14', 'adx_14',
                'macd_histogram_pct',
                'price_vs_sma20_zscore',
                'market_breadth',
            ]
        available_cols = [c for c in feature_cols if c in df.columns]
        feature_cols = available_cols
        
        graph_builder = StockGraphBuilder(
            window=self.graph_window,
            corr_threshold=self.corr_threshold,
        )
        
        all_dates = sorted(df['date'].unique())
        valid_dates = all_dates[self.graph_window:]
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for i, td in enumerate(valid_dates):
                edge_index, symbols, _ = graph_builder.build_graph(df, td)
                
                if len(symbols) < 5 or edge_index.shape[1] == 0:
                    continue
                
                node_features = graph_builder.build_node_features(
                    df, symbols, td, feature_cols
                )
                
                x = torch.tensor(node_features, dtype=torch.float32)
                ei = torch.tensor(edge_index, dtype=torch.long)
                
                embeddings = self.model(x, ei).numpy()
                
                for j, sym in enumerate(symbols):
                    row = {'symbol': sym, 'date': td}
                    for k in range(self.embed_dim):
                        row[f'gnn_emb_{k}'] = float(embeddings[j, k])
                    results.append(row)
                
                if (i + 1) % 50 == 0:
                    print(f"  [GNN-v2] Generated embeddings for {i+1}/{len(valid_dates)} dates")
        
        emb_df = pd.DataFrame(results)
        
        if output_path is None:
            output_path = 'data/cache/gnn_embeddings.parquet'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        emb_df.to_parquet(output_path, index=False)
        print(f"[GNN-v2] Saved {len(emb_df)} embeddings to {output_path}")
        
        return emb_df

    def _build_graph_snapshots(
        self,
        df: pd.DataFrame,
        graph_builder,
        dates: list,
        feature_cols: list,
    ) -> list:
        """Build PyG Data objects with CONTINUOUS return labels for BPR."""
        graphs = []
        all_dates_sorted = sorted(df['date'].unique())
        
        for td in dates:
            edge_index, symbols, _ = graph_builder.build_graph(df, td)
            
            if len(symbols) < 5 or edge_index.shape[1] == 0:
                continue
            
            node_features = graph_builder.build_node_features(
                df, symbols, td, feature_cols
            )
            
            # CONTINUOUS return labels (not binary) for BPR ranking loss
            td_idx = list(all_dates_sorted).index(td)
            
            if td_idx + self.fwd_days >= len(all_dates_sorted):
                continue
            
            fwd_date = all_dates_sorted[td_idx + self.fwd_days]
            today_prices = df[df['date'] == td][['symbol', 'adj_close']].set_index('symbol')
            fwd_prices = df[df['date'] == fwd_date][['symbol', 'adj_close']].set_index('symbol')
            
            labels = np.zeros(len(symbols))
            for i, sym in enumerate(symbols):
                if sym in today_prices.index and sym in fwd_prices.index:
                    ret = fwd_prices.loc[sym, 'adj_close'] / today_prices.loc[sym, 'adj_close'] - 1
                    labels[i] = float(ret)  # continuous return, not binary
                else:
                    labels[i] = 0.0
            
            x = torch.tensor(node_features, dtype=torch.float32)
            ei = torch.tensor(edge_index, dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.float32)
            
            graphs.append(Data(x=x, edge_index=ei, y=y))
        
        return graphs


# Keep backward compatibility — alias old name
GraphSAGEEmbedder = GATEmbedder
