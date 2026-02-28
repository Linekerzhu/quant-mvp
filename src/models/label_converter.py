"""
Label Conversion Module

Implements Meta-Label conversion logic and Triple Barrier label handling.

Author: 李得勤
Date: 2026-02-28
"""

from typing import Dict, Optional

import pandas as pd

from src.ops.event_logger import get_logger

logger = get_logger()


class LabelConverter:
    """
    标签转换器。
    
    将 Triple Barrier 标签转换为 Meta-Label:
    - Meta-Label 问的是"信号是否盈利？"
    - profit → 1, loss → 0
    """
    
    def __init__(self, config: Dict):
        """
        Initialize LabelConverter.
        
        Args:
            config: Label configuration dictionary
        """
        self.config = config
        self.strategy = config.get('strategy', 'binary_filtered')
        self.mapping = config.get('mapping', {-1: 0, 1: 1})
        
        logger.info(f"LabelConverter initialized: strategy={self.strategy}")
    
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 Triple Barrier 标签转换为 Meta-Label。
        
        Meta-Label 逻辑:
        - 信号是否盈利？profit → 1, loss → 0
        - label=-1 和 label=0 都映射为 0（寇连材审计建议）
        
        Args:
            df: DataFrame with 'label' column from Triple Barrier
        
        Returns:
            DataFrame with added 'meta_label' column (0 or 1)
        """
        if self.strategy == 'binary_filtered':
            # Filter out label=0 (time barriers)
            # BUG-03 Fix: 先过滤NaN，再过滤label=0
            df = df[df['label'].notna() & (df['label'] != 0)].copy()
            
            # Map {-1: 0, +1: 1} for binary classification
            df['meta_label'] = df['label'].map(self.mapping)
            
            logger.info(f"Meta-labels: {len(df)} samples, "
                       f"positive={(df['meta_label']==1).sum()}, "
                       f"negative={(df['meta_label']==0).sum()}")
        else:
            raise ValueError(f"Unknown label strategy: {self.strategy}")
        
        return df
    
    def get_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        获取标签统计信息。
        
        Args:
            df: DataFrame with 'meta_label' column
        
        Returns:
            Dictionary with label counts
        """
        if 'meta_label' not in df.columns:
            return {'total': len(df)}
        
        return {
            'total': len(df),
            'positive': int((df['meta_label'] == 1).sum()),
            'negative': int((df['meta_label'] == 0).sum()),
            'positive_ratio': float((df['meta_label'] == 1).mean())
        }
