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
        # D3 Fix: 添加防御性映射，避免 label=0 静默丢失
        self.mapping = config.get('mapping', {-1: 0, 0: 0, 1: 1})
        
        logger.info(f"LabelConverter initialized: strategy={self.strategy}")
    
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 Triple Barrier 标签转换为 Meta-Label。
        
        Meta-Label 逻辑:
        - FATAL-2 修复: meta_label = 1 if (side × label > 0)
        - 即：Base Model 的方向预测是否正确
        
        Args:
            df: DataFrame with 'label' column from Triple Barrier
        
        Returns:
            DataFrame with added 'meta_label' column (0 or 1)
        """
        if self.strategy == 'binary_filtered':
            # H2 Fix: Time Barrier 过滤监控
            total_events = len(df)
            
            # Filter out label=0 (time barriers)
            # BUG-03 Fix: 先过滤NaN，再过滤label=0
            df_filtered = df[df['label'].notna() & (df['label'] != 0)].copy()
            filtered_count = total_events - len(df_filtered)
            df = df_filtered

            # ============================================================
            # FATAL-2 Fix: Meta-Label = "Base Model方向是否正确"
            # ============================================================
            # AFML定义: meta_label = 1 if (side × label > 0)
            # - side=+1,label=+1 → 做多赚钱 → 1
            # - side=+1,label=-1 → 做多亏钱 → 0
            # - side=-1,label=-1 → 做空赚钱 → 1  ← 旧代码错误映射为0
            # - side=-1,label=+1 → 做空亏钱 → 0  ← 旧代码错误映射为1
            # ============================================================
            if 'side' in df.columns:
                df['meta_label'] = ((df['side'] * df['label']) > 0).astype(int)
                logger.info("FATAL-2: meta_label使用side×label方向感知映射")
            else:
                # 向后兼容: 无side列时退回旧映射 (假定全部做多)
                df['meta_label'] = df['label'].map(self.mapping)
                logger.warn("FATAL-2: 无side列，退回label直接映射 (假定全部做多)")

            logger.info(f"Meta-labels: {len(df)} samples, "
                       f"positive={(df['meta_label']==1).sum()}, "
                       f"negative={(df['meta_label']==0).sum()}")
            
            # H2 Fix: Time Barrier 过滤监控
            logger.info("time_barrier_stats", {
                "total_events": total_events,
                "time_barrier_filtered": filtered_count,
                "filter_ratio": filtered_count / total_events if total_events > 0 else 0,
                "positive_after": int((df['meta_label'] == 1).sum()),
                "negative_after": int((df['meta_label'] == 0).sum())
            })
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
