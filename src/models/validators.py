"""
Base Validator Abstract Class

Defines the interface for cross-validation strategies.

Author: 李得勤
Date: 2026-02-28
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import numpy as np
import pandas as pd


class BaseValidator(ABC):
    """
    交叉验证器的抽象基类。
    
    所有验证器（ CPCV、Walk-Forward、KFold等）都需要实现此接口。
    """
    
    @abstractmethod
    def split(
        self, 
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成分割索引。
        
        Args:
            df: 输入数据
            date_col: 日期列名
        
        Yields:
            (train_indices, test_indices) 元组
        """
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """
        返回分割数量。
        
        Returns:
            分割数量
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
