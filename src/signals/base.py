"""
Base Signal Generator Abstract Class

Defines the interface for all Base Model signal generators.

Author: 李得勤
Date: 2026-02-28
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import pandas as pd


class BaseSignalGenerator(ABC):
    """
    所有 Base Model 的抽象基类。
    
    定义生成交易信号的统一接口。
    """
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。
        
        Args:
            df: OHLCV 数据，至少包含 [symbol, date, adj_close] 列
        
        Returns:
            DataFrame with added 'side' column:
            - +1: 做多信号 (long)
            - -1: 做空信号 (short)  
            -  0: 无信号 (neutral/cold start)
        """
        pass
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """
        验证输入数据。
        
        Args:
            df: Input DataFrame
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['symbol', 'date', 'adj_close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ========== Model Registry ==========

class SignalModelRegistry:
    """
    Base Model 注册表。
    
    使用装饰器自动注册:
        @SignalModelRegistry.register('sma')
        class BaseModelSMA(BaseSignalGenerator):
            ...
    """
    
    _models: Dict[str, Type[BaseSignalGenerator]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        装饰器：注册 Base Model。
        
        用法:
            @SignalModelRegistry.register('sma')
            class BaseModelSMA(BaseSignalGenerator):
                ...
        """
        def decorator(model_class: Type[BaseSignalGenerator]):
            cls._models[name.lower()] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseSignalGenerator:
        """
        创建 Base Model 实例。
        
        Args:
            name: 模型名称 ('sma', 'momentum')
            **kwargs: 构造参数
        
        Returns:
            BaseSignalGenerator 实例
        """
        name = name.lower()
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册的模型。"""
        return list(cls._models.keys())
    
    @classmethod
    def get(cls, name: str) -> Type[BaseSignalGenerator]:
        """获取模型类。"""
        name = name.lower()
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name]
