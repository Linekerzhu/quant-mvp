"""
Fractional Differentiation (FracDiff) Features

Implements AFML Ch5: Fractional differentiation for feature engineering.
Used to find the minimum differencing order d that makes a series stationary
while retaining memory.

Author: 李得勤
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.ops.event_logger import get_logger

logger = get_logger()

# ADF 测试最小样本量（统计可靠性要求 30-50 个样本）
MIN_ADF_SAMPLES = 50


def _validate_series(series: pd.Series, allow_na: str = 'error'):
    """
    验证输入 series，处理 NaN 值
    
    Args:
        series: 输入序列
        allow_na: NaN 处理策略 'error' | 'drop' | 'fill'
            - error: 包含 NaN 时抛出异常（默认）
            - drop: 删除 NaN 值
            - fill: 使用前向填充后向填充
    
    Returns:
        处理后的 series
    """
    if series.isna().any():
        nan_count = series.isna().sum()
        if allow_na == 'error':
            raise ValueError(
                f"Input series contains {nan_count} NaN values. "
                "Use allow_na='drop' or 'fill' to handle NaN."
            )
        elif allow_na == 'drop':
            logger.debug(f"Dropping {nan_count} NaN values from series")
            return series.dropna()
        elif allow_na == 'fill':
            logger.debug(f"Filling {nan_count} NaN values in series")
            return series.ffill().bfill()
        else:
            raise ValueError(f"Unknown allow_na mode: {allow_na}")
    return series


def _safe_adf_test(series: pd.Series, window: int) -> Optional[tuple]:
    """
    执行安全的 ADF 测试
    
    Args:
        series: 输入序列（已去除 NaN）
        window: 窗口大小，用于计算 maxlag
    
    Returns:
        ADF 测试结果元组，如果测试无法执行则返回 None
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("statsmodels is required for stationarity test")
    
    nobs = len(series)
    
    # 样本量检查
    if nobs < MIN_ADF_SAMPLES:
        logger.debug(f"样本量不足: {nobs} < {MIN_ADF_SAMPLES}，跳过 ADF 测试")
        return None
    
    # 动态计算 maxlag：确保 maxlag < nobs - 1，建议不超过 nobs // 3
    maxlag = min(window // 10, nobs // 3, 100)  # 上限 100
    
    try:
        return adfuller(series, maxlag=maxlag)
    except Exception as e:
        logger.warning(f"ADF 测试失败: {e}")
        return None


def fracdiff_weights(d: float, window: int) -> np.ndarray:
    """
    Calculate fractional differentiation weights.
    
    Formula (AFML Ch5):
        w[0] = 1
        w[k] = w[k-1] * (k - 1 - d) / k
    
    Args:
        d: Differencing order (0 <= d <= 1)
        window: Number of weights to compute
    
    Returns:
        Array of weights
    """
    weights = np.zeros(window)
    weights[0] = 1.0
    
    for k in range(1, window):
        weights[k] = weights[k-1] * (k - 1 - d) / k
    
    return weights


def fracdiff_fixed_window(
    series: pd.Series,
    d: float,
    window: int = 100,
    allow_na: str = 'error'
) -> pd.Series:
    """
    Fixed window fractional differentiation.
    
    Uses a fixed window of past prices to compute fractional difference.
    This is the simplest implementation but may have boundary effects.
    
    Args:
        series: Price series (e.g., adj_close)
        d: Differencing order, 0 <= d <= 1 (can include 0 and 1 for edge cases)
        window: Weight truncation window (default 100)
        allow_na: NaN 处理策略 'error' | 'drop' | 'fill' (默认: error)
    
    Returns:
        Fractionally differenced series (first window-1 values are NaN)
    
    Algorithm:
        fracdiff[t] = sum(weights[k] * price[t-k] for k in range(window))
    
    Note:
        - Purely causal: only uses data at t and before
        - No look-ahead bias
    """
    if not 0 <= d <= 1:
        raise ValueError("d must be between 0 and 1 (inclusive)")
    
    # 验证并处理 NaN
    series = _validate_series(series, allow_na)
    
    # Calculate weights
    weights = fracdiff_weights(d, window)
    
    # Compute fractional difference using rolling dot product
    frac_diff = series.rolling(window=window).apply(
        lambda x: np.dot(weights, x[::-1]),
        raw=True
    )
    
    return frac_diff


def fracdiff_expand_window(
    series: pd.Series,
    d: float,
    max_window: int = 100,
    allow_na: str = 'error'
) -> pd.Series:
    """
    Expanding window fractional differentiation.
    
    Uses all available past data with expanding window.
    More accurate at the beginning but computationally heavier.
    
    Args:
        series: Price series
        d: Differencing order, 0 <= d <= 1
        max_window: Maximum window size (for performance)
        allow_na: NaN 处理策略 'error' | 'drop' | 'fill' (默认: error)
    
    Returns:
        Fractionally differenced series
    """
    if not 0 <= d <= 1:
        raise ValueError("d must be between 0 and 1 (inclusive)")
    
    # 验证并处理 NaN
    series = _validate_series(series, allow_na)
    
    n = len(series)
    result = pd.Series(index=series.index, dtype=float)
    
    for t in range(n):
        # Use expanding window up to max_window
        w = min(t + 1, max_window)
        
        if w < 2:
            result.iloc[t] = np.nan
            continue
        
        # Calculate weights for this window
        weights = fracdiff_weights(d, w)
        
        # Get the price window
        price_window = series.iloc[t - w + 1:t + 1].values
        
        # Compute dot product
        result.iloc[t] = np.dot(weights, price_window[::-1])
    
    return result


def fracdiff_online(
    series: pd.Series,
    d: float,
    window: int = 100,
    allow_na: str = 'error'
) -> pd.Series:
    """
    Online/incremental fractional differentiation.
    
    More memory efficient for streaming data.
    
    Args:
        series: Price series
        d: Differencing order, 0 <= d <= 1
        window: Window size
        allow_na: NaN 处理策略 'error' | 'drop' | 'fill' (默认: error)
    
    Returns:
        Fractionally differenced series
    """
    if not 0 <= d <= 1:
        raise ValueError("d must be between 0 and 1 (inclusive)")
    
    # 验证并处理 NaN
    series = _validate_series(series, allow_na)
    
    return fracdiff_fixed_window(series, d, window, allow_na='error')


def find_min_d_stationary(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    threshold: float = 0.01,
    window: int = 100,
    allow_na: str = 'error'
) -> float:
    """
    Find minimum d that makes the series stationary.
    
    Uses ADF (Augmented Dickey-Fuller) test to check stationarity.
    
    Args:
        series: Price series to test
        d_range: Range of d to search (min, max)
        threshold: p-value threshold for stationarity
        window: Window size for fracdiff
        allow_na: NaN 处理策略 'error' | 'drop' | 'fill' (默认: error)
    
    Returns:
        Minimum d that achieves stationarity
    """
    # 验证并处理 NaN
    series = _validate_series(series, allow_na)
    
    # Check if already stationary at d=0
    # For d=0, just use the original series (after dropping NaN)
    clean_series = series.dropna()
    test_result = _safe_adf_test(clean_series, window)
    if test_result is not None and test_result[1] < threshold:
        return 0.0
    
    # Check if stationary at d=1
    fracdiff_1 = fracdiff_fixed_window(series, 1.0, window, allow_na='error')
    clean_1 = fracdiff_1.dropna()
    
    # 增强样本量检查：从 20 提升到 MIN_ADF_SAMPLES (50)
    if len(clean_1) < MIN_ADF_SAMPLES:
        logger.debug(f"d=1 时样本量不足: {len(clean_1)} < {MIN_ADF_SAMPLES}")
        return 1.0
    
    test_result = _safe_adf_test(clean_1, window)
    if test_result is None or test_result[1] >= threshold:
        # Even d=1 is not stationary or test failed, return 1.0
        return 1.0
    
    # Binary search
    d_min, d_max = d_range
    
    while d_max - d_min > 0.01:
        d_mid = (d_min + d_max) / 2
        
        fracdiff_mid = fracdiff_fixed_window(series, d_mid, window, allow_na='error')
        clean_mid = fracdiff_mid.dropna()
        
        # 样本量检查
        if len(clean_mid) < MIN_ADF_SAMPLES:
            logger.debug(f"d={d_mid:.2f} 时样本量不足: {len(clean_mid)} < {MIN_ADF_SAMPLES}")
            d_min = d_mid
            continue
        
        test_result = _safe_adf_test(clean_mid, window)
        
        if test_result is None:
            # 测试失败，假设非平稳，增加 d
            d_min = d_mid
        elif test_result[1] < threshold:
            d_max = d_mid
        else:
            d_min = d_mid
    
    return round(d_max, 2)


class FracDiffTransformer:
    """
    sklearn-compatible FracDiff transformer.
    
    Can be used in sklearn pipelines.
    """
    
    def __init__(
        self,
        d: float = 0.5,
        window: int = 100,
        method: str = 'fixed'
    ):
        """
        Initialize FracDiff transformer.
        
        Args:
            d: Differencing order
            window: Window size
            method: 'fixed', 'expand', or 'online'
        """
        self.d = d
        self.window = window
        self.method = method
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op, for sklearn compatibility)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame columns.
        
        Args:
            X: DataFrame with numeric columns
        
        Returns:
            DataFrame with fractional diff applied
        """
        result = X.copy()
        
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if self.method == 'fixed':
                    result[col] = fracdiff_fixed_window(X[col], self.d, self.window)
                elif self.method == 'expand':
                    result[col] = fracdiff_expand_window(X[col], self.d, self.window)
                elif self.method == 'online':
                    result[col] = fracdiff_online(X[col], self.d, self.window)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
    
    def __repr__(self):
        return f"FracDiffTransformer(d={self.d}, window={self.window}, method='{self.method}')"


def create_fracdiff_features(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    d_values: list = None,
    window: int = 100
) -> pd.DataFrame:
    """
    Create multiple FracDiff features with different d values.
    
    Args:
        df: DataFrame with price column
        price_col: Name of price column
        d_values: List of d values (default: [0.3, 0.4, 0.5, 0.6, 0.7])
        window: Window size
    
    Returns:
        DataFrame with added fracdiff columns
    """
    if d_values is None:
        d_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    result = df.copy()
    price = df[price_col]
    
    for d in d_values:
        col_name = f'fracdiff_{int(d*10)}'
        result[col_name] = fracdiff_fixed_window(price, d, window)
    
    return result
