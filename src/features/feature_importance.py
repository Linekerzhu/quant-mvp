"""
Feature Importance Module

Implements Information Coefficient (IC) tracking and feature drift detection.
Part of Phase B feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from src.ops.event_logger import get_logger

logger = get_logger()


@dataclass
class ICMetrics:
    """IC metrics for a single feature."""
    feature_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float  # Information Ratio = IC_mean / IC_std
    t_stat: float
    p_value: float
    n_obs: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DriftAlert:
    """Feature drift alert."""
    feature_name: str
    drift_type: str  # 'ic_drop', 'sign_flip', 'volatility_spike'
    severity: str  # 'warning', 'critical'
    baseline_ic: float
    current_ic: float
    timestamp: datetime = field(default_factory=datetime.now)


class FeatureImportanceTracker:
    """
    Track feature importance via Information Coefficient (IC).
    
    IC = correlation between feature values and forward returns.
    Monitors for drift and degradation over time.
    """
    
    # Thresholds for drift detection
    IC_DROP_THRESHOLD = 0.3  # 30% drop in IC magnitude
    SIGN_FLIP_THRESHOLD = 0.1  # IC sign flipped and magnitude > threshold
    VOLATILITY_SPIKE_THRESHOLD = 2.0  # IC std increased 2x
    
    def __init__(self, lookback_days: int = 63):  # ~3 months
        self.lookback_days = lookback_days
        self.baseline_metrics: Dict[str, ICMetrics] = {}
        self.current_metrics: Dict[str, ICMetrics] = {}
        self.drift_history: List[DriftAlert] = []
        
    def calculate_ic(
        self,
        df: pd.DataFrame,
        feature_col: str,
        return_col: str = 'label_return',
        forward_period: int = 1
    ) -> Optional[ICMetrics]:
        """
        Calculate Information Coefficient for a feature.
        
        IC = Spearman correlation between feature and forward returns.
        
        Args:
            df: DataFrame with features and returns
            feature_col: Feature column name
            return_col: Return column name
            forward_period: Forward return period (days)
            
        Returns:
            ICMetrics or None if insufficient data
        """
        # Filter valid events with feature values
        valid_mask = (
            (df['event_valid'] == True) &
            df[feature_col].notna() &
            df[return_col].notna()
        )
        
        valid_df = df.loc[valid_mask]
        
        if len(valid_df) < 30:  # Minimum sample size
            logger.warn("insufficient_data_for_ic", {
                "feature": feature_col,
                "n_obs": len(valid_df)
            })
            return None
        
        # Calculate Spearman correlation
        feature_values = valid_df[feature_col].values
        returns = valid_df[return_col].values
        
        # Spearman rank correlation
        ic, p_value = stats.spearmanr(feature_values, returns)
        
        # Calculate rolling IC for stability metrics
        if len(valid_df) >= 100:
            # Split into chunks and calculate IC for each
            n_chunks = min(10, len(valid_df) // 20)
            chunk_ics = []
            
            for i in range(n_chunks):
                start_idx = i * len(valid_df) // n_chunks
                end_idx = (i + 1) * len(valid_df) // n_chunks
                
                chunk_ic, _ = stats.spearmanr(
                    feature_values[start_idx:end_idx],
                    returns[start_idx:end_idx]
                )
                chunk_ics.append(chunk_ic)
            
            ic_std = np.std(chunk_ics)
        else:
            ic_std = 0.1  # Default if insufficient data for chunking
        
        # Information Ratio
        ic_ir = ic / max(ic_std, 0.01)
        
        # T-statistic
        t_stat = ic * np.sqrt(len(valid_df) - 2) / np.sqrt(1 - ic**2)
        
        metrics = ICMetrics(
            feature_name=feature_col,
            ic_mean=float(ic),
            ic_std=float(ic_std),
            ic_ir=float(ic_ir),
            t_stat=float(t_stat),
            p_value=float(p_value),
            n_obs=len(valid_df)
        )
        
        return metrics
    
    def calculate_all_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        return_col: str = 'label_return'
    ) -> Dict[str, ICMetrics]:
        """
        Calculate IC for all features.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns (auto-detected if None)
            return_col: Return column name
            
        Returns:
            Dict mapping feature_name -> ICMetrics
        """
        if feature_cols is None:
            feature_cols = self._auto_detect_features(df)
        
        results = {}
        
        for feature in feature_cols:
            metrics = self.calculate_ic(df, feature, return_col)
            if metrics:
                results[feature] = metrics
        
        self.current_metrics = results
        
        logger.info("ic_calculation_complete", {
            "n_features": len(results),
            "features_with_ic_gt_0.05": sum(1 for m in results.values() if abs(m.ic_mean) > 0.05)
        })
        
        return results
    
    def detect_drift(
        self,
        baseline_metrics: Optional[Dict[str, ICMetrics]] = None
    ) -> List[DriftAlert]:
        """
        Detect feature drift by comparing current vs baseline IC.
        
        Args:
            baseline_metrics: Baseline metrics (uses stored baseline if None)
            
        Returns:
            List of DriftAlert
        """
        if baseline_metrics is None:
            baseline_metrics = self.baseline_metrics
        
        if not baseline_metrics or not self.current_metrics:
            logger.warn("cannot_detect_drift_no_baseline", {})
            return []
        
        alerts = []
        
        for feature, current in self.current_metrics.items():
            if feature not in baseline_metrics:
                continue
            
            baseline = baseline_metrics[feature]
            
            # Check 1: IC magnitude drop
            baseline_mag = abs(baseline.ic_mean)
            current_mag = abs(current.ic_mean)
            
            if baseline_mag > 0.01 and current_mag < baseline_mag * (1 - self.IC_DROP_THRESHOLD):
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_type='ic_drop',
                    severity='critical' if current_mag < baseline_mag * 0.5 else 'warning',
                    baseline_ic=baseline.ic_mean,
                    current_ic=current.ic_mean
                ))
            
            # Check 2: Sign flip with magnitude
            if (baseline.ic_mean * current.ic_mean < 0 and  # Sign flipped
                current_mag > self.SIGN_FLIP_THRESHOLD):
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_type='sign_flip',
                    severity='critical',
                    baseline_ic=baseline.ic_mean,
                    current_ic=current.ic_mean
                ))
            
            # Check 3: Volatility spike
            if current.ic_std > baseline.ic_std * self.VOLATILITY_SPIKE_THRESHOLD:
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_type='volatility_spike',
                    severity='warning',
                    baseline_ic=baseline.ic_std,
                    current_ic=current.ic_std
                ))
        
        self.drift_history.extend(alerts)
        
        logger.info("drift_detection_complete", {
            "n_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.severity == 'critical')
        })
        
        return alerts
    
    def set_baseline(self, metrics: Optional[Dict[str, ICMetrics]] = None) -> None:
        """Set baseline metrics for drift detection."""
        if metrics is None:
            metrics = self.current_metrics
        
        self.baseline_metrics = metrics.copy()
        
        logger.info("baseline_set", {
            "n_features": len(metrics),
            "mean_ic": np.mean([m.ic_mean for m in metrics.values()])
        })
    
    def get_top_features(
        self,
        n: int = 10,
        min_ic: float = 0.02
    ) -> List[Tuple[str, float]]:
        """
        Get top features by absolute IC.
        
        Returns:
            List of (feature_name, ic_mean) sorted by |IC|
        """
        filtered = [
            (name, metrics.ic_mean)
            for name, metrics in self.current_metrics.items()
            if abs(metrics.ic_mean) >= min_ic
        ]
        
        return sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:n]
    
    def get_drift_report(self) -> Dict:
        """Generate drift detection report."""
        return {
            'n_features_tracked': len(self.current_metrics),
            'n_baseline_features': len(self.baseline_metrics),
            'drift_alerts': [
                {
                    'feature': a.feature_name,
                    'type': a.drift_type,
                    'severity': a.severity,
                    'baseline': a.baseline_ic,
                    'current': a.current_ic
                }
                for a in self.drift_history[-20:]  # Last 20 alerts
            ]
        }
    
    def _auto_detect_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from DataFrame."""
        exclude = ['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 'raw_close',
                   'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume',
                   'label', 'label_barrier', 'label_return', 'label_holding_days',
                   'event_valid', 'features_valid', 'feature_version', 'sample_weight',
                   'dummy_noise']
        
        # Include only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return [col for col in numeric_cols if col not in exclude]
