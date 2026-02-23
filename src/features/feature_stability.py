"""
Feature Stability Module

Implements feature stability gating and monitoring.
Features that are too volatile or unstable are flagged for exclusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats

from src.ops.event_logger import get_logger

logger = get_logger()


@dataclass
class StabilityMetrics:
    """Stability metrics for a feature."""
    feature_name: str
    turn_over: float  # Feature turnover (correlation with previous period)
    auto_corr_1d: float  # 1-day autocorrelation
    auto_corr_5d: float  # 5-day autocorrelation
    unique_value_ratio: float  # Ratio of unique values to total observations
    outlier_ratio: float  # Ratio of values beyond 3 std
    nan_ratio: float  # Ratio of NaN values
    stability_score: float  # Composite score (0-1, higher = more stable)
    is_stable: bool  # Pass/fail flag
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StabilityGate:
    """Gate result for a feature."""
    feature_name: str
    passed: bool
    failed_checks: List[str]
    metrics: StabilityMetrics


class FeatureStabilityMonitor:
    """
    Monitor feature stability over time.
    
    Unstable features (high turnover, high outlier ratio) are flagged
    and may be excluded from the model.
    
    Thresholds based on Marcos Lopez de Prado's research on
    feature selection in financial ML.
    """
    
    # Stability thresholds
    MIN_TURNOVER = 0.60  # Minimum correlation with previous period
    MAX_OUTLIER_RATIO = 0.05  # Maximum 5% outliers
    MAX_NAN_RATIO = 0.10  # Maximum 10% NaN
    MIN_UNIQUE_RATIO = 0.01  # At least 1% unique values (not constant)
    
    def __init__(self):
        self.metrics_history: Dict[str, List[StabilityMetrics]] = {}
        self.gate_results: Dict[str, StabilityGate] = {}
    
    def calculate_stability(
        self,
        df: pd.DataFrame,
        feature_col: str,
        lookback_days: int = 63
    ) -> Optional[StabilityMetrics]:
        """
        Calculate stability metrics for a feature.
        
        Args:
            df: DataFrame with feature values
            feature_col: Feature column name
            lookback_days: Lookback window for stability calculation
            
        Returns:
            StabilityMetrics or None if insufficient data
        """
        # Filter for valid values
        valid_mask = df[feature_col].notna() & (df['features_valid'] if 'features_valid' in df.columns else True)
        valid_df = df.loc[valid_mask].sort_values('date')
        
        if len(valid_df) < lookback_days:
            logger.warn("insufficient_data_for_stability", {
                "feature": feature_col,
                "n_obs": len(valid_df)
            })
            return None
        
        # Use most recent data
        recent_df = valid_df.iloc[-lookback_days:]
        values = recent_df[feature_col].values
        
        # 1. Turnover: correlation with previous period
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        if len(first_half) > 10 and len(second_half) > 10:
            # Calculate mean values for each period
            first_mean = np.mean(first_half)
            second_mean = np.mean(second_half)
            
            # Simple turnover metric: normalized difference
            turnover = 1.0 - abs(second_mean - first_mean) / (abs(first_mean) + 1e-6)
            turnover = max(0, min(1, turnover))  # Clip to [0, 1]
        else:
            turnover = 0.5  # Default if insufficient data
        
        # 2. Autocorrelation (1-day and 5-day)
        auto_corr_1d = self._safe_autocorr(values, lag=1)
        auto_corr_5d = self._safe_autocorr(values, lag=5)
        
        # 3. Unique value ratio
        n_unique = len(np.unique(values))
        unique_ratio = n_unique / len(values)
        
        # 4. Outlier ratio (beyond 3 std)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            outlier_mask = np.abs(values - mean_val) > 3 * std_val
            outlier_ratio = np.mean(outlier_mask)
        else:
            outlier_ratio = 0.0
        
        # 5. NaN ratio
        nan_ratio = 1.0 - (len(values) / len(df))
        
        # Calculate composite stability score
        # Weighted combination of metrics
        stability_score = (
            0.3 * turnover +
            0.2 * max(0, auto_corr_1d) +
            0.2 * max(0, auto_corr_5d) +
            0.1 * (1.0 if unique_ratio > self.MIN_UNIQUE_RATIO else 0) +
            0.1 * (1.0 if outlier_ratio < self.MAX_OUTLIER_RATIO else 0) +
            0.1 * (1.0 if nan_ratio < self.MAX_NAN_RATIO else 0)
        )
        
        # Determine if stable
        is_stable = (
            turnover >= self.MIN_TURNOVER and
            outlier_ratio <= self.MAX_OUTLIER_RATIO and
            nan_ratio <= self.MAX_NAN_RATIO and
            unique_ratio >= self.MIN_UNIQUE_RATIO
        )
        
        metrics = StabilityMetrics(
            feature_name=feature_col,
            turn_over=float(turnover),
            auto_corr_1d=float(auto_corr_1d),
            auto_corr_5d=float(auto_corr_5d),
            unique_value_ratio=float(unique_ratio),
            outlier_ratio=float(outlier_ratio),
            nan_ratio=float(nan_ratio),
            stability_score=float(stability_score),
            is_stable=is_stable
        )
        
        return metrics
    
    def check_all_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, StabilityGate]:
        """
        Check stability for all features.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns (auto-detected if None)
            
        Returns:
            Dict mapping feature_name -> StabilityGate
        """
        if feature_cols is None:
            feature_cols = self._auto_detect_features(df)
        
        results = {}
        
        for feature in feature_cols:
            metrics = self.calculate_stability(df, feature)
            
            if metrics is None:
                continue
            
            # Determine which checks failed
            failed_checks = []
            
            if metrics.turn_over < self.MIN_TURNOVER:
                failed_checks.append('high_turnover')
            
            if metrics.outlier_ratio > self.MAX_OUTLIER_RATIO:
                failed_checks.append('too_many_outliers')
            
            if metrics.nan_ratio > self.MAX_NAN_RATIO:
                failed_checks.append('too_many_nans')
            
            if metrics.unique_value_ratio < self.MIN_UNIQUE_RATIO:
                failed_checks.append('near_constant')
            
            gate = StabilityGate(
                feature_name=feature,
                passed=len(failed_checks) == 0,
                failed_checks=failed_checks,
                metrics=metrics
            )
            
            results[feature] = gate
            
            # Store in history
            if feature not in self.metrics_history:
                self.metrics_history[feature] = []
            self.metrics_history[feature].append(metrics)
        
        self.gate_results = results
        
        # Log summary
        passed = sum(1 for g in results.values() if g.passed)
        failed = len(results) - passed
        
        logger.info("stability_check_complete", {
            "n_features": len(results),
            "passed": passed,
            "failed": failed
        })
        
        return results
    
    def get_stable_features(self) -> List[str]:
        """Get list of features that passed stability check."""
        return [
            name for name, gate in self.gate_results.items()
            if gate.passed
        ]
    
    def get_unstable_features(self) -> List[Tuple[str, List[str]]]:
        """
        Get list of unstable features with failure reasons.
        
        Returns:
            List of (feature_name, failed_checks)
        """
        return [
            (name, gate.failed_checks)
            for name, gate in self.gate_results.items()
            if not gate.passed
        ]
    
    def get_stability_report(self) -> Dict:
        """Generate stability monitoring report."""
        if not self.gate_results:
            return {'error': 'No stability checks performed yet'}
        
        stable_features = self.get_stable_features()
        unstable_features = self.get_unstable_features()
        
        # Calculate average stability score
        avg_score = np.mean([
            gate.metrics.stability_score
            for gate in self.gate_results.values()
        ])
        
        return {
            'total_features': len(self.gate_results),
            'stable_features': len(stable_features),
            'unstable_features': len(unstable_features),
            'average_stability_score': round(avg_score, 4),
            'stability_rate': round(len(stable_features) / len(self.gate_results), 4),
            'unstable_details': [
                {'feature': name, 'failed_checks': checks}
                for name, checks in unstable_features
            ]
        }
    
    def _safe_autocorr(self, values: np.ndarray, lag: int = 1) -> float:
        """Calculate safe autocorrelation."""
        if len(values) <= lag:
            return 0.0
        
        try:
            # Pearson correlation with lag
            x1 = values[:-lag]
            x2 = values[lag:]
            
            if np.std(x1) == 0 or np.std(x2) == 0:
                return 0.0
            
            corr = np.corrcoef(x1, x2)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _auto_detect_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from DataFrame."""
        exclude = ['symbol', 'date', 'raw_open', 'raw_high', 'raw_low', 'raw_close',
                   'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume',
                   'label', 'label_barrier', 'label_return', 'label_holding_days',
                   'event_valid', 'features_valid', 'feature_version', 'sample_weight',
                   'dummy_noise']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude]
