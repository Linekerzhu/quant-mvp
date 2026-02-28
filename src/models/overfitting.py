"""
Overfitting Detection Module

Implements PBO (Probability of Backtest Overfitting) detection
and Dummy Feature Sentinel for detecting overfitting in Meta-Labeling.

Author: 李得勤
Date: 2026-02-28
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np

from src.ops.event_logger import get_logger

logger = get_logger()


class OverfittingDetector:
    """
    过拟合检测器。
    
    提供两种检测机制:
    1. PBO (Probability of Backtest Overfitting) - 路径级过拟合检测
    2. Dummy Feature Sentinel - 特征级过拟合检测
    """
    
    def __init__(self, config: Dict):
        """
        Initialize OverfittingDetector.
        
        Args:
            config: Overfitting configuration dictionary
        """
        self.config = config
        self.pbo_reject = config.get('pbo_reject', 0.5)
        self.pbo_threshold = config.get('pbo_threshold', 0.3)
        
        dummy_config = config.get('dummy_feature_sentinel', {})
        self.dummy_threshold = dummy_config.get('ranking_threshold', 0.25)
        
        logger.info("overfitting_detector_init", {
            "pbo_threshold": self.pbo_threshold,
            "pbo_reject": self.pbo_reject,
            "dummy_threshold": self.dummy_threshold
        })
    
    def calculate_pbo(self, path_results: List[Dict]) -> float:
        """
        计算 PBO（Probability of Backtest Overfitting）。
        
        方法：IS-OOS Gap 检测（简化版，过AFML定义）
        - 计算每条路径的 IS AUC 和 OOS AUC 差距
        - 如果 IS 明显高于 OOS → 过拟合风险高
        
        Args:
            path_results: List of result dicts from each CPCV path
        
        Returns:
            PBO value between 0 and 1
            - PBO > 0.5: 高过拟合风险
            - PBO < 0.3: 低过拟合风险
        """
        # OR2-01 Fix: 使用 IS-OOS Gap 方法
        is_aucs = []
        oos_aucs = []
        
        for r in path_results:
            is_auc = r.get('is_auc', r.get('auc', 0.5))
            oos_auc = r.get('oos_auc', r.get('auc', 0.5))
            is_aucs.append(is_auc)
            oos_aucs.append(oos_auc)
        
        n = len(is_aucs)
        
        if n == 0:
            return 1.0
        
        # 计算 IS-OOS Gap
        gaps = [is_auc - oos_auc for is_auc, oos_auc in zip(is_aucs, oos_aucs)]
        
        # 平均 gap
        mean_gap = np.mean(gaps)
        
        # 标准差
        std_gap = np.std(gaps, ddof=1) if len(gaps) > 1 else 0
        
        # 计算 PBO：gap > 0.05 视为过拟合
        # PBO = 过拟合路径的比例
        gap_threshold = 0.05
        pbo = np.mean([gap > gap_threshold for gap in gaps])
        
        logger.info("pbo_is_oos_gap_method", {
            "n_paths": n,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
            "pbo": pbo
        })
        
        return float(pbo)
    
    def check_pbo_gate(self, pbo: float) -> Tuple[bool, str]:
        """
        PBO 门控检查。
        
        Args:
            pbo: Calculated PBO value
        
        Returns:
            Tuple of (passed, message)
        """
        if pbo >= self.pbo_reject:
            return False, f"HARD REJECT: PBO={pbo:.2f} >= {self.pbo_reject}"
        elif pbo >= self.pbo_threshold:
            return True, f"WARNING: PBO={pbo:.2f} in [{self.pbo_threshold}, {self.pbo_reject})"
        else:
            return True, f"PASS: PBO={pbo:.2f} < {self.pbo_threshold}"
    
    def dummy_feature_sentinel(
        self, 
        feature_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Dummy Feature 过拟合哨兵。
        
        检查 dummy 特征是否进入前 25% 排名。
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
        
        Returns:
            Dictionary with sentinel results
        """
        dummy_col = 'dummy_noise'
        
        if dummy_col not in feature_importance:
            return {'passed': True, 'note': 'dummy not found'}
        
        # Sort features by importance (descending)
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate ranks
        ranks = {name: i + 1 for i, (name, _) in enumerate(sorted_features)}
        
        dummy_rank = ranks[dummy_col]
        total_features = len(feature_importance)
        ranking_ratio = dummy_rank / total_features
        
        # Check threshold
        passed = ranking_ratio > self.dummy_threshold
        
        return {
            'dummy_rank': dummy_rank,
            'total_features': total_features,
            'ranking_ratio': ranking_ratio,
            'threshold': self.dummy_threshold,
            'passed': passed
        }
    
    def calculate_deflated_sharpe(self, path_results: List[Dict]) -> float:
        """
        计算 DSR 检验的 z-score（统计显著性检验）。
        
        注意：这不是真正的 Deflated Sharpe Ratio！
        真正的 DSR 需要用 norm.cdf() 转换，这里直接返回 z-score。
        
        公式: z = (SR - SR₀) / SE(SR)
        
        判定标准 (使用 z-score):
        - z > 1.645: 95% 置信度通过
        - z > 1.282: 90% 置信度通过
        - z <= 1.282: 拒绝
        
        Args:
            path_results: List of CPCV path results
        
        Returns:
            z-score 值 (用于 check_dsr_gate 判定)
        """
        # Baseline assumption:
        # - For accuracy: 0.5 = random guessing (balanced classes)
        # - For AUC: 0.5 = random ranking
        # - For Sharpe: 0.0 = zero excess return
        # Note: If classes are imbalanced, baseline should be max(class_prior)
        
        # OR2-04 Fix: 分别收集metrics和baselines
        metrics = []
        baselines = []
        
        for r in path_results:
            metrics.append(r.get('accuracy', r.get('auc', 0.5)))
            # 从实际数据获取baseline
            baselines.append(r.get('positive_ratio', 0.5))
        
        baseline = np.mean(baselines) if baselines else 0.5
        
        if len(metrics) < 2:
            logger.warn("deflated_sharpe_insufficient_data", {"n_paths": len(metrics)})
            return 0.0
        
        mean_sr = np.mean(metrics)
        std_sr = np.std(metrics, ddof=1)
        n = len(metrics)
        
        # Standard error
        if std_sr == 0 or n < 2:
            logger.warn("deflated_sharpe_zero_variance", {"std": std_sr, "n": n})
            return 0.0
        
        se_sr = std_sr / np.sqrt(n)
        
        # OR2-04 Fix: baseline已在上方计算
        dsr = (mean_sr - baseline) / se_sr
        
        logger.info("deflated_sharpe", {
            "mean_metric": mean_sr,
            "std_metric": std_sr,
            "n_paths": n,
            "dsr": dsr
        })
        
        return float(dsr)
    
    def check_dsr_gate(self, dsr: float) -> Tuple[bool, str]:
        """
        DSR z-score 门控检查。
        
        使用 z-score 判定统计显著性：
        - z > 1.645: 策略有效 (95% 置信度)
        - z > 1.282: 策略可能有效 (90% 置信度)
        - z <= 1.282: 策略无效 (拒绝)
        
        Args:
            dsr: Calculated z-score value from calculate_deflated_sharpe()
        
        Args:
            dsr: Calculated DSR value
        
        Returns:
            Tuple of (passed, message)
        """
        from scipy.stats import norm
        
        # Convert DSR to p-value like score
        # DSR > 0.95 means > 95% confidence
        if dsr > norm.ppf(0.95):  # ~1.645
            return True, f"PASS: DSR={dsr:.2f} > 1.645 (95% confidence)"
        elif dsr > norm.ppf(0.90):  # ~1.282
            return True, f"WARNING: DSR={dsr:.2f} in [1.282, 1.645) (90% confidence)"
        else:
            return False, f"REJECT: DSR={dsr:.2f} <= 1.282 (< 90% confidence)"
    
    def check_overfitting(
        self, 
        path_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        执行完整的过拟合检查。
        
        Args:
            path_results: List of CPCV path results
        
        Returns:
            Dictionary with overfitting check results
        """
        # PBO check
        pbo = self.calculate_pbo(path_results)
        pbo_passed, pbo_message = self.check_pbo_gate(pbo)
        
        # DSR check
        dsr = self.calculate_deflated_sharpe(path_results)
        dsr_passed, dsr_message = self.check_dsr_gate(dsr)
        
        # Check dummy feature on average importance
        avg_importance = {}
        for r in path_results:
            for feat, imp in r.get('feature_importance', {}).items():
                if feat not in avg_importance:
                    avg_importance[feat] = []
                avg_importance[feat].append(imp)
        
        avg_importance = {k: np.mean(v) for k, v in avg_importance.items()}
        dummy_result = self.dummy_feature_sentinel(avg_importance)
        
        return {
            'pbo': pbo,
            'pbo_passed': pbo_passed,
            'pbo_message': pbo_message,
            'dsr': dsr,
            'dsr_passed': dsr_passed,
            'dsr_message': dsr_message,
            'dummy_sentinel': dummy_result,
            'overall_passed': pbo_passed and dsr_passed and dummy_result.get('passed', True)
        }


class DataPenaltyApplier:
    """
    数据技术债惩罚性拨备。
    
    plan.md v4.2 要求:
    - CAGR: -3% (survivorship bias 2% + lookahead bias 1%)
    - MDD: +10%
    """
    
    # Penalty constants
    SURVIVORSHIP_CAGR_PENALTY = 0.02
    LOOKAHEAD_CAGR_PENALTY = 0.01
    MDD_INFLATION = 0.10
    
    def apply(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        应用数据技术债惩罚性拨备。
        
        Args:
            metrics: Dictionary of performance metrics
        
        Returns:
            Dictionary with adjusted metrics
        """
        adjusted = metrics.copy()
        
        total_cagr_penalty = self.SURVIVORSHIP_CAGR_PENALTY + self.LOOKAHEAD_CAGR_PENALTY
        
        if 'cagr' in metrics:
            adjusted['cagr'] = (
                metrics['cagr'] - total_cagr_penalty
            )
            adjusted['cagr_raw'] = metrics['cagr']
            adjusted['cagr_penalty'] = total_cagr_penalty
        
        if 'mdd' in metrics:
            adjusted['mdd'] = metrics['mdd'] + self.MDD_INFLATION
            adjusted['mdd_raw'] = metrics['mdd']
            adjusted['mdd_inflation'] = self.MDD_INFLATION
        
        return adjusted
