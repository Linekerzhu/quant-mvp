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
        
        基于 Bailey & López de Prado (2017) 的简化方法：
        - 对每条路径，比较 IS (in-sample) 和 OOS (out-of-sample) 性能
        - 如果最优 IS 模型在 OOS 上表现差 → 过拟合
        
        方法：计算 IS vs OOS 性能的排名差异
        - IS 排名高但 OOS 排名低 → 过拟合
        - PBO = OOS排名低于IS排名的概率
        
        Args:
            path_results: List of result dicts from each CPCV path
                        Each dict should contain:
                        - 'is_auc': IS (in-sample) AUC
                        - 'oos_auc': OOS (out-of-sample) AUC
                        (fallback to 'auc' if not present)
        
        Returns:
            PBO value between 0 and 1
            - PBO > 0.5: 高过拟合风险
            - PBO < 0.3: 低过拟合风险
        """
        # 提取 IS 和 OOS AUC
        is_aucs = []
        oos_aucs = []
        
        for r in path_results:
            # 优先使用分离的 IS/OOS，如果没有则使用整个路径的 AUC
            is_auc = r.get('is_auc', r.get('auc', 0.5))
            oos_auc = r.get('oos_auc', r.get('auc', 0.5))
            is_aucs.append(is_auc)
            oos_aucs.append(oos_auc)
        
        n = len(is_aucs)
        
        if n == 0:
            return 1.0
        
        # 如果 IS 和 OOS 相同（未分离），使用简化方法：基于AUC方差
        if len(set(is_aucs)) == 1 and len(set(oos_aucs)) == 1:
            # 使用简化方法：基于方差的保守估计
            all_aucs = is_aucs + oos_aucs
            mean_auc = np.mean(all_aucs)
            std_auc = np.std(all_aucs, ddof=1)
            
            # 变异系数
            cv = std_auc / mean_auc if mean_auc > 0 else 0
            
            # 映射到 PBO：CV越高，过拟合风险越高
            # CV < 0.05 -> PBO ≈ 0
            # CV > 0.2 -> PBO ≈ 1
            pbo = min(1.0, max(0.0, (cv - 0.05) / 0.15))
            logger.info("pbo_variance_method", {"cv": cv, "pbo": pbo})
            return float(pbo)
        
        # 真正的 PBO 方法：比较 IS vs OOS 排名
        # 1. IS 排名（降序，最好的 IS 排第一）
        is_ranking = np.argsort(np.argsort(is_aucs)[::-1])
        
        # 2. OOS 排名（降序，最好的 OOS 排第一）
        oos_ranking = np.argsort(np.argsort(oos_aucs)[::-1])
        
        # 3. 计算排名差异：IS排名 - OOS排名
        # 如果 IS 排名高（数字小）但 OOS 排名低（数字大）→ 过拟合
        rank_diff = is_ranking - oos_ranking
        
        # 4. PBO = 排名下降的比例（IS比OOS差的概率）
        pbo = np.mean(rank_diff > 0)  # IS比OOS差的概率
        
        logger.info("pbo_is_oos_method", {
            "n_paths": n,
            "mean_is_auc": np.mean(is_aucs),
            "mean_oos_auc": np.mean(oos_aucs),
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
        
        # 使用 accuracy 作为业绩代理指标（替代 Sharpe Ratio）
        # 在 meta-labeling 中，accuracy 比 sharpe 更稳定
        metrics = [r.get('accuracy', r.get('auc', 0.5)) for r in path_results]
        
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
        
        # Deflated Sharpe (assuming SR₀ = 0.5 for accuracy, 0 for raw sharpe)
        # For accuracy, baseline is 0.5 (random guessing)
        baseline = 0.5
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
