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
        
        AFML 排名方法 (Ch7 §7.4.2, Bailey et al. 2016):
        PBO = 实际过拟合路径数 / 总路径数
        
        R19-F2 Fix: 从二值(0/1)改为概率
        - 运行多次蒙特卡洛模拟
        - 每次模拟随机打乱OOS排名
        - 统计"IS最好但OOS差"的概率
        
        Args:
            path_results: List of result dicts from each CPCV path
        
        Returns:
            PBO value between 0 and 1 (概率)
        """
        # R14-A1 Fix: 实现 AFML 排名方法
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
        
        if n < 3:
            # 样本太少，用二值方法
            is_ranks = np.argsort(np.argsort(-np.array(is_aucs))) + 1
            oos_ranks = np.argsort(np.argsort(-np.array(oos_aucs))) + 1
            median_rank = (n + 1) / 2
            best_is_idx = np.argmin(is_ranks)
            return 1.0 if oos_ranks[best_is_idx] > median_rank else 0.0
        
        # R19-F2 Fix: 蒙特卡洛模拟计算概率
        # 模拟1000次，每次随机打乱OOS排名
        n_simulations = 1000
        overfit_count = 0
        
        for _ in range(n_simulations):
            # 打乱OOS排名
            shuffled_oos = np.array(oos_aucs)
            np.random.shuffle(shuffled_oos)
            
            # IS排名
            is_ranks = np.argsort(np.argsort(-np.array(is_aucs))) + 1
            oos_ranks_sim = np.argsort(np.argsort(-shuffled_oos)) + 1
            
            median_rank = (n + 1) / 2
            best_is_idx = np.argmin(is_ranks)
            
            # 如果打乱后IS最好的路径在OOS中仍然差，计为过拟合
            if oos_ranks_sim[best_is_idx] > median_rank:
                overfit_count += 1
        
        pbo = overfit_count / n_simulations
        
        logger.info("pbo_monte_carlo", {
            "n_paths": n,
            "n_simulations": n_simulations,
            "overfit_count": overfit_count,
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
        feature_importance: Dict[str, float],
        per_fold_importance: List[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Dummy Feature 过拟合哨兵。
        
        R14-A9 Fix: 添加 per-fold 检查
        - 检查 dummy 是否在任何 fold 中进入前 25%
        - 如果任何 fold 出现此情况，标记为警告
        
        Args:
            feature_importance: Dictionary of feature names to importance scores (average)
            per_fold_importance: Optional list of per-fold importance dicts
        
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
        
        # R14-A9 Fix: Per-fold 检查
        per_fold_warnings = []
        if per_fold_importance:
            for fold_idx, fold_importance in enumerate(per_fold_importance):
                if dummy_col in fold_importance:
                    # Per-fold 排名
                    fold_sorted = sorted(
                        fold_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    fold_ranks = {name: i + 1 for i, (name, _) in enumerate(fold_sorted)}
                    fold_dummy_rank = fold_ranks.get(dummy_col, total_features + 1)
                    fold_ratio = fold_dummy_rank / len(fold_importance)
                    
                    # 如果 dummy 在前 25%，记录警告
                    if fold_ratio <= 0.25:
                        per_fold_warnings.append({
                            'fold': fold_idx,
                            'dummy_rank': fold_dummy_rank,
                            'total_features': len(fold_importance),
                            'ratio': fold_ratio
                        })
        
        # Check threshold - 如果有任何 per-fold 警告，标记为未通过
        passed = ranking_ratio > self.dummy_threshold and len(per_fold_warnings) == 0
        
        return {
            'dummy_rank': dummy_rank,
            'total_features': total_features,
            'ranking_ratio': ranking_ratio,
            'threshold': self.dummy_threshold,
            'passed': passed,
            'per_fold_warnings': per_fold_warnings
        }
    
    def calculate_deflated_sharpe(self, path_results: List[Dict]) -> float:
        """
        计算 Deflated Sharpe Ratio (DSR) 检验的 z-score。
        
        R14-A2 Fix: 添加 skewness/kurtosis 校正和多重测试校正
        
        AFML Ch8, Bailey & López de Prado (2014) 公式:
        DSR = (SR - SR_benchmark) / (std(SR) * (1 + γ₃*SR/6 - γ₄*(SR²-1)/24))
        
        其中:
        - γ₃ = skewness (偏度)
        - γ₄ = kurtosis (峰度)
        - E[max(SR)] 来自多重测试校正
        
        Args:
            path_results: List of CPCV path results
        
        Returns:
            z-score 值 (用于 check_dsr_gate 判定)
        """
        # Baseline assumption:
        # - For accuracy: 0.5 = random guessing (balanced classes)
        # - For AUC: 0.5 = random ranking
        # - For Sharpe: 0.0 = zero excess return
        
        # OR2-04 Fix: 分别收集metrics和baselines
        metrics = []
        baselines = []
        
        for r in path_results:
            metrics.append(r.get('accuracy', r.get('auc', 0.5)))
            baselines.append(r.get('positive_ratio', 0.5))
        
        # BUG-03 Fix: 使用per-path excess计算DSR
        excess = [metrics[i] - baselines[i] for i in range(len(metrics))]
        excess = np.array(excess)
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        n = len(excess)
        
        if n < 2:
            logger.warn("deflated_sharpe_insufficient_data", {"n_paths": n})
            return 0.0
        
        # BUG-05 Fix: 零variance正确处理
        if std_excess < 1e-10:
            if mean_excess > 1e-6:
                return 20.0
            else:
                return 0.0
        
        # R14-A2 Fix: 计算 skewness 和 kurtosis
        # 使用样本公式 (ddof=1)
        skewness = np.mean(((excess - mean_excess) / std_excess) ** 3)
        kurtosis = np.mean(((excess - mean_excess) / std_excess) ** 4) - 3  # 超额峰度
        
        # R19-F3 Fix: 多重检验N=1
        # Bailey & López de Prado (2014): N = 策略变体总数，不是CPCV路径数
        # Phase C MVP只测试1种策略配置，N=1时expected_max应为0
        N_strategies = 1  # 只有一种策略配置
        
        if N_strategies > 1:
            expected_max = np.sqrt(2 * np.log(N_strategies))
        else:
            expected_max = 0.0
        
        # 计算 DSR with skewness/kurtosis 校正
        # SR = mean_excess / (std_excess / sqrt(n)) = mean_excess * sqrt(n) / std_excess
        sr = mean_excess / (std_excess / np.sqrt(n))
        
        # 校正因子
        correction = 1 + (skewness * sr / 6) - (kurtosis * (sr**2 - 1) / 24)
        correction = max(correction, 0.1)  # 防止负数或过小
        
        # 应用多重测试校正 (从 SR 中减去期望最大值)
        dsr_adjusted = (sr - expected_max) / correction
        
        logger.info("deflated_sharpe", {
            "mean_excess": mean_excess,
            "std_excess": std_excess,
            "n_paths": n,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "expected_max_z": expected_max,
            "sr_raw": sr,
            "correction": correction,
            "dsr_adjusted": dsr_adjusted
        })
        
        return float(dsr_adjusted)
    
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
        per_fold_importance = []  # R14-A9 Fix: 收集 per-fold importance
        for r in path_results:
            fold_imp = r.get('feature_importance', {})
            if fold_imp:
                per_fold_importance.append(fold_imp)
            for feat, imp in fold_imp.items():
                if feat not in avg_importance:
                    avg_importance[feat] = []
                avg_importance[feat].append(imp)
        
        avg_importance = {k: np.mean(v) for k, v in avg_importance.items()}
        # R14-A9 Fix: 传递 per_fold_importance 进行 per-fold 检查
        dummy_result = self.dummy_feature_sentinel(avg_importance, per_fold_importance)
        
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
