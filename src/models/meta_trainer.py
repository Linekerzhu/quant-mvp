"""
Meta-Labeling Trainer

Implements the complete Meta-Labeling training pipeline:
1. Base Model generates directional signals (side: +1/-1/0)
2. Filter samples where side != 0
3. Convert Triple Barrier labels to Meta-Labels (profit=1, loss=0)
4. CPCV training with FracDiff features
5. LightGBM meta-model training
6. Overfitting detection (PBO, Dummy Feature Sentinel)
7. Data debt penalty application

Author: 李得勤
Date: 2026-02-27
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml

from src.ops.event_logger import get_logger
from src.models.overfitting import OverfittingDetector, DataPenaltyApplier
from src.models.label_converter import LabelConverter

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()


class MetaTrainer:
    """
    Meta-Labeling 训练管道。
    
    完整流程:
    1. 加载 Phase A-B 产出的特征+标签数据
    2. Base Model 生成方向信号 side
    3. 过滤: 只保留 side != 0 的样本
    4. 标签转换: {profit → 1, loss → 0}（Meta-Label: 信号是否盈利）
    5. 对每个 CPCV fold:
       a. 在 train 集上用二分法找最优 FracDiff d
       b. 用该 d 值计算 train 和 test 的 FracDiff 特征
       c. 训练 LightGBM (从 training.yaml 读参数)
       d. 在 test 集上预测概率 p
    6. 汇总 15 条 path 的结果
    7. 输出: 概率校准曲线、AUC、PBO 估计
    """
    
    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Initialize MetaTrainer with configuration.
        
        Args:
            config_path: Path to training.yaml configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # LightGBM parameters
        self.lgb_params = self.config.get('lightgbm', {}).copy()
        
        # OR5-CODE T5: Extract callback parameters
        self.n_estimators = self.lgb_params.pop('n_estimators', 500)
        self.early_stopping_rounds = self.lgb_params.pop('early_stopping_rounds', 50)
        
        # Validate OR5 hardened parameters
        self._validate_or5_params()
        
        # CPCV configuration
        self.cpcv_config = self.config.get('validation', {}).get('cpcv', {})
        
        # Initialize sub-modules
        self.overfitting_detector = OverfittingDetector(
            self.config.get('overfitting', {})
        )
        self.label_converter = LabelConverter(
            self.config.get('label', {})
        )
        self.data_penalty = DataPenaltyApplier()
        
        logger.info(f"MetaTrainer initialized with config: {config_path}")
        logger.info(f"OR5 Hardened: max_depth={self.lgb_params.get('max_depth')}, "
                   f"num_leaves={self.lgb_params.get('num_leaves')}, "
                   f"min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            # Try relative to project root
            path = Path(__file__).parent.parent.parent / config_path
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_or5_params(self):
        """
        Validate OR5 Anti-Kaggle Hardening parameters.
        
        Raises:
            ValueError: If hardened parameters are violated
        """
        max_depth = self.lgb_params.get('max_depth', 0)
        num_leaves = self.lgb_params.get('num_leaves', 0)
        min_data_in_leaf = self.lgb_params.get('min_data_in_leaf', 0)
        
        # H-05 Fix: 使用显式检查，替代assert（可被-O绕过）
        if max_depth > 3:
            raise ValueError(f"OR5 VIOLATION: max_depth={max_depth} > 3")
        if num_leaves > 7:
            raise ValueError(f"OR5 VIOLATION: num_leaves={num_leaves} > 7")
        if min_data_in_leaf < 100:
            raise ValueError(f"OR5 VIOLATION: min_data_in_leaf={min_data_in_leaf} < 100")
        
        logger.info("OR5 hardened parameters validated successfully")
    
    def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算样本权重（基于 uniqueness）。
        
        根据 AFML Ch4，样本权重应基于：
        1. Uniqueness: 样本的独立程度
        2. Return: 样本的收益贡献（可选）
        
        Args:
            df: DataFrame with 'uniqueness' column (from Phase B)
        
        Returns:
            Array of sample weights (normalized to mean=1)
        """
        weight_config = self.config.get('sample_weights', {})
        method = weight_config.get('method', 'uniqueness')
        
        if method == 'uniqueness':
            # 使用 uniqueness 列（应由 Phase B 生成）
            if 'uniqueness' in df.columns:
                weights = df['uniqueness'].values.copy()
            else:
                # Fallback: 均匀权重
                logger.warn("uniqueness_column_missing", {})
                weights = np.ones(len(df))
        elif method == 'equal':
            weights = np.ones(len(df))
        else:
            weights = np.ones(len(df))
        
        # 应用 min/max 限制
        min_weight = weight_config.get('min_weight', 0.01)
        max_weight = weight_config.get('max_weight', 10.0)
        weights = np.clip(weights, min_weight, max_weight)
        
        # 归一化（保持均值=1）
        weights = weights / weights.mean()
        
        return weights
    
    def _generate_base_signals(
        self, 
        df: pd.DataFrame, 
        base_model
    ) -> pd.DataFrame:
        """
        使用 Base Model 生成方向信号。
        
        Args:
            df: DataFrame with at least [symbol, date, adj_close] columns
            base_model: Base model instance (e.g., BaseModelSMA, BaseModelMomentum)
        
        Returns:
            DataFrame with added 'side' column (+1, -1, 0)
        """
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = base_model.generate_signals(symbol_df)
            results.append(symbol_df)
        
        df_with_signals = pd.concat(results, ignore_index=True)
        
        # 过滤 side != 0（只保留有明确信号的样本）
        df_filtered = df_with_signals[df_with_signals['side'] != 0].copy()
        
        n_total = len(df_with_signals)
        n_filtered = len(df_filtered)
        logger.info(f"Base signals: {n_total} samples, {n_filtered} after filtering side!=0 "
                   f"({n_total - n_filtered} removed)")
        
        return df_filtered
    
    def _convert_to_meta_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 Triple Barrier 标签转换为 Meta-Label。
        
        Args:
            df: DataFrame with 'label' column from Triple Barrier
        
        Returns:
            DataFrame with added 'meta_label' column (0 or 1)
        """
        return self.label_converter.convert(df)
    
    def _train_cpcv_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: List[str],
        target_col: str = 'meta_label',
        price_col: str = 'adj_close'
    ) -> Dict[str, Any]:
        """
        训练单个 CPCV fold，包含 FracDiff 特征计算。
        
        Args:
            train_df: Training data
            test_df: Test data
            features: Feature column names
            target_col: Target column name
            price_col: Price column name for FracDiff
        
        Returns:
            Dictionary with training results
        """
        # OR2-02 Fix: 集成 FracDiff 特征计算
        from src.features.fracdiff import find_min_d_stationary, fracdiff_fixed_window
        
        # Step 1: 在训练集上找最优 d
        fracdiff_config = self.config.get('fracdiff', {})
        window = fracdiff_config.get('window', 100)
        
        try:
            # MEDIUM-01 Fix: d在log(price)空间搜索，与应用空间一致
            optimal_d = find_min_d_stationary(
                np.log(train_df[price_col]),  # log space
                threshold=0.05
            )
        except Exception as e:
            logger.warn("find_min_d_failed", {"error": str(e)})
            optimal_d = 0.5  # fallback to default
        
        logger.info("fracdiff_optimal_d", {"optimal_d": optimal_d})
        
        # Step 2: 计算 FracDiff 特征（train + test）
        # OR2-02 Fix #2: 传Series不是.values
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        train_df['fracdiff'] = fracdiff_fixed_window(
            np.log(train_df[price_col]), optimal_d, window  # OR2-03: 使用log(price)
        )
        test_df['fracdiff'] = fracdiff_fixed_window(
            np.log(test_df[price_col]), optimal_d, window  # OR2-03: 使用log(price)
        )
        
        # Step 3: 添加到特征列表（只用fracdiff，不是fracdiff_5）
        current_features = features + ['fracdiff']
        
        # Step 4: 去除 NaN（FracDiff burn-in period）
        train_df = train_df.dropna(subset=['fracdiff'])
        test_df = test_df.dropna(subset=['fracdiff'])
        
        # 确保有足够数据
        if len(train_df) < 50 or len(test_df) < 10:
            logger.warn("insufficient_data_after_fracdiff", {
                "n_train": len(train_df),
                "n_test": len(test_df)
            })
        
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for training")
        
        # Prepare data - C-02 Fix: 使用 current_features 包含 fracdiff
        X_train = train_df[current_features]
        y_train = train_df[target_col]
        X_test = test_df[current_features]
        y_test = test_df[target_col]
        
        # C-03 Fix: 计算样本权重
        train_weights = self._calculate_sample_weights(train_df)
        test_weights = self._calculate_sample_weights(test_df)
        
        # Create datasets with sample weights
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            weight=train_weights  # C-03: 传入样本权重
        )
        valid_data = lgb.Dataset(
            X_test, 
            label=y_test, 
            reference=train_data,
            weight=test_weights  # C-03: 传入样本权重
        )
        
        # Log weight statistics
        logger.info("sample_weights_stats", {
            "train_mean": float(np.mean(train_weights)),
            "train_std": float(np.std(train_weights)),
            "train_min": float(np.min(train_weights)),
            "train_max": float(np.max(train_weights)),
            "test_mean": float(np.mean(test_weights)),
            "test_std": float(np.std(test_weights))
        })
        
        # Train model
        model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
        
        # OR2-01 Fix: 计算 IS (train) AUC 和 OOS (test) AUC
        # 对训练集做预测得到 IS AUC
        y_train_pred_proba = model.predict(X_train, num_iteration=model.best_iteration)
        try:
            is_auc = roc_auc_score(y_train, y_train_pred_proba)
        except:
            is_auc = 0.5  # fallback
        
        oos_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            loss = log_loss(y_test, y_pred_proba)
        except:
            loss = None
        
        # Feature importance
        importance = dict(zip(current_features, model.feature_importance(importance_type='gain')))
        
        return {
            'auc': oos_auc,
            'accuracy': accuracy,
            'log_loss': loss,
            'best_iteration': model.best_iteration,
            'feature_importance': importance,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'optimal_d': optimal_d,  # C-02: 返回最优d值
            'is_auc': is_auc,  # OR2-01 Fix: 真实的IS AUC
            'oos_auc': oos_auc,  # OOS AUC
            'positive_ratio': float((y_test == 1).mean())  # HIGH-02: for DSR baseline
        }
    
    def apply_data_penalty(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        应用数据技术债惩罚性拨备。
        
        Args:
            metrics: Dictionary of performance metrics
        
        Returns:
            Dictionary with adjusted metrics
        """
        return self.data_penalty.apply(metrics)
    
    def train(
        self,
        df: pd.DataFrame,
        base_model,
        features: List[str],
        price_col: str = 'adj_close'
    ) -> Dict[str, Any]:
        """
        执行完整的 Meta-Labeling 训练管道。
        
        Args:
            df: DataFrame with features and labels
            base_model: Base model instance for signal generation
            features: List of feature column names
            price_col: Price column name for FracDiff
        
        Returns:
            Dictionary with complete training results
        """
        from src.models.purged_kfold import CombinatorialPurgedKFold
        
        logger.info("=" * 60)
        logger.info("Starting Meta-Labeling Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Generate Base Model signals
        logger.info("Step 1: Generating Base Model signals...")
        df_signals = self._generate_base_signals(df, base_model)
        
        # Step 2: Convert to Meta-Labels
        logger.info("Step 2: Converting to Meta-Labels...")
        df_meta = self._convert_to_meta_labels(df_signals)
        
        if len(df_meta) == 0:
            raise ValueError("No samples remaining after meta-label conversion")
        
        # Step 3: Initialize CPCV
        logger.info("Step 3: Initializing CPCV...")
        cpcv = CombinatorialPurgedKFold.from_config(self.config)
        n_paths = cpcv.get_n_paths()
        logger.info(f"CPCV: {n_paths} paths, n_splits={cpcv.n_splits}, "
                   f"n_test_splits={cpcv.n_test_splits}")
        
        # Step 4: CPCV Training Loop
        logger.info("Step 4: Training CPCV paths...")
        path_results = []
        
        for path_idx, (train_idx, test_idx) in enumerate(
            cpcv.split(df_meta), 1
        ):
            logger.info(f"  Path {path_idx}/{n_paths}: "
                       f"train={len(train_idx)}, test={len(test_idx)}")
            
            train_df = df_meta.iloc[train_idx].copy()
            test_df = df_meta.iloc[test_idx].copy()
            
            # OR2-02 Fix: 直接传原始features给fold，fold内部负责FracDiff计算
            # 不要在此处构造fracdiff_5列名
            
            # Train this fold
            result = self._train_cpcv_fold(train_df, test_df, features)  # 用原始features
            result['path_idx'] = path_idx
            path_results.append(result)
        
        logger.info(f"Completed {len(path_results)} paths")
        
        # Step 5: Calculate PBO
        logger.info("Step 5: Calculating PBO...")
        overfitting_result = self.overfitting_detector.check_overfitting(path_results)
        
        pbo = overfitting_result['pbo']
        pbo_passed = overfitting_result['pbo_passed']
        pbo_message = overfitting_result['pbo_message']
        
        # HIGH-04 Fix: Use overall_passed (PBO + DSR + Dummy)
        if not overfitting_result['overall_passed']:
            logger.error(f"Overfitting Gate BLOCKED: PBO={pbo_message}, DSR={overfitting_result.get('dsr_message', 'N/A')}")
            raise RuntimeError(f"Overfitting Gate BLOCKED: PBO={pbo_message}, DSR={overfitting_result.get('dsr_message', 'N/A')}")
        
        # Step 6: Aggregate results
        aucs = [r['auc'] for r in path_results]
        accuracies = [r['accuracy'] for r in path_results]
        
        results = {
            'paths': path_results,
            'n_paths': len(path_results),
            'pbo': pbo,
            'pbo_status': pbo_message,
            'overfitting_check': overfitting_result,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'min_auc': np.min(aucs),
            'max_auc': np.max(aucs),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
        }
        
        logger.info("=" * 60)
        logger.info("Meta-Labeling Training Complete")
        logger.info(f"  Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        logger.info(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")
        logger.info(f"  PBO: {pbo:.2f} ({pbo_message})")
        logger.info("=" * 60)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成训练报告。
        
        Args:
            results: Results dictionary from train()
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "Meta-Labeling Training Report",
            "=" * 60,
            "",
            f"CPCV Paths: {results['n_paths']}",
            f"PBO: {results['pbo']:.2f} ({results['pbo_status']})",
            "",
            "Performance Metrics:",
            f"  AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}",
            f"       [{results['min_auc']:.4f}, {results['max_auc']:.4f}]",
            f"  Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# Convenience function for quick training
def train_meta_model(
    df: pd.DataFrame,
    base_model,
    features: List[str],
    config_path: str = "config/training.yaml"
) -> Dict[str, Any]:
    """
    Quick training function.
    
    Args:
        df: Training data
        base_model: Base model instance
        features: Feature columns
        config_path: Config file path
    
    Returns:
        Training results
    """
    trainer = MetaTrainer(config_path)
    return trainer.train(df, base_model, features)
