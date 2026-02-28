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
            AssertionError: If hardened parameters are violated
        """
        max_depth = self.lgb_params.get('max_depth', 0)
        num_leaves = self.lgb_params.get('num_leaves', 0)
        min_data_in_leaf = self.lgb_params.get('min_data_in_leaf', 0)
        
        assert max_depth <= 3, f"OR5: max_depth must be <= 3, got {max_depth}"
        assert num_leaves <= 7, f"OR5: num_leaves must be <= 7, got {num_leaves}"
        assert min_data_in_leaf >= 100, f"OR5: min_data_in_leaf should be >= 100, got {min_data_in_leaf}"
        
        logger.info("OR5 hardened parameters validated successfully")
    
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
        target_col: str = 'meta_label'
    ) -> Dict[str, Any]:
        """
        训练单个 CPCV fold。
        
        Args:
            train_df: Training data
            test_df: Test data
            features: Feature column names
            target_col: Target column name
        
        Returns:
            Dictionary with training results
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for training")
        
        # Prepare data
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
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
        
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            loss = log_loss(y_test, y_pred_proba)
        except:
            loss = None
        
        # Feature importance
        importance = dict(zip(features, model.feature_importance(importance_type='gain')))
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'log_loss': loss,
            'best_iteration': model.best_iteration,
            'feature_importance': importance,
            'n_train': len(train_df),
            'n_test': len(test_df)
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
            
            # Find optimal d for FracDiff (placeholder - actual implementation in Step 3)
            # For now, use default d=0.5
            optimal_d = 0.5
            frac_col = f'fracdiff_{int(optimal_d*10)}'
            
            # Add fracdiff feature if not present
            if frac_col not in features:
                current_features = features + [frac_col]
            else:
                current_features = features
            
            # Train this fold
            result = self._train_cpcv_fold(train_df, test_df, current_features)
            result['path_idx'] = path_idx
            result['optimal_d'] = optimal_d
            path_results.append(result)
        
        logger.info(f"Completed {len(path_results)} paths")
        
        # Step 5: Calculate PBO
        logger.info("Step 5: Calculating PBO...")
        overfitting_result = self.overfitting_detector.check_overfitting(path_results)
        
        pbo = overfitting_result['pbo']
        pbo_passed = overfitting_result['pbo_passed']
        pbo_message = overfitting_result['pbo_message']
        
        logger.info(f"  PBO = {pbo:.2f}: {pbo_message}")
        
        if not pbo_passed:
            logger.error(f"PBO Gate BLOCKED: {pbo_message}")
            raise RuntimeError(f"PBO Gate BLOCKED: {pbo_message}")
        
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
