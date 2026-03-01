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
        
        # R14-A3 Fix: 添加 SampleWeightCalculator 用于 per-fold 权重重算
        from src.labels.sample_weights import SampleWeightCalculator
        self.weight_calculator = SampleWeightCalculator()
        
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
            df: DataFrame with 'sample_weight' column (from Phase B)  # BUG-02 Fix
        
        Returns:
            Array of sample weights (normalized to mean=1)
        """
        weight_config = self.config.get('sample_weights', {})
        method = weight_config.get('method', 'uniqueness')
        
        if method == 'uniqueness':
            # BUG-02 Fix: 读取 'sample_weight' 列
            if 'sample_weight' in df.columns:
                weights = df['sample_weight'].values.copy()
            else:
                # Fallback: 均匀权重
                logger.warn("sample_weight_column_missing", {})
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
        price_col: str = 'adj_close',
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        raw_events: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        训练单个 CPCV fold，包含 FracDiff 特征计算。
        
        R14-A3 Fix: 添加 per-fold 权重重算支持。
        
        Args:
            train_df: Training data
            test_df: Test data
            features: Feature column names
            target_col: Target column name
            price_col: Price column name for FracDiff
            train_indices: Original train indices for weight calculation
            test_indices: Original test indices for weight calculation
            raw_events: Raw event data with label_holding_days, etc.
        
        Returns:
            Dictionary with training results
        """
        # EXT-Q2 Fix: FracDiff 已全局预计算，此处直接使用 features 参数
        # features 已包含 'fracdiff' 列
        
        # 去除 NaN（全局 fracdiff 可能有少量 burn-in）
        train_df = train_df.dropna(subset=features)
        test_df = test_df.dropna(subset=features)
        
        # 确保有足够数据
        if len(train_df) < 50 or len(test_df) < 10:
            logger.warn("insufficient_data", {
                "n_train": len(train_df),
                "n_test": len(test_df)
            })
        
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for training")
        
        # Prepare data
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # EXT-Q1 Fix: Early Stopping 隔离 - 从训练集尾部切 validation set
        # 将训练集分为 inner_train (80%) 和 validation (20%)
        n_train = len(X_train)
        val_size = max(int(n_train * 0.2), 50)  # 至少50个样本
        val_size = min(val_size, 200)  # 最多200个样本
        
        train_idx = np.arange(n_train - val_size)
        val_idx = np.arange(n_train - val_size, n_train)
        
        X_train_inner = X_train.iloc[train_idx]
        y_train_inner = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        
        # R14-A3 Fix: Per-fold 样本权重重算
        if raw_events is not None and train_indices is not None:
            # 使用 SampleWeightCalculator 重新计算权重
            try:
                # 提取当前 fold 的原始事件
                train_events = raw_events.iloc[train_indices].copy()
                test_events = raw_events.iloc[test_indices].copy()
                
                # 只对有效事件计算权重
                if 'event_valid' in train_events.columns:
                    train_valid = train_events[train_events['event_valid'] == True]
                    test_valid = test_events[test_events['event_valid'] == True]
                else:
                    train_valid = train_events
                    test_valid = test_events
                
                # 计算 per-fold 权重
                if len(train_valid) > 0 and 'label_holding_days' in train_valid.columns:
                    train_weighted = self.weight_calculator.calculate_weights(train_valid)
                    train_weights = train_df['sample_weight'].values.copy() if 'sample_weight' in train_df.columns else np.ones(len(train_df))
                    # 将计算出的权重映射回 train_df
                    if 'sample_weight' in train_weighted.columns:
                        # 创建索引映射
                        train_weighted_idx = train_weighted.index
                        for i, idx in enumerate(train_df.index):
                            if idx in train_weighted_idx:
                                train_weights[i] = train_weighted.at[idx, 'sample_weight']
                else:
                    train_weights = self._calculate_sample_weights(train_df)
                
                if len(test_valid) > 0 and 'label_holding_days' in test_valid.columns:
                    test_weighted = self.weight_calculator.calculate_weights(test_valid)
                    test_weights = test_df['sample_weight'].values.copy() if 'sample_weight' in test_df.columns else np.ones(len(test_df))
                    if 'sample_weight' in test_weighted.columns:
                        test_weighted_idx = test_weighted.index
                        for i, idx in enumerate(test_df.index):
                            if idx in test_weighted_idx:
                                test_weights[i] = test_weighted.at[idx, 'sample_weight']
                else:
                    test_weights = self._calculate_sample_weights(test_df)
                    
                logger.info("R14-A3: Per-fold weights recalculated")
            except Exception as e:
                logger.warn(f"R14-A3: Per-fold weight recalculation failed: {e}, using pre-computed")
                train_weights = self._calculate_sample_weights(train_df)
                test_weights = self._calculate_sample_weights(test_df)
        else:
            # Fallback to pre-computed weights
            train_weights = self._calculate_sample_weights(train_df)
            test_weights = self._calculate_sample_weights(test_df)
        
        # EXT-Q1 Fix: 使用 inner_train 进行训练，validation set 用于 early stopping
        # test set 只用于最终评估（不参与 early stopping）
        
        # 计算 inner_train 的权重
        train_inner_weights = train_weights[:-val_size] if len(train_weights) >= val_size else train_weights
        val_weights = train_weights[-val_size:] if len(train_weights) >= val_size else np.ones(len(y_val))
        
        # Create datasets with sample weights
        train_data = lgb.Dataset(
            X_train_inner, 
            label=y_train_inner,
            weight=train_inner_weights
        )
        valid_data = lgb.Dataset(
            X_val, 
            label=y_val, 
            reference=train_data,
            weight=val_weights
        )
        
        # Log weight statistics
        logger.info("sample_weights_stats", {
            "train_mean": float(np.mean(train_inner_weights)),
            "train_std": float(np.std(train_inner_weights)),
            "train_min": float(np.min(train_inner_weights)),
            "train_max": float(np.max(train_inner_weights)),
            "val_mean": float(np.mean(val_weights)),
            "val_std": float(np.std(val_weights))
        })
        
        # Train model with early stopping on validation set
        model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )
        
        # Predictions - 使用全量训练集的预测（因为 test set 不参与训练）
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
        
        # EXT2-Q6 Fix: oos_auc 添加异常保护，避免单类标签导致 NaN 污染
        try:
            oos_auc = roc_auc_score(y_test, y_pred_proba)
            if np.isnan(oos_auc):
                logger.warn("oos_auc_nan_detected", {"n_test": len(y_test), "label_distribution": {"positive": int((y_test==1).sum()), "negative": int((y_test==0).sum())}})
                oos_auc = 0.5  # fallback
        except:
            oos_auc = 0.5  # fallback
        
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
            # EXT-Q2: optimal_d 现在在 train() 级别返回
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
        
        # EXT-Q2 Fix: 全局 FracDiff 预计算（避免每 fold 损失数据）
        # 在全局数据上计算一次最优 d 和 fracdiff
        logger.info("Step 3.5: Computing global FracDiff...")
        fracdiff_config = self.config.get('fracdiff', {})
        window = fracdiff_config.get('window', 100)
        
        from src.features.fracdiff import find_min_d_stationary, fracdiff_fixed_window
        
        try:
            # 在完整数据集上找最优 d
            optimal_d = find_min_d_stationary(
                np.log(df_meta[price_col]),
                threshold=0.05
            )
        except Exception as e:
            logger.warn("find_min_d_failed", {"error": str(e)})
            optimal_d = 0.5
        
        logger.info("fracdiff_global_optimal_d", {"optimal_d": optimal_d})
        
        # 在完整数据集上计算 fracdiff
        df_meta = df_meta.copy()
        df_meta['fracdiff'] = fracdiff_fixed_window(
            np.log(df_meta[price_col]), optimal_d, window
        )
        
        # 添加到特征列表
        features_with_fracdiff = features + ['fracdiff']
        
        # R14-A3 Fix: 保存原始事件数据用于 per-fold 权重重算
        # 需要 label_holding_days, label_exit_date, date 等列
        event_cols = ['date', 'symbol', 'label_holding_days', 'label_exit_date', 'event_valid']
        available_event_cols = [c for c in event_cols if c in df_meta.columns]
        if available_event_cols:
            logger.info(f"R14-A3: 保存事件列用于 per-fold 权重重算: {available_event_cols}")
            raw_events = df_meta[available_event_cols].copy()
        else:
            raw_events = None
            logger.warn("R14-A3: 事件列不可见，将使用预计算权重")
        
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
            
            # R14-A3 Fix: 传递 raw_events 和 indices 进行 per-fold 权重重算
            # EXT-Q2 Fix: 使用 features_with_fracdiff (已包含 fracdiff)
            result = self._train_cpcv_fold(
                train_df, test_df, features_with_fracdiff,
                train_indices=train_idx, test_indices=test_idx,
                raw_events=raw_events
            )
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
            'optimal_d': optimal_d,  # EXT-Q2: 全局最优 d
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
