#!/usr/bin/env python3
"""
Meta-Labeling Training Pipeline Entry Point

Runs the complete Meta-Labeling training pipeline:
1. Load processed data with features and labels
2. Initialize Base Model and MetaTrainer
3. Run CPCV training with all safety checks
4. Generate performance report
5. Export results

Usage:
    python run_pipeline.py --config config/training.yaml --data data/processed/
    python run_pipeline.py --model-type sma --symbols AAPL,MSFT

Author: 李得勤
Date: 2026-02-27
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Meta-Labeling Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_pipeline.py
  
  # Use specific config and data
  python run_pipeline.py --config config/training.yaml --data data/processed/features.parquet
  
  # Use SMA base model with specific symbols
  python run_pipeline.py --model-type sma --symbols AAPL,MSFT,GOOGL
  
  # Enable verbose logging
  python run_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/training.yaml',
        help='Path to training configuration file (default: config/training.yaml)'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/processed/',
        help='Path to processed data directory or file (default: data/processed/)'
    )
    
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        choices=['sma', 'momentum'],
        default='sma',
        help='Base model type: sma (moving average) or momentum (default: sma)'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default=None,
        help='Comma-separated list of symbols to use (default: all available)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/meta_training_results.yaml',
        help='Output file path for results (default: output/meta_training_results.yaml)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated list of feature columns (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running training'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_data(data_path: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load processed data from file or directory.
    
    Args:
        data_path: Path to data file or directory
        symbols: Optional list of symbols to filter
    
    Returns:
        DataFrame with processed data
    """
    path = Path(data_path)
    
    if path.is_file():
        # Load single file
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    elif path.is_dir():
        # Load all files in directory
        files = list(path.glob('*.parquet')) + list(path.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"No data files found in {data_path}")
        
        dfs = []
        for f in files:
            if f.suffix == '.parquet':
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f, parse_dates=['date']))
        df = pd.concat(dfs, ignore_index=True)
    
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Filter symbols if specified
    if symbols:
        df = df[df['symbol'].isin(symbols)].copy()
    
    logger.info(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
    
    return df


def get_base_model(model_type: str, config: dict):
    """
    Initialize base model based on type.
    
    Args:
        model_type: 'sma' or 'momentum'
        config: Training configuration
    
    Returns:
        Base model instance
    """
    from src.signals.base import SignalModelRegistry
    
    # Use registry to create model
    try:
        return SignalModelRegistry.create(model_type)
    except ValueError:
        # Fallback to old behavior for backward compatibility
        from src.signals.base_models import BaseModelSMA, BaseModelMomentum
        
        if model_type == 'sma':
            return BaseModelSMA(fast_window=20, slow_window=60)
        elif model_type == 'momentum':
            return BaseModelMomentum(window=20)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def auto_detect_features(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect feature columns from DataFrame.
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_cols = {
        'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close',
        'volume', 'label', 'side', 'meta_label', 'label_exit_date',
        'label_take_profit', 'label_stop_loss', 'entry_date', 'exit_date'
    }
    
    # Select numeric columns not in exclude list
    features = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    logger.info(f"Auto-detected {len(features)} features: {features[:5]}...")
    
    return features


def validate_config(config_path: str) -> bool:
    """
    Validate training configuration.
    
    Args:
        config_path: Path to config file
    
    Returns:
        True if valid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required = ['lightgbm', 'label', 'validation']
        for section in required:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check LightGBM parameters
        lgb = config['lightgbm']
        assert lgb.get('max_depth', 0) <= 3, "OR5: max_depth must be <= 3"
        assert lgb.get('num_leaves', 0) <= 7, "OR5: num_leaves must be <= 7"
        
        logger.info("Configuration validation passed")
        return True
    
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def save_results(results: dict, output_path: str):
    """
    Save training results to file.
    
    Args:
        results: Training results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python for YAML serialization
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    results_clean = convert(results)
    
    with open(output_path, 'w') as f:
        yaml.dump(results_clean, f, default_flow_style=False)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Meta-Labeling Training Pipeline")
    logger.info("=" * 60)
    
    # Validate configuration
    logger.info(f"Loading configuration from {args.config}...")
    if not validate_config(args.config):
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - configuration validated, exiting")
        sys.exit(0)
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    symbols = args.symbols.split(',') if args.symbols else None
    try:
        df = load_data(args.data, symbols)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Initialize base model
    logger.info(f"Initializing {args.model_type} base model...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    base_model = get_base_model(args.model_type, config)
    
    # Detect features
    if args.features:
        features = args.features.split(',')
    else:
        features = auto_detect_features(df)
    
    if not features:
        logger.error("No features detected! Please specify features manually.")
        sys.exit(1)
    
    logger.info(f"Using {len(features)} features")
    
    # Run training
    logger.info("Starting training...")
    try:
        from src.models.meta_trainer import MetaTrainer
        
        trainer = MetaTrainer(config_path=args.config)
        results = trainer.train(df, base_model, features)
        
        # Generate and print report
        report = trainer.generate_report(results)
        print("\n" + report)
        
        # Save results
        save_results(results, args.output)
        
        logger.info("Training pipeline completed successfully")
        
    except RuntimeError as e:
        logger.error(f"Training blocked: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
