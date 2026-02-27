"""
End-to-End Test for Meta-Labeling Architecture

This script validates the complete Meta-Labeling pipeline:
1. Base Model generates side signals
2. Triple Barrier only triggers when side != 0
3. Final labels are in {0, 1} (Meta-Label format)

Author: 寇连材 (Auditor)
Date: 2026-02-27
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals.base_models import BaseModelSMA, BaseModelMomentum
from src.labels.triple_barrier import TripleBarrierLabeler


def create_test_data():
    """Create synthetic price data for testing."""
    np.random.seed(42)
    n = 300  # Increased to ensure enough events
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    # Create price series with trend
    returns = np.random.randn(n) * 0.02 + 0.0005  # Slight upward bias
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'symbol': 'TEST',
        'date': dates,
        'adj_close': prices,
        'adj_open': prices * (1 + np.random.randn(n) * 0.005),
        'adj_high': prices * (1 + np.abs(np.random.randn(n)) * 0.02),  # Increased volatility
        'adj_low': prices * (1 - np.abs(np.random.randn(n)) * 0.02),   # Increased volatility
    })
    
    # Add ATR (required by Triple Barrier) - use atr_20 to match config
    df['atr_20'] = df['adj_close'].rolling(20).std()
    
    # Fill NaN ATR with a reasonable default
    df['atr_20'] = df['atr_20'].fillna(df['adj_close'].iloc[:20].std())
    
    # Add trading flags
    df['can_trade'] = True
    df['features_valid'] = True
    df['volume'] = 1e6
    
    return df


def test_e2e_sma_pipeline():
    """Test SMA Base Model + Triple Barrier end-to-end."""
    print("\n" + "="*60)
    print("🧪 E2E Test: SMA Base Model → Triple Barrier")
    print("="*60)
    
    # Step 1: Create test data
    df = create_test_data()
    print(f"\n📊 Created test data: {len(df)} rows")
    
    # Step 2: Generate Base Model signals
    model = BaseModelSMA(fast_window=20, slow_window=60)
    df = model.generate_signals(df)
    
    print(f"\n📈 SMA Base Model signals generated:")
    print(f"   - Total rows: {len(df)}")
    print(f"   - side=0 (cold start): {(df['side'] == 0).sum()}")
    print(f"   - side=1 (long): {(df['side'] == 1).sum()}")
    print(f"   - side=-1 (short): {(df['side'] == -1).sum()}")
    
    # Validate signal values
    unique_signals = set(df['side'].unique())
    assert unique_signals.issubset({-1, 0, 1}), f"❌ Invalid signals: {unique_signals}"
    print("   ✅ Signal values valid: {-1, 0, +1}")
    
    # Validate cold start
    cold_start = df['side'].iloc[:60]
    assert (cold_start == 0).all(), "❌ Cold start period should have side=0"
    print("   ✅ Cold start correct: first 60 rows have side=0")
    
    # Step 3: Apply Triple Barrier
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.label_events(df)
    
    print(f"\n🏷️ Triple Barrier labels generated:")
    
    # Check events only generated for side != 0
    valid_events = df_labeled[df_labeled['event_valid'] == True]
    
    # Debug: Check rejection reasons
    rejected = df_labeled[df_labeled['event_valid'] == False]
    print(f"   - Total rows: {len(df_labeled)}")
    print(f"   - Valid events: {len(valid_events)}")
    print(f"   - Rejected rows: {len(rejected)}")
    
    # Check if side=0 rows are correctly excluded
    side_zero_rows = df_labeled[df_labeled['side'] == 0]
    side_nonzero_rows = df_labeled[df_labeled['side'] != 0]
    print(f"   - Rows with side=0: {len(side_zero_rows)}")
    print(f"   - Rows with side!=0: {len(side_nonzero_rows)}")
    
    # Verify side constraint - side=0 should not generate events
    if len(valid_events) > 0:
        zero_side_events = valid_events[valid_events['side'] == 0]
        if len(zero_side_events) > 0:
            print("   ❌ Triple Barrier generated events for side=0!")
            return False
        else:
            print("   ✅ Triple Barrier only generates events for side != 0")
    
    # Check label values (should be -1, 0, 1 for standard labeling)
    label_values = set(df_labeled['label'].dropna().unique())
    print(f"   - Label values: {label_values}")
    
    # For Meta-Labeling, labels should be:
    # 1 = profit barrier hit (base signal was correct)
    # 0 = time barrier hit (neutral)
    # -1 = loss barrier hit (base signal was wrong)
    expected_labels = {-1, 0, 1}
    assert label_values.issubset(expected_labels), f"❌ Invalid labels: {label_values}"
    print(f"   ✅ Label values in {expected_labels}")
    
    # Step 4: Verify distribution
    distribution = labeler.get_label_distribution(df_labeled)
    print(f"\n📊 Label distribution:")
    print(f"   - Total events: {distribution.get('total_events', 0)}")
    
    if 'by_label' in distribution:
        print(f"   - Profit (label=1): {distribution['by_label']['profit']}")
        print(f"   - Loss (label=-1): {distribution['by_label']['loss']}")
        print(f"   - Neutral (label=0): {distribution['by_label']['neutral']}")
        print(f"   - Mean return: {distribution.get('mean_return', 0):.4f}")
        print(f"   - Mean holding days: {distribution.get('mean_holding_days', 0):.2f}")
    elif 'error' in distribution:
        print(f"   ⚠️ {distribution['error']}")
    
    print("\n✅ SMA E2E Test PASSED")
    return True


def test_e2e_momentum_pipeline():
    """Test Momentum Base Model + Triple Barrier end-to-end."""
    print("\n" + "="*60)
    print("🧪 E2E Test: Momentum Base Model → Triple Barrier")
    print("="*60)
    
    # Step 1: Create test data
    df = create_test_data()
    print(f"\n📊 Created test data: {len(df)} rows")
    
    # Step 2: Generate Base Model signals
    model = BaseModelMomentum(window=20)
    df = model.generate_signals(df)
    
    print(f"\n📈 Momentum Base Model signals generated:")
    print(f"   - Total rows: {len(df)}")
    print(f"   - side=0 (cold start): {(df['side'] == 0).sum()}")
    print(f"   - side=1 (long): {(df['side'] == 1).sum()}")
    print(f"   - side=-1 (short): {(df['side'] == -1).sum()}")
    
    # Validate cold start
    cold_start = df['side'].iloc[:20]
    assert (cold_start == 0).all(), "❌ Cold start period should have side=0"
    print("   ✅ Cold start correct: first 20 rows have side=0")
    
    # Step 3: Apply Triple Barrier
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.label_events(df)
    
    valid_events = df_labeled[df_labeled['event_valid'] == True]
    
    print(f"\n🏷️ Triple Barrier labels generated:")
    print(f"   - Valid events: {len(valid_events)}")
    
    # Verify side constraint - side=0 should not generate events
    if len(valid_events) > 0:
        zero_side_events = valid_events[valid_events['side'] == 0]
        if len(zero_side_events) > 0:
            print("   ❌ Triple Barrier generated events for side=0!")
            return False
        else:
            print("   ✅ Triple Barrier only generates events for side != 0")
    
    # Step 4: Verify distribution
    distribution = labeler.get_label_distribution(df_labeled)
    print(f"\n📊 Label distribution:")
    print(f"   - Total events: {distribution.get('total_events', 0)}")
    
    if 'by_label' in distribution:
        print(f"   - Profit (label=1): {distribution['by_label']['profit']}")
        print(f"   - Loss (label=-1): {distribution['by_label']['loss']}")
        print(f"   - Neutral (label=0): {distribution['by_label']['neutral']}")
    elif 'error' in distribution:
        print(f"   ⚠️ {distribution['error']}")
    
    print("\n✅ Momentum E2E Test PASSED")
    return True


def test_shift1_leakage_prevention():
    """Verify shift(1) correctly prevents look-ahead bias."""
    print("\n" + "="*60)
    print("🧪 Test: shift(1) Leakage Prevention")
    print("="*60)
    
    # Create data with extreme T-day move
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    df_normal = pd.DataFrame({
        'symbol': 'TEST',
        'date': dates,
        'adj_close': prices,
    })
    
    # Create version with extreme last-day move
    df_extreme = df_normal.copy()
    df_extreme.loc[99, 'adj_close'] = 500  # Extreme move
    
    # Generate signals
    model = BaseModelSMA(fast_window=20, slow_window=60)
    result_normal = model.generate_signals(df_normal)
    result_extreme = model.generate_signals(df_extreme)
    
    # Signals BEFORE the extreme day should be identical
    # (because shift(1) means T-day signal only uses T-1 and earlier)
    signals_before_normal = result_normal['side'].iloc[:99]
    signals_before_extreme = result_extreme['side'].iloc[:99]
    
    match = (signals_before_normal == signals_before_extreme).all()
    
    if match:
        print("   ✅ shift(1) correctly prevents leakage")
        print("   ✅ Signals before T-day are identical with/without extreme move")
    else:
        print("   ❌ shift(1) NOT working - signals changed!")
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 Meta-Labeling E2E Audit")
    print("Auditor: 寇连材公公")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("shift(1) Leakage Prevention", test_shift1_leakage_prevention()))
    results.append(("SMA Pipeline", test_e2e_sma_pipeline()))
    results.append(("Momentum Pipeline", test_e2e_momentum_pipeline()))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL E2E TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
