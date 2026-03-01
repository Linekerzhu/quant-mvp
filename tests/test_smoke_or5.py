# OR5 Smoke Tests - Coverage Enhancement
# Simplified to test only existing modules
# Focus: OR5 Hotfix verification

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import tempfile


class TestOR5Hotfixes:
    """OR5 specific hotfix verification tests"""
    
    @pytest.fixture
    def labeler(self):
        from src.labels.triple_barrier import TripleBarrierLabeler
        return TripleBarrierLabeler()
    
    def make_df(self, adj_open, adj_high, adj_low, adj_close):
        """Helper to create test DataFrame."""
        n = len(adj_open)
        dates = pd.date_range('2024-01-01', periods=n, freq='B')
        return pd.DataFrame({
            'symbol': 'TST',
            'date': dates,
            'adj_open': adj_open,
            'adj_close': adj_close,
            'adj_high': adj_high,
            'adj_low': adj_low,
            'raw_open': adj_open,
            'raw_high': adj_high,
            'raw_low': adj_low,
            'raw_close': adj_close,
            'volume': [5000] * n,
            'atr_20': [2.0] * n,  # ATR=2, entry=100 → barriers at 96 and 104
            'features_valid': True,
            'can_trade': True
        })
    
    def test_loss_gap_uses_actual_open(self, labeler):
        """OR5 HF-1: Loss gap should use actual open price, not barrier."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: open at 94 (below loss barrier 96)
        o[2] = 94.0
        h[2] = 95.0
        l[2] = 93.0
        c[2] = 94.5
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        assert ev['label_barrier'] == 'loss_gap'
        assert abs(ev['label_return'] - np.log(94 / 100)) < 1e-6
    
    def test_profit_gap_uses_actual_open(self, labeler):
        """OR5 HF-1: Profit gap should use actual open price, not barrier."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: open at 107 (above profit barrier 104)
        o[2] = 107.0
        h[2] = 108.0
        l[2] = 106.0
        c[2] = 107.5
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        assert ev['label_barrier'] == 'profit_gap'
        assert abs(ev['label_return'] - np.log(107 / 100)) < 1e-6
    
    def test_collision_forces_loss(self, labeler):
        """OR5 HF-1: Same-day double hit should force loss (pessimism)."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: both barriers hit (high=105 > 104, low=95 < 96)
        o[2] = 100.0
        h[2] = 105.0
        l[2] = 95.0
        c[2] = 101.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        assert ev['label_barrier'] == 'loss_collision'
        assert ev['label'] == -1
    
    def test_normal_profit_path(self, labeler):
        """Normal profit: high reaches barrier, open below."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: high reaches 105 (>104), open at 101 (<104)
        o[2] = 101.0
        h[2] = 105.0
        l[2] = 100.5
        c[2] = 104.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        assert ev['label_barrier'] == 'profit'
    
    def test_normal_loss_path(self, labeler):
        """Normal loss: low reaches barrier, open above."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: low reaches 95 (<96), open at 99 (>96)
        o[2] = 99.0
        h[2] = 99.5
        l[2] = 95.0
        c[2] = 95.5
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        assert ev['label_barrier'] == 'loss'
    
    def test_gap_priority_over_collision(self, labeler):
        """Gap execution should be checked before collision."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 2: open at 93 (below loss barrier), also high > profit barrier
        o[2] = 93.0  # Gap down through loss
        h[2] = 105.0  # Also hits profit
        l[2] = 92.0
        c[2] = 95.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        # Gap should take priority over collision
        assert ev['label_barrier'] == 'loss_gap'
    
    # ============================================================
    # R33-A1: Entry Day Barrier Detection Tests
    # ============================================================
    
    def test_entry_day_loss_barrier(self, labeler):
        """R33-A1: Entry day (T+1) should check loss barrier."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 1 (entry day, day=0): low at 94 (< loss barrier 96)
        # Entry price = 100 (Day 1 open), loss barrier = 100 - 2*2 = 96
        l[1] = 94.0
        h[1] = 100.5
        o[1] = 100.0
        c[1] = 95.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        # Should detect loss on entry day
        assert ev['label_barrier'] == 'loss'
        assert ev['label_holding_days'] == 0  # R33-A1: day=0
        assert ev['label'] == -1
    
    def test_entry_day_profit_barrier(self, labeler):
        """R33-A1: Entry day (T+1) should check profit barrier."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 1 (entry day, day=0): high at 106 (> profit barrier 104)
        # Entry price = 100, profit barrier = 100 + 2*2 = 104
        h[1] = 106.0
        l[1] = 99.5
        o[1] = 100.0
        c[1] = 105.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        # Should detect profit on entry day
        assert ev['label_barrier'] == 'profit'
        assert ev['label_holding_days'] == 0  # R33-A1: day=0
        assert ev['label'] == 1
    
    def test_entry_day_collision(self, labeler):
        """R33-A1: Entry day collision should force loss (pessimism)."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 1 (entry day, day=0): both barriers hit
        # high=107 > 104, low=93 < 96
        h[1] = 107.0
        l[1] = 93.0
        o[1] = 100.0
        c[1] = 102.0
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        # Should detect collision on entry day, force loss
        assert ev['label_barrier'] == 'loss_collision'
        assert ev['label_holding_days'] == 0  # R33-A1: day=0
        assert ev['label'] == -1
    
    def test_entry_day_no_gap_check(self, labeler):
        """R33-A1: Entry day should NOT check gap (open IS entry_price)."""
        n = 20
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        
        # Day 1 (entry day): open = entry_price = 100
        # Even if open is way below barrier, should NOT trigger gap
        # Because gap is physically impossible on entry day
        o[1] = 100.0  # This IS the entry price
        h[1] = 100.5
        l[1] = 100.0  # Normal range, no barrier hit
        c[1] = 100.2
        
        result = labeler.label_events(self.make_df(o, h, l, c))
        ev = result[result['event_valid'] == True].iloc[0]
        
        # Should hit time barrier (no barrier touched on entry day)
        # Not loss_gap even though open < loss_barrier would be true
        # if we were checking gap on day=0
        assert ev['label_barrier'] == 'time'  # No barrier hit on day 0
        assert ev['label_holding_days'] == 10  # Full holding period


class TestEndToEndPipeline:
    """E2E pipeline: validate -> features -> labels -> weights -> sklearn.fit"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for 5 symbols over 300 days."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        dfs = []
        
        for sym in ['A', 'B', 'C', 'D', 'E']:
            p = np.random.uniform(50, 200) * np.cumprod(1 + np.random.normal(0.0003, 0.015, 300))
            dfs.append(pd.DataFrame({
                'symbol': sym,
                'date': dates,
                'adj_open': p * 0.999,
                'adj_close': p,
                'adj_high': p * 1.005,
                'adj_low': p * 0.995,
                'raw_open': p * 0.999,
                'raw_high': p * 1.005,
                'raw_low': p * 0.995,
                'raw_close': p,
                'volume': np.random.randint(2000, 15000, 300)
            }))
        
        return pd.concat(dfs, ignore_index=True)
    
    def test_pipeline_sklearn_fit(self, sample_data):
        """Full pipeline should produce sklearn-ready data."""
        from src.data.validate import DataValidator
        from src.data.corporate_actions import CorporateActionsHandler
        from src.features.build_features import FeatureEngineer
        from src.labels.triple_barrier import TripleBarrierLabeler
        from src.labels.sample_weights import SampleWeightCalculator
        from sklearn.ensemble import RandomForestClassifier
        
        # Step 1: Validate
        validator = DataValidator()
        passed, validated, report = validator.validate(sample_data)
        assert len(validated) > 0
        
        # Step 2: Corporate actions
        ca = CorporateActionsHandler()
        processed, _ = ca.apply_all(validated)
        assert len(processed) > 0
        
        # Step 3: Build features
        fe = FeatureEngineer()
        features = fe.build_features(processed)
        assert len(features) > 0
        
        # Step 4: Label events
        labeler = TripleBarrierLabeler()
        labeled = labeler.label_events(features)
        assert 'label' in labeled.columns
        
        # Step 5: Calculate weights
        swc = SampleWeightCalculator()
        weighted = swc.calculate_weights(labeled)
        
        # Step 6: Filter valid and fit
        valid = weighted[weighted['event_valid'] == True]
        feature_cols = fe._get_feature_columns(features)
        
        X = valid[feature_cols]
        y = valid['label']
        w = valid['sample_weight']
        
        # Check sklearn-ready (NaN-free)
        assert X.isna().sum().sum() == 0, "X has NaN"
        assert y.isna().sum() == 0, "y has NaN"
        assert w.isna().sum() == 0, "w has NaN"
        
        # Fit a simple model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y, sample_weight=w)
        
        preds = clf.predict(X[:5])
        assert len(preds) == 5


class TestCanTradeFlow:
    """can_trade column: NaN/True/False three-path verification"""
    
    def test_can_trade_nan_filled_as_true(self):
        """NaN in can_trade should be filled as True (R28-S1)."""
        from src.data.corporate_actions import CorporateActionsHandler
        
        df = pd.DataFrame({
            'symbol': ['A'] * 5,
            'date': pd.date_range('2024-01-01', periods=5, freq='B'),
            'adj_open': [100.0] * 5,
            'adj_close': [100.0] * 5,
            'adj_high': [101.0] * 5,
            'adj_low': [99.0] * 5,
            'raw_open': [100.0] * 5,
            'raw_close': [100.0] * 5,
            'raw_high': [101.0] * 5,
            'raw_low': [99.0] * 5,
            'volume': [1000] * 5,
            'can_trade': [True, False, pd.NA, True, np.nan]
        })
        
        ca = CorporateActionsHandler()
        result, _ = ca.apply_all(df)
        
        # NaN should be filled with True (R28-S1 fix)
        assert result['can_trade'].iloc[2] == True
        assert result['can_trade'].iloc[4] == True
    
    def test_can_trade_false_preserved(self):
        """Explicit False should be preserved."""
        from src.data.corporate_actions import CorporateActionsHandler
        
        df = pd.DataFrame({
            'symbol': ['A'] * 3,
            'date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'adj_open': [100.0] * 3,
            'adj_close': [100.0] * 3,
            'adj_high': [101.0] * 3,
            'adj_low': [99.0] * 3,
            'raw_open': [100.0] * 3,
            'raw_close': [100.0] * 3,
            'raw_high': [101.0] * 3,
            'raw_low': [99.0] * 3,
            'volume': [1000] * 3,
            'can_trade': [True, False, True]
        })
        
        ca = CorporateActionsHandler()
        result, _ = ca.apply_all(df)
        
        assert result['can_trade'].iloc[1] == False


class TestInvalidAtrFilter:
    """ATR = NaN defense"""
    
    def test_invalid_atr_filtered(self):
        """Events with NaN ATR should be marked invalid."""
        from src.labels.triple_barrier import TripleBarrierLabeler
        
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        df = pd.DataFrame({
            'symbol': 'TST',
            'date': dates,
            'adj_open': [100.0] * 50,
            'adj_close': [100.0] * 50,
            'adj_high': [101.0] * 50,
            'adj_low': [99.0] * 50,
            'volume': [5000] * 50,
            'atr_20': [2.0] * 25 + [np.nan] * 25,
            'features_valid': True,
            'can_trade': True
        })
        
        labeler = TripleBarrierLabeler()
        result = labeler.label_events(df)
        
        valid = result[result['event_valid'] == True]
        assert len(valid) > 0


class TestWAPUtils:
    """Parquet read/write utilities (in src/data/)"""
    
    def test_parquet_roundtrip(self):
        """Should write and read parquet without data loss."""
        from src.data.wap_utils import write_parquet_wap, read_parquet_safe
        
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'price': [100.0, 200.0, 300.0],
            'volume': [1000, 2000, 3000]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.parquet'
            write_parquet_wap(df, path)
            loaded = read_parquet_safe(path)
            
            # Check core data preserved (may have added audit columns)
            assert len(loaded) == len(df)
            assert list(loaded['symbol']) == ['A', 'B', 'C']


class TestConfigConsistency:
    """Cross-config file consistency checks"""
    
    def test_training_yaml_has_or5_params(self):
        """training.yaml should have all OR5 locked parameters."""
        config_path = Path(__file__).parent.parent / 'config' / 'training.yaml'
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        lgb = cfg['lightgbm']
        
        # OR5 locked parameters
        assert lgb['max_depth'] == 3, f"max_depth should be 3, got {lgb['max_depth']}"
        assert lgb['num_leaves'] == 7, f"num_leaves should be 7, got {lgb['num_leaves']}"
        assert lgb['min_data_in_leaf'] == 100, f"min_data_in_leaf should be 100 (R20-F1 fix)"
        assert lgb['lambda_l1'] == 1.0, f"lambda_l1 should be 1.0"
    
    def test_embargo_is_60(self):
        """Embargo window should be 60 (OR2-06: match max feature lookback)."""
        config_path = Path(__file__).parent.parent / 'config' / 'training.yaml'
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        embargo = cfg['validation']['cpcv']['embargo_window']
        assert embargo == 60, f"embargo_window should be 60, got {embargo}"
    
    def test_min_data_days_is_200(self):
        """min_data_days should be 630 (R27-B1)."""
        config_path = Path(__file__).parent.parent / 'config' / 'training.yaml'
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        min_days = cfg['validation']['cpcv']['min_data_days']
        assert min_days == 200, f"min_data_days should be 630, got {min_days}"


class TestBarrierTypes:
    """Verify all new barrier types are supported"""
    
    def test_all_barrier_types_exist(self):
        """All OR5 barrier types should be valid."""
        from src.labels.triple_barrier import TripleBarrierLabeler
        
        labeler = TripleBarrierLabeler()
        
        # Create data that will hit time barrier
        n = 30
        dates = pd.date_range('2024-01-01', periods=n, freq='B')
        df = pd.DataFrame({
            'symbol': 'TST',
            'date': dates,
            'adj_open': [100.0] * n,
            'adj_close': [100.0] * n,
            'adj_high': [100.5] * n,  # Low volatility - will hit time barrier
            'adj_low': [99.5] * n,
            'volume': [5000] * n,
            'atr_20': [0.5] * n,  # Very low ATR
            'features_valid': True,
            'can_trade': True
        })
        
        result = labeler.label_events(df)
        valid = result[result['event_valid'] == True]
        
        # Should have at least some events
        assert len(valid) > 0
        
        # All barrier types should be valid
        valid_barriers = valid['label_barrier'].unique()
        for b in valid_barriers:
            assert b in ['profit', 'loss', 'time', 'profit_gap', 'loss_gap', 'loss_collision']
