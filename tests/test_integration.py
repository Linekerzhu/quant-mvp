"""
End-to-End Integration Tests

Tests the full pipeline from Phase A to Phase B.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.build_features import FeatureEngineer
from src.features.regime_detector import RegimeDetector
from src.labels.triple_barrier import TripleBarrierLabeler
from src.labels.sample_weights import SampleWeightCalculator


class TestPhaseAtoBPipeline:
    """Test integration between Phase A and Phase B."""
    
    @pytest.fixture
    def mock_data_phase_a_output(self):
        """
        Simulate Phase A output (processed data with corporate actions).
        
        This represents the data structure after:
        - Data ingestion
        - Validation
        - Integrity checks
        - Corporate action processing
        """
        path = Path("tests/fixtures/mock_prices.parquet")
        df = pd.read_parquet(path)
        
        # Add Phase A processed columns
        df['can_trade'] = ~df['is_suspended'] if 'is_suspended' in df.columns else True
        df['detected_split'] = False
        df['split_ratio'] = 1.0
        
        # Simulate split detection for MOCK000
        df.loc[df['symbol'] == 'MOCK000', 'detected_split'] = True
        df.loc[df['symbol'] == 'MOCK000', 'split_ratio'] = 2.0
        
        # Add suspension for MOCK001
        df['is_suspended'] = False
        df.loc[(df['symbol'] == 'MOCK001') & 
               (df['date'] >= '2024-03-15') & 
               (df['date'] <= '2024-03-21'), 'is_suspended'] = True
        
        return df
    
    def test_full_pipeline_integration(self, mock_data_phase_a_output):
        """
        Test full pipeline: Phase A output -> Features -> Labels -> Weights.
        """
        df = mock_data_phase_a_output.copy()
        
        # Phase B Step 1: Feature Engineering
        engineer = FeatureEngineer()
        df = engineer.build_features(df)
        
        # Verify features created
        assert 'returns_5d' in df.columns
        assert 'rsi_14' in df.columns
        assert 'atr_14' in df.columns
        assert 'dummy_noise' in df.columns
        
        # Phase B Step 2: Regime Detection (optional)
        detector = RegimeDetector()
        df = detector.detect_regime(df)
        
        assert 'regime_volatility' in df.columns
        assert 'regime_trend' in df.columns
        
        # Phase B Step 3: Triple Barrier Labeling
        labeler = TripleBarrierLabeler()
        df = labeler.label_events(df)
        
        assert 'label' in df.columns
        assert 'event_valid' in df.columns
        
        # Phase B Step 4: Sample Weights
        calculator = SampleWeightCalculator()
        df = calculator.calculate_weights(df)
        
        assert 'sample_weight' in df.columns
        
        # Verify final output
        valid_events = df[df['event_valid'] == True]
        assert len(valid_events) > 0
        assert valid_events['sample_weight'].notna().all()
        assert (valid_events['sample_weight'] > 0).all()
        
        print(f"\nPipeline Summary:")
        print(f"  Input rows: {len(mock_data_phase_a_output)}")
        print(f"  Valid events: {len(valid_events)}")
        print(f"  Features: {len([c for c in df.columns if c not in mock_data_phase_a_output.columns])}")
        print(f"  Mean weight: {valid_events['sample_weight'].mean():.4f}")
    
    def test_data_flow_consistency(self, mock_data_phase_a_output):
        """Test that data flows correctly through all stages."""
        df = mock_data_phase_a_output.copy()
        
        # Track row count through pipeline
        initial_rows = len(df)
        
        # Features (should not drop rows)
        engineer = FeatureEngineer()
        df = engineer.build_features(df)
        assert len(df) == initial_rows, "Feature engineering should not drop rows"
        
        # Labels (may mark some invalid, but shouldn't drop)
        labeler = TripleBarrierLabeler()
        df = labeler.label_events(df)
        assert len(df) == initial_rows, "Labeling should not drop rows"
        
        # Weights (should not drop rows)
        calculator = SampleWeightCalculator()
        df = calculator.calculate_weights(df)
        assert len(df) == initial_rows, "Weight calculation should not drop rows"
    
    def test_column_preservation(self, mock_data_phase_a_output):
        """Test that required columns are preserved through pipeline."""
        df = mock_data_phase_a_output.copy()
        required_cols = ['symbol', 'date', 'adj_close', 'can_trade']
        
        # Run pipeline
        engineer = FeatureEngineer()
        labeler = TripleBarrierLabeler()
        calculator = SampleWeightCalculator()
        
        df = engineer.build_features(df)
        df = labeler.label_events(df)
        df = calculator.calculate_weights(df)
        
        # Check all required columns preserved
        for col in required_cols:
            assert col in df.columns, f"Required column '{col}' was dropped"
    
    def test_mock_split_scenario_in_pipeline(self, mock_data_phase_a_output):
        """Test that split scenario (MOCK000) is handled correctly."""
        df = mock_data_phase_a_output.copy()
        
        # Run pipeline
        engineer = FeatureEngineer()
        labeler = TripleBarrierLabeler()
        calculator = SampleWeightCalculator()
        
        df = engineer.build_features(df)
        
        # Check split is reflected in raw/adj ratio
        mock000 = df[df['symbol'] == 'MOCK000'].sort_values('date')
        pre_split = mock000[mock000['date'] < '2024-06-17']
        
        if len(pre_split) > 0:
            # Raw should be ~2x adj before split
            ratio = pre_split.iloc[0]['raw_close'] / pre_split.iloc[0]['adj_close']
            assert 1.9 < ratio < 2.1, f"Split ratio should be ~2.0, got {ratio}"
        
        df = labeler.label_events(df)
        df = calculator.calculate_weights(df)
        
        # Check that split stock has valid events
        mock000_valid = df[(df['symbol'] == 'MOCK000') & (df['event_valid'] == True)]
        assert len(mock000_valid) > 0, "Split stock should have valid events"
    
    def test_suspended_stock_handling(self, mock_data_phase_a_output):
        """Test that suspended stock (MOCK001) is handled correctly."""
        df = mock_data_phase_a_output.copy()
        
        # Run pipeline
        engineer = FeatureEngineer()
        labeler = TripleBarrierLabeler()
        calculator = SampleWeightCalculator()
        
        df = engineer.build_features(df)
        df = labeler.label_events(df)
        
        # Check suspended periods have no valid events
        # Use raw_close.isna() to identify actual suspension days
        suspended = df[(df['symbol'] == 'MOCK001') & 
                       (df['raw_close'].isna())]
        
        if len(suspended) > 0:
            assert suspended['event_valid'].sum() == 0, "Suspended days should have no valid events"


class TestConfigurationConsistency:
    """Test that configurations are consistent across modules."""
    
    def test_atr_window_consistency(self):
        """Test ATR window is consistent between features and labels."""
        import yaml
        
        with open("config/features.yaml") as f:
            features_cfg = yaml.safe_load(f)
        
        with open("config/event_protocol.yaml") as f:
            protocol_cfg = yaml.safe_load(f)
        
        # FIX B1: Actually check ATR window consistency
        # Get expected ATR window from protocol
        expected_window = protocol_cfg['triple_barrier']['atr']['window']
        
        # Check that features.yaml has matching atr_N feature
        volatility_features = features_cfg['categories']['volatility']['features']
        atr_features = [f for f in volatility_features if f['name'].startswith('atr_')]
        
        assert len(atr_features) == 1, f"Expected exactly one ATR feature, got {atr_features}"
        actual_window = int(atr_features[0]['name'].split('_')[1])
        
        assert actual_window == expected_window, \
            f"ATR window mismatch: features.yaml has {actual_window}, event_protocol.yaml expects {expected_window}"
    
    def test_max_holding_days_consistency(self):
        """Test max holding days is consistent."""
        import yaml
        
        with open("config/event_protocol.yaml") as f:
            protocol_cfg = yaml.safe_load(f)
        
        max_holding = protocol_cfg['triple_barrier']['max_holding_days']
        assert max_holding == 10, "Max holding days should be 10"
    
    def test_feature_version_tracking(self):
        """Test that feature version is tracked."""
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        metadata = engineer.get_feature_metadata()
        
        assert 'version' in metadata
        assert 'dummy_feature' in metadata
        assert metadata['dummy_feature'] == 'dummy_noise'
