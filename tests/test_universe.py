"""
Universe module smoke tests.

OR5-CODE T4: Coverage gap fill - basic smoke tests for universe.py
"""

import pytest
import pandas as pd
import numpy as np


class TestUniverseLoader:
    """Smoke tests for universe loading and filtering."""

    def test_load_universe_returns_symbols(self):
        """Universe loader should return > 0 symbols."""
        from src.data.universe import load_universe
        
        symbols = load_universe()
        assert len(symbols) > 0, "Universe should not be empty"
        assert all(isinstance(s, str) for s in symbols), "All symbols should be strings"

    def test_filters_apply(self):
        """ADV filter should reduce or maintain symbol count."""
        from src.data.universe import load_universe, filter_universe
        
        all_symbols = load_universe()
        
        # Mock market data with varying ADV
        mock_adv_data = pd.DataFrame({
            'symbol': all_symbols[:100] if len(all_symbols) > 100 else all_symbols,
            'adv_usd': [5_000_000 * (1 + i % 10) for i in range(min(100, len(all_symbols)))]
        })
        
        filtered = filter_universe(
            all_symbols[:100] if len(all_symbols) > 100 else all_symbols,
            mock_adv_data,
            min_adv_usd=5_000_000
        )
        
        # Filtered should be <= original
        assert len(filtered) <= min(100, len(all_symbols)), "Filter should not add symbols"

    def test_universe_config_loaded(self):
        """Universe config should be loadable."""
        import yaml
        from pathlib import Path
        
        config_path = Path("config/universe.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert 'min_history_days' in config, "Config should have min_history_days"
            assert 'min_adv_usd' in config, "Config should have min_adv_usd"
            assert config['min_history_days'] >= 30, "min_history_days should be reasonable"
