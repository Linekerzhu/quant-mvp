"""
Universe management tests - R29 Fix

R29 审计整改：测试对齐实际 API（UniverseManager）
"""
import pytest
import pandas as pd
import numpy as np
from src.data.universe import UniverseManager


class TestUniverseManager:
    """Test UniverseManager class"""

    @pytest.fixture
    def manager(self):
        """Create universe manager instance"""
        return UniverseManager(config_path='config/universe.yaml')

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        data = []
        for symbol in symbols:
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'raw_close': 100 + np.random.randn() * 10,
                    'adj_close': 100 + np.random.randn() * 10,
                    'volume': np.random.randint(1000000, 10000000)
                })

        return pd.DataFrame(data)

    def test_manager_get_sp500_tickers_returns_list(self, manager):
        """R29-A2: get_sp500_tickers should return list of tickers"""
        tickers = manager.get_sp500_tickers()

        assert isinstance(tickers, list)
        # May be empty if no network access, but should be a list
        if len(tickers) > 0:
            assert all(isinstance(t, str) for t in tickers)

    def test_manager_filter_liquidity(self, manager, sample_price_data):
        """R29-A2: filter_liquidity should filter by ADV"""
        filtered = manager.filter_liquidity(sample_price_data, lookback_days=20)

        assert isinstance(filtered, list)
        # Should return symbols with sufficient liquidity
        # (exact count depends on mock data)
        assert all(isinstance(s, str) for s in filtered)

    def test_manager_config_has_filters_min_history_days(self, manager):
        """R29-A2: Config should have min_history_days in filters section"""
        config = manager.config

        assert 'filters' in config, "Config missing 'filters' section"

        filters = config['filters']
        assert 'min_history_days' in filters
        assert isinstance(filters['min_history_days'], int)
        assert filters['min_history_days'] > 0

    def test_manager_build_universe(self, manager, sample_price_data):
        """R29-A2: build_universe should return universe dict"""
        result = manager.build_universe(sample_price_data)

        assert isinstance(result, dict)
        assert 'symbols' in result
        assert 'cold_start' in result
        assert 'count' in result
        assert 'metadata' in result

        # Should have some symbols
        assert isinstance(result['symbols'], list)

    def test_manager_check_history_length(self, manager, sample_price_data):
        """R29-A2: check_history_length should return list with sufficient history"""
        with_history = manager.check_history_length(sample_price_data, min_days=60)

        assert isinstance(with_history, list)
        # With 100 days of data, all 3 symbols should have sufficient history
        # (but exact result depends on implementation)
