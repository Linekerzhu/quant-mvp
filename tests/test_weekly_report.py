import pytest
import os
import pandas as pd
from src.ops.weekly_report import WeeklyReportGenerator

def test_weekly_report_generation(tmp_path):
    generator = WeeklyReportGenerator(report_dir=str(tmp_path))
    
    # Mock data
    port_df = pd.DataFrame({
        'date': ['2026-03-01', '2026-03-02'],
        'daily_return': [0.01, -0.005],
        'cost_usd': [5.0, 3.5]
    })
    
    spy_df = pd.DataFrame({
        'date': ['2026-03-01', '2026-03-02'],
        'adj_close': [400.0, 404.0] # +1%
    })
    
    consistency = {
        'total_signals': 10,
        'total_orders': 10,
        'inconsistencies': 0,
        'inconsistency_rate': 0.0
    }
    
    report_path = generator.generate('2026-03-06', port_df, spy_df, consistency)
    
    assert os.path.exists(report_path)
    with open(report_path, 'r') as f:
        content = f.read()
        
    assert "Portfolio Return" in content
    assert "SPY Return" in content
    assert "1.00%" in content # SPY +1%
    assert "$8.50" in content # total cost
