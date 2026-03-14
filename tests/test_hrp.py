import numpy as np
import pandas as pd
from src.risk.hrp import HierarchicalRiskParity

def test_hrp_deconcentration():
    """
    Test that HRP successfully decentralizes capital away from highly
    correlated assets compared to naive equal weighting.
    """
    np.random.seed(42)
    
    # Simulate 5 assets
    # A, B, C are highly correlated (e.g. 3 semiconductor stocks)
    # D, E are independent
    
    # 120 days of returns
    n_days = 120
    
    # Base market factor
    market = np.random.normal(0, 0.01, n_days)
    
    # The semi cluster factor
    semi_factor = np.random.normal(0, 0.02, n_days)
    
    # Generate returns
    # A, B, C share the semi factor -> high correlation
    ret_A = market + semi_factor + np.random.normal(0, 0.005, n_days)
    ret_B = market + semi_factor + np.random.normal(0, 0.005, n_days)
    ret_C = market + semi_factor + np.random.normal(0, 0.005, n_days)
    
    # D, E are mostly independent / low vol
    ret_D = market + np.random.normal(0, 0.015, n_days)
    ret_E = market + np.random.normal(0, 0.015, n_days)
    
    returns_df = pd.DataFrame({
        'A': ret_A,
        'B': ret_B,
        'C': ret_C,
        'D': ret_D,
        'E': ret_E
    })
    
    # Print correlation matrix
    print("Correlation Matrix:")
    print(returns_df.corr().round(2))
    print("\n-------------------------")
    
    # Let's say our base Kelly sizer gave them all 10% weight
    raw_weights = {
        'A': 0.10,
        'B': 0.10,
        'C': 0.10,
        'D': 0.10,
        'E': 0.10
    }
    print(f"Raw Weights: {raw_weights}")
    
    # Portfolio Variance before HRP
    raw_w_arr = np.array(list(raw_weights.values()))
    cov_mat = returns_df.cov().values
    var_before = np.dot(raw_w_arr.T, np.dot(cov_mat, raw_w_arr))
    
    # Apply HRP
    hrp = HierarchicalRiskParity(history_window=120, max_weight=0.20)
    opt_weights = hrp.optimize(raw_weights, returns_df)
    
    print("\nOptimized HRP Weights:")
    for k, v in opt_weights.items():
        print(f"  {k}: {v:.3f}")
        
    opt_w_arr = np.array([opt_weights[sym] for sym in raw_weights.keys()])
    var_after = np.dot(opt_w_arr.T, np.dot(cov_mat, opt_w_arr))
    
    semi_total_raw = raw_weights['A'] + raw_weights['B'] + raw_weights['C']
    semi_total_opt = opt_weights['A'] + opt_weights['B'] + opt_weights['C']
    
    print("\n-------------------------")
    print(f"Semi-Conductor Cluster Total Weight (Raw): {semi_total_raw:.2f}")
    print(f"Semi-Conductor Cluster Total Weight (HRP): {semi_total_opt:.2f}")
    
    print(f"Portfolio Volatility (Raw): {np.sqrt(var_before)*np.sqrt(252)*100:.2f}%")
    print(f"Portfolio Volatility (HRP): {np.sqrt(var_after)*np.sqrt(252)*100:.2f}%")
    
    assert semi_total_opt < semi_total_raw, "HRP failed to deconcentrate correlated cluster"
    assert var_after < var_before, "HRP failed to reduce portfolio variance"
    
    print("\n✅ HRP Simulation PASSED: Successfully penalized correlated cluster and reduced variance.")

if __name__ == '__main__':
    test_hrp_deconcentration()
