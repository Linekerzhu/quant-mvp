import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# Set up path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.overfitting import OverfittingDetector

def run_simulation():
    print("=" * 60)
    print("🧠 数学仿真与量化评估：Deflated Sharpe Ratio (DSR) & PBO")
    print("=" * 60)

    config = {
        'pbo_reject': 0.5,
        'pbo_threshold': 0.3,
        'dummy_feature_sentinel': {'ranking_threshold': 0.25}
    }
    detector = OverfittingDetector(config)

    import numpy as np
    
    # Generate correlated random variables for the scenarios
    def gen_scenario_paths(mean_is, mean_oos, is_oos_corr=0.0, std=0.02, n=15):
        paths = []
        for _ in range(n):
            # Generate two correlated standard normals
            z1 = np.random.normal(0, 1)
            z2 = is_oos_corr * z1 + np.sqrt(1 - is_oos_corr**2) * np.random.normal(0, 1)
            
            is_auc = mean_is + z1 * std
            oos_auc = mean_oos + z2 * std
            paths.append({
                "is_auc": is_auc,
                "oos_auc": oos_auc,
                "auc": oos_auc
            })
        return paths

    scenarios = [
        {
            "name": "Scenario 1: 纯随机噪声 (Random Noise)",
            "desc": "IS 和 OOS 表现均为 0.5 左右的随机波动。预期：不过度拟合但也通不过 DSR（无显著收益）。",
            "paths": gen_scenario_paths(0.5, 0.5, is_oos_corr=0.0)
        },
        {
            "name": "Scenario 2: 严重过拟合 (Severe Overfitting)",
            "desc": "模型在训练集 (IS) 表现极好，但在验证集 (OOS) 表现极差，且二者呈负相关。预期：触及 PBO 门控拒绝，DSR 也随之崩溃。",
            "paths": gen_scenario_paths(0.65, 0.45, is_oos_corr=-0.8)
        },
        {
            "name": "Scenario 3: 真实稳健 Alpha (True Robust Alpha)",
            "desc": "IS 和 OOS 同样表现优秀，且高度正相关（泛化能力强）。预期：PBO < 0.3 (PASS)，DSR 具有高置信度。",
            "paths": gen_scenario_paths(0.53, 0.53, is_oos_corr=0.8, std=0.01)
        },
        {
            "name": "Scenario 3.5: 极强真实 Alpha (Very Strong True Alpha)",
            "desc": "极强的信号，平均 AUC 达到 0.58+。预期：PBO PASS 且 DSR PASS (Z > 1.645)。",
            "paths": gen_scenario_paths(0.58, 0.58, is_oos_corr=0.9, std=0.02)
        },
        {
            "name": "Scenario 4: 微弱但不显著的 Alpha (Weak & Flawed Alpha)",
            "desc": "表现稍微大于 0.5，但方差很大（偏度和峰度不好）。预期：DSR 警告或拒绝，惩罚其高方差。",
            "paths": gen_scenario_paths(0.51, 0.51, is_oos_corr=0.1, std=0.06)
        }
    ]

    for sc in scenarios:
        print(f"\n{sc['name']}")
        print(f"描述: {sc['desc']}")
        
        # Check overall
        result = detector.check_overfitting(sc['paths'])
        
        pbo = result['pbo']
        pbo_msg = result['pbo_message']
        dsr = result['dsr']
        dsr_msg = result['dsr_message']
        
        # Calc some stats manually for display
        aucs = [r['auc'] for r in sc['paths']]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        print(f"  [统计量] Mean OOS AUC: {mean_auc:.4f}, Std OOS AUC: {std_auc:.4f}")
        print(f"  [PBO 评估] 值: {pbo:.4f} -> {pbo_msg}")
        print(f"  [DSR 评估] Z-Score: {dsr:.4f} -> {dsr_msg}")
        print(f"  [综合判定] Passed: {result['overall_passed']}")

    print("\n" + "=" * 60)
    print("数学仿真完成。")
    print("=" * 60)

if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    run_simulation()
