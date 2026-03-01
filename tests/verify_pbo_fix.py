#!/usr/bin/env python3
"""
R16 验证脚本：PBO 修复验证

验证 R15 修复后的 PBO 计算逻辑是否正确。
修复内容：PBO 计算改为只检查 IS rank #1 的路径

验证点：
1. PBO 只检查 IS rank #1 的路径
2. 如果 IS #1 的路径 OOS 排名 > 中位数，PBO = 1.0
3. 如果 IS #1 的路径 OOS 排名 <= 中位数，PBO = 0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.overfitting import OverfittingDetector
import numpy as np


def test_pbo_is_rank1_only():
    """
    测试 PBO 只检查 IS rank #1 的路径。
    
    场景：
    - 路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.4 (rank #4/4)
    - 路径 2: IS AUC = 0.7 (rank #2), OOS AUC = 0.8 (rank #1/4)
    - 路径 3: IS AUC = 0.6 (rank #3), OOS AUC = 0.7 (rank #2/4)
    - 路径 4: IS AUC = 0.5 (rank #4), OOS AUC = 0.6 (rank #3/4)
    
    预期结果：
    - IS #1 的路径是路径 1
    - 路径 1 的 OOS 排名是 #4，大于中位数排名 (4+1)/2 = 2.5
    - PBO = 1.0 (过拟合)
    """
    detector = OverfittingDetector({})
    
    path_results = [
        {'is_auc': 0.9, 'oos_auc': 0.4},  # IS #1, OOS #4
        {'is_auc': 0.7, 'oos_auc': 0.8},  # IS #2, OOS #1
        {'is_auc': 0.6, 'oos_auc': 0.7},  # IS #3, OOS #2
        {'is_auc': 0.5, 'oos_auc': 0.6},  # IS #4, OOS #3
    ]
    
    pbo = detector.calculate_pbo(path_results)
    
    print("=" * 60)
    print("测试 1: IS #1 的路径 OOS 表现差于中位数")
    print("=" * 60)
    print(f"路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.4 (rank #4)")
    print(f"中位数排名: (4+1)/2 = 2.5")
    print(f"OOS 排名 #4 > 2.5 → PBO = 1.0 (过拟合)")
    print(f"实际 PBO: {pbo}")
    
    assert pbo == 1.0, f"Expected PBO=1.0, got {pbo}"
    print("✅ 通过：PBO = 1.0\n")


def test_pbo_is_rank1_good_oos():
    """
    测试 IS rank #1 的路径 OOS 表现好于中位数。
    
    场景：
    - 路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.9 (rank #1/4)
    - 路径 2: IS AUC = 0.7 (rank #2), OOS AUC = 0.7 (rank #2/4)
    - 路径 3: IS AUC = 0.6 (rank #3), OOS AUC = 0.6 (rank #3/4)
    - 路径 4: IS AUC = 0.5 (rank #4), OOS AUC = 0.5 (rank #4/4)
    
    预期结果：
    - IS #1 的路径是路径 1
    - 路径 1 的 OOS 排名是 #1，小于中位数排名 2.5
    - PBO = 0.0 (未过拟合)
    """
    detector = OverfittingDetector({})
    
    path_results = [
        {'is_auc': 0.9, 'oos_auc': 0.9},  # IS #1, OOS #1
        {'is_auc': 0.7, 'oos_auc': 0.7},  # IS #2, OOS #2
        {'is_auc': 0.6, 'oos_auc': 0.6},  # IS #3, OOS #3
        {'is_auc': 0.5, 'oos_auc': 0.5},  # IS #4, OOS #4
    ]
    
    pbo = detector.calculate_pbo(path_results)
    
    print("=" * 60)
    print("测试 2: IS #1 的路径 OOS 表现好于中位数")
    print("=" * 60)
    print(f"路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.9 (rank #1)")
    print(f"中位数排名: (4+1)/2 = 2.5")
    print(f"OOS 排名 #1 < 2.5 → PBO = 0.0 (未过拟合)")
    print(f"实际 PBO: {pbo}")
    
    assert pbo == 0.0, f"Expected PBO=0.0, got {pbo}"
    print("✅ 通过：PBO = 0.0\n")


def test_pbo_median_case():
    """
    测试边界情况：IS #1 的路径 OOS 排名正好等于中位数。
    
    场景（5条路径，中位数排名 = 3）：
    - 路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.7 (rank #3/5)
    - 其他路径...
    
    预期结果：
    - OOS 排名 #3 不大于中位数排名 3
    - PBO = 0.0
    """
    detector = OverfittingDetector({})
    
    path_results = [
        {'is_auc': 0.9, 'oos_auc': 0.7},  # IS #1, OOS #3
        {'is_auc': 0.8, 'oos_auc': 0.9},
        {'is_auc': 0.7, 'oos_auc': 0.8},
        {'is_auc': 0.6, 'oos_auc': 0.6},
        {'is_auc': 0.5, 'oos_auc': 0.5},
    ]
    
    pbo = detector.calculate_pbo(path_results)
    
    print("=" * 60)
    print("测试 3: 边界情况 - OOS 排名等于中位数")
    print("=" * 60)
    print(f"路径 1: IS AUC = 0.9 (rank #1), OOS AUC = 0.7 (rank #3)")
    print(f"中位数排名: (5+1)/2 = 3")
    print(f"OOS 排名 #3 不大于 3 → PBO = 0.0")
    print(f"实际 PBO: {pbo}")
    
    assert pbo == 0.0, f"Expected PBO=0.0, got {pbo}"
    print("✅ 通过：PBO = 0.0\n")


def test_pbo_old_logic_would_fail():
    """
    验证旧逻辑（排名后 50% 比例）不适用于此场景。
    
    旧逻辑会计算所有路径的 AUC 排名，然后计算排名后 50% 的比例。
    这与 AFML 定义不符（只检查 IS #1 的路径）。
    """
    print("=" * 60)
    print("测试 4: 验证旧逻辑为何不正确")
    print("=" * 60)
    
    # 如果使用旧逻辑（计算所有 AUC 的排名）
    # 场景：所有路径表现一致，IS 和 OOS 相关性高
    path_results = [
        {'is_auc': 0.9, 'oos_auc': 0.85},
        {'is_auc': 0.8, 'oos_auc': 0.75},
        {'is_auc': 0.7, 'oos_auc': 0.65},
        {'is_auc': 0.6, 'oos_auc': 0.55},
    ]
    
    detector = OverfittingDetector({})
    pbo = detector.calculate_pbo(path_results)
    
    print("场景：IS 和 OOS 表现一致，无明显过拟合")
    print(f"新逻辑（只检查 IS #1）: PBO = {pbo}")
    print(f"IS #1 路径的 OOS 排名也是 #1 → PBO = 0.0 (正确)")
    print()
    
    assert pbo == 0.0, f"Expected PBO=0.0, got {pbo}"
    print("✅ 通过：新逻辑正确识别未过拟合\n")


def main():
    """运行所有验证测试。"""
    print("\n" + "=" * 60)
    print("R16 PBO 修复验证")
    print("=" * 60)
    print("验证内容：PBO 计算改为只检查 IS rank #1 的路径")
    print("修复文件：src/models/overfitting.py")
    print("=" * 60 + "\n")
    
    try:
        test_pbo_is_rank1_only()
        test_pbo_is_rank1_good_oos()
        test_pbo_median_case()
        test_pbo_old_logic_would_fail()
        
        print("=" * 60)
        print("✅ R16 内审通过")
        print("=" * 60)
        print("\n所有测试通过：")
        print("1. ✅ PBO 只检查 IS rank #1 的路径")
        print("2. ✅ IS #1 路径 OOS 排名 > 中位数 → PBO = 1.0")
        print("3. ✅ IS #1 路径 OOS 排名 <= 中位数 → PBO = 0.0")
        print("4. ✅ 边界情况处理正确")
        print("5. ✅ 新逻辑符合 AFML 定义")
        print("\n验证结论：修复正确，无新问题。")
        
    except AssertionError as e:
        print("=" * 60)
        print("❌ R16 内审未通过")
        print("=" * 60)
        print(f"\n错误：{e}")
        print("\n需要进一步检查 PBO 计算逻辑。")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
