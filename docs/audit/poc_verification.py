"""
PoC验证代码 - 寇连材审计

本文件包含所有审计发现的PoC验证代码。
运行方式: python docs/audit/poc_verification.py
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay, BusinessDay
import sys
import os

# 添加项目根目录到path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_c01_loc_vs_iloc():
    """
    PoC C-01: 验证loc和iloc混用的问题
    """
    print("\n" + "="*60)
    print("PoC C-01: loc vs iloc混用问题")
    print("="*60)
    
    # 场景1: reset_index后可以工作（当前情况）
    print("\n场景1: reset_index后（当前代码状态）")
    df1 = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'value': range(10)
    })
    df1 = df1.reset_index(drop=True)
    
    idx = 5
    print(f"  df.loc[{idx}, 'date'] = {df1.loc[idx, 'date']}")  # ✓ 可以工作
    print(f"  df.iloc[{idx}]['date'] = {df1.iloc[idx]['date']}")  # ✓ 可以工作
    
    # 场景2: 索引不是0-n连续整数（潜在bug）
    print("\n场景2: 索引不是0-n连续（潜在问题）")
    df2 = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'value': range(10)
    })
    df2.index = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    
    idx = 5  # 位置索引
    print(f"  位置索引 idx={idx}")
    print(f"  df.iloc[{idx}]['date'] = {df2.iloc[idx]['date']}")  # ✓ 正确
    
    try:
        print(f"  df.loc[{idx}, 'date'] = ", end="")
        result = df2.loc[idx, 'date']  # ✗ KeyError
        print(f"{result}")
    except KeyError as e:
        print(f"❌ KeyError: {e}")
    
    # 场景3: 索引包含重复值
    print("\n场景3: 索引包含重复值（隐蔽bug）")
    df3 = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'value': range(10)
    })
    df3.index = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # 重复索引
    
    idx = 0  # 期望访问第0行，但loc会返回多行
    print(f"  位置索引 idx={idx}")
    print(f"  df.iloc[{idx}]['date'] = {df3.iloc[idx]['date']}")  # ✓ 正确（第0行）
    print(f"  df.loc[{idx}, 'date'] = ")  # ✗ 返回多行
    print(f"    {df3.loc[idx, 'date'].values}")  # 返回第0行和第5行
    
    print("\n结论:")
    print("  ✓ 当前代码有reset_index(drop=True)，所以loc可以工作")
    print("  ❌ 但如果后续有人删除了reset_index，会导致隐蔽bug")
    print("  💡 建议: 统一使用iloc访问位置索引\n")


def test_c02_purge_logic_inconsistency():
    """
    PoC C-02: 验证split和split_with_info的purge逻辑不一致
    """
    print("\n" + "="*60)
    print("PoC C-02: purge逻辑不一致问题")
    print("="*60)
    
    # 模拟CPCV场景
    print("\n假设CPCV配置:")
    print("  n_splits=6, n_test_splits=2")
    print("  当前path: test段 = [1, 3]")
    print("  purge_window = 10天")
    
    # 模拟时间线
    dates = pd.date_range('2020-01-01', periods=120, freq='B')
    
    # 段划分
    segment_size = 20
    segments = [(i*segment_size, (i+1)*segment_size) for i in range(6)]
    
    print("\n段划分:")
    for i, (start, end) in enumerate(segments):
        print(f"  段{i}: [{start}, {end}) = {dates[start].date()} ~ {dates[end-1].date()}")
    
    # split()方法的purge（正确）
    print("\nsplit()方法的purge（正确）:")
    test_seg_indices = [1, 3]
    purge_windows = []
    
    for seg_idx in test_seg_indices:
        seg_start, seg_end = segments[seg_idx]
        seg_start_date = dates[seg_start]
        seg_end_date = dates[seg_end - 1]
        
        purge_start = seg_start_date - BDay(10)
        purge_end = seg_end_date + BDay(10)
        
        purge_windows.append((purge_start, purge_end))
        print(f"  段{seg_idx} purge: {purge_start.date()} ~ {purge_end.date()}")
    
    print(f"  → 两个独立的purge窗口")
    
    # split_with_info()方法的purge（简化）
    print("\nsplit_with_info()方法的purge（简化，可能过度）:")
    test_min_date = dates[segments[1][0]]  # 段1开始
    test_max_date = dates[segments[3][1] - 1]  # 段3结束
    
    purge_start_global = test_min_date - BDay(10)
    purge_end_global = test_max_date + BDay(10)
    
    print(f"  全局purge: {purge_start_global.date()} ~ {purge_end_global.date()}")
    print(f"  → 一个连续窗口，覆盖段2！")
    
    # 计算差异
    print("\n差异分析:")
    seg2_start = dates[segments[2][0]]
    seg2_end = dates[segments[2][1] - 1]
    print(f"  段2范围: {seg2_start.date()} ~ {seg2_end.date()}")
    
    # 检查段2是否在全局purge范围内
    if purge_start_global <= seg2_start and purge_end_global >= seg2_end:
        print(f"  ❌ 段2完全在全局purge范围内！")
        print(f"  ❌ split_with_info()会错误地purge段2的样本")
    
    print("\n结论:")
    print("  ✓ split()方法正确实现")
    print("  ❌ split_with_info()方法使用简化逻辑，会过度purge")
    print("  💡 建议: 统一两个方法的purge逻辑\n")


def test_m01_code_duplication():
    """
    PoC M-01: 验证代码重复
    """
    print("\n" + "="*60)
    print("PoC M-01: 代码重复统计")
    print("="*60)
    
    file_path = 'src/labels/sample_weights.py'
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    # 统计重复代码
    pattern1 = "entry_date = trigger_date + BDay(1)"
    pattern2 = "if 'label_exit_date' in row and pd.notna(row['label_exit_date'])"
    
    count1 = code.count(pattern1)
    count2 = code.count(pattern2)
    
    print(f"\n重复代码统计:")
    print(f"  'entry_date = trigger_date + BDay(1)' 出现次数: {count1}")
    print(f"  'if label_exit_date in row...' 出现次数: {count2}")
    print(f"  总重复代码行数: ~{count1 * 6} 行")
    
    print("\n结论:")
    print(f"  ❌ entry_date计算重复{count1}次")
    print(f"  ❌ exit_date计算重复{count2}次")
    print("  💡 建议: 提取为_get_event_dates()方法\n")


def test_m02_performance():
    """
    PoC M-02: 验证性能问题
    """
    print("\n" + "="*60)
    print("PoC M-02: iterrows性能问题")
    print("="*60)
    
    import time
    
    # 创建测试数据
    n = 5000
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='B'),
        'label_exit_date': pd.date_range('2020-01-11', periods=n, freq='B'),
        'label_holding_days': 10,
        'symbol': 'AAPL'
    })
    
    print(f"\n测试数据: {n}行")
    
    # 方法1: iterrows
    print("\n方法1: iterrows")
    start = time.time()
    dates1 = []
    for idx, row in df.iterrows():
        entry_date = row['date'] + BDay(1)
        dates1.append(entry_date)
    time_iterrows = time.time() - start
    print(f"  耗时: {time_iterrows:.4f}s")
    
    # 方法2: 向量化
    print("\n方法2: 向量化")
    start = time.time()
    dates2 = df['date'] + BDay(1)
    time_vectorized = time.time() - start
    print(f"  耗时: {time_vectorized:.4f}s")
    
    # 加速比
    speedup = time_iterrows / time_vectorized if time_vectorized > 0 else 0
    print(f"\n加速比: {speedup:.0f}x")
    
    print("\n结论:")
    print(f"  ❌ iterrows耗时 {time_iterrows:.4f}s")
    print(f"  ✓ 向量化耗时 {time_vectorized:.4f}s")
    print(f"  💡 建议: 使用向量化操作代替iterrows\n")


def test_m03_magic_numbers():
    """
    PoC M-03: 验证magic number
    """
    print("\n" + "="*60)
    print("PoC M-03: Magic Number问题")
    print("="*60)
    
    file_path = 'src/models/meta_trainer.py'
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    # 查找magic number
    import re
    
    # 查找 len(train_df) < 50 或 len(test_df) < 10
    pattern = r'len\((train|test)_df\) < (\d+)'
    matches = re.findall(pattern, code)
    
    print("\nMagic Numbers:")
    for var, num in matches:
        print(f"  len({var}_df) < {num}")
    
    print("\n问题:")
    print("  ❌ 数字50和10没有配置化")
    print("  ❌ 没有注释说明这些数字的来源")
    print("  💡 建议: 移到config/training.yaml\n")


def test_m04_purgedkfold_inconsistency():
    """
    PoC M-04: 验证PurgedKFold的purge逻辑
    """
    print("\n" + "="*60)
    print("PoC M-04: PurgedKFold purge逻辑")
    print("="*60)
    
    print("\n对比两个类的purge逻辑:")
    
    print("\nCombinatorialPurgedKFold.split():")
    print("  ✓ 对每个test段分别计算purge范围")
    print("  ✓ 使用: seg_start_date - BDay(purge_window)")
    print("  ✓ 使用: seg_end_date + BDay(purge_window)")
    
    print("\nPurgedKFold.split():")
    print("  ⚠️  使用简化逻辑")
    print("  ⚠️  使用: test_min_date")
    print("  ⚠️  使用: test_max_date + BDay(purge_window)")
    
    print("\n虽然PurgedKFold只有一个test段，但逻辑应该保持一致")
    print("  💡 建议: 统一两个类的purge逻辑\n")


def run_all_pocs():
    """
    运行所有PoC验证
    """
    print("\n" + "="*60)
    print("寇连材审计 - PoC验证")
    print("="*60)
    
    test_c01_loc_vs_iloc()
    test_c02_purge_logic_inconsistency()
    test_m01_code_duplication()
    test_m02_performance()
    test_m03_magic_numbers()
    test_m04_purgedkfold_inconsistency()
    
    print("\n" + "="*60)
    print("PoC验证完成")
    print("="*60)
    print("\n总结:")
    print("  🔴 严重问题: 2个（C-01, C-02）")
    print("  🟡 中等问题: 4个（M-01~M-04）")
    print("  🟢 轻微问题: 4个（m-01~m-04）")
    print("\n建议:")
    print("  1. 立即修复: C-01, C-02")
    print("  2. 本周修复: M-01, M-02, M-04")
    print("  3. 有时间时: M-03, m-01~m-04")


if __name__ == '__main__':
    run_all_pocs()
