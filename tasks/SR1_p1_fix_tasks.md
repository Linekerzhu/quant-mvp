# SR1 P1 问题修复计划

> 文档创建：张德功（八品领侍）
> 日期：2026-02-27
> 审计阶段：SR1 (Security & Reliability Audit 1)

---

## 问题概述

本次 P1 问题涉及两个模块的输入验证和逻辑优化：

| 模块 | 文件路径 | 问题类型 | 风险等级 |
|------|----------|----------|----------|
| 分数差分 | `src/features/fracdiff.py` | 输入验证缺失 | P1 |
| 清洗交叉验证 | `src/models/purged_kfold.py` | 逻辑漏洞 | P1 |

---

## 问题 1：fracdiff.py 输入验证

### 1.1 问题描述

#### 问题 A：ADF 最小样本量不足

**代码位置**：`find_min_d_stationary()` 函数（第 136-181 行）

**具体问题**：
- 第 155 行直接调用 `adfuller(series.dropna(), ...)`，未检查样本量是否满足 ADF 测试要求
- ADF 测试对样本量有严格要求：
  - `maxlag` 参数默认值为 12*(nobs/100)^{1/4}
  - 当 nobs < 20 时，maxlag 计算可能为 0 或负数
  - statsmodels 会抛出 `ValueError: maxlag must be positive` 或警告样本不足

**风险**：
- 生产环境中遇到短序列会触发异常中断
- 影响回测系统的稳定性

#### 问题 B：NaN 输入未验证

**代码位置**：所有 `fracdiff_*` 函数及 `find_min_d_stationary()`

**具体问题**：
- 输入 `series` 可能包含大量 NaN
- `fracdiff_fixed_window` 返回序列的前 `window-1` 个值默认为 NaN
- 没有前置验证确保输入有足够有效数据

**风险**：
- 静默产生全 NaN 结果
- 下游模块无法检测数据质量问题

### 1.2 修复方案

#### 修复 A：ADF 样本量验证

```python
def find_min_d_stationary(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    threshold: float = 0.01,
    window: int = 100,
    min_adf_samples: int = 30  # 新增：ADF 最小样本要求
) -> float:
    """
    ...
    Args:
        ...
        min_adf_samples: Minimum samples required for ADF test (default: 30)
                         Based on statsmodels recommendation for reliable results.
    """
    # 新增：前置验证
    clean_series = series.dropna()
    if len(clean_series) < min_adf_samples:
        raise ValueError(
            f"Insufficient data for ADF test: got {len(clean_series)} samples, "
            f"minimum required: {min_adf_samples}. "
            f"Consider reducing window size or using more data."
        )
    
    # 检查 maxlag 有效性
    maxlag = min(window // 10, (len(clean_series) - 1) // 2)
    if maxlag < 1:
        raise ValueError(
            f"Cannot compute ADF: maxlag={maxlag}, "
            f"need at least 3 samples after differencing"
        )
    
    # 原有逻辑...
    test_result = adfuller(clean_series, maxlag=maxlag)
    # ...
```

#### 修复 B：NaN 输入验证

```python
def _validate_series(series: pd.Series, min_valid_ratio: float = 0.5) -> pd.Series:
    """
    Validate series has sufficient non-NaN data.
    
    Args:
        series: Input series to validate
        min_valid_ratio: Minimum ratio of non-NaN values (default: 0.5)
    
    Returns:
        Cleaned series
    
    Raises:
        ValueError: If validation fails
    """
    if series is None or len(series) == 0:
        raise ValueError("Series is empty or None")
    
    n_total = len(series)
    n_valid = series.notna().sum()
    valid_ratio = n_valid / n_total
    
    if valid_ratio < min_valid_ratio:
        raise ValueError(
            f"Series has too many NaN values: {n_valid}/{n_total} valid "
            f"({valid_ratio:.1%}), minimum required: {min_valid_ratio:.0%}"
        )
    
    return series

# 在每个 public 函数开头添加：
def fracdiff_fixed_window(series: pd.Series, ...):
    series = _validate_series(series)
    # ... existing code

def find_min_d_stationary(series: pd.Series, ...):
    series = _validate_series(series, min_valid_ratio=0.7)  # 更严格
    # ... existing code
```

### 1.3 测试用例

```python
def test_adf_insufficient_samples():
    """Test ADF with too few samples raises error."""
    short_series = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Insufficient data"):
        find_min_d_stationary(short_series)

def test_nan_input_validation():
    """Test NaN-heavy input raises error."""
    nan_series = pd.Series([1, np.nan, np.nan, np.nan, 2])
    with pytest.raises(ValueError, match="too many NaN"):
        find_min_d_stationary(nan_series)

def test_valid_input_passes():
    """Test valid input works correctly."""
    np.random.seed(42)
    valid_series = pd.Series(np.random.randn(100).cumsum() + 100)
    result = find_min_d_stationary(valid_series)
    assert 0 <= result <= 1
```

### 1.4 工作量估算

| 任务项 | 预估时间 | 负责人 |
|--------|----------|--------|
| 实现 _validate_series 辅助函数 | 30 min | 李得勤 |
| 修改 find_min_d_stationary 样本量检查 | 45 min | 李得勤 |
| 在 fracdiff_fixed_window 等函数添加验证 | 30 min | 李得勤 |
| 编写单元测试 | 60 min | 李得勤 |
| 回归测试 | 30 min | 李得勤 |
| **总计** | **~4 小时** | |

---

## 问题 2：purged_kfold.py Purge 逻辑优化

### 2.1 问题描述

**代码位置**：`CombinatorialPurgedKFold.split()` 和 `PurgedKFold.split()`

**具体问题**：

当前 purge 逻辑仅检查 exit_date 是否在 purge 窗口内（第 155-162 行）：

```python
if exit_date_col in df.columns:
    exit_date = df.loc[idx, exit_date_col]
    if pd.notna(exit_date):
        if exit_date >= purge_start:
            if exit_date <= purge_end:
                continue  # 仅基于 exit_date 判断
```

**遗漏的场景**：

考虑以下时间线（holding period = 5天, purge_window = 10天）：

```
Sample A: entry=Day1, exit=Day6
Sample B: entry=Day15, exit=Day20
Test Period: Day10-15

Purge window: Day0-25 (Day10±10)

当前逻辑：
- Sample A: exit=Day6 < purge_start(Day0)? 否，Day6 >= Day0 ✓, Day6 <= Day25 ✓ → 跳过
- Sample B: exit=Day20 >= Day0 ✓, Day20 <= Day25 ✓ → 跳过

但 Sample A 实际上 entry=Day1, exit=Day6，完全不与 Test Period 重叠！
问题：entry_date 在 Test Period 之前很远，不应被 purge。
```

**正确逻辑**：
Purge 应基于 **entry date**，确保训练样本的 entry 不在 Test Period + purge window 内。

### 2.2 修复方案

#### 修复 Purge 逻辑

```python
def split(self, df: pd.DataFrame, ...):
    # ... 前面代码不变 ...
    
    for test_seg_indices in test_combinations:
        # ... 构建 test_indices ...
        
        test_dates = df.loc[test_indices, date_col]
        test_min_date = test_dates.min()
        test_max_date = test_dates.max()
        
        # 【修复】正确定义 purge 范围：
        # 训练集中的样本如果 entry_date 落在 [test_min - purge, test_max + purge] 应被排除
        # 这确保没有信息泄露
        purge_start = test_min_date - pd.Timedelta(days=self.purge_window)
        purge_end = test_max_date + pd.Timedelta(days=self.purge_window)
        
        # Embargo：测试集结束后的一段时间完全排除
        embargo_end = test_max_date + pd.Timedelta(days=self.embargo_window)
        
        train_indices = []
        for idx in range(n_samples):
            if idx in test_indices:
                continue
            
            row_entry_date = df.loc[idx, date_col]
            row_exit_date = df.loc[idx, exit_date_col] if exit_date_col in df.columns else None
            
            # 【修复】Embargo 检查：entry_date 在 embargo 期间
            if row_entry_date > test_max_date and row_entry_date <= embargo_end:
                continue
            
            # 【修复】Purge 检查：
            # 情况 1：entry_date 在 purge 范围内 → 必须排除
            if purge_start <= row_entry_date <= purge_end:
                continue
            
            # 情况 2：entry_date 在 test period 前，但 exit_date 与 test period 重叠
            if row_entry_date < test_min_date and pd.notna(row_exit_date):
                if row_exit_date >= test_min_date:
                    # 交易跨越了 test period 开始 → 排除
                    continue
            
            # 情况 3：entry_date 在 test period 后，但 entry 在 purge 范围内
            # 已由情况 1 覆盖
            
            train_indices.append(idx)
        
        # ... 后续代码 ...
```

#### 更清晰的实现（推荐）

```python
def _should_exclude(
    self,
    entry_date: pd.Timestamp,
    exit_date: Optional[pd.Timestamp],
    test_min: pd.Timestamp,
    test_max: pd.Timestamp
) -> bool:
    """
    Determine if a sample should be excluded from training set.
    
    Exclusion rules (AFML Ch7):
    1. Entry date falls within purge window around test period
    2. Trade overlaps with test period (entry before test, exit during/after)
    3. Entry falls within embargo period after test
    """
    purge_start = test_min - pd.Timedelta(days=self.purge_window)
    purge_end = test_max + pd.Timedelta(days=self.purge_window)
    embargo_end = test_max + pd.Timedelta(days=self.embargo_window)
    
    # Rule 1: Entry in purge window
    if purge_start <= entry_date <= purge_end:
        return True
    
    # Rule 2: Trade overlaps with test period
    if entry_date < test_min and pd.notna(exit_date):
        if exit_date >= test_min:
            return True
    
    # Rule 3: Entry in embargo period
    if test_max < entry_date <= embargo_end:
        return True
    
    return False
```

### 2.3 测试用例

```python
def test_purge_entry_overlap():
    """Test that entry overlap is properly purged."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=20, freq='D'),
        'label_exit_date': pd.date_range('2024-01-03', periods=20, freq='D'),  # 2-day holding
    })
    
    # Test period: Day 10-12
    # Purge window: Day 5-17 (±5 days)
    
    splitter = CombinatorialPurgedKFold(
        n_splits=4, n_test_splits=1,
        purge_window=5, embargo_window=2
    )
    
    for train_idx, test_idx in splitter.split(df):
        test_dates = df.loc[test_idx, 'date']
        test_min, test_max = test_dates.min(), test_dates.max()
        
        for idx in train_idx:
            entry = df.loc[idx, 'date']
            exit_ = df.loc[idx, 'label_exit_date']
            
            # Verify no overlap
            purge_start = test_min - pd.Timedelta(days=5)
            purge_end = test_max + pd.Timedelta(days=5)
            
            assert not (purge_start <= entry <= purge_end), \
                f"Entry {entry} should be purged (in purge window)"
            
            if entry < test_min and pd.notna(exit_):
                assert exit_ < test_min, \
                    f"Trade [{entry}, {exit_}] overlaps test period starting {test_min}"

def test_purge_scenarios():
    """Explicit test cases for purge logic."""
    # Scenario matrix:
    # Case 1: entry=Day1, exit=Day3 | Test=Day10-15 | Expected: Keep (far before)
    # Case 2: entry=Day8, exit=Day12 | Test=Day10-15 | Expected: Purge (overlap)
    # Case 3: entry=Day12, exit=Day18 | Test=Day10-15 | Expected: Purge (entry in purge)
    # Case 4: entry=Day18, exit=Day20 | Test=Day10-15 | Expected: Purge (entry in purge)
    # Case 5: entry=Day25, exit=Day27 | Test=Day10-15 | Expected: Keep (far after, beyond embargo)
    pass
```

### 2.4 工作量估算

| 任务项 | 预估时间 | 负责人 |
|--------|----------|--------|
| 分析现有 purge 逻辑问题 | 30 min | 李得勤 |
| 重构 _should_exclude 方法 | 45 min | 李得勤 |
| 更新 CombinatorialPurgedKFold.split() | 30 min | 李得勤 |
| 更新 PurgedKFold.split() | 30 min | 李得勤 |
| 编写边界测试用例 | 90 min | 李得勤 |
| 与原始 AFML Ch7 逻辑对照验证 | 45 min | 李得勤 |
| 回归测试 | 30 min | 李得勤 |
| **总计** | **~6 小时** | |

---

## 总体计划

### 执行顺序

```
Day 1 (上午)
├── fracdiff.py 修复
│   ├── 实现输入验证
│   └── 单元测试

Day 1 (下午)
└── purged_kfold.py 修复
    ├── 重构 purge 逻辑
    └── 边界测试

Day 2
└── 集成测试 & 回归验证
```

### 验收标准

| 检查项 | 标准 |
|--------|------|
| 代码覆盖率 | > 90% for modified lines |
| 测试通过率 | 100% (包括新增边界测试) |
| AFML 对照 | Purge 逻辑与 AFML Ch7 描述一致 |
| 文档更新 | docstring 和类型注解完整 |

### 依赖项

- statsmodels >= 0.14.0 (ADF 测试)
- pandas >= 2.0.0 (时间序列处理)
- pytest >= 7.0 (测试框架)

---

*张德功 敬上*
*长春宫量化团队*
