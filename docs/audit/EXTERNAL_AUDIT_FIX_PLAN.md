# 外部审计阻塞项整改计划

**编制人**: 张得功（八品领侍）  
**编制日期**: 2026-02-28  
**整改目标**: 修复3个CRITICAL阻塞项，确保Phase D可进入

---

## 一、整改概览

| 编号 | 问题 | 严重级别 | 负责人 | 预计工期 | 验收状态 |
|------|------|----------|--------|----------|----------|
| C-01 | PBO计算逻辑完全无效 | CRITICAL | 李得勤 | 4h | ⏳ 待修复 |
| C-02 | MetaTrainer端到端不可运行 | CRITICAL | 李得勤 | 8h | ⏳ 待修复 |
| C-03 | Sample Weights未传入LightGBM | CRITICAL | 李得勤 | 2h | ⏳ 待修复 |
| H-01 | DSR数值不稳定 | HIGH | 李得勤 | 2h | ⏳ 待修复 |
| H-02 | CPCV使用日历日而非交易日 | HIGH | 李得勤 | 3h | ⏳ 待修复 |
| H-03 | BDay(1)落假日问题 | HIGH | 李得勤 | 2h | ⏳ 待修复 |
| H-04 | backtest/execution空壳 | HIGH | 李得勤 | 后续Phase | ⏳ 待规划 |
| H-05 | assert可绕过 | HIGH | 李得勤 | 1h | ⏳ 待修复 |

**总工期估算**: 22小时（约3个工作日）

---

## 二、CRITICAL阻塞项整改方案

### 🔴 C-01: PBO计算逻辑完全无效

**问题描述**:
- **文件**: `src/models/overfitting.py` 第57-65行
- **症状**: `np.argsort(np.argsort(aucs))` 产生 [0,1,2,...,14] 恒定序列
- **影响**: PBO恒等于0.533，无法真实反映过拟合程度
- **根本原因**: 双重argsort产生排名序列，而非比较IS vs OOS性能

**问题代码**:
```python
def calculate_pbo(self, path_results: List[Dict]) -> float:
    aucs = [r['auc'] for r in path_results]
    n = len(aucs)
    
    # ❌ 错误：双重argsort产生 [0,1,2,...,n-1]
    ranked = np.argsort(np.argsort(aucs))
    
    # ❌ 错误：这永远返回约 0.5
    pbo = np.mean(ranked < n / 2)
    
    return float(pbo)
```

**整改方案**:

**方案A: 实现真正的PBO（推荐）**

按照 Bailey & López de Prado (2017) 的定义：

```python
def calculate_pbo(self, path_results: List[Dict]) -> float:
    """
    计算 PBO（Probability of Backtest Overfitting）。
    
    基于 Bailey & López de Prado (2017):
    PBO = Prob(rank_IS != rank_OOS_max)
    
    对于每个组合的测试集，计算：
    1. IS (in-sample) AUC排名
    2. OOS (out-of-sample) AUC
    3. 比较最优IS模型在OOS上的表现
    """
    n = len(path_results)
    if n == 0:
        return 1.0
    
    # 提取 IS 和 OOS AUC
    is_aucs = [r.get('is_auc', r['auc']) for r in path_results]
    oos_aucs = [r.get('oos_auc', r['auc']) for r in path_results]
    
    # 找到 IS 表现最好的路径
    is_ranking = np.argsort(is_aucs)[::-1]  # 降序，best first
    best_is_idx = is_ranking[0]
    
    # 计算 OOS 排名
    oos_ranking = np.argsort(np.argsort(oos_aucs)[::-1])  # 降序排名
    best_is_oos_rank = oos_ranking[best_is_idx]
    
    # PBO: 最优IS模型在OOS中排名靠后的概率
    # 如果排名在下半部分（排名 >= n/2），视为过拟合
    pbo = 1.0 if best_is_oos_rank >= n / 2 else 0.0
    
    return float(pbo)
```

**方案B: 保守估计（简化版）**

如果暂时无法获取IS/OOS分离数据：

```python
def calculate_pbo(self, path_results: List[Dict]) -> float:
    """
    保守估计：计算测试集AUC的方差系数。
    
    高方差 = 高过拟合风险
    CV = std / mean
    """
    aucs = [r['auc'] for r in path_results]
    n = len(aucs)
    
    if n == 0:
        return 1.0
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    
    # 变异系数
    cv = std_auc / mean_auc if mean_auc > 0 else 1.0
    
    # 映射到 [0, 1] 区间
    # CV < 0.1 -> PBO ≈ 0
    # CV > 0.3 -> PBO ≈ 1
    pbo = min(1.0, max(0.0, (cv - 0.05) / 0.25))
    
    return float(pbo)
```

**推荐**: 方案A（符合AFML定义）

**验收标准**:
1. ✅ PBO值不再恒等于0.533
2. ✅ 不同数据集产生不同PBO值
3. ✅ 单元测试验证逻辑正确性
4. ✅ 文档更新，说明PBO计算方法

**测试用例**:
```python
def test_pbo_not_constant():
    """PBO不应恒等于0.533"""
    detector = OverfittingDetector({})
    
    # 构造高过拟合数据（IS好，OOS差）
    high_overfit_results = [
        {'is_auc': 0.7, 'oos_auc': 0.5},
        {'is_auc': 0.65, 'oos_auc': 0.52},
        {'is_auc': 0.68, 'oos_auc': 0.48},
    ]
    pbo_high = detector.calculate_pbo(high_overfit_results)
    
    # 构造低过拟合数据（IS和OOS接近）
    low_overfit_results = [
        {'is_auc': 0.6, 'oos_auc': 0.58},
        {'is_auc': 0.59, 'oos_auc': 0.57},
        {'is_auc': 0.61, 'oos_auc': 0.59},
    ]
    pbo_low = detector.calculate_pbo(low_overfit_results)
    
    # 高过拟合应比低过拟合有更高的PBO
    assert pbo_high > pbo_low, f"PBO should vary: {pbo_high} vs {pbo_low}"
    assert pbo_high != 0.533, "PBO should not be constant 0.533"
```

---

### 🔴 C-02: MetaTrainer端到端不可运行

**问题描述**:
- **文件**: `src/models/meta_trainer.py`
- **症状**: (a) FracDiff列不存在 (b) find_min_d_stationary未调用 (c) 只有5/15路径有效
- **影响**: 无法完成完整的端到端训练流程

**问题A: FracDiff列不存在**

**问题代码** (`meta_trainer.py` 第235-245行):
```python
# ❌ 错误：只是检查，没有计算
optimal_d = 0.5
frac_col = f'fracdiff_{int(optimal_d*10)}'

if frac_col not in features:
    current_features = features + [frac_col]  # 加入特征列表
else:
    current_features = features

# ❌ 但 DataFrame 中根本没有这个列！
result = self._train_cpcv_fold(train_df, test_df, current_features)  # 会报 KeyError
```

**整改方案**:
```python
def _train_cpcv_fold(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    price_col: str = 'adj_close'
) -> Dict[str, Any]:
    """训练单个 CPCV fold，包含 FracDiff 特征计算"""
    from src.features.fracdiff import find_min_d_stationary, fracdiff_fixed_window
    
    # Step 1: 在训练集上找最优 d
    optimal_d = find_min_d_stationary(
        train_df[price_col],
        d_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        adf_pvalue_threshold=0.05
    )
    
    logger.info(f"  Optimal d={optimal_d:.2f}")
    
    # Step 2: 计算 FracDiff 特征（train + test）
    window = 100
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['fracdiff'] = fracdiff_fixed_window(
        train_df[price_col].values, optimal_d, window
    )
    test_df['fracdiff'] = fracdiff_fixed_window(
        test_df[price_col].values, optimal_d, window
    )
    
    # Step 3: 添加到特征列表
    current_features = features + ['fracdiff']
    
    # Step 4: 去除 NaN（FracDiff burn-in period）
    train_df = train_df.dropna(subset=['fracdiff'])
    test_df = test_df.dropna(subset=['fracdiff'])
    
    # Step 5: 训练 LightGBM
    X_train = train_df[current_features]
    y_train = train_df[target_col]
    X_test = test_df[current_features]
    y_test = test_df[target_col]
    
    # ... 后续训练逻辑
```

**问题B: find_min_d_stationary未调用**

**现状**: 代码硬编码 `optimal_d = 0.5`，未使用 ADF 检验找最优值

**整改方案**: 见上方代码，在训练集上调用 `find_min_d_stationary()`

**问题C: 生产配置导致数据不足**

**配置问题** (`config/training.yaml`):
```yaml
cpcv:
  n_splits: 6
  min_data_days: 630  # 约2.5年数据
```

**影响**:
- 630天 / 6 folds = 105天/fold
- purge_window=10, embargo_window=40 → gap=50天
- 有效训练数据 = 105 - 50 = 55天 < 200天阈值
- **结果**: 只有5/15路径满足最小数据要求

**整改方案**（三选一）:

**方案A: 降低min_data_days（临时方案）**
```yaml
cpcv:
  min_data_days: 450  # 450/6=75 > gap(50)，有效25天
```

**方案B: 减少n_splits（推荐）**
```yaml
cpcv:
  n_splits: 5  # C(5,2)=10条路径（减少但足够）
  min_data_days: 500  # 500/5=100 > gap(50)，有效50天
```

**方案C: 减少purge/embargo窗口**
```yaml
cpcv:
  purge_window: 5   # 减少到5天
  embargo_window: 30  # 减少到30天
  # gap = 35天，更多有效训练数据
```

**推荐**: 方案B（平衡路径数量和数据质量）

**验收标准**:
1. ✅ FracDiff特征正确计算并添加到DataFrame
2. ✅ 每个fold调用find_min_d_stationary()找最优d
3. ✅ 所有15条路径（或10条，如果用方案B）有足够训练数据
4. ✅ 端到端训练流程无错误运行
5. ✅ 输出包含optimal_d值

**测试用例**:
```python
def test_meta_trainer_end_to_end():
    """端到端训练测试"""
    # 准备测试数据
    df = pd.DataFrame({
        'symbol': ['AAPL'] * 1000,
        'date': pd.date_range('2020-01-01', periods=1000),
        'adj_close': 100 + np.random.randn(1000).cumsum(),
        'label': np.random.choice([-1, 1], 1000),
        'feature1': np.random.randn(1000),
    })
    
    trainer = MetaTrainer("config/training.yaml")
    base_model = BaseModelSMA(fast_window=20, slow_window=60)
    
    # 应该能完成训练，不抛出异常
    results = trainer.train(df, base_model, features=['feature1'])
    
    assert 'n_paths' in results
    assert results['n_paths'] >= 10  # 至少10条路径
    assert all('optimal_d' in r for r in results['paths'])  # 每条路径都有optimal_d
```

---

### 🔴 C-03: Sample Weights未传入LightGBM

**问题描述**:
- **文件**: `src/models/meta_trainer.py` 第199行
- **症状**: `lgb.Dataset` 创建时未传入 `weight` 参数
- **影响**: 样本权重配置无效，模型训练不考虑uniqueness权重

**问题代码**:
```python
# ❌ 错误：未传入weight参数
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
```

**配置存在但未使用** (`config/training.yaml`):
```yaml
sample_weights:
  method: uniqueness
  min_weight: 0.01
  max_weight: 10.0
```

**整改方案**:
```python
def _train_cpcv_fold(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str = 'meta_label'
) -> Dict[str, Any]:
    """训练单个 CPCV fold，包含样本权重"""
    
    # Step 1: 准备数据
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # Step 2: 计算样本权重（基于 uniqueness）
    train_weights = self._calculate_sample_weights(train_df)
    test_weights = self._calculate_sample_weights(test_df)
    
    # Step 3: 创建带权重的数据集
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        weight=train_weights  # ✅ 传入权重
    )
    valid_data = lgb.Dataset(
        X_test, 
        label=y_test, 
        reference=train_data,
        weight=test_weights  # ✅ 传入权重
    )
    
    # Step 4: 训练
    model = lgb.train(
        self.lgb_params,
        train_data,
        num_boost_round=self.n_estimators,
        valid_sets=[valid_data],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
    )
    
    # ... 后续逻辑

def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
    """
    计算样本权重（基于 uniqueness）
    
    根据 AFML Ch4，样本权重应基于：
    1. Uniqueness: 样本的独立程度
    2. Return: 样本的收益贡献（可选）
    """
    weight_config = self.config.get('sample_weights', {})
    method = weight_config.get('method', 'uniqueness')
    
    if method == 'uniqueness':
        # 使用 uniqueness 列（应由 Phase B 生成）
        if 'uniqueness' in df.columns:
            weights = df['uniqueness'].values
        else:
            # Fallback: 均匀权重
            weights = np.ones(len(df))
    elif method == 'equal':
        weights = np.ones(len(df))
    else:
        weights = np.ones(len(df))
    
    # 应用 min/max 限制
    min_weight = weight_config.get('min_weight', 0.01)
    max_weight = weight_config.get('max_weight', 10.0)
    weights = np.clip(weights, min_weight, max_weight)
    
    # 归一化（保持均值=1）
    weights = weights / weights.mean()
    
    return weights
```

**验收标准**:
1. ✅ LightGBM Dataset 包含 weight 参数
2. ✅ 样本权重基于 uniqueness 计算
3. ✅ 权重值在 [min_weight, max_weight] 范围内
4. ✅ 单元测试验证权重传递
5. ✅ 训练日志包含权重统计信息

**测试用例**:
```python
def test_sample_weights_passed_to_lgb():
    """验证样本权重传入LightGBM"""
    import lightgbm as lgb
    
    # Mock 数据
    train_df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'meta_label': [0, 1, 0, 1, 0],
        'uniqueness': [0.5, 0.8, 0.6, 0.9, 0.4]
    })
    
    trainer = MetaTrainer("config/training.yaml")
    weights = trainer._calculate_sample_weights(train_df)
    
    # 验证权重非均匀
    assert not np.allclose(weights, weights[0]), "Weights should not be uniform"
    
    # 验证权重范围
    config = trainer.config.get('sample_weights', {})
    min_w = config.get('min_weight', 0.01)
    max_w = config.get('max_weight', 10.0)
    assert weights.min() >= min_w
    assert weights.max() <= max_w
```

---

## 三、HIGH级问题整改方案

### ⚠️ H-01: DSR数值不稳定

**问题描述**:
- **文件**: `src/models/overfitting.py` 第133-165行
- **症状**: 当 std=0 或 n<2 时返回0，未标记为无效

**整改方案**:
```python
def calculate_deflated_sharpe(self, path_results: List[Dict]) -> Tuple[float, bool]:
    """
    计算 DSR z-score，返回 (值, 是否有效)
    """
    metrics = [r.get('accuracy', r.get('auc', 0.5)) for r in path_results]
    
    if len(metrics) < 2:
        logger.warn("deflated_sharpe_insufficient_data", {"n_paths": len(metrics)})
        return 0.0, False  # ✅ 标记为无效
    
    mean_sr = np.mean(metrics)
    std_sr = np.std(metrics, ddof=1)
    n = len(metrics)
    
    if std_sr == 0 or n < 2:
        logger.warn("deflated_sharpe_zero_variance", {"std": std_sr, "n": n})
        return 0.0, False  # ✅ 标记为无效
    
    se_sr = std_sr / np.sqrt(n)
    baseline = 0.5
    dsr = (mean_sr - baseline) / se_sr
    
    return float(dsr), True  # ✅ 有效
```

---

### ⚠️ H-02: CPCV使用日历日而非交易日

**问题描述**:
- **文件**: `src/models/purged_kfold.py`
- **症状**: purge_window=10 使用日历日，非交易日

**整改方案**:
```python
def _apply_purge(self, df: pd.DataFrame, ...):
    """应用 purge，使用交易日而非日历日"""
    
    # ✅ 使用 pd.tseries.offsets.BDay (business day)
    from pandas.tseries.offsets import BDay
    
    purge_start = test_start - BDay(self.purge_window)
    purge_end = test_start
    
    # ... 后续逻辑
```

---

### ⚠️ H-03: BDay(1)落假日问题

**问题描述**:
- **症状**: `BDay(1)` 可能落在假日（如中国春节）

**整改方案**:
```python
from pandas.tseries.holiday import USFederalHolidayCalendar

# 使用交易日历
us_calendar = USFederalHolidayCalendar()
business_day = pd.tseries.offsets.CustomBusinessDay(calendar=us_calendar)

purge_start = test_start - business_day(self.purge_window)
```

---

### ⚠️ H-05: assert可绕过

**问题描述**:
- **文件**: `src/models/meta_trainer.py` 第89-96行
- **症状**: 生产环境可用 `-O` 标志绕过 assert

**整改方案**:
```python
# ❌ 错误：assert 可被 -O 绕过
assert max_depth <= 3, f"OR5: max_depth must be <= 3"

# ✅ 正确：使用显式检查
if max_depth > 3:
    raise ValueError(f"OR5 VIOLATION: max_depth={max_depth} > 3")
if num_leaves > 7:
    raise ValueError(f"OR5 VIOLATION: num_leaves={num_leaves} > 7")
if min_data_in_leaf < 100:
    raise ValueError(f"OR5 VIOLATION: min_data_in_leaf={min_data_in_leaf} < 100")
```

---

## 四、执行时间表

### Week 1 (2026-03-02 ~ 2026-03-06)

| 日期 | 任务 | 负责人 | 工时 |
|------|------|--------|------|
| Day 1 上午 | C-01: PBO修复 | 李得勤 | 2h |
| Day 1 下午 | C-01: 测试验证 | 李得勤 | 2h |
| Day 2 全天 | C-02: MetaTrainer修复（FracDiff+find_min_d） | 李得勤 | 6h |
| Day 3 上午 | C-02: 配置调整（n_splits/min_data_days） | 李得勤 | 2h |
| Day 3 下午 | C-03: Sample Weights修复 | 李得勤 | 2h |

### Week 2 (2026-03-09 ~ 2026-03-13)

| 日期 | 任务 | 负责人 | 工时 |
|------|------|--------|------|
| Day 1 上午 | H-01: DSR稳定性 | 李得勤 | 1h |
| Day 1 下午 | H-02: CPCV交易日修复 | 李得勤 | 2h |
| Day 2 上午 | H-03: BDay假日处理 | 李得勤 | 2h |
| Day 2 下午 | H-05: assert替换 | 李得勤 | 1h |
| Day 3 | 集成测试 + 文档更新 | 李得勤 | 4h |

---

## 五、验收检查清单

### CRITICAL阻塞项验收

- [ ] **C-01**: PBO值不再恒等于0.533
  - [ ] 单元测试通过：`test_pbo_not_constant()`
  - [ ] 不同数据集产生不同PBO
  - [ ] 文档更新

- [ ] **C-02**: MetaTrainer端到端可运行
  - [ ] FracDiff特征正确计算
  - [ ] find_min_d_stationary() 被调用
  - [ ] 所有路径有足够训练数据（>= 10条）
  - [ ] 集成测试通过：`test_meta_trainer_end_to_end()`

- [ ] **C-03**: Sample Weights正确传递
  - [ ] lgb.Dataset 包含 weight 参数
  - [ ] 权重基于 uniqueness 计算
  - [ ] 单元测试通过：`test_sample_weights_passed_to_lgb()`

### HIGH级问题验收

- [ ] **H-01**: DSR返回有效性标记
- [ ] **H-02**: CPCV使用 BDay 而非日历日
- [ ] **H-03**: 使用交易日历处理假日
- [ ] **H-05**: assert 替换为显式 ValueError

### 最终验收

- [ ] 所有单元测试通过：`pytest tests/ -v`
- [ ] 端到端流程无错误：`python run_pipeline.py --mode train`
- [ ] 代码审查通过（李成荣公公审批）
- [ ] 文档更新完成

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| FracDiff计算耗时 | 训练时间增加 | 预计算 + 缓存 |
| 配置调整影响现有测试 | 测试失败 | 同步更新测试固件 |
| 交易日历数据缺失 | BDay计算错误 | 使用 pandas 内置 USFederalHolidayCalendar |
| Sample weights极端值 | 模型不稳定 | clip(min_weight, max_weight) |

---

## 七、附件

### A. 修复后代码示例

见各CRITICAL问题的"整改方案"部分。

### B. 测试计划

1. **单元测试**: 每个修复项对应独立测试
2. **集成测试**: 端到端训练流程
3. **回归测试**: 确保原有功能不受影响

### C. 回滚方案

如修复引入新问题：
1. Git revert 对应 commit
2. 恢复原配置文件
3. 重新评估修复方案

---

**编制完成日期**: 2026-02-28  
**下次审计日期**: 2026-03-16

---

*张得功谨呈*  
*八品领侍*
