# Phase B 深度安全与可用性审计报告

**审计时间**: 2026-02-24  
**审计员**: 李得勤  
**审计范围**: Phase B 全部代码（特征工程、标签、市场状态）  
**审计维度**: 安全性、可用性、健壮性、性能、可维护性  

---

## 🔴 严重安全问题

### 1. 数据泄漏风险（PIT 违规）

**问题**: `triple_barrier.py` 第 97-98 行

```python
# 当前代码
entry_price = symbol_df.loc[entry_idx + 1, 'adj_open']  # T+1 open
atr = symbol_df.loc[entry_idx, 'atr_14']  # T 日的 ATR
```

**风险**: ATR 计算使用 `rolling(window=14)`，在 T 日包含 T-13 到 T 的数据，
这是正确的 PIT。但**标签生成时**使用了 T+1 的价格，如果特征也使用 T+1 价格会造成泄漏。

**状态**: ✅ **实际上安全** - 标签使用 T+1 是 Triple Barrier 定义所需，
特征是 T 日计算的，与标签计算分离。

**建议**: 添加注释说明这是标签定义的一部分，非泄漏。

---

### 2. 除零风险

**位置**: `build_features.py` 多处

| 行号 | 代码 | 风险 |
|------|------|------|
| 114 | `(price - sma) / std` | std=0 时除零 |
| 120 | `(price - ema) / std` | std=0 时除零 |
| 162 | `gain / loss` | loss=0 时除零 |

**当前处理**:
- 114/120 行有 `std.replace(0, np.nan)` ✅
- 162 行有 `loss.replace(0, np.nan)` ✅

**状态**: ✅ **已处理**

---

### 3. 随机种子可预测性（安全 vs 可复现性权衡）

**位置**: `build_features.py` 第 25 行

```python
self.dummy_seed = 42  # 硬编码种子
```

**风险**: 硬编码种子导致 dummy_noise 可预测，攻击者可能利用。

**评估**: 
- 这是**有意为之**的设计（Plan v4 要求可复现性）
- dummy_noise 仅用于过拟合检测，不参与实际交易决策
- 可预测性不构成实际安全威胁

**状态**: ✅ **可接受**（设计决策）

---

## 🟡 中等问题

### 4. 内存效率（大数据集）

**位置**: `build_features.py` 第 32-42 行

```python
df = df.copy()  # 完整复制
df = self._calc_momentum_features(df)  # 又复制
df = self._calc_volatility_features(df)  # 又复制
```

**问题**: 多次完整 DataFrame 复制，内存使用 O(n×m)。

**影响**: 
- 10 只股票 × 5 年数据 (~12,500 行 × 50 列) ≈ 50MB
- 500 只股票 × 10 年数据 (~1,260,000 行 × 50 列) ≈ 5GB

**建议优化**:
```python
# 原地修改模式
with pd.option_context('mode.chained_assignment', None):
    self._calc_momentum_features_inplace(df)
```

**优先级**: 🟡 中（当前规模可接受，扩展时需优化）

---

### 5. 循环效率（Symbol-wise 计算）

**位置**: `build_features.py` 多处 `for symbol in df['symbol'].unique()`

**问题**: Python 循环处理每个 symbol，而非向量化。

**当前**: 
```python
for symbol in df['symbol'].unique():
    mask = df['symbol'] == symbol
    df.loc[mask, 'feature'] = calculation
```

**优化方案**（GroupBy）:
```python
df['feature'] = df.groupby('symbol')['adj_close'].transform(
    lambda x: x.rolling(20).mean()
)
```

**性能对比**:
- 当前: O(n_symbols × n_rows) 循环
- 优化: 底层 C 实现，快 10-100 倍

**状态**: 🟡 需优化（Phase C 前）

---

### 6. NaN 传播风险

**位置**: `build_features.py` 第 114 行

```python
df.loc[mask, f'price_vs_sma{window}_zscore'] = (
    (df.loc[mask, 'adj_close'] - sma) / std.replace(0, np.nan)
)
```

**问题**: `std=0` 时替换为 `np.nan`，但其他 NaN（如停牌）也会传播。

**影响**: 模型可能无法处理 NaN 特征。

**建议**: 添加显式 NaN 填充策略
```python
# 在 build_features 最后添加
df[self._get_feature_columns(df)] = df[self._get_feature_columns(df)].fillna(0)
```

---

## 🟢 轻微问题

### 7. 魔法数字

**位置**: `regime_detector.py` 第 21-26 行

```python
self.low_vol_threshold = 0.15  # 15% annualized
self.high_vol_threshold = 0.25  # 25% annualized
self.strong_trend_threshold = 25
self.weak_trend_threshold = 15
```

**问题**: 硬编码阈值，应移入配置文件。

**建议**: 添加到 `config/features.yaml`

---

### 8. 缺少输入验证

**位置**: `build_features.py` 第 28 行

```python
def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
```

**问题**: 无输入列检查，如果缺少 `adj_close` 等列会报错。

**建议**: 添加前置检查
```python
required_cols = ['symbol', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")
```

---

### 9. 日志信息可能泄露敏感数据

**位置**: `build_features.py` 第 175 行

```python
logger.info("dummy_noise_injected", {
    "seed": self.dummy_seed,
    "mean": float(df['dummy_noise'].mean()),
    "std": float(df['dummy_noise'].std())
})
```

**问题**: 当前是统计信息，安全。但如果未来添加原始数据采样会泄漏。

**建议**: 添加注释提醒未来开发者
```python
# SECURITY: Do not log raw price samples or feature values
```

---

## ✅ 优秀实践

### 10. Dummy Feature 正确隔离

**位置**: `build_features.py` 第 166-179 行

```python
def _inject_dummy_noise(self, df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(self.dummy_seed)
    df['dummy_noise'] = np.random.normal(0, 1, size=len(df))
    # ... 明确文档说明这是 sentinel_only
```

**优点**:
- ✅ 使用固定种子确保可复现性
- ✅ 明确文档说明不用于预测
- ✅ 版本号追踪

---

### 11. 防御性拷贝

**位置**: `build_features.py` 第 31 行

```python
df = df.copy()
```

**优点**: 防止修改原始数据，符合函数式编程原则。

---

### 12. 配置驱动设计

**位置**: 所有模块都使用 YAML 配置

**优点**:
- ✅ 参数可调整无需改代码
- ✅ 版本控制追踪配置变更
- ✅ 符合 Plan v4 要求

---

## 📊 综合评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **安全性** | 85/100 | 无严重漏洞，有优化空间 |
| **健壮性** | 80/100 | 处理了大部分边界情况 |
| **性能** | 65/100 | 可优化为向量化/GroupBy |
| **可维护性** | 85/100 | 结构清晰，配置驱动 |
| **可用性** | 90/100 | API 简洁，文档充分 |

---

## 🛠️ 建议修复清单（按优先级）

### P0（Phase C 前必须修复）

1. **添加输入验证** (`build_features.py`)
   ```python
   def _validate_input(self, df: pd.DataFrame) -> None:
       required = ['symbol', 'date', 'adj_close', 'adj_high', 'adj_low', 'volume']
       missing = set(required) - set(df.columns)
       if missing:
           raise ValueError(f"Missing columns: {missing}")
   ```

2. **优化循环为 GroupBy** (`build_features.py`)
   - 将所有 `for symbol in df['symbol'].unique()` 改为 `groupby().transform()`
   - 预计性能提升 10-100 倍

3. **统一 NaN 处理策略**
   ```python
   # 在 build_features 末尾添加
   feature_cols = self._get_feature_columns(df)
   df[feature_cols] = df[feature_cols].fillna(0)
   ```

### P1（Phase D 前修复）

4. **魔法数字配置化** (`regime_detector.py`)
   - 将阈值移入 `config/features.yaml`

5. **添加性能监控日志**
   ```python
   import time
   start = time.time()
   # ... 计算
   logger.info("feature_calc_time", {"elapsed_ms": (time.time() - start) * 1000})
   ```

### P2（可选优化）

6. **内存优化**
   - 使用 `float32` 代替 `float64`
   - 原地修改减少拷贝

---

## 🔒 安全审计结论

**Phase B 代码整体安全，可进入 Phase C。**

主要风险已控制：
- ✅ 无数据泄漏（PIT 合规）
- ✅ 除零已防护
- ✅ 随机种子可预测性为设计决策

建议在 Phase C 前完成 P0 优化项。

---

*审计完成时间: 2026-02-24*  
*审计员: 李得勤*  
