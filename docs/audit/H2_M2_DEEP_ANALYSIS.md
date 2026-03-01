# H2/M2 问题深度分析报告

**分析者**: 张德功（长春宫八品领侍）  
**日期**: 2026-03-01  
**审计来源**: 寇连材内审发现  
**问题级别**: HIGH (H2), MEDIUM (M2)

---

## 执行摘要

寇连材公公在内审中发现了两个关键问题：

| 问题 | 严重度 | 核心问题 | 当前建议 |
|------|--------|----------|----------|
| **H2** | HIGH | Time Barrier 过滤导致35%样本丢失 | 添加监控 |
| **M2** | MEDIUM | 样本权重在稀疏序列上计算偏差 | 完整序列预计算 |

本报告从**举一反三**、**次生灾害评估**、**最佳方案评估**三个维度进行深度分析。

---

## 第一部分：举一反三分析

### 1.1 H2: Time Barrier 过滤偏差 —— 是否代表一类问题？

#### 核心洞察：这是典型的"样本选择偏差"模式

```
数据流:
原始数据 (100%)
    ↓
Triple Barrier 标注 → label ∈ {-1, 0, 1}
    ↓
Base Model 生成 side ∈ {-1, 0, +1}
    ↓
过滤 side != 0 → 移除"无信号"样本
    ↓
LabelConverter 过滤 label != 0 → 移除 time barrier (35%)
    ↓
训练数据 (约 65%)
```

**问题本质**：
- 不是代码 bug，而是**策略选择**
- 但策略带来了**系统性偏差**
- 模型只学习"有明确趋势"的场景，在震荡市中表现未知

#### 举一反三：系统中是否存在类似模式？

| 位置 | 现象 | 风险 |
|------|------|------|
| `event_valid` 过滤 | 无效事件被静默丢弃 | 可能过滤"困难样本" |
| `can_trade` 过滤 | 停牌期间事件被移除 | 可能引入幸存者偏差 |
| `features_valid` 过滤 | 特征不完整样本被移除 | 可能过滤异常但重要的样本 |
| `side=0` 冷启动过滤 | 冷启动期无信号 | 已处理，但逻辑类似 |

**关键发现**：
> 整个系统存在**多层过滤链**，每层都声称"移除无效数据"，但累积效应可能导致：
> 1. **训练分布 ≠ 实际分布**
> 2. **模型在边缘情况表现未知**
> 3. **回测结果可能过于乐观**

#### 预防性修复建议

```python
# 建议添加系统级监控 - data_filter_monitor.py
class DataFilterMonitor:
    """监控所有数据过滤点的累积效应"""
    
    def __init__(self):
        self.filter_stages = []
    
    def record_filter(self, stage_name: str, n_before: int, n_after: int, 
                      filter_reason: str):
        """记录每个过滤阶段"""
        self.filter_stages.append({
            'stage': stage_name,
            'n_before': n_before,
            'n_after': n_after,
            'filter_ratio': 1 - n_after/n_before,
            'reason': filter_reason
        })
    
    def get_cumulative_filter_ratio(self) -> float:
        """计算累积过滤比例"""
        if not self.filter_stages:
            return 0.0
        initial = self.filter_stages[0]['n_before']
        final = self.filter_stages[-1]['n_after']
        return 1 - final/initial
    
    def warn_if_excessive(self, threshold: float = 0.5):
        """如果累积过滤超过阈值，发出警告"""
        ratio = self.get_cumulative_filter_ratio()
        if ratio > threshold:
            logger.warning(f"累积过滤比例 {ratio:.1%} 超过阈值 {threshold:.1%}")
            for stage in self.filter_stages:
                logger.warning(f"  {stage['stage']}: -{stage['filter_ratio']:.1%}")
```

---

### 1.2 M2: 样本权重时间连续性假设 —— 是否代表一类问题？

#### 核心洞察：这是典型的"算法假设被数据流破坏"模式

```
问题数据流（当前）:
原始事件序列（时间连续）
    ↓
Triple Barrier 标注
    ↓
Base Model 生成 side
    ↓
side != 0 过滤 → 序列变稀疏
    ↓
SampleWeightCalculator 计算权重
    ↓
问题：算法假设时间连续，但输入数据已不连续
```

**问题本质**：
- 算法实现正确
- 但**输入数据不满足算法前置假设**
- Sweep-line 算法假设：事件序列时间连续，可以计算并发数
- 实际输入：经过过滤后的稀疏序列

#### 举一反三：系统中还有哪些算法假设可能被破坏？

| 算法 | 隐含假设 | 可能被破坏的场景 | 风险 |
|------|----------|------------------|------|
| **FracDiff** | 等间距时间序列 | side!=0 过滤后稀疏 | 差分阶数计算偏差 |
| **CPCV** | 时间排序的数据 | 错误排序或索引错乱 | 信息泄漏 |
| **Triple Barrier** | 连续的OHLC数据 | 数据源缺失某天 | 屏障计算错误 |
| **ATR计算** | 连续的high/low | 停牌期间 | 波动性估计偏差 |
| **样本权重** | 事件时间连续 | side!=0 过滤后 | 并发数低估 |

**关键发现**：
> 这是一个**系统性风险模式**：
> 1. 算法 A 有隐含假设 X
> 2. 上游处理 B 破坏了假设 X
> 3. 算法 A 在错误输入上运行，结果不可靠
> 4. 由于算法 A 不报错，问题被静默引入

#### 预防性修复建议

```python
# 建议添加假设验证装饰器 - algorithm_assumptions.py
from functools import wraps
import pandas as pd

def validate_temporal_continuity(required_col: str = 'date'):
    """验证时间连续性假设"""
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            if required_col in df.columns:
                dates = pd.to_datetime(df[required_col])
                # 检查是否有异常大的时间间隔
                time_diffs = dates.diff().dropna()
                median_diff = time_diffs.median()
                max_diff = time_diffs.max()
                
                if max_diff > median_diff * 3:
                    logger.warning(
                        f"{func.__name__}: 时间序列可能不连续，"
                        f"最大间隔 {max_diff} 远超中位数 {median_diff}"
                    )
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

# 使用示例
class SampleWeightCalculator:
    
    @validate_temporal_continuity('date')
    def calculate_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        # 原始实现...
        pass
```

---

### 1.3 修复 H2/M2 能否预防其他类似 Bug？

#### 正向收益

| 修复内容 | 预防的潜在问题 |
|----------|----------------|
| H2 监控 | 所有"选择性过滤导致分布偏移"问题 |
| M2 预计算权重 | 所有"算法假设被破坏"问题 |
| 数据流追踪文档 | 跨模块数据对齐问题 |
| 假设验证机制 | 静默错误引入问题 |

#### 具体预防效果

```
如果实施 H2 监控方案:
├── 能发现 Triple Barrier 中 time barrier 比例异常
├── 能发现 event_valid 过滤比例过高
├── 能发现特定市场环境下样本选择偏差
└── 能评估模型在不同波动区间的表现

如果实施 M2 预计算权重:
├── 能确保样本权重不受过滤影响
├── 能建立"完整序列计算 → 过滤 → 训练"的标准模式
├── 能预防 FracDiff 等算法在稀疏序列上的问题
└── 能明确数据处理的正确顺序
```

---

## 第二部分：次生灾害评估

### 2.1 H2 修复后的潜在负面影响

#### 方案 A: 仅添加监控（推荐短期方案）

**潜在负面影响**：
1. **性能开销**：每次训练都需要计算并记录 time barrier 统计
   - 影响程度：低（仅统计计算，O(n)）
   - 缓解措施：异步日志记录

2. **日志膨胀**：增加新的监控指标导致日志量增加
   - 影响程度：中
   - 缓解措施：结构化日志，按需查询

3. **心理影响**：开发者看到高比例 time barrier 可能过度反应
   - 影响程度：低
   - 缓解措施：文档说明正常范围

**次生灾害风险**：**LOW**

#### 方案 B: 三分类 Meta-Label（中期方案）

**潜在负面影响**：

1. **模型复杂度增加**
   ```python
   # 二分类
   objective: binary
   metric: binary_logloss
   
   # 三分类 - 更复杂
   objective: multiclass
   num_class: 3
   metric: multi_logloss  # 更难优化
   ```

2. **类别不平衡问题**
   - 如果 time barrier 占 35%，三类分布可能不均衡
   - 需要类别权重调整
   - 评估指标需要重新设计（不能简单用 AUC）

3. **决策阈值复杂化**
   ```python
   # 二分类决策
   if prob > 0.5: trade
   
   # 三分类决策 - 需要两个阈值
   if prob_profit > threshold_high: trade
   elif prob_loss > threshold_low: avoid
   else: uncertain → 观望
   ```

4. **与现有回测框架兼容性问题**
   - 回测逻辑需要处理"观望"决策
   - 仓位管理需要支持"不交易"

**次生灾害风险**：**MEDIUM-HIGH**

#### 方案 C: Base Model 增强（长期方案）

**潜在负面影响**：

1. **回测结果不可比**
   - Base Model 逻辑改变 → 信号生成改变 → 无法与历史结果对比

2. **需要重新调参**
   - 新增的波动性阈值、趋势强度阈值需要重新优化
   - 可能引入过拟合

3. **周期延长**
   - 需要大量回测验证
   - 上线风险高

**次生灾害风险**：**HIGH**

### 2.2 M2 修复后的潜在负面影响

#### 方案 A: 完整序列预计算（推荐方案）

**潜在负面影响**：

1. **内存开销增加**
   ```python
   # 当前流程
   df_filtered = df[df['side'] != 0]  # 例如 1000 → 800 条
   weights = calculate_weights(df_filtered)  # 800 条
   
   # 修复后流程
   weights_full = calculate_weights(df)  # 1000 条
   df_filtered = df[df['side'] != 0]  # 800 条
   df_filtered['sample_weight'] = weights_full.loc[df_filtered.index]  # 需要保留 1000 条权重
   ```
   - 需要临时存储全量权重
   - 影响程度：低（权重只是一个 float 列）

2. **索引对齐风险**
   ```python
   # 风险点：merge 或 loc 赋值时索引对齐
   df_signals = df_signals.merge(
       df_full_weights,
       on=['date', 'symbol'],
       how='left'
   )
   # 如果 date/symbol 组合不唯一，可能产生 Cartesian product
   ```
   - 需要确保 merge keys 的唯一性
   - 需要添加验证检查

3. **计算时间增加**
   - 全量计算权重 vs 过滤后计算
   - 影响程度：低（权重计算是 O(n log n)，数据量增加 25% 不会显著增加时间）

**次生灾害风险**：**LOW**

#### 方案 B: 并发事件校正因子（备选方案）

**潜在负面影响**：

1. **统计假设可能不成立**
   ```python
   # 假设：被过滤事件的并发分布与保留事件相似
   correction_factor = 1 / (1 - side_0_ratio)
   ```
   - 实际上 side=0 事件可能集中在低波动期，与保留事件分布不同
   - 校正因子可能过度或不足校正

2. **引入系统性偏差**
   - 如果校正不准确，可能比不校正更糟糕
   - 难以验证校正效果

**次生灾害风险**：**MEDIUM**

#### 方案 C: side=0 事件纳入权重计算

**潜在负面影响**：

1. **接口变更**
   ```python
   # 当前接口
def calculate_weights(self, df: pd.DataFrame) -> pd.DataFrame:
       
   # 新接口需要额外参数
   def calculate_weights(
       self, 
       df: pd.DataFrame,
       filter_col: str = 'side',
       filter_value: int = 0
   ) -> pd.DataFrame:
   ```
   - 所有调用方需要修改
   - 可能遗漏某些调用点

2. **语义混淆**
   - 返回的 DataFrame 包含 side=0 的行（有权重）
   - 调用方需要记得再次过滤
   - 容易出错

**次生灾害风险**：**MEDIUM**

### 2.3 边界情况处理评估

#### H2 边界情况

| 边界情况 | 当前处理 | 修复后处理 | 评估 |
|----------|----------|------------|------|
| time barrier = 0% | 无问题 | 监控显示 0% | ✓ 正确 |
| time barrier = 50% | 无问题 | 监控警告 | ✓ 正确 |
| time barrier = 100% | 训练数据为空 | 监控报警，提前终止 | 需要处理 |
| 多空 time barrier 分布不均 | 无感知 | 分别监控 | ✓ 更好 |

#### M2 边界情况

| 边界情况 | 当前处理 | 修复后处理 | 评估 |
|----------|----------|------------|------|
| side=0 比例 = 0% | 无问题 | 预计算无额外开销 | ✓ 正确 |
| side=0 比例 = 90% | 权重严重偏差 | 预计算正确 | ✓ 修复有效 |
| 单 symbol 全 side=0 | 该 symbol 被过滤 | 预计算保留权重 | ✓ 正确 |
| 索引重复 | 可能错误 | merge 前验证唯一性 | 需要添加检查 |

---

## 第三部分：最佳方案评估

### 3.1 当前建议方案是否是最佳方案？

#### H2 当前建议：添加监控

**评估**：

| 维度 | 评分 | 理由 |
|------|------|------|
| 正确性 | ⭐⭐⭐⭐⭐ | 不改变现有逻辑，仅增加可见性 |
| 风险 | ⭐⭐⭐⭐⭐ | 零风险，不修改训练逻辑 |
| 成本 | ⭐⭐⭐⭐⭐ | 2小时实现 |
| 效果 | ⭐⭐⭐ | 治标不治本，但提供决策数据 |
| **综合** | **⭐⭐⭐⭐** | **短期最佳方案** |

**结论**：监控先行是**正确的短期策略**，理由：
1. 问题本质是策略选择，而非代码错误
2. 在修改策略前，需要先量化问题影响
3. 35% 的 time barrier 是否导致性能高估，需要数据支持

#### M2 当前建议：完整序列预计算

**评估**：

| 维度 | 评分 | 理由 |
|------|------|------|
| 正确性 | ⭐⭐⭐⭐⭐ | 符合算法假设，逻辑严谨 |
| 风险 | ⭐⭐⭐⭐ | 需要处理索引对齐，低风险 |
| 成本 | ⭐⭐⭐⭐⭐ | 4小时实现，改动量小 |
| 效果 | ⭐⭐⭐⭐⭐ | 根治问题 |
| **综合** | **⭐⭐⭐⭐⭐** | **最佳方案** |

**结论**：完整序列预计算是**最佳方案**，理由：
1. 保留算法假设的完整性
2. 改动量适中，风险可控
3. 不引入额外的统计假设（如校正因子）

### 3.2 有没有更好的替代方案？

#### H2 替代方案对比

| 方案 | 正确性 | 风险 | 成本 | 效果 | 综合 |
|------|--------|------|------|------|------|
| **A: 监控（当前）** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2h | ⭐⭐⭐ | **⭐⭐⭐⭐** |
| B: 三分类 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 6h | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| C: Base Model 增强 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 2周 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| D: 保留 time barrier 作为回归 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4h | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**方案 D 详细说明**（新增替代方案）：

```python
# 不是三分类，而是用回归预测 meta_label ∈ [0, 1]
# time barrier 的 meta_label 可以这样定义:

def compute_meta_label_regression(df):
    """
    回归模式：meta_label ∈ [0, 1]
    - 1.0: 明确盈利（hit profit barrier）
    - 0.0: 明确亏损（hit loss barrier）
    - 0.5: 不确定（hit time barrier, return ≈ 0）
    - (0, 0.5) 或 (0.5, 1): time barrier 但 return 有方向
    """
    if 'side' in df.columns:
        # 方向正确性 × 收益幅度
        # side=+1, label_return=+0.05 → meta_label ≈ 0.75
        # side=-1, label_return=-0.05 → meta_label ≈ 0.75 (做空正确)
        # side=+1, label_return=-0.02 → meta_label ≈ 0.4
        direction_correct = (df['side'] * df['label_return']) > 0
        magnitude = df['label_return'].abs() / df['label_return'].abs().max()
        
        df['meta_label'] = 0.5 + (direction_correct.astype(float) - 0.5) * magnitude
    return df
```

**方案 D 优点**：
- 保留 time barrier 信息（通过 return 幅度）
- 使用 LightGBM 回归模式，无需改多分类
- 更细粒度地表达"不确定程度"

**方案 D 缺点**：
- 需要验证回归目标的设计合理性
- 回测逻辑需要支持概率阈值

#### M2 替代方案对比

| 方案 | 正确性 | 风险 | 成本 | 效果 | 综合 |
|------|--------|------|------|------|------|
| **A: 预计算（当前）** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4h | ⭐⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| B: 校正因子 | ⭐⭐⭐ | ⭐⭐⭐ | 2h | ⭐⭐⭐ | ⭐⭐⭐ |
| C: 纳入 side=0 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 6h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| D: 重新设计权重算法 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 1周 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**方案 D 详细说明**（长期架构优化）：

```python
# 重新设计权重算法，使其对稀疏序列鲁棒
class RobustSampleWeightCalculator:
    """
    鲁棒的样本权重计算，显式处理稀疏序列
    """
    
    def calculate_weights(self, df: pd.DataFrame, 
                          full_timeline: pd.DataFrame = None) -> pd.DataFrame:
        """
        Args:
            df: 可能稀疏的事件序列
            full_timeline: 完整的时间线（可选，如果提供则基于此计算）
        """
        if full_timeline is not None:
            # 使用完整时间线计算并发
            return self._calculate_with_timeline(df, full_timeline)
        else:
            # 检测稀疏性，如果稀疏则警告
            if self._is_sparse(df):
                logger.warning("事件序列稀疏，权重可能不准确")
            return self._calculate_weights_sweep_line(df)
    
    def _is_sparse(self, df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """检测序列是否稀疏"""
        dates = pd.to_datetime(df['date'])
        time_diffs = dates.diff().dropna()
        # 如果存在异常大的间隔，认为是稀疏的
        median_diff = time_diffs.median()
        large_gaps = (time_diffs > median_diff * 3).sum()
        return large_gaps / len(time_diffs) > threshold
```

### 3.3 成本收益比分析

#### H2 各方案成本收益比

```
方案 A（监控）:
├── 成本: 2小时
├── 收益: 
│   ├── 量化 time barrier 影响（价值高）
│   ├── 为策略决策提供数据支持
│   └── 零风险
└── 收益/成本比: ⭐⭐⭐⭐⭐ (极高)

方案 B（三分类）:
├── 成本: 6小时 + 回测验证时间
├── 收益:
│   ├── 保留完整样本
│   ├── 但模型复杂度增加
│   └── 效果不确定（可能更好，可能更差）
└── 收益/成本比: ⭐⭐⭐ (中等)

方案 C（Base Model 增强）:
├── 成本: 2周
├── 收益:
│   ├── 从源头解决问题
│   └── 架构更优雅
└── 收益/成本比: ⭐⭐ (低，周期太长)

方案 D（回归模式）:
├── 成本: 4小时
├── 收益:
│   ├── 细粒度表达不确定性
│   ├── 保留 time barrier 信息
│   └── 无需改多分类
└── 收益/成本比: ⭐⭐⭐⭐ (高)
```

**推荐路径**：
```
立即（今天） → 方案 A（监控）
    ↓
本周 → 方案 D（回归模式）试点
    ↓
验证效果后 → 决定是否推广或尝试方案 B
    ↓
长期 → 方案 C（Base Model 增强）
```

#### M2 各方案成本收益比

```
方案 A（预计算）:
├── 成本: 4小时
├── 收益:
│   ├── 根治问题
│   ├── 逻辑严谨
│   └── 风险低
└── 收益/成本比: ⭐⭐⭐⭐⭐ (极高)

方案 B（校正因子）:
├── 成本: 2小时
├── 收益:
│   ├── 快速缓解
│   └── 但可能不准确
└── 收益/成本比: ⭐⭐⭐ (中等，治标不治本)

方案 C（纳入 side=0）:
├── 成本: 6小时
├── 收益:
│   ├── 精确考虑被过滤事件
│   └── 但接口变更大
└── 收益/成本比: ⭐⭐⭐⭐ (高，但不如 A)

方案 D（重新设计算法）:
├── 成本: 1周
├── 收益:
│   └── 架构更优雅，但收益与 A 相同
└── 收益/成本比: ⭐⭐ (低)
```

**推荐**：**方案 A（预计算）** 是最佳性价比选择

---

## 第四部分：综合建议

### 4.1 分阶段实施计划

```
Phase 1: 立即（今天，2小时）
├── H2-A: 添加 Time Barrier 监控
│   └── 记录 time barrier 比例、收益分布
├── 收益: 立即获得问题量化数据
└── 风险: 零

Phase 2: 本周（6小时）
├── M2-A: 完整序列预计算权重
│   └── 修改 meta_trainer.py，先计算全量权重
├── H2-D: 试点回归模式（可选）
│   └── 在一个品种上测试回归 meta_label
└── 收益: 修复 M2，验证 H2 替代方案

Phase 3: 下周（4小时）
├── 评估监控数据
│   └── 分析 time barrier 分布、与波动性关系
├── 评估回归模式效果
│   └── 对比二分类 vs 回归的 AUC/夏普比率
└── 收益: 数据驱动的策略决策

Phase 4: 本月（视评估结果）
├── 如果 H2 确实是问题 → 推广回归模式或三分类
├── 如果 M2 修复有效 → 关闭问题
└── 添加回归测试，防止回退
```

### 4.2 关键决策点

| 决策点 | 决策依据 | 决策标准 |
|--------|----------|----------|
| 是否实施 H2-B/C/D? | Phase 1 监控数据 | time barrier > 40% 且与波动性负相关 |
| 是否实施 M2-C? | Phase 2 修复验证 | 预计算与过滤后计算差异 > 10% |
| 是否长期方案 C? | Phase 3 效果评估 | 当前方案在震荡市表现差 |

### 4.3 风险缓解措施

| 风险 | 缓解措施 |
|------|----------|
| M2 修复导致索引对齐错误 | 添加 merge 验证检查，确保一对一映射 |
| H2 回归模式效果差 | 保留原有二分类作为 fallback |
| 监控数据不足 | 记录完整分布，不只是比例 |
| 修复引入性能问题 | 添加性能基准测试 |

---

## 第五部分：总结

### 5.1 核心结论

1. **H2 问题**是**策略层面的样本选择偏差**，不是代码错误
   - 当前建议（监控）是**正确的短期策略**
   - 中期推荐尝试**回归模式**（方案 D）

2. **M2 问题**是**算法假设被数据流破坏**
   - 当前建议（预计算）是**最佳方案**
   - 性价比高，风险低，效果确定

3. **举一反三**：发现了系统中存在的**系统性风险模式**
   - 多层过滤链导致的累积效应
   - 算法假设被上游处理破坏
   - 建议建立系统性的防御机制

### 5.2 关键行动项

| 优先级 | 行动项 | 负责人 | 工时 | 验收标准 |
|--------|--------|--------|------|----------|
| P0 | H2-A 监控方案 | 待分配 | 2h | time_barrier_stats 日志输出 |
| P0 | M2-A 预计算权重 | 待分配 | 4h | 全量权重计算正确，测试通过 |
| P1 | H2-D 回归试点 | 待分配 | 4h | 单品种回归模式测试完成 |
| P1 | 添加假设验证机制 | 待分配 | 4h | 关键算法添加装饰器验证 |
| P2 | 数据流追踪文档 | 待分配 | 4h | 数据流图文档完成 |

### 5.3 长期架构建议

```
建议建立"算法契约"机制:

每个算法模块显式声明:
├── 输入假设（Input Assumptions）
│   ├── 时间连续性
│   ├── 等间距
│   └── 数据完整性
├── 前置条件（Preconditions）
│   ├── 最小样本量
│   ├── 必要列存在
│   └── 数据类型
└── 验证机制（Validation）
    ├── 输入验证装饰器
    ├── 运行时检查
    └── 警告/错误机制
```

---

**报告完成**  
*张德功恭呈*  
*长春宫*

---

## 附录：关键代码片段

### H2 监控方案代码

```python
# src/models/label_converter.py - H2-A 监控

def convert(self, df: pd.DataFrame) -> pd.DataFrame:
    if self.strategy == 'binary_filtered':
        # === H2 监控 ===
        n_total = len(df)
        n_time_barrier = (df['label'] == 0).sum()
        
        if n_time_barrier > 0:
            tb_df = df[df['label'] == 0]
            stats = {
                'count': n_time_barrier,
                'ratio': n_time_barrier / n_total,
                'mean_return': tb_df['label_return'].mean(),
                'mean_holding_days': tb_df['label_holding_days'].mean(),
                'return_near_zero': (tb_df['label_return'].abs() < 0.01).mean(),
            }
            logger.info("time_barrier_stats", stats)
            
            # 警告阈值
            if stats['ratio'] > 0.40:
                logger.warning("time_barrier_ratio_high", stats)
        # === H2 监控结束 ===
        
        df = df[df['label'].notna() & (df['label'] != 0)].copy()
        # ... 后续逻辑
```

### M2 预计算权重代码

```python
# src/models/meta_trainer.py - M2-A 预计算

def train(self, df, base_model, features, price_col='adj_close'):
    # ... 前置逻辑 ...
    
    # === M2 修复：完整序列预计算权重 ===
    from src.labels.sample_weights import SampleWeightCalculator
    weight_calc = SampleWeightCalculator()
    
    # Step 1: 在完整序列上计算权重（过滤前）
    df_with_weights = weight_calc.calculate_weights(df)
    df_full_weights = df_with_weights[['date', 'symbol', 'sample_weight']].copy()
    
    # Step 2: Base Model 生成 side
    df_signals = self._generate_base_signals(df, base_model)  # 内部过滤 side!=0
    
    # Step 3: 将完整序列的权重合并回去
    df_signals = df_signals.merge(
        df_full_weights,
        on=['date', 'symbol'],
        how='left'
    )
    # === M2 修复结束 ===
    
    # ... 后续训练逻辑 ...
```

---

*文档版本: v1.0*  
*最后更新: 2026-03-01*
