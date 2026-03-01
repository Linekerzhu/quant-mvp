# H2/M2 问题修复方案设计

**设计者**: 张德功  
**日期**: 2026-03-01  
**项目**: quant-mvp  
**审计员**: 寇连材（内审发现）

---

## 一、问题概述

| 问题ID | 严重度 | 描述 | 影响 |
|--------|--------|------|------|
| H2 | HIGH | Time Barrier 过滤偏差 | 35% 样本被过滤，可能导致性能高估 |
| M2 | MEDIUM | 样本权重时间连续性假设破坏 | 过滤后序列稀疏，权重可能偏高 |

---

## 二、H2 [HIGH]: Time Barrier 过滤偏差

### 2.1 问题根因分析

**数据流追踪**:

```
原始数据 (100%)
    ↓
Triple Barrier 标注 → 产生 label ∈ {-1, 0, 1}
    ↓                    - label=-1: loss barrier (亏损)
                         - label=0: time barrier (无明确结果)
                         - label=+1: profit barrier (盈利)
    ↓
Base Model 生成 side ∈ {-1, 0, +1}
    ↓
过滤 side != 0 → 移除 "无信号" 样本
    ↓
LabelConverter.convert() 过滤 label != 0 → 移除 time barrier
    ↓
训练数据 (约 65%)
```

**根因**:
- `LabelConverter.convert()` 方法中的 `binary_filtered` 策略主动过滤 `label == 0` 的样本
- 代码位置：`src/models/label_converter.py:60`

```python
# label_converter.py
if self.strategy == 'binary_filtered':
    # Filter out label=0 (time barriers)
    df = df[df['label'].notna() & (df['label'] != 0)].copy()
```

**为什么过滤 time barrier**:
- Meta-Label 的定义是"信号是否盈利"，只有盈利(label=+1)和亏损(label=-1)是明确结果
- time barrier 代表"无明确结果"，不属于盈利也不属于亏损
- AFML Ch3.4 建议：time barrier 作为 neutral class，不应简单映射为 0 或 1

### 2.2 影响分析

| 影响类型 | 描述 | 严重度 |
|----------|------|--------|
| 样本选择偏差 | 模型只学习"有明确趋势"的场景，在震荡市中表现未知 | HIGH |
| 性能高估风险 | 过滤掉"困难样本"后，AUC/准确率可能虚高 | MEDIUM |
| 市场适应性 | 模型可能在低波动环境中过度自信 | MEDIUM |

**量化评估**（基于审计数据）:
- Time barrier 占比：约 35%
- time barrier 的 label_return 分布：|return| < 1% 占 38%
- 这意味着 35% 的"无明确趋势"样本被排除在训练之外

### 2.3 修复方案

#### 方案 A：监控先行（短期，2小时）

**目标**: 不改变现有逻辑，但增加监控指标，量化 time barrier 过滤的影响

**实施步骤**:

1. **添加 Time Barrier 统计指标**

```python
# src/models/label_converter.py

def convert(self, df: pd.DataFrame) -> pd.DataFrame:
    # 记录过滤前的统计
    n_total = len(df)
    n_time_barrier = (df['label'] == 0).sum()
    n_profit = (df['label'] == 1).sum()
    n_loss = (df['label'] == -1).sum()
    
    # 记录 time barrier 的特征分布
    time_barrier_df = df[df['label'] == 0]
    if len(time_barrier_df) > 0:
        time_barrier_stats = {
            'count': len(time_barrier_df),
            'ratio': len(time_barrier_df) / n_total,
            'mean_return': time_barrier_df['label_return'].mean(),
            'std_return': time_barrier_df['label_return'].std(),
            'mean_holding_days': time_barrier_df['label_holding_days'].mean(),
            'return_near_zero': (time_barrier_df['label_return'].abs() < 0.01).mean(),  # |r| < 1%
        }
        logger.info("time_barrier_stats", time_barrier_stats)
    
    # 过滤 time barrier
    df = df[df['label'].notna() & (df['label'] != 0)].copy()
    
    # 记录过滤后的统计
    logger.info("time_barrier_filtered", {
        "n_before": n_total,
        "n_after": len(df),
        "n_filtered": n_total - len(df),
        "filter_ratio": (n_total - len(df)) / n_total
    })
    
    # ... 后续 meta_label 转换逻辑
```

2. **添加监控配置**

```yaml
# config/monitoring.yaml
time_barrier_monitoring:
  enabled: true
  warn_threshold: 0.30  # 超过 30% 警告
  alert_threshold: 0.40  # 超过 40% 报警
```

**优点**:
- 不改变训练逻辑，风险最低
- 量化 time barrier 的影响
- 为后续优化提供数据支持

**缺点**:
- 不解决根本问题
- 仍存在性能高估风险

---

#### 方案 B：三分类 Meta-Label（中期，4-6小时）

**目标**: 保留 time barrier 作为第三类，让模型学习"何时不确定"

**Meta-Label 定义变更**:
```
原定义: meta_label ∈ {0, 1}
- 1: 信号盈利
- 0: 信号亏损

新定义: meta_label ∈ {0, 0.5, 1}
- 1: 信号盈利（hit profit barrier）
- 0: 信号亏损（hit loss barrier）
- 0.5: 无明确结果（time barrier）
```

**实施步骤**:

1. **修改 LabelConverter**

```python
# src/models/label_converter.py

def convert(self, df: pd.DataFrame) -> pd.DataFrame:
    if self.strategy == 'ternary':
        # 保留 time barrier 作为 neutral class
        df = df.copy()
        
        if 'side' in df.columns:
            # meta_label = side × label（方向感知）
            # time barrier: side × 0 = 0
            df['meta_label'] = (df['side'] * df['label'])
            # 归一化到 [0, 0.5, 1]
            # side=+1, label=+1 → +1 → 1 (盈利)
            # side=-1, label=-1 → +1 → 1 (盈利)
            # side=+1, label=-1 → -1 → 0 (亏损)
            # side=-1, label=+1 → -1 → 0 (亏损)
            # side=*, label=0 → 0 → 0.5 (不确定)
            df['meta_label'] = df['meta_label'].map({
                1: 1,   # 盈利
                -1: 0,  # 亏损
                0: 0.5  # 不确定
            })
        else:
            # 向后兼容
            df['meta_label'] = df['label'].map({
                1: 1,
                -1: 0,
                0: 0.5
            })
        
        logger.info("ternary_meta_labels", {
            "profit": (df['meta_label'] == 1).sum(),
            "loss": (df['meta_label'] == 0).sum(),
            "neutral": (df['meta_label'] == 0.5).sum()
        })
    elif self.strategy == 'binary_filtered':
        # 原有逻辑...
```

2. **修改 LightGBM 训练配置**

```yaml
# config/training.yaml
lightgbm:
  # 三分类模式
  objective: multiclass
  num_class: 3
  
  # 或者使用回归模式（预测 meta_label ∈ [0, 1]）
  # objective: regression
  # metric: mse
```

**优点**:
- 保留完整样本，避免选择偏差
- 模型学习"何时应该观望"
- 更符合 AFML 的理念

**缺点**:
- 需要修改训练流程
- 三分类问题更复杂
- 可能降低分类性能

---

#### 方案 C：Base Model 增强（长期，1-2周）

**目标**: 让 Base Model 自己决定"何时不signal"，而不是在 Meta Model 层面处理

**思路**:
- Base Model 原本生成 `side ∈ {-1, 0, +1}`
- 0 代表"无信号"，即"不确定"
- 增强 Base Model 的 side=0 判断逻辑

**示例**:

```python
# src/signals/base_models.py

class BaseModelSMA:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... 原有逻辑 ...
        
        # 新增：波动性过滤
        df['volatility'] = df['adj_close'].rolling(20).std()
        vol_threshold = df['volatility'].quantile(0.25)  # 低 25% 波动性
        
        # 低波动环境下不产生信号
        df.loc[df['volatility'] < vol_threshold, 'side'] = 0
        
        # 新增：趋势强度过滤
        df['trend_strength'] = (df['sma_fast'] - df['sma_slow']).abs() / df['sma_slow']
        df.loc[df['trend_strength'] < 0.02, 'side'] = 0  # 趋势不明显
        
        return df
```

**优点**:
- 从源头解决问题
- 符合 AFML "Primary Model + Meta Model" 架构
- Meta Model 专注学习"信号质量"，不处理"何时signal"

**缺点**:
- 需要大量回测验证
- 改动范围大
- 周期长

---

### 2.4 推荐方案

**分阶段实施**:

| 阶段 | 方案 | 工时 | 优先级 | 风险 |
|------|------|------|--------|------|
| 立即 | A（监控） | 2h | HIGH | 低 |
| 本周 | B（三分类）| 4-6h | MEDIUM | 中 |
| 下月 | C（Base Model 增强）| 1-2周 | LOW | 高 |

**理由**:
1. 先量化问题（方案 A），收集数据
2. 再尝试保守修复（方案 B），验证效果
3. 最后考虑架构级优化（方案 C）

---

## 三、M2 [MEDIUM]: 样本权重时间连续性假设破坏

### 3.1 问题根因分析

**数据流追踪**:

```
原始事件序列（时间连续）
E1, E2, E3, E4, E5, E6, E7, E8, E9, E10
    ↓
Base Model 生成 side
    ↓
过滤 side != 0 → 稀疏序列
E1, E3, E5, E7, E9
    ↓
SampleWeightCalculator.calculate_weights()
    ↓
 Sweep-line 算法计算并发事件数
    ↓
问题：E2, E4, E6, E8 被过滤掉，但它们实际存在！
     算法误判 E1 和 E3 的并发数偏低
```

**根因**:
- `SampleWeightCalculator._calculate_weights_sweep_line()` 在过滤后的数据上计算权重
- 算法假设事件序列是完整的，但实际上已经过 side!=0 过滤
- 代码位置：`src/labels/sample_weights.py:95-130`

**算法假设**:
```python
# sweep-line 算法计算的是"当前数据集中"的并发事件数
# 而不是"原始完整数据集中"的并发事件数
daily_concurrent = np.cumsum(diff)  # 基于过滤后的数据
```

### 3.2 影响分析

**场景示例**:

```
原始数据（某 symbol 的连续10天）:
Day 1:  E1 触发 (side=+1, entry=Day2, exit=Day5)
Day 2:  E2 触发 (side=0, entry=Day3, exit=Day6)  ← 被过滤
Day 3:  E3 触发 (side=-1, entry=Day4, exit=Day7)
Day 4:  E4 触发 (side=0, entry=Day5, exit=Day8)  ← 被过滤
Day 5:  E5 触发 (side=+1, entry=Day6, exit=Day9)

过滤后（side != 0）:
E1 (Day1-5), E3 (Day3-7), E5 (Day5-9)

算法计算的并发数:
- Day 2: 只有 E1 活跃 → 并发=1
- Day 4: E1 + E3 活跃 → 并发=2
- Day 6: E3 + E5 活跃 → 并发=2

实际并发数（包含被过滤的 E2, E4）:
- Day 2: E1 + E2 活跃 → 并发=2
- Day 4: E1 + E2 + E3 + E4 活跃 → 并发=4
- Day 6: E2 + E3 + E4 + E5 活跃 → 并发=4
```

**影响**:
- 并发事件数被低估 → 权重被高估
- 权重高估 → 训练时这些样本被"过度重视"
- 可能导致过拟合

**量化评估**:
- 如果 side=0 比例为 20%，并发数可能被低估 20-40%
- 权重可能高估 10-30%（因为 weight = 1/(1+concurrent)）

### 3.3 修复方案

#### 方案 A：完整序列预计算（短期，4小时）

**目标**: 在完整事件序列上计算权重，然后再过滤

**实施步骤**:

1. **修改 MetaTrainer.train()**

```python
# src/models/meta_trainer.py

def train(self, df: pd.DataFrame, base_model, features: List[str], price_col: str = 'adj_close') -> Dict:
    # ... 前置逻辑 ...
    
    # ============================================================
    # M2 FIX: 在完整序列上计算样本权重
    # ============================================================
    # 1. Triple Barrier 先标注（不包含 side 信息）
    # 2. 计算样本权重（基于完整事件序列）
    # 3. Base Model 生成 side
    # 4. 过滤 side != 0（此时权重已计算完毕）
    # ============================================================
    
    # Step 1: Triple Barrier 已完成（来自 Phase B）
    # df 已包含 event_valid, label_holding_days, label_exit_date
    
    # Step 2: 在完整序列上计算样本权重
    from src.labels.sample_weights import SampleWeightCalculator
    weight_calc = SampleWeightCalculator()
    df = weight_calc.calculate_weights(df)  # 全量计算
    
    # 保存完整序列的权重
    df_full_weights = df[['date', 'symbol', 'sample_weight']].copy()
    df_full_weights.rename(columns={'sample_weight': 'full_sample_weight'}, inplace=True)
    
    # Step 3: Base Model 生成信号
    df_signals = self._generate_base_signals(df, base_model)  # 过滤 side != 0
    
    # Step 4: 将完整序列的权重合并回去
    # side!=0 过滤后的样本，使用完整序列计算的权重
    df_signals = df_signals.merge(
        df_full_weights,
        on=['date', 'symbol'],
        how='left'
    )
    df_signals['sample_weight'] = df_signals['full_sample_weight']
    df_signals.drop(columns=['full_sample_weight'], inplace=True)
    
    # ... 后续训练逻辑 ...
```

2. **添加配置开关**

```yaml
# config/training.yaml
sample_weights:
  method: uniqueness
  pre_filter_calculation: true  # M2 FIX: 在过滤前计算权重
```

**优点**:
- 保留算法假设的完整性
- 权重计算不受过滤影响
- 改动量适中

**缺点**:
- 增加一次全量权重计算
- 需要处理索引对齐问题

---

#### 方案 B：并发事件校正因子（中期，2-3小时）

**目标**: 不改变计算顺序，但对权重进行统计校正

**思路**:
- 假设被过滤的 side=0 事件的并发分布与保留事件相似
- 使用校正因子调整权重

```python
# 估算过滤前的并发数
side_0_ratio = (df['side'] == 0).mean()
correction_factor = 1 / (1 - side_0_ratio)  # 上调并发数

# 调整权重
adjusted_weight = original_weight / correction_factor
```

**优点**:
- 改动量最小
- 不需要重新计算权重

**缺点**:
- 假设可能不成立
- 只能缓解问题，不能根治

---

#### 方案 C：side=0 事件纳入权重计算（长期，6-8小时）

**目标**: 在权重计算时，考虑 side=0 事件对并发的影响

**思路**:
- 保留 side=0 事件的元数据（entry_date, exit_date）
- 权重计算时，side=0 事件参与并发计算
- 但 side=0 事件不参与训练

```python
# src/labels/sample_weights.py

def calculate_weights_with_filtered_events(
    self,
    df: pd.DataFrame,
    filter_col: str = 'side',
    filter_value: int = 0
) -> pd.DataFrame:
    """
    计算样本权重时，考虑被过滤事件对并发的影响。
    
    Args:
        df: 包含所有事件的 DataFrame（包括 side=0）
        filter_col: 过滤列名
        filter_value: 要过滤掉的值
    """
    df = df.copy()
    df['sample_weight'] = 1.0
    
    # 使用所有事件（包括 side=0）计算并发数
    all_events = df[df['event_valid'] == True].copy()
    weights_all = self._calculate_weights_sweep_line(all_events)
    
    # 将权重映射回原始 df
    df.loc[weights_all.index, 'sample_weight'] = weights_all
    
    # 过滤 side=0（这些事件有权重，但不用于训练）
    df_train = df[df[filter_col] != filter_value].copy()
    
    return df_train
```

**优点**:
- 精确考虑被过滤事件的影响
- 不需要两次计算

**缺点**:
- 接口变更较大
- 需要修改调用方

---

### 3.4 推荐方案

**优先选择方案 A（完整序列预计算）**

**理由**:
1. 逻辑清晰，风险可控
2. 符合 AFML 的样本权重定义
3. 不引入额外假设
4. 工时适中（4小时）

**实施步骤**:
1. 在 `MetaTrainer.train()` 中，先计算全量权重
2. Base Model 生成 side 后，保留全量权重
3. 添加配置开关，支持回滚
4. 编写测试验证

---

## 四、改动量与风险评估

### 4.1 改动量评估

| 问题 | 推荐方案 | 文件变更 | 代码行数 | 工时 |
|------|----------|----------|----------|------|
| H2 | A（监控） | `label_converter.py` | +30 行 | 2h |
| M2 | A（预计算） | `meta_trainer.py` | +40 行 | 4h |
| **总计** | - | 2 文件 | +70 行 | **6h** |

### 4.2 风险评估

| 风险类型 | H2 方案 | M2 方案 | 缓解措施 |
|----------|---------|---------|----------|
| 功能回归 | 低 | 中 | 添加测试用例 |
| 性能影响 | 无 | 低 | 全量计算有缓存 |
| 数据对齐 | 无 | 中 | 使用 merge 而非直接赋值 |
| 配置兼容 | 低 | 低 | 添加配置开关 |

### 4.3 测试计划

**H2 测试**:
1. 验证 time barrier 统计指标正确记录
2. 验证过滤逻辑不变
3. 验证日志输出格式

**M2 测试**:
1. 构造测试数据：已知并发数的事件序列
2. 验证预计算权重正确
3. 验证过滤后权重保留
4. 对比修复前后的训练效果

---

## 五、实施计划

### 5.1 里程碑

| 阶段 | 内容 | 预计完成 |
|------|------|----------|
| Phase 1 | H2 监控方案实施 | 今天（2小时） |
| Phase 2 | M2 预计算方案实施 | 明天（4小时） |
| Phase 3 | 集成测试 | 后天（2小时） |
| Phase 4 | 回归测试 | 后天（1小时） |

### 5.2 验收标准

**H2 验收**:
- [ ] time_barrier_stats 日志正常输出
- [ ] 包含 5 个关键指标：count, ratio, mean_return, std_return, mean_holding_days
- [ ] 过滤比例 < 40% 无警告

**M2 验收**:
- [ ] 全量权重计算正常
- [ ] 过滤后权重正确保留
- [ ] 训练效果对比（修复前后 AUC 差异 < 0.05）
- [ ] 165/165 测试通过

---

## 六、总结

### 6.1 关键洞察

1. **H2 问题本质**: 样本选择偏差，而非算法错误
2. **M2 问题本质**: 算法假设被数据流破坏
3. **共同点**: 都是"数据处理顺序"导致的问题

### 6.2 核心原则

1. **监控先行**: 先量化问题，再决定是否修复
2. **保持假设完整**: 算法计算应在满足假设的数据上进行
3. **配置化**: 所有修复都应可通过配置开关控制

### 6.3 后续优化方向

1. **H2 中期**: 尝试三分类 Meta-Label
2. **M2 长期**: 优化 Base Model，减少 side=0 的比例
3. **架构优化**: 建立"数据流图"文档，明确每一步的假设

---

**设计者签名**: 张德功  
**日期**: 2026-03-01  
**长春宫**
