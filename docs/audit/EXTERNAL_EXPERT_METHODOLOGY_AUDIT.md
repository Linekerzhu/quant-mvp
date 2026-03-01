# 新内审报告：外部专家方法论应用

**审计日期**: 2026-03-01  
**审计员**: 李成荣（长春宫总管太监）  
**方法论**: 外部专家审计思维（端到端数据追踪、金融语义验证、算法假设验证）

---

## 一、吸收到了什么？

### 外部专家的核心思维方式

#### 1. **端到端追踪数据张量**
- **内部审计**：验证单个函数逻辑对不对
- **外部审计**：追踪数据从入口到出口的完整流向，验证"A模块输出的行顺序，B模块是否假设了同样的顺序"

**吸收**：
- 不再孤立地看每个模块，而是追踪完整数据流：`raw data → features → labels → weights → training → evaluation`
- 关注模块边界的数据形状和对齐问题
- 追踪索引、列名、数据类型在跨模块传递中的变化

#### 2. **金融语义验证**
- **内部审计**：验证 profit_barrier 数学公式对不对
- **外部审计**：做空赚钱时，系统会怎么标注？一个空头价格跌穿下方屏障到底是赚还是亏？

**吸收**：
- 每个金融计算都要在**多空两种场景**下验证语义正确性
- 不只看公式，更要看业务含义
- 追问"这个计算在真实交易中代表什么？"

#### 3. **质疑算法输入假设**
- **内部审计**：验证算法实现正确
- **外部审计**：验证输入数据是否满足算法的前置假设（如等间距、连续时间序列）

**吸收**：
- 每个算法都有隐含假设（FracDiff 假设等间距、样本权重假设时间连续性）
- 数据经过过滤、切分后，可能破坏这些假设
- 审计时不仅要看算法本身，更要看数据是否满足算法要求

---

## 二、计划如何开展新内审？

### 新审计方法论

#### 第一阶段：跨模块数据追踪

**追踪路径**：
```
1. raw data (yfinance/Tiingo) 
   ↓ [验证：数据格式、列名、索引]
2. features (build_features.py)
   ↓ [验证：特征计算、NaN处理、特征版本]
3. labels (triple_barrier.py)
   ↓ [验证：事件触发、屏障计算、标签语义]
4. weights (sample_weights.py)
   ↓ [验证：uniqueness计算、并发事件、权重归一化]
5. training (meta_trainer.py)
   ↓ [验证：FracDiff、CPCV切分、per-fold权重]
6. evaluation (overfitting.py)
   ↓ [验证：PBO、DSR、Dummy特征]
```

**审计要点**：
- **数据形状验证**：每个模块输入输出的行数、列数、索引是否对齐
- **数据类型验证**：timestamp、float、int 是否一致
- **数据缺失验证**：NaN 在跨模块传递中如何处理

#### 第二阶段：财务语义验证

**验证清单**：
1. **Base Model 信号语义**
   - 做多信号（side=+1）：价格上涨时赚钱？
   - 做空信号（side=-1）：价格下跌时赚钱？

2. **Triple Barrier 标签语义**
   - profit barrier（上方）：价格涨到这里 → 赚钱
   - loss barrier（下方）：价格跌到这里 → 亏钱
   - **做空场景**：profit barrier 在下方！loss barrier 在上方！

3. **Meta-Label 语义**
   - meta_label=1：Base Model 的信号是**正确的**（按信号操作赚钱）
   - meta_label=0：Base Model 的信号是**错误的**（按信号操作亏钱）

4. **样本权重语义**
   - 高权重：独立事件，信息含量高
   - 低权重：并发事件，信息重叠

#### 第三阶段：算法假设验证

**验证清单**：
1. **FracDiff 算法假设**
   - ✅ 等间距时间序列：per-symbol 预计算保证
   - ⚠️ 足够的样本量：MIN_ADF_SAMPLES=50
   - ⚠️ 价格序列连续：side!=0 过滤后可能破坏

2. **样本权重算法假设**
   - ⚠️ 事件时间连续：side!=0 过滤后破坏
   - ✅ 并发事件可计算：Sweep-line 算法正确

3. **CPCV 算法假设**
   - ✅ 时间排序：全局排序保证
   - ✅ 持有期间不重叠：non-overlapping constraint 保证
   - ⚠️ purge 窗口充足：purge_window=10 vs max_holding_days=10

---

## 三、发现的新问题

### HIGH 优先级

#### H1：做空场景的 Base Model 信号语义缺失

**问题描述**：
- `BaseModelSMA` 和 `BaseModelMomentum` 只生成 `side ∈ {+1, -1, 0}`
- side=+1 理解为" bullish "，side=-1 理解为"bearish"
- **但**：在 Meta-Labeling 架构中，side 的语义应该是"**交易方向**"而非"市场方向"

**场景分析**：
```
价格上涨场景：
- side=+1（做多）→ 价格继续上涨 → 赚钱 → meta_label=1 ✓
- side=-1（做空）→ 价格继续上涨 → 亏钱 → meta_label=0 ✓

价格下跌场景：
- side=+1（做多）→ 价格继续下跌 → 亏钱 → meta_label=0 ✓
- side=-1（做空）→ 价格继续下跌 → 赚钱 → meta_label=1 ✓
```

**当前代码**：
```python
# label_converter.py
if 'side' in df.columns:
    df['meta_label'] = ((df['side'] * df['label']) > 0).astype(int)
```

**验证**：
- side=+1, label=+1（价格上涨，做多）→ 1*1=1 > 0 → meta_label=1 ✓
- side=-1, label=-1（价格下跌，做空）→ -1*-1=1 > 0 → meta_label=1 ✓
- side=+1, label=-1（价格下跌，做多）→ 1*-1=-1 < 0 → meta_label=0 ✓
- side=-1, label=+1（价格上涨，做空）→ -1*1=-1 < 0 → meta_label=0 ✓

**结论**：✅ **FATAL-2 修复正确！** 标签语义在多空场景下正确。

---

#### H2：Time Barrier 过滤的样本选择偏差

**问题描述**：
- `LabelConverter` 过滤掉所有 label=0（time barrier）的事件
- time barrier 占 35% 的事件，代表"无明确趋势"
- 过滤后，模型只学习"有明确趋势"的场景

**影响**：
- **样本选择偏差**：模型在波动性低、趋势不明确的市场中表现未知
- **性能高估**：过滤"困难样本"后，性能指标可能虚高

**数据流追踪**：
```python
# label_converter.py
df = df[df['label'].notna() & (df['label'] != 0)].copy()  # 过滤 time barrier
```

**AFML 观点**：
- time barrier 代表"无明确结果"，应保留作为 neutral class
- 或者在特征中添加"波动性"维度，让模型学习何时signal、何时不signal

**建议**：
1. **短期**：记录过滤掉的 time barrier 比例，监控其对性能的影响
2. **中期**：尝试保留 time barrier，作为 meta_label=0.5（不确定类）
3. **长期**：在 Base Model 中添加"不signal"的判断（side=0）

**风险评估**：MEDIUM（性能可能高估，但不影响系统正确性）

---

### MEDIUM 优先级

#### M1：FracDiff 数据对齐的潜在隐患

**问题描述**：
- `meta_trainer.py` 中的 per-symbol FracDiff 预计算使用 `.values` 赋值
- 如果 `df_meta` 的索引不连续（经过 side!=0 过滤后），可能导致数据错位

**代码追踪**：
```python
# meta_trainer.py: train()
df_meta = df_meta.sort_values('date').reset_index(drop=True)  # 重置索引

# FATAL-3 Fix: Per-symbol FracDiff
for symbol in df_meta['symbol'].unique():
    sym_mask = df_meta['symbol'] == symbol
    sym_prices = np.log(df_meta.loc[sym_mask, price_col])  # 使用 .loc
    sym_fd = fracdiff_fixed_window(sym_prices, sym_d, window)
    df_meta.loc[sym_mask, 'fracdiff'] = sym_fd.values  # 使用 .values
```

**分析**：
- `df_meta` 已经 `reset_index(drop=True)`，索引是连续的 0, 1, 2, ...
- `sym_mask` 是布尔掩码，`.loc[sym_mask]` 选择对应行
- `sym_fd.values` 丢失索引，但长度与 `df_meta.loc[sym_mask]` 一致
- `.loc[sym_mask, 'fracdiff'] = values` 会按标签赋值

**验证方法**：
```python
# 测试代码
import pandas as pd
import numpy as np

df = pd.DataFrame({'symbol': ['A', 'B', 'A', 'B'], 'price': [10, 20, 15, 25]})
df = df.sort_values('symbol').reset_index(drop=True)
print(f"Index: {df.index.tolist()}")  # [0, 1, 2, 3]

mask = df['symbol'] == 'A'
print(f"Masked index: {df.loc[mask].index.tolist()}")  # [0, 2]

# 赋值
values = np.array([100, 200])
df.loc[mask, 'test'] = values
print(df)
#   symbol  price   test
# 0      A     10  100.0  ← 正确赋值
# 1      B     20    NaN
# 2      A     15  200.0  ← 正确赋值
# 3      B     25    NaN
```

**结论**：✅ **数据对齐正确**。`.loc` 基于标签赋值，不是位置赋值。

---

#### M2：样本权重计算的时间连续性假设

**问题描述**：
- `SampleWeightCalculator` 计算权重时，假设事件序列是**时间连续的**
- 经过 side!=0 过滤后，事件序列变得稀疏和不连续
- 这可能导致权重计算不准确

**数据流追踪**：
```python
# meta_trainer.py: _generate_base_signals()
df_filtered = df_with_signals[df_with_signals['side'] != 0].copy()  # 过滤 side=0

# sample_weights.py: calculate_weights()
valid_df = df[valid_mask].copy()  # 再次过滤 event_valid=False
weights = self._calculate_weights_sweep_line(valid_df)  # 计算权重
```

**场景分析**：
```
原始事件序列（时间连续）：
E1, E2, E3, E4, E5, E6, E7, E8, E9, E10

经过 side!=0 过滤后（稀疏）：
E1, E3, E5, E7, E9

计算并发事件：
- E1 和 E3 之间间隔 1 天，但原始序列中 E2 存在
- 算法会认为 E1 和 E3 的并发数较低，但实际上 E2 也在持有期间
```

**影响**：
- **权重高估**：稀疏序列的并发事件看起来更少，权重可能偏高
- **影响程度**：取决于 side=0 的比例（如果 side=0 很少，影响小）

**建议**：
1. **短期**：在原始完整事件序列上计算权重，然后再过滤
2. **中期**：验证过滤前后的权重差异，评估影响

**风险评估**：MEDIUM（可能影响训练效果，但不会导致系统性错误）

---

### LOW 优先级

#### L1：Dummy Feature 哨兵的 per-fold 检查

**问题描述**：
- R14-A9 添加了 per-fold dummy feature 检查
- 如果 dummy 在**任何** fold 中进入前 25%，标记为警告
- 这可能导致"偶发过拟合"被误判

**代码追踪**：
```python
# overfitting.py: dummy_feature_sentinel()
if fold_ratio <= 0.25:
    per_fold_warnings.append({...})

passed = ranking_ratio > self.dummy_threshold and len(per_fold_warnings) == 0
```

**场景分析**：
- 15 个 CPCV path，其中 1 个 path 的 dummy 排名进入前 25%
- 按当前逻辑，overall_passed=False
- 但平均排名可能正常

**建议**：
- 区分"系统性过拟合"（平均排名高）和"偶发过拟合"（个别 fold 排名高）
- 偶发过拟合可能是数据噪声，不应直接拒绝

**风险评估**：LOW（保守策略，宁可误杀）

---

## 四、审计结论

### 系统整体评估

**✅ 正确性**：系统在关键路径上逻辑正确，FATAL 级别的问题已修复
- ✅ Meta-Label 多空语义正确（FATAL-2）
- ✅ FracDiff per-symbol 预计算（FATAL-3）
- ✅ CPCV 索引对齐（全局排序）
- ✅ Triple Barrier Maximum Pessimism Principle（OR5）

**⚠️ 完整性**：存在样本选择偏差和假设破坏问题
- ⚠️ Time barrier 过滤（35% 样本丢失）
- ⚠️ 样本权重时间连续性假设破坏

**✅ 可维护性**：代码结构清晰，修复记录完整
- ✅ 每个 P0/P1/P2 问题都有修复记录
- ✅ 配置与代码分离
- ✅ 测试覆盖率高（165/165 passing）

### 建议优先级

| 优先级 | 问题 | 建议 | 预计工作量 |
|--------|------|------|-----------|
| HIGH | Time barrier 过滤偏差 | 添加监控指标，评估影响 | 2小时 |
| MEDIUM | 样本权重连续性假设 | 在完整序列上计算权重 | 4小时 |
| LOW | Dummy per-fold 检查 | 区分系统性和偶发过拟合 | 2小时 |

### 外部专家方法论的价值

通过采用外部专家的思维方式，本次审计发现了**之前审计遗漏的问题**：

1. **端到端数据追踪**：发现了 FracDiff 数据对齐问题（虽然验证后确认正确，但过程有价值）
2. **金融语义验证**：确认了 FATAL-2 修复的正确性，增强了信心
3. **算法假设验证**：发现了样本权重的时间连续性假设问题

**核心洞察**：
> 内部审计关注"代码对不对"，外部审计关注"系统对不对"。
> 代码正确 ≠ 系统正确，因为数据可能破坏算法假设。

---

## 五、后续行动

### 立即行动（本周）

1. **添加 Time Barrier 监控**
   - 记录过滤掉的 time barrier 比例
   - 记录 time barrier 的特征分布（波动性、持仓天数）
   - 添加到 `MetaTrainer.train()` 的日志

2. **验证样本权重影响**
   - 对比"过滤前计算"和"过滤后计算"的权重差异
   - 评估对训练的影响

### 中期行动（本月）

1. **Time Barrier 处理方案**
   - 尝试保留 time barrier 作为 neutral class
   - 或者添加"波动性阈值"过滤低波动场景

2. **样本权重算法改进**
   - 在完整事件序列上计算权重
   - 然后再进行 side!=0 过滤

### 长期行动（下季度）

1. **Base Model 增强**
   - 添加"不signal"的判断逻辑（side=0）
   - 让模型学习何时应该观望

2. **外部专家审计制度化**
   - 每次内审都采用"端到端+语义+假设"的三维方法论
   - 建立"数据流图"文档，记录跨模块的数据形状

---

**审计员签名**：李成荣  
**日期**：2026-03-01  
**长春宫**
