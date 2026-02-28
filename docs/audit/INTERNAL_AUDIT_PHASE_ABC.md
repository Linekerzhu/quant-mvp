# Phase A-C 金融数学内审报告

**审计人**: 寇连材（八品监斋）
**审计日期**: 2026-02-28
**审计范围**: Phase A-C 全面内审，重点关注刚修复的外部审计问题
**测试状态**: ✅ 165/165 测试通过

---

## 执行摘要

本次内审对 Phase A-C 进行了全面的金融数学层面审查，特别关注外部审计发现的 5 个问题（C-01, C-02, C-03, H-02, H-05）。

**关键发现**：
- ✅ **3个修复已确认有效**：C-03, H-02, H-05
- ❌ **1个修复存在严重BUG**：C-02 (FracDiff特征未被使用)
- ⚠️ **1个修复逻辑存疑**：C-01 (PBO计算方法与AFML定义不完全一致)

**风险等级**: 🔴 **CRITICAL** - C-02 修复引入新BUG，导致 FracDiff 特征完全无效

---

## 一、外部审计问题修复验证

### 1.1 C-01: PBO 计算改用 IS vs OOS 排名比较

**文件**: `src/models/overfitting.py` 第 51-108 行

**修复代码**:
```python
def calculate_pbo(self, path_results: List[Dict]) -> float:
    """计算 PBO（Probability of Backtest Overfitting）"""
    # 提取 IS 和 OOS AUC
    is_aucs = [r.get('is_auc', r.get('auc', 0.5)) for r in path_results]
    oos_aucs = [r.get('oos_auc', r.get('auc', 0.5)) for r in path_results]
    
    # IS vs OOS 排名比较
    is_ranking = np.argsort(np.argsort(is_aucs)[::-1])
    oos_ranking = np.argsort(np.argsort(oos_aucs)[::-1])
    
    rank_diff = is_ranking - oos_ranking
    pbo = np.mean(rank_diff > 0)
    
    return float(pbo)
```

**审计结论**: ⚠️ **逻辑存疑，但实际运行正常**

**分析**:
1. **定义偏差**: 
   - AFML 定义的 PBO：**最优 IS 模型在 OOS 上排名靠后的概率**
   - 当前实现：**平均而言，IS 排名比 OOS 差的概率**
   - 两者概念不完全一致

2. **实际测试结果**:
   ```
   高过拟合场景（IS好OOS差）: PBO = 0.33
   低过拟合场景（IS≈OOS）: PBO = 0.00
   ```
   ✅ 能够区分高/低过拟合场景

3. **金融数学评估**:
   - 虽然不是严格的 AFML PBO 定义，但提供了合理的过拟合风险评估
   - 使用排名比较避免了 AUC 绝对值的影响
   - 当 IS/OOS 未分离时，fallback 到方差方法（合理）

**建议**:
- 短期：当前实现可接受，无需修改
- 长期：考虑实现真正的 AFML PBO（关注最优 IS 模型的 OOS 表现）

**风险等级**: 🟡 **MEDIUM** - 逻辑有偏差但不影响实际使用

---

### 1.2 C-02: MetaTrainer 集成 FracDiff

**文件**: `src/models/meta_trainer.py` 第 238-343 行

**❌ 严重发现**: **修复引入新BUG，FracDiff 特征完全无效**

#### 问题 1: 特征使用错误（CRITICAL）

**Bug 位置**: 第 284-286 行
```python
# ❌ 错误：使用原始特征列表
X_train = train_df[features]
X_test = test_df[features]
```

**问题分析**:
1. 第 259-262 行：正确计算了 `fracdiff` 特征
2. 第 263 行：正确将 `fracdiff` 加入 `current_features`
3. **但第 284-286 行训练时使用的是 `features`（不含 fracdiff）**
4. **结果**：FracDiff 特征被计算但从未被使用

**影响**:
- ❌ C-02 修复完全无效
- ❌ LightGBM 训练未使用 FracDiff 特征
- ❌ 输出的 `optimal_d` 值毫无意义（特征未被使用）

**修复方案**:
```python
# ✅ 修复：使用包含 fracdiff 的特征列表
X_train = train_df[current_features]
X_test = test_df[current_features]
```

#### 问题 2: 特征重要性丢失（CRITICAL）

**Bug 位置**: 第 340 行
```python
# ❌ 错误：使用原始特征列表
importance = dict(zip(features, model.feature_importance(...)))
```

**问题分析**:
- `model.feature_importance()` 返回 N+1 个值（包含 fracdiff）
- `zip(features, ...)` 只配对前 N 个值
- **fracdiff 的重要性丢失**

**修复方案**:
```python
# ✅ 修复：使用正确的特征列表
importance = dict(zip(current_features, model.feature_importance(...)))
```

**验证脚本**:
```python
# test_feature_bug.py 验证结果
Original features: ['f1', 'f2', 'f3']
Features used for training: ['f1', 'f2', 'f3', 'fracdiff']
Model feature importance (4 values): [10, 20, 15, 25]

❌ Buggy result (using 'features'):
   {'f1': 10, 'f2': 20, 'f3': 15}  # fracdiff 丢失！
```

**风险等级**: 🔴 **CRITICAL** - FracDiff 集成完全无效

---

### 1.3 C-03: Sample Weights 传入 LightGBM

**文件**: `src/models/meta_trainer.py` 第 212-236, 291-296 行

**修复代码**:
```python
# 计算样本权重（第 212-236 行）
def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
    """计算样本权重（基于 uniqueness）"""
    if 'uniqueness' in df.columns:
        weights = df['uniqueness'].values.copy()
    else:
        weights = np.ones(len(df))
    
    # 应用 min/max 限制并归一化
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
    return weights

# 传入 LightGBM（第 291-296 行）
train_data = lgb.Dataset(
    X_train, 
    label=y_train,
    weight=train_weights  # ✅ 正确传入
)
```

**审计结论**: ✅ **修复正确，符合 AFML 要求**

**金融数学评估**:
1. ✅ 基于 uniqueness 计算权重（AFML Ch4 要求）
2. ✅ 应用 min/max 限制防止极端权重
3. ✅ 归一化保持均值=1（标准做法）
4. ✅ 正确传入 LightGBM Dataset

**代码质量**:
- 有 fallback 机制（uniqueness 列缺失时使用均匀权重）
- 有日志记录权重统计信息
- 符合配置文件中的参数设置

**风险等级**: 🟢 **LOW** - 修复正确，无问题

---

### 1.4 H-02: CPCV 改用 BDay（交易日）

**文件**: `src/models/purged_kfold.py` 第 17, 150-151 行

**修复代码**:
```python
# 导入 BDay（第 17 行）
from pandas.tseries.offsets import BDay

# 使用 BDay（第 150-151 行）
purge_start = test_min_date - BDay(self.purge_window)
purge_end = test_max_date + BDay(self.purge_window)
```

**审计结论**: ✅ **修复正确，符合金融实践**

**金融数学评估**:
1. ✅ 使用 BDay（交易日）而非日历日
2. ✅ 正确处理周末和节假日
3. ✅ purge 窗口计算更准确

**代码一致性**:
- split() 和 split_with_info() 方法都使用 BDay
- 与配置文件中的 purge_window 参数一致

**风险等级**: 🟢 **LOW** - 修复正确，无问题

---

### 1.5 H-05: assert 改用 ValueError

**文件**: `src/models/meta_trainer.py` 第 119-125 行

**修复代码**:
```python
# H-05 Fix: 使用显式检查，替代assert（可被-O绕过）
if max_depth > 3:
    raise ValueError(f"OR5 VIOLATION: max_depth={max_depth} > 3")
if num_leaves > 7:
    raise ValueError(f"OR5 VIOLATION: num_leaves={num_leaves} > 7")
if min_data_in_leaf < 100:
    raise ValueError(f"OR5 VIOLATION: min_data_in_leaf={min_data_in_leaf} < 100")
```

**审计结论**: ✅ **修复正确，符合最佳实践**

**工程评估**:
1. ✅ 使用显式 ValueError 替代 assert
2. ✅ 生产环境无法绕过（不受 -O 标志影响）
3. ✅ 错误消息清晰，包含违规参数值
4. ✅ OR5 参数验证逻辑完整

**风险等级**: 🟢 **LOW** - 修复正确，无问题

---

## 二、新发现的问题

### 2.1 DSR 计算注释不准确（LOW）

**文件**: `src/models/overfitting.py` 第 206 行

**问题**:
```python
def calculate_deflated_sharpe(self, path_results: List[Dict]) -> float:
    """
    计算 DSR 检验的 z-score（统计显著性检验）。
    
    注意：这不是真正的 Deflated Sharpe Ratio！
    真正的 DSR 需要用 norm.cdf() 转换，这里直接返回 z-score。
    """
```

**分析**:
- 注释已经说明这不是真正的 DSR
- 实际计算的是 z-score，用于 `check_dsr_gate` 判定
- 判定标准使用 z-score 阈值（1.645, 1.282）是正确的

**建议**:
- 考虑重命名方法为 `calculate_sharpe_zscore` 以避免混淆
- 当前实现可接受，注释已充分说明

**风险等级**: 🟢 **LOW** - 文档问题，不影响功能

---

## 三、测试验证

### 3.1 单元测试状态

```bash
$ python3 -m pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.0
rootdir: /Users/zjz/.openclaw/workspace/changchungong/quant-mvp

collected 165 items

tests/test_base_models.py .......................... [ 16%]
tests/test_corporate_actions.py ........ [ 21%]
tests/test_cpcv.py .................... [ 34%]
tests/test_data.py ....... [ 38%]
tests/test_event_logger.py ..... [ 41%]
tests/test_feature_importance.py ..... [ 44%]
tests/test_features.py .......... [ 50%]
tests/test_fracdiff.py ................ [ 60%]
tests/test_integration.py ......... [ 65%]
tests/test_integrity.py ..... [ 68%]
tests/test_labels.py ....... [ 72%]
tests/test_no_leakage.py .......... [ 78%]
tests/test_overfit_sentinels.py ....... [ 82%]
tests/test_reproducibility.py ..... [ 85%]
tests/test_sample_weights.py ..... [ 88%]
tests/test_smoke_or5.py ...................... [100%]
tests/test_universe.py ..... [100%]

======================= 165 passed, 3 warnings in 18.68s =======================
```

✅ **所有 165 个单元测试通过**

### 3.2 测试覆盖度评估

**缺失的测试**:
- ❌ PBO 计算逻辑没有单元测试
- ❌ MetaTrainer 的 FracDiff 集成没有端到端测试
- ❌ 特征重要性计算没有测试

**建议**:
1. 添加 `tests/test_pbo.py` - 测试 PBO 计算的各种场景
2. 添加 `tests/test_meta_trainer_integration.py` - 测试端到端训练流程
3. 添加特征重要性正确性的断言

---

## 四、修复优先级

### 🔴 CRITICAL（立即修复）

#### BUG-1: FracDiff 特征未被使用

**文件**: `src/models/meta_trainer.py` 第 284-286 行

**修复**:
```python
# 当前（错误）
X_train = train_df[features]
X_test = test_df[features]

# 修复
X_train = train_df[current_features]
X_test = test_df[current_features]
```

**同时修复特征重要性**（第 340 行）:
```python
# 当前（错误）
importance = dict(zip(features, model.feature_importance(...)))

# 修复
importance = dict(zip(current_features, model.feature_importance(...)))
```

**预期影响**:
- FracDiff 特征将真正参与训练
- 模型性能可能变化（需要重新评估）
- `optimal_d` 参数将真正影响模型

---

### 🟡 MEDIUM（建议修复）

#### ISSUE-1: PBO 定义与 AFML 不完全一致

**建议**:
- 短期：当前实现可接受，能够区分过拟合程度
- 长期：实现真正的 AFML PBO（关注最优 IS 模型的 OOS 排名）

---

### 🟢 LOW（可选改进）

#### ISSUE-2: DSR 方法命名可能引起混淆

**建议**: 重命名为 `calculate_sharpe_zscore`

#### ISSUE-3: 缺少 PBO 和 MetaTrainer 集成测试

**建议**: 添加相应的单元测试和集成测试

---

## 五、整改建议

### 5.1 立即行动（24小时内）

1. **修复 C-02 BUG**（预计 1 小时）
   - 修改 `meta_trainer.py` 第 284-286 行
   - 修改 `meta_trainer.py` 第 340 行
   - 运行测试验证修复

2. **回归测试**（预计 2 小时）
   - 运行完整测试套件
   - 手动测试 MetaTrainer 端到端流程
   - 验证 FracDiff 特征确实被使用

### 5.2 短期改进（1周内）

1. **添加缺失测试**
   - 创建 `tests/test_pbo.py`
   - 创建 `tests/test_meta_trainer_integration.py`
   - 覆盖关键金融数学逻辑

2. **文档更新**
   - 更新 PBO 计算方法的说明
   - 记录 FracDiff 集成的正确用法
   - 更新 PHASE_C_IMPL_GUIDE.md

### 5.3 长期改进（可选）

1. **实现真正的 AFML PBO**
2. **添加交易日历支持**（H-03 问题，当前使用 USFederalHolidayCalendar）
3. **优化 FracDiff 计算性能**（缓存机制）

---

## 六、结论

### 6.1 总体评估

**修复质量**: ⚠️ **部分有效**
- ✅ C-03, H-02, H-05 修复正确
- ❌ C-02 修复引入严重BUG
- ⚠️ C-01 逻辑存疑但实际可用

**测试覆盖**: ✅ **165/165 通过**
- 单元测试覆盖良好
- 但缺少关键逻辑的专项测试（PBO, MetaTrainer集成）

**金融数学正确性**: ⚠️ **存在风险**
- PBO 计算方法与 AFML 定义有偏差
- FracDiff 集成完全无效（严重）
- 其他修复的金融数学逻辑正确

### 6.2 阻塞项

**🔴 C-02 FracDiff BUG 必须立即修复**，否则：
- Phase C 的核心特性（FracDiff 集成）完全无效
- 模型训练结果不可信
- 无法进入 Phase D

### 6.3 审批意见

**❌ 暂不批准进入 Phase D**

**理由**:
1. C-02 存在 CRITICAL 级别 BUG
2. FracDiff 特征完全未被使用
3. 修复后需要重新验证所有性能指标

**批准条件**:
1. ✅ 修复 C-02 BUG（X_train/X_test 使用 current_features）
2. ✅ 修复特征重要性计算
3. ✅ 重新运行测试（165/165 通过）
4. ✅ 验证 FracDiff 特征确实被使用（日志+特征重要性）

---

## 七、附录

### A. 验证脚本

#### A.1 PBO 逻辑验证
```bash
$ python3 test_pbo_logic.py
High overfit PBO: 0.333
Low overfit PBO: 0.0
✅ PBO correctly identifies higher overfitting risk
```

#### A.2 特征重要性 BUG 验证
```bash
$ python3 test_feature_bug.py
❌ Buggy result (using 'features'):
   {'f1': 10, 'f2': 20, 'f3': 15}
   Missing: fracdiff importance = 25
```

### B. 相关文件清单

**核心文件**:
- `src/models/overfitting.py` - PBO 和过拟合检测
- `src/models/meta_trainer.py` - Meta-Labeling 训练管道
- `src/models/purged_kfold.py` - CPCV 交叉验证
- `config/training.yaml` - 训练配置

**审计文档**:
- `docs/audit/EXTERNAL_AUDIT_FIX_PLAN.md` - 外部审计修复计划
- `docs/PHASE_C_IMPL_GUIDE.md` - Phase C 实现指南

---

**审计完成时间**: 2026-02-28 13:45  
**下次审计**: C-02 BUG 修复后重新审计

---

*寇连材谨呈*  
*八品监斋*
