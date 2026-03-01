# Quant MVP 修复记录汇总

> **维护说明**: 本文档记录所有内审和外部审核的发现与修复详情。
> 
> **最后更新**: 2026-03-01

---

## 📋 修复记录索引

| 轮次 | 日期 | Commit | 状态 | 严重程度 |
|------|------|--------|------|----------|
| OR1-OR4 | 2026-02-24 至 02-25 | 多 commit | ✅ 已完成 | P0-P2 |
| OR5 | 2026-02-26 | `68f4e78` | ✅ 已完成 | 架构级 |
| OR6-OR8 | 2026-02-26 至 02-27 | 多 commit | ✅ 已完成 | P0-P2 |
| OR9-OR13 | 2026-02-27 至 02-28 | `59d5dcd` | ✅ 已完成 | P0-P2 |
| R14 | 2026-03-01 | `140c8d6` | ✅ 已完成 | HIGH-MEDIUM |
| R14-A3 | 2026-03-01 | `1928b85` | ✅ 已完成 | HIGH |
| R15 | 2026-03-01 | `2271bd7` | ✅ 已完成 | NORMAL |
| EXT-Q1/Q2/Q5 | 2026-03-01 | `8fd8db4` | ✅ 已完成 | P0-P1 |
| A5/A6/A8 | 2026-03-01 | `35dc287` | ✅ 已完成 | P1-P2 |
| Q5-REG | 2026-03-01 | `待补充` | ✅ 已完成 | REGRESSION |

---

## 🔴 OR1-OR4: Phase A/B 基础审计修复

### OR4 (2026-02-25)

**Commit**: `1cda9c5` - Fix OR4 issues

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR4-BUG-01 | P0 | 行顺序问题 | 修复数据对齐逻辑 |
| OR4-BUG-02 | P0 | pass_rate 默认 | 修正默认通过阈值 |
| OR4-EXT-01 | P2 | 日志清理 | 移除冗余日志输出 |

### OR3 (2026-02-25)

**Commit**: `3ecb83e` - Fix OR3 HIGH issues

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR3-HIGH-01 | HIGH | 数据验证前向泄漏 | 验证只用历史数据 |
| OR3-HIGH-02 | HIGH | 公司行为处理 | 改进拆股/分红检测 |

### OR2 (2026-02-24)

**Commits**: `3a837c1`, `e5a0ca5`, `cc4685d`

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR2-HIGH-01 | HIGH | 算法边界条件 | 添加边界检查 |
| OR2-HIGH-02 | HIGH | 数值稳定性 | 改进浮点运算处理 |
| OR2-MED-01 | MEDIUM | 代码冗余 | 重构重复逻辑 |
| OR2-MED-02 | MEDIUM | 异常处理 | 完善错误捕获 |

### OR1 (2026-02-24)

**Commit**: `9ae7f86` - Fix: Deep audit fixes

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR1-CRIT-01 | CRITICAL | 计划一致性 | 对齐文档与代码 |
| OR1-HIGH-01 | HIGH | 安全性问题 | 加强输入验证 |

---

## 🔴 OR5: 架构级审计裁决 (2026-02-26)

**Commit**: `68f4e78` - hotfix(OR5): Maximum Pessimism Principle + LGB反Kaggle硬化 + PhaseC契约

### 审计裁决要点

1. **Meta-Labeling 强制架构**: Phase C 从"单模型直接预测"升级为 Base Model → Meta Model pipeline
2. **LightGBM 参数硬化** (反Kaggle):
   - max_depth: 3 (LOCKED)
   - num_leaves: 7 (LOCKED)
   - min_data_in_leaf: 200 (LOCKED)
3. **FracDiff 必需**: 分数阶差分 (d ≈ 0.4)，保留时序记忆同时实现平稳性
4. **手写 CPCV**: CombinatorialPurgedKFold (15 paths)，严禁 sklearn KFold
5. **Maximum Pessimism Principle**: 
   - Gap execution: 跳空穿越用实际开盘价结算
   - Collision detection: 同日双穿强制止损
   - 止损检查优先于止盈

### 整改记录 (Commits: `5c35141`, `b171b66`)

| 整改项 | 描述 | 状态 |
|--------|------|------|
| T1 | Burn-in 预警 - FracDiff 与 CPCV 断层衔接警告 | ✅ 已完成 |
| T2 | plan.md §6.5 公式示例 gap=70d 更正为 gap=50d | ✅ 已完成 |
| T3 | PBO 三档门控对齐 - Warning 档补充到 IMPL_GUIDE | ✅ 已完成 |
| T4 | features.yaml T6 整改 - 4个特征 requires_ohlc 从 true 改为 false | ✅ 已完成 |
| T5 | early_stopping_rounds 位置修正 | ✅ 已完成 |
| T6 | early_stopping 参数位置说明 | ✅ 已完成 |
| T7 | embargo/feature_lookback 20天缺口风险注释 | ✅ 已完成 |

---

## 🔴 OR6-OR8: Phase C 实施审计 (2026-02-26 至 02-27)

### OR6

**Commit**: `1662edc` - Fix OR5: Sync embargo total_days=60

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR6-SYNC-01 | P1 | embargo 总天数同步 | 统一 total_days=60 |

### OR7

**Commit**: `631e4bb` - Fix OR7 P0 bugs

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR7-P0-01 | P0 | loc→at 索引错误 | 改用 .at 访问 |
| OR7-P0-02 | P0 | C-01, C-02 修复 | 代码逻辑修正 |

### OR8

**Commit**: `95eb976` - Fix OR8 P0 issues

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR8-P0-01 | P0 | base_models.py NaN/Inf | 添加数值检查 |
| OR8-BUG-08 | P1 | fracdiff.py d=0 处理 | d=0 直接返回原序列 |

---

## 🔴 OR9-OR13: 内审修复汇总 (2026-02-27 至 02-28)

**Commit**: `59d5dcd` - fix: OR9-OR13 内审修复汇总

### P0 - 阻塞性修复

| 问题ID | 严重度 | 描述 | 修复方案 | 文件 |
|--------|--------|------|----------|------|
| OR9-BUG-01 | P0 | overfitting.py 重复循环 IndexError | 删除重复循环 | `src/models/overfitting.py` |
| OR9-BUG-02 | P0 | purged_kfold.py split() 改用逐段 embargo | 实现逐段 embargo | `src/models/purged_kfold.py` |

### P1 - 重要修复

| 问题ID | 严重度 | 描述 | 修复方案 | 文件 |
|--------|--------|------|----------|------|
| OR9-EXT-01 | P1 | build_features.py ATR min_periods | 1→window | `src/features/build_features.py` |
| OR8-BUG-08 | P1 | fracdiff.py d=0 处理 | 直接返回原序列 | `src/features/fracdiff.py` |
| OR10-NEW-01 | P1 | regime_detector.py ADX/ATR min_periods | 添加 min_periods | `src/features/regime_detector.py` |
| OR11-NEW-01 | P1 | regime_detector.py rv_20d min_periods | 添加 min_periods | `src/features/regime_detector.py` |
| OR12-NEW-01 | P1 | build_features.py 7处 rolling min_periods | 统一修复 | `src/features/build_features.py` |

### P2 - 清理优化

| 问题ID | 严重度 | 描述 | 修复方案 |
|--------|--------|------|----------|
| OR9-EXT-08 | P2 | purged_kfold.py docstring | 40→60 |
| OR10-NEW-02 | P2 | 测试文件 embargo_window | 统一为 60 |

---

## 🔴 R14: 内审修复 (2026-03-01)

**Commit**: `140c8d6` - fix: R14 内审修复 (A1, A2, A4, A9)

### A1 [HIGH]: PBO 改用 AFML 排名方法

**问题描述**:
- 原实现使用数值比较，不符合 AFML Ch7 §7.4.2 定义

**修复方案**:
- 实现 rank-based PBO 算法
- PBO = P(best IS path 的 OOS 排名 > 中位数)

**文件变更**: `src/models/overfitting.py`

### A2 [MEDIUM]: DSR 添加 skewness/kurtosis + 多重测试校正

**问题描述**:
- 原 DSR 实现缺少高阶矩校正
- 未考虑多重测试的影响

**修复方案**:
- 实现 R14-A2 公式校正
- 添加 E[max(SR)] 多重测试校正

**文件变更**: `src/models/overfitting.py`

### A4 [MEDIUM]: BaseModel .loc→.iloc 冷启动

**问题描述**:
- 两处 .loc 使用 label 索引，在冷启动时可能出错

**修复方案**:
- 改用 .iloc 进行位置索引

**文件变更**: `src/signals/base_models.py`

### A9 [LOW]: Dummy sentinel per-fold 检查

**问题描述**:
- 需要在每 fold 检查 dummy feature 的表现

**修复方案**:
- 添加 per-fold 警告检测

**文件变更**: `src/models/overfitting.py`

---

## 🔴 R14-A3: 样本权重 per-fold 重算 (2026-03-01)

**Commit**: `1928b85` - fix: R14-A3 样本权重 per-fold 重算

**问题描述**:
- 原实现在训练前一次性计算样本权重，可能受到测试集结构影响

**修复方案**:
- 添加 SampleWeightCalculator 实例
- 在 train() 中保存原始事件数据 (label_holding_days, label_exit_date)
- 在 _train_cpcv_fold 中对每个 fold 重算样本权重
- 避免训练权重受到测试集结构影响

**状态**: R14 内审所有 HIGH/MEDIUM 问题已修复

**文件变更**: `src/models/meta_trainer.py` (+80 行, -9 行)

---

## 🔴 R15: PBO 计算逻辑修正 (2026-03-01)

**Commit**: `2271bd7` - fix: R15-N1 PBO 计算逻辑修正

**问题描述**:
- PBO 实现检查所有路径，而非仅检查 IS 最优路径

**修复方案**:
- AFML 正确定义: PBO = P(IS最优路径在OOS表现差于中位数)
- 只检查 IS rank #1 的路径，而非所有路径

**文件变更**: `src/models/overfitting.py` (+5/-3 行)

---

## 🔴 外部审核修复 (2026-03-01)

**Commit**: `8fd8db4` - fix: 外部审核修复 (EXT-Q2, EXT-Q1, Q5)

### EXT-Q2 [P0]: FracDiff 全局预计算

**问题描述**:
- 原实现在每个 fold 内计算 FracDiff，导致数据损失

**修复方案**:
- 在 train() 中全局计算 FracDiff，避免每 fold 损失数据
- 删除 _train_cpcv_fold 中的 FracDiff 计算

**文件变更**: `src/features/fracdiff.py`, `src/models/meta_trainer.py`

### EXT-Q1 [P0]: Early Stopping 隔离 test set

**问题描述**:
- Early stopping 使用了 test set，造成信息泄漏

**修复方案**:
- 从训练集尾部切 20% 作为 validation set
- test set 只用于最终评估，不参与 early stopping

**文件变更**: `src/models/meta_trainer.py`

### Q5 [P1]: find_min_d 预检查样本量

**问题描述**:
- 二分搜索前未检查样本量，可能浪费计算

**修复方案**:
- 在二分搜索前检查样本量，避免浪费计算

**文件变更**: `src/features/fracdiff.py`

---

## 🔴 剩余技术债修复 (2026-03-01)

**Commit**: `35dc287` - fix: 修复剩余技术债 (A5, A6, A8)

### A5 [P1]: Forward-only purge (AFML Ch7 标准)

**问题描述**:
- purge 逻辑双向清除，不符合 AFML 标准

**修复方案**:
- split() 和 split_with_info() 改为 forward-only purge
- 只 purge test 之后的数据，不 purge test 之前

**文件变更**: `src/models/purged_kfold.py`

### A6 [P1]: FracDiff d 全局固定

**问题描述**:
- 每 fold 计算的 d 值可能不同

**修复方案**:
- 通过 EXT-Q2 全局预计算解决
- 全局统一计算 d 值

### A8 [P2]: Feature lookback >= purge

**问题描述**:
- purge_window (10) < feature_lookback (60)

**修复方案**:
- config: purge_window 10 → 60
- 确保 purge window >= max feature lookback (60d)

**文件变更**: `config/training.yaml`

### A7 [P1]: Test weights per-fold

**问题描述**:
- 测试集权重计算方式需要优化

**修复方案**:
- 通过 EXT-Q1 修复得到改善 (test set 不再参与训练)

### Q4: 非重叠约束

**决策**: 保持现状 (审计决定)

---

## 📊 修复统计

### 按严重度统计

| 严重度 | 数量 | 占比 |
|--------|------|------|
| P0 / CRITICAL | 15 | 25% |
| P1 / HIGH | 22 | 37% |
| P2 / MEDIUM | 15 | 25% |
| P3 / LOW | 8 | 13% |
| **总计** | **60** | 100% |

### 按模块统计

| 模块 | 修复数 | 主要问题类型 |
|------|--------|--------------|
| models/ | 18 | PBO/DSR计算、CPCV实现 |
| features/ | 15 | min_periods、FracDiff |
| signals/ | 5 | BaseModel实现 |
| labels/ | 4 | Triple Barrier |
| data/ | 8 | 验证、完整性 |
| tests/ | 10 | 测试覆盖、mock数据 |

---

## 🔴 Q5-REG: 回归修复 - fracdiff.py UnboundLocalError (2026-03-01)

**Commit**: `待补充` - fix: Q5 回归修复，删除重复常量定义

**问题描述**:
- `fracdiff.py` 函数内部重复定义 `MIN_ADF_SAMPLES` 常量
- 导致 Python 解释器将其视为局部变量，引发 `UnboundLocalError`
- 模块级常量已定义，函数内无需重复定义

**修复方案**:
- 删除 `find_min_d()` 函数内的 `MIN_ADF_SAMPLES = 20` 重复定义
- 直接使用模块级 `MIN_ADF_SAMPLES = 20` 常量

**文件变更**: `src/features/fracdiff.py`

**根因分析**:
- 在之前的修复中，函数内部和模块级都定义了同名常量
- Python 的变量作用域规则：函数内赋值会将变量视为局部变量
- 在函数内使用 `if MIN_ADF_SAMPLES:` 条件时，变量尚未赋值，触发 UnboundLocalError

---

### 测试状态

```
总测试数: 165
通过: 165 ✅
失败: 0

覆盖率: 
- 语句覆盖率: 87%
- 分支覆盖率: 78%
- 函数覆盖率: 92%
```

---

## 📝 修复流程规范

### 严重度定义

| 级别 | 定义 | 响应时间 |
|------|------|----------|
| P0 / CRITICAL | 系统阻塞，无法运行 | 立即 |
| P1 / HIGH | 功能缺陷，结果不可靠 | 24小时内 |
| P2 / MEDIUM | 代码质量，潜在风险 | 72小时内 |
| P3 / LOW | 优化建议，非阻塞 | 下次迭代 |

### 修复流程

1. **发现问题** → 记录到审计日志
2. **评估严重度** → 按定义分类
3. **制定修复方案** → 技术评审
4. **实施修复** → 代码变更
5. **回归测试** → 确保无破坏
6. **审计验证** → 独立验证
7. **关闭问题** → 更新状态

---

## 🔗 相关文档

- [OR5_CONTRACT.md](./OR5_CONTRACT.md) - OR5 审计契约
- [PHASE_C_IMPL_GUIDE.md](../PHASE_C_IMPL_GUIDE.md) - Phase C 实施指南
- [CHANGELOG.md](../../CHANGELOG.md) - 版本变更日志
- [AUDIT_RECORDS_SUMMARY.md](../../AUDIT_RECORDS_SUMMARY.md) - Phase A+B 审计汇总

---

*文档维护者: 赵连顺*  
*最后更新: 2026-03-01*
