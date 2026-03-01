# 修复日志 (Fix Log)

> 记录所有生产环境修复，按 H/R 编号追踪

---

## R22 - 样本权重公式修正 (AFML Ch4)

| 属性 | 值 |
|------|-----|
| **修复编号** | R22 |
| **修复日期** | 2026-03-02 |
| **Commit** | `17189fb` |
| **涉及文件** | `src/labels/sample_weights.py` |
| **修复类型** | Bug Fix |
| **严重程度** | P2→Phase D |

### 问题描述

`_calculate_weights_sweep_line()` 使用 `1/(1+avg_concurrent)` 计算权重。
AFML Ch4 标准公式为 `mean(1/c(t))`。

两个差异：
1. `1/(1+c)` vs `1/c` — concurrent已含自身，额外+1导致系统性偏低
2. `1/mean(c)` vs `mean(1/c)` — Jensen不等式，非线性差异

### 修复内容

- 将权重计算改为 AFML Ch4 average uniqueness: `mean(1/c(t))`
- 单独事件: weight=1.0 (原0.5)
- 2个完全重叠事件: weight=0.5 (原0.33)

### 审计结果

| 审计项 | 结果 | 备注 |
|--------|------|------|
| 测试通过 | ✅ 165/165 | |
| 公式修正 | ✅ | 符合AFML Ch4标准 |

**状态**: ✅ 已修复

---

## R21 - 死路径检测 & DSR Baseline 修复

| 属性 | 值 |
|------|-----|
| **修复编号** | R21 |
| **修复日期** | 2026-03-02 |
| **Commit** | `a4c0f8f` |
| **涉及文件** | `src/models/meta_trainer.py`, `src/models/overfitting.py` |
| **修复类型** | Bug Fix |
| **严重程度** | P2 (R21-F1), P1→Phase D (R21-F2) |

### 问题描述

R20-F1修复后，通过深度审计发现：
- R21-F1: 5/15 CPCV paths因bagging有效样本不足产出死模型，污染PBO统计
- R21-F2: DSR baseline=positive_ratio，当pos_ratio<0.5时可被trivial模型骗过

### 修复内容

| 子修复 | 描述 | 涉及文件 |
|--------|------|----------|
| **R21-F1** | 死路径检测 - dead_ratio>0.5时抛出F8错误 | `meta_trainer.py` |
| **R21-F2** | DSR baseline改为max(pos_ratio, 1-pos_ratio) | `overfitting.py` |

### 审计结果

| 审计项 | 结果 | 备注 |
|--------|------|------|
| 测试通过 | ✅ 165/165 | |
| 死路径检测 | ✅ | dead_ratio≈0.33 < 0.5 |
| DSR baseline | ✅ | 当前数据误差≤0.016 |
| Commit 验证 | ✅ | `a4c0f8f` 已推送 |

**状态**: ✅ 已修复

---

## R20 - min_data_in_leaf 修复

| 属性 | 值 |
|------|-----|
| **修复编号** | R20 |
| **修复日期** | 2026-03-01 |
| **Commit** | `f64fd2a` |
| **涉及文件** | `src/models/meta_trainer.py` |
| **修复类型** | Bug Fix |
| **严重程度** | High |

### 问题描述

min_data_in_leaf=200过大，导致15/15 CPCV paths全部产出死模型。

### 修复内容

- 将 min_data_in_leaf 从 200 降至 100

### 审计结果

| 审计项 | 结果 | 备注 |
|--------|------|------|
| 测试通过 | ✅ 165/165 | |
| 存活路径 | ✅ 10/15 | 从0/15提升至10/15 |

**状态**: ✅ 已修复

---

## R19 - 数据饥饿修复 & 代码清理

| 属性 | 值 |
|------|-----|
| **修复编号** | R19 |
| **修复日期** | 2026-03-01 |
| **Commit** | `e397040` |
| **涉及文件** | `src/training/meta_trainer.py`, `tests/e2e/`, 多模块 |
| **修复类型** | Bug Fix + Refactor |
| **严重程度** | High |

### 问题描述

MetaTrainer 存在数据饥饿问题，e2e测试异常处理不完善，代码库中存在死代码需要清理。

### 修复内容

| 子修复 | 描述 | 涉及文件 |
|--------|------|----------|
| **R19-F1** | 数据饥饿修复 - MetaTrainer延迟过滤 | `src/training/meta_trainer.py` |
| **R19-F2** | e2e测试异常处理 | `tests/e2e/` 目录 |
| **R19-F3** | 死代码清理 | 多模块 |

### 审计结果

| 审计项 | 结果 | 备注 |
|--------|------|------|
| 代码审查 | ✅ 通过 | R19-F1/F2/F3 修复逻辑正确 |
| Commit 验证 | ✅ 通过 | `e397040` 已提交 |

**状态**: ✅ 已修复

---

## H2 - Time Barrier 过滤监控修复

| 属性 | 值 |
|------|-----|
| **修复编号** | H2 |
| **修复日期** | 2026-03-01 |
| **Commit** | `fb6252b` |
| **涉及文件** | `src/models/label_converter.py` |
| **修复类型** | Bug Fix |
| **严重程度** | Medium |

### 问题描述

Time Barrier 过滤监控逻辑存在问题，需要修复以确保标签转换过程中的时间屏障正确过滤。

### 修复内容

- 修复了 `label_converter.py` 中的 Time Barrier 过滤监控功能
- Commit: `fb6252b`

### 审计结果

| 审计项 | 结果 | 备注 |
|--------|------|------|
| 代码审查 | ✅ 通过 | 修复逻辑正确 |
| 文件验证 | ✅ 通过 | `src/models/label_converter.py` 已更新 |
| Commit 验证 | ✅ 通过 | `fb6252b` 已提交 |

**审计员**: 连顺  
**审计日期**: 2026-03-01  
**状态**: ✅ 已修复并通过审计

---

## 修复历史

| 编号 | 日期 | 描述 | 文件 | 状态 |
|------|------|------|------|------|
| R22 | 2026-03-02 | 样本权重公式修正为AFML标准 | `sample_weights.py` | ✅ 已修复 |
| R21 | 2026-03-02 | 死路径检测 + DSR majority-class baseline | `meta_trainer.py`, `overfitting.py` | ✅ 已修复 |
| R20 | 2026-03-01 | min_data_in_leaf 200→100 | `meta_trainer.py` | ✅ 已修复 |
| R19 | 2026-03-01 | 数据饥饿修复 + e2e异常处理 + 死代码清理 | `meta_trainer.py`, `tests/e2e/`, 多模块 | ✅ 已修复 |
| H2 | 2026-03-01 | Time Barrier 过滤监控 | `src/models/label_converter.py` | ✅ 已修复 |
