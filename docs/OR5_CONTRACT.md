# OR5 架构契约 — Phase C 红线

**签署日期**: 2026-02-25
**审计基线**: commit `7fddb78`
**审计官**: 外部审计

---

## 契约概述

本契约定义了 Phase C 开发必须遵守的架构红线。任何违反本契约的代码提交将被一票否决。

---

## 契约 1: LightGBM 参数锁死

**状态**: ✅ 已实施 (training.yaml)

```yaml
lightgbm:
  max_depth: 3        # ❗ 严禁超过 3
  num_leaves: 7       # ❗ <= 2^max_depth - 1
  min_data_in_leaf: 200  # ❗ 强制统计显著性
  feature_fraction: 0.5  # ❗ 双重随机化
  learning_rate: 0.01    # ❗ 降速学习
  lambda_l1: 1.0         # ❗ 特征稀疏化
```

**允许的搜索空间**:
- `max_depth` ∈ {2, 3}
- `num_leaves` ∈ {3, 5, 7}
- `min_data_in_leaf` ∈ {100, 200, 300}

**违约后果**: 即使 CV 分数提升，超过此范围的参数也不可接受。

---

## 契约 2: Meta-Labeling 强制架构

**状态**: ⏳ Phase C 待实施

**要求**: Phase C 必须实现 Meta-Labeling 架构，LightGBM 不直接预测涨跌方向。

**架构**:
```
原始特征 → Primary Model (三屏障标签) → Meta-Features → Secondary Model (Meta-Label)
```

**Meta-Features 包括**:
- Primary model 的预测概率
- 置信度区间
- 预测一致性指标

**违约后果**: Phase C 不可进入回测阶段。

---

## 契约 3: FracDiff 特征基座

**状态**: ⏳ Phase C 待实施

**要求**: Phase C 必须实现分数阶差分 (Fractional Differentiation) 特征。

**目的**: 解决价格序列的非平稳性问题，同时保留长期记忆。

**实现要求**:
- d ∈ [0, 1] 可调参数
- 默认 d = 0.4 (AFML 推荐)
- 必须验证平稳性 (ADF test p < 0.05)

**违约后果**: 任何使用原始价格或简单收益率的模型不可进入生产。

---

## 契约 4: CPCV 手写 Purge + Embargo

**状态**: ⏳ Phase C 待实施

**要求**: Phase C 必须实现手写的 CPCV 切分器，包含 Purge 和 Embargo 逻辑。

**当前配置**:
```yaml
cpcv:
  n_splits: 6
  n_test_splits: 2
  purge_window: 10   # 天
  embargo_window: 40 # 天
```

**实现检查清单**:
- [ ] Purge: 训练集末尾 N 天移除，防止泄露
- [ ] Embargo: 测试集开头 N 天移除，防止泄露
- [ ] 路径生成: 所有 15 条路径等长 (>= 200 天训练数据)
- [ ] 无重叠: 训练/测试集合无交集

**违约后果**: 任何使用 sklearn KFold 的交叉验证不可接受。

---

## 拨备: 数据技术债扣减

**状态**: ⏳ 回测报告必须硬编码

**要求**: 所有回测报告必须硬编码以下扣减:

| 指标 | 扣减 | 说明 |
|------|------|------|
| CAGR | -3% | 未来与现实差距 |
| MDD | +10% | 极端市场风险 |

**示例**:
```python
# 回测报告
raw_cagr = 0.25
raw_mdd = 0.15

# 必须报告扣减后
reported_cagr = raw_cagr - 0.03  # 22%
reported_mdd = raw_mdd + 0.10    # 25%
```

**违约后果**: 任何未扣减的回测报告不可用于投资决策。

---

## 契约签署

- [x] 契约 1: LightGBM 参数锁死 ✅ (training.yaml)
- [x] 契约 2: Meta-Labeling 架构 ✅ (src/models/meta_trainer.py)
- [x] 契约 3: FracDiff 特征 ✅ (src/features/fracdiff.py)
- [x] 契约 4: CPCV 手写切分器 ✅ (src/models/purged_kfold.py)
- [x] 拨备: 回测扣减 ✅ (src/models/overfitting.py DataPenaltyApplier)

**更新日期**: 2026-02-28
**更新人**: 李得勤

---

*审计官签名: OR5 External Audit*
*工程负责人: quant-mvp team*
*日期: 2026-02-25*
*契约完成: 2026-02-28*
