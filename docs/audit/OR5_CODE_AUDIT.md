# OR5 代码级审计验证与整改记录

> **审计轮次**: OR5
> **审计日期**: 2026-02-26
> **基准 commit**: `ceb90cb` (docs(v4.2): OR5 审计裁决)
> **测试基线**: 97/97 passing，覆盖率 57%
> **状态**: ✅ 已完成

---

## 一、审计专家声称 vs 代码验证结论

### ✅ 验证通过（代码与声称一致）

#### HF-1：Maximum Pessimism Principle

- **文件**: `src/labels/triple_barrier.py` L302-356
- **验证**: Gap 检查 → Collision 检查 → Normal，优先级正确
- **关键点**:
  - `day_open <= loss_barrier` 使用 `<=`（精确触碰也判 gap）
  - collision 返回 `loss_barrier` 作为 exit_price，标记 `loss_collision`
  - loss gap 优先于 profit gap，符合悲观原则
  - `if pd.isna(day_open): continue` 防止脏数据穿透
- **测试覆盖**: `test_smoke_or5.py` 6 个测试精确验证全部路径

#### HF-2：LightGBM Anti-Kaggle 硬化

- **文件**: `config/training.yaml` lightgbm 段
- **验证**: 所有硬化参数到位
  - max_depth: 3, num_leaves: 7, min_data_in_leaf: 200
  - learning_rate: 0.01, lambda_l1: 1.0, feature_fraction: 0.5
  - n_estimators: 500
- **测试覆盖**: `test_training_yaml_has_or5_params`

#### Q4C：ATR 无前视泄漏

- **文件**: `src/features/build_features.py` L381-390
- **验证**: `_calc_atr` 使用 `df['adj_close'].shift(1)`
- **结论**: ATR[T] = f(H[T], L[T], C[T-1])，无泄漏

#### Q5：Sample Weights Entry-Date 修复

- **文件**: `src/labels/sample_weights.py` L105-130
- **验证**: R29-A1 修复到位
  - `entry_date = trigger_date + BDay(1)`
  - 活跃区间 = `[entry_date, label_exit_date]`，不包含 trigger day

#### CPCV 配置参数

- **文件**: `config/training.yaml` cpcv 段
- **验证**: n_splits=6, n_test_splits=2, purge_window=10, embargo_window=40
- **结论**: min_data_days=630，每 fold 充足

### ⚠️ 审计专家声称正确，但尚无代码（Phase C 任务）

| 声称 | 状态 | 目标文件 |
|------|------|----------|
| Q3: FracDiff 未实现 | ✅ 确认无代码 | `src/features/fracdiff.py`（待建）|
| Q6: CPCV Purge 未实现 | ✅ 确认无代码 | `src/models/purged_kfold.py`（待建）|
| Q7A: Meta-Labeling 未实现 | ✅ 确认无代码 | `src/signals/base_models.py` + `src/models/meta_trainer.py`（待建）|
| Q2: hash_frozen 未实现 | ✅ 确认无代码 | Phase A 技术债，不阻塞 Phase C |

### 🔍 验证中发现的新问题

审计专家未提及，但代码审查中发现的问题，已整理为整改任务 T1-T7。

---

## 二、OR4 外部审计交叉验证

### ❌ 误报（代码实际无此问题）

#### OR4-P0-1：validate.py OHLC Low 校验

- **审计声称**: Low 检查只对 `adj_` 生效
- **验证结论**: ❌ 误报。Check 2（High）和 Check 3（Low）缩进一致，均在循环内
- **处置**: 无需修改

#### OR4-P0-2：corporate_actions.py 退市检测

- **审计声称**: 未按 date 排序
- **验证结论**: ❌ 误报。R25 已修复，L123-124 有排序
- **处置**: 无需修改

### ✅ 确认属实（需要整改）

#### OR4-P0-3：features.yaml requires_ohlc 元数据错误

- **审计声称**: 4个特征错误标记 requires_ohlc: true
- **验证结论**: ✅ 属实
- **影响**: 备源降级时错误禁用这些特征
- **整改**: 见 T6

### ⚠️ 部分属实（降级处理）

- **OR4-P1-1**: ffill + pct_change 假阳性 → 降为 P3
- **OR4-P1-2**: missing_values passed 计数近似 → 降为 P3
- **OR4-P2**: embargo 术语混用 → 见 T7

---

## 三、整改任务清单

### 优先级 P1（必须在 Phase C 开工前完成）

#### T1: Burn-in 预警推送 ✅

- **问题**: FracDiff 的 burn-in 与 CPCV 断层衔接陷阱未文档化
- **操作**: 
  - `docs/PHASE_C_IMPL_GUIDE.md` Step 3 添加 Burn-in 预警
  - `plan.md` Step 3 添加 Burn-in 衔接规则
- **完成**: commit `5c35141`

#### T6: features.yaml requires_ohlc 修正 ✅

- **问题**: 4个特征错误标记为 requires_ohlc: true
- **操作**: 
  - rsi_14, macd_line_pct, macd_histogram_pct, pv_correlation_5d
  - requires_ohlc 从 true 改为 false
- **完成**: commit `5c35141`

#### T7: embargo/feature_lookback 决策记录 ✅

- **问题**: embargo=40 < feature_lookback=60，存在 20 天缺口
- **操作**: 
  - `config/event_protocol.yaml` 添加风险注释
  - `docs/PHASE_C_IMPL_GUIDE.md` Step 2 添加两个解决方案
- **完成**: commit `5c35141`

### 优先级 P2

#### T2: plan.md §6.5 公式示例过时 ✅

- **问题**: gap=70d 应更新为 gap=50d
- **完成**: commit `5c35141`

#### T3: PBO 三档门控对齐 ✅

- **问题**: IMPL_GUIDE 缺少 Warning 档 (0.3-0.5)
- **完成**: commit `5c35141`

#### T4: 覆盖率盲区冒烟测试 ✅

- **问题**: universe.py 和 feature_importance.py 覆盖率 0%
- **操作**: 新增 `tests/test_universe.py` + `tests/test_feature_importance.py`
- **完成**: commit `5c35141`

### 优先级 P3

#### T5: early_stopping_rounds 位置 ✅

- **问题**: 是 callback 参数，不是模型参数
- **操作**: 添加注释说明
- **完成**: commit `5c35141`

---

## 四、完成确认

### 提交记录

```bash
git log --oneline -3
```

```
5c35141 fix(OR5-CODE): 审计整改 T1-T7 全部完成
ceb90cb docs(v4.2): OR5 审计裁决 - Phase C 强制 Meta-Labeling 架构
75283d4 docs(OR5): 将架构契约签署进 plan.md 作为 Phase C S0 前置条件
```

### 验证检查表

- [x] `grep "Burn-in" docs/PHASE_C_IMPL_GUIDE.md plan.md` 各至少 1 行
- [x] `grep "gap=70d" plan.md` 只出现在历史注记中（已更正为 gap=50d）
- [x] `grep "0.3" docs/PHASE_C_IMPL_GUIDE.md` 出现 PBO warning 逻辑
- [x] `tests/test_universe.py` 和 `tests/test_feature_importance.py` 已创建
- [x] `grep "requires_ohlc: false" config/features.yaml` 包含 4 个修正的特征
- [x] `grep "embargo.*feature_lookback" config/event_protocol.yaml` 存在
- [x] `git log --oneline -1` 显示 OR5-CODE commit

---

## 五、后续行动

- ✅ OR5 审计整改完成
- 🚧 **Phase C 待开工**: 按照 `docs/PHASE_C_IMPL_GUIDE.md` 4步 SOP 实施
- ⏸️ Phase D/E/F 等待 Phase C 完成后启动

---

*审计完成日期: 2026-02-26*
*整改完成日期: 2026-02-26*
*下一步: Phase C 开工*
