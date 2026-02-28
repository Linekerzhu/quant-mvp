# Phase C 更新计划（基于实际进度）

> **制定人**: 张得功(八品领侍)  
> **更新日期**: 2026-02-28  
> **版本**: v2.0  
> **状态**: 执行中  

---

## 一、执行摘要

### 1.1 实际完成情况

**代码实现 100% 完成：**

| Step | 任务 | 状态 | 文件 | 测试 |
|------|------|------|------|------|
| Step 1 | Base Models | ✅ 已完成 | `src/signals/base_models.py` | ✅ `test_base_models.py` |
| Step 2 | CPCV | ✅ 已完成 | `src/models/purged_kfold.py` | ✅ `test_cpcv.py` |
| Step 3 | FracDiff | ✅ 已完成 | `src/features/fracdiff.py` | ✅ `test_fracdiff.py` |
| Step 4 | Meta-MVP | ✅ 已完成 | `src/models/meta_trainer.py` | ❌ **缺失** |

**刚完成的代码整改（提升质量）：**

- ✅ **拆分 meta_trainer.py** → `overfitting.py` + `label_converter.py`
- ✅ **添加 Base Model 抽象基类** + 注册表 (`src/signals/base.py`)
- ✅ **配置依赖倒置** - 从 `training.yaml` 读取参数
- ✅ **统一日志** - 使用 `event_logger`
- ✅ **代码质量提升** - 165测试全通过，金融审计通过

**当前状态：**
- ✅ 165测试全通过
- ✅ 金融审计通过
- ⚠️ Meta Trainer 单元测试缺失
- ⚠️ 验收报告文档缺失

---

## 二、剩余工作清单

### 2.1 代码层面（优先级 P0）

| 编号 | 任务 | 文件 | 预计工时 | 状态 |
|------|------|------|----------|------|
| C1 | 补充 Meta Trainer 单元测试 | `tests/test_meta_trainer.py` | 3h | ❌ 待办 |
| C2 | 跑通完整训练管道（合成数据） | - | 1h | ❌ 待办 |
| C3 | 验证 OR5 参数硬化 | - | 0.5h | ❌ 待办 |

**C1 详细要求：**

测试清单（参照原计划 Step 4）：
- [ ] `test_full_pipeline_runs`: 合成数据跑通完整流程
- [ ] `test_meta_label_binary`: 标签只有 0 和 1
- [ ] `test_sample_weight_passed`: LGB 接收到 sample_weight
- [ ] `test_dummy_sentinel_catches_overfit`: 人造过拟合场景触发哨兵
- [ ] `test_cpcv_15_paths`: 确认产出 15 条 path
- [ ] `test_lgb_params_from_config`: 参数从 YAML 读取

**C2 详细要求：**

使用合成数据（或真实数据）跑通完整训练管道：
- Base Model 生成信号
- Triple Barrier 标签转换
- FracDiff 特征计算
- CPCV 15 条路径训练
- PBO 过拟合检测
- Dummy Feature 哨兵检查

**C3 详细要求：**

验证 OR5 参数硬化（from `config/training.yaml`）：
- `max_depth ≤ 3`
- `num_leaves ≤ 7`
- `min_data_in_leaf ≥ 200`
- `feature_fraction = 0.5`
- `bagging_fraction = 0.7`
- `learning_rate = 0.01`
- `lambda_l1 = 1.0`

---

### 2.2 文档层面（优先级 P1）

| 编号 | 任务 | 文件 | 预计工时 | 状态 |
|------|------|------|----------|------|
| D1 | CPCV 验证报告 | `reports/cpcv_validation.md` | 2h | ❌ 待办 |
| D2 | PBO 分析报告 | `reports/pbo_analysis.md` | 2h | ❌ 待办 |
| D3 | Dummy Feature 哨兵报告 | `reports/dummy_sentinel.md` | 1h | ❌ 待办 |
| D4 | 回测性能报告（含扣减） | `reports/backtest_performance.md` | 2h | ❌ 待办 |
| D5 | Phase C 完成总结 | `docs/PHASE_C_SUMMARY.md` | 1h | ❌ 待办 |

**D1 详细要求：**

报告内容：
- 15 条 CPCV path 的详细信息
- 每条 path 的训练/测试样本数
- Purge 和 Embargo 统计
- 有效训练天数验证（≥ 200 天）
- 无泄漏确认

**D2 详细要求：**

报告内容：
- PBO 计算方法
- PBO 值和门控结果
- 路径级性能分布（AUC）
- Deflated Sharpe Ratio
- 过拟合风险评估

**D3 详细要求：**

报告内容：
- Dummy Feature 哨兵检查结果
- `dummy_noise` 特征排名
- 相对贡献度（vs 中位数真实特征）
- 过拟合判定

**D4 详细要求：**

报告内容：
- 原始性能指标（CAGR, MDD, Sharpe, etc.）
- 数据技术债扣减：
  - CAGR: -3%（survivorship 2% + lookahead 1%）
  - MDD: +10%
- 调整后性能指标
- 与 SPY 基准对比

**D5 详细要求：**

总结内容：
- Phase C 目标回顾
- 实施成果总结
- 遇到的挑战和解决方案
- 性能验收结果
- 下一步建议（Phase D）

---

### 2.3 集成测试（优先级 P2）

| 编号 | 任务 | 预计工时 | 状态 |
|------|------|----------|------|
| I1 | 跑通真实数据训练（非合成） | 2h | ❌ 待办 |
| I2 | 验证与 Phase A/B 数据合约 | 1h | ❌ 待办 |
| I3 | 性能基准测试（时间、内存） | 1h | ❌ 待办 |

**I1 详细要求：**

使用真实数据（非合成数据）跑通完整训练：
- 加载 Phase A 产出的 `features.parquet`
- 加载 Phase B 产出的 `labels.parquet`
- 执行完整 Meta-Labeling 训练
- 验证输出格式和内容

**I2 详细要求：**

验证数据合约：
- Phase A → Phase C 特征对接
- Phase B → Phase C 标签对接
- `sample_weights` 权重传递
- `label_exit_date` 传递（用于 CPCV Purge）

**I3 详细要求：**

性能基准：
- 15 条 CPCV path 训练总时间
- 单个 fold 训练时间
- 内存峰值使用
- 优化建议

---

## 三、时间估算

### 3.1 总体时间

| 阶段 | 预计工时 | 日历天数 | 备注 |
|------|----------|----------|------|
| C1-C3: 代码补全 | 4.5h | 0.5 天 | Meta Trainer 测试 + 验证 |
| D1-D5: 文档生成 | 8h | 1 天 | 5 份验收报告 |
| I1-I3: 集成测试 | 4h | 0.5 天 | 真实数据测试 |
| **总计** | **16.5h** | **2 天** | 含缓冲 |

### 3.2 建议排期

**Day 1（上午）：**
- ✅ C1: 补充 `test_meta_trainer.py` (3h)
- ✅ C2: 跑通完整管道 (1h)

**Day 1（下午）：**
- ✅ C3: OR5 参数验证 (0.5h)
- ✅ D1-D3: CPCV/PBO/Dummy 报告 (5h)

**Day 2（上午）：**
- ✅ D4-D5: 回测报告 + 总结 (3h)
- ✅ I1-I3: 集成测试 (4h)

**Day 2（下午）：**
- ✅ 最终验收
- ✅ 提交主子审批

---

## 四、验收标准

### 4.1 代码验收

| 编号 | 验收项 | 通过标准 |
|------|--------|----------|
| C1 | Meta Trainer 测试 | `pytest tests/test_meta_trainer.py` 100% 通过 |
| C2 | 完整管道 | 合成数据训练成功，产出 15 条 CPCV path |
| C3 | OR5 参数 | 所有硬化参数符合 `training.yaml` 配置 |
| Q1 | 全量测试 | `pytest tests/` → 100% 通过 (165+ 测试) |

### 4.2 文档验收

| 编号 | 验收项 | 交付物 |
|------|--------|--------|
| D1 | CPCV 报告 | `reports/cpcv_validation.md` (15 path 详情) |
| D2 | PBO 报告 | `reports/pbo_analysis.md` (PBO < 0.3 或 Warning) |
| D3 | Dummy 哨兵 | `reports/dummy_sentinel.md` (哨兵通过) |
| D4 | 回测报告 | `reports/backtest_performance.md` (含扣减) |
| D5 | 总结文档 | `docs/PHASE_C_SUMMARY.md` |

### 4.3 性能验收

| 编号 | 验收项 | 通过标准 | 门控等级 |
|------|--------|----------|----------|
| P1 | PBO | PBO < 0.3 | **Pass** |
| P2 | PBO (Warning) | 0.3 ≤ PBO < 0.5 | **Warning** |
| P3 | PBO (Reject) | PBO ≥ 0.5 | **Hard Reject** |
| P4 | Dummy Feature | 排名 > 25% 且 相对贡献 ≤ 1.0 | **Pass** |

---

## 五、执行策略

### 5.1 优先级顺序

**执行顺序（严格按照）：**
1. **C1 (Meta Trainer 测试)** - 阻塞后续所有工作
2. **C2 (完整管道)** - 验证代码可用性
3. **C3 (OR5 验证)** - 合规检查
4. **D1-D5 (文档生成)** - 产出验收证据
5. **I1-I3 (集成测试)** - 真实数据验证

**并行任务：**
- D1, D2, D3 可并行（基于 C2 的训练结果）
- I1, I2 可并行（真实数据测试）

### 5.2 风险应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| Meta Trainer 测试发现 Bug | 中 | 高 | 立即修复，重新跑测试 |
| PBO ≥ 0.5 (Hard Reject) | 低 | 极高 | 回退 Phase B，调整特征/标签 |
| 真实数据训练失败 | 中 | 高 | 检查数据合约，修复兼容性 |
| 文档生成耗时超预期 | 低 | 中 | 使用模板，减少手工编写 |

### 5.3 回退策略

**Hard Reject 场景（PBO ≥ 0.5）：**
1. 立即停止 Phase C
2. 回退到 Phase B 重新调整特征
3. 检查 Base Model 信号质量
4. 重新评估标签生成逻辑

**数据合约失败场景：**
1. 检查 Phase A 产出格式
2. 检查 Phase B 产出格式
3. 修复数据管道兼容性
4. 重新运行 Phase A/B

---

## 六、关键决策点

### 6.1 PBO 门控判定

**判定规则（from `training.yaml`）：**
- `pbo_threshold = 0.30`
- `pbo_reject = 0.50`

**判定结果：**
- PBO < 0.30 → **Pass** → 直接进入 Phase D
- 0.30 ≤ PBO < 0.50 → **Warning** → 人工复核后可继续
- PBO ≥ 0.50 → **Hard Reject** → 回退 Phase B

### 6.2 Dummy Feature 哨兵判定

**判定规则：**
- `ranking_threshold = 0.25` (top 25% 拒绝)
- `relative_threshold = 1.0` (相对中位数贡献 > 1.0 拒绝)

**判定结果：**
- 排名 > 25% 且 相对贡献 ≤ 1.0 → **Pass**
- 排名 ≤ 25% 或 相对贡献 > 1.0 → **Reject** → 检查特征质量

### 6.3 数据技术债拨备

**硬编码扣减（from `overfitting.py`）：**
- CAGR: -3% (survivorship 2% + lookahead 1%)
- MDD: +10%

**展示规则：**
- 报告中必须同时展示原始值和调整后值
- 调整后值为最终验收标准

---

## 七、交付清单

### 7.1 代码交付物

- [x] `src/signals/base_models.py` (SMA + Momentum)
- [x] `src/signals/base.py` (Base Model 抽象基类 + 注册表)
- [x] `src/models/purged_kfold.py` (CPCV 隔离器)
- [x] `src/features/fracdiff.py` (FracDiff 特征)
- [x] `src/models/meta_trainer.py` (Meta Trainer 主类)
- [x] `src/models/overfitting.py` (过拟合检测)
- [x] `src/models/label_converter.py` (标签转换器)
- [ ] `tests/test_meta_trainer.py` **（待补充）**

### 7.2 测试交付物

- [x] `tests/test_base_models.py` (4+ 个测试)
- [x] `tests/test_cpcv.py` (5+ 个测试)
- [x] `tests/test_fracdiff.py` (5+ 个测试)
- [ ] `tests/test_meta_trainer.py` (6+ 个测试) **（待补充）**

### 7.3 文档交付物

- [ ] `reports/cpcv_validation.md` (CPCV 验证报告)
- [ ] `reports/pbo_analysis.md` (PBO 分析报告)
- [ ] `reports/dummy_sentinel.md` (Dummy Feature 哨兵报告)
- [ ] `reports/backtest_performance.md` (回测性能报告，含扣减)
- [ ] `docs/PHASE_C_SUMMARY.md` (Phase C 完成总结)

### 7.4 模型交付物

- [ ] 训练好的 Meta-Labeling LightGBM 模型 (`.joblib`)
- [ ] 15 条 CPCV path 的验证结果 (`.json`)
- [ ] 特征重要性排名 (`.csv`)
- [ ] 概率校准曲线 (`.png`)

---

## 八、附录

### 8.1 已完成代码清单

**Step 1: Base Models**
- `src/signals/base_models.py` - SMA Cross + Momentum 信号生成器
- `src/signals/base.py` - Base Model 抽象基类 + 注册表

**Step 2: CPCV**
- `src/models/purged_kfold.py` - Combinatorial Purged K-Fold 实现
  - Purge 算法（基于 `label_exit_date`）
  - Embargo 算法（40 天窗口）
  - 15 条 CPCV path 生成

**Step 3: FracDiff**
- `src/features/fracdiff.py` - 分数阶差分特征
  - Fixed-Window FracDiff
  - ADF 平稳性检验
  - 最优 d 搜索

**Step 4: Meta-MVP**
- `src/models/meta_trainer.py` - Meta-Labeling 训练管道
- `src/models/overfitting.py` - 过拟合检测（PBO + Dummy Sentinel）
- `src/models/label_converter.py` - Triple Barrier → Meta-Label 转换

**代码质量改进：**
- 抽象基类 + 注册表模式
- 配置依赖倒置（从 YAML 读取）
- 统一日志（`event_logger`）
- 代码拆分（降低耦合）

### 8.2 测试覆盖情况

**现有测试（165 个全通过）：**
- ✅ `test_base_models.py` - Base Model 信号生成测试
- ✅ `test_cpcv.py` - CPCV 隔离器测试
- ✅ `test_fracdiff.py` - FracDiff 特征测试
- ❌ `test_meta_trainer.py` - **缺失，待补充**

### 8.3 参考文档

- `docs/PHASE_C_PLAN.md` v1.0 - 原始计划
- `docs/PHASE_C_IMPL_GUIDE.md` - 工程实施指南
- `docs/OR5_CONTRACT.md` - OR5 审计契约
- `config/training.yaml` - 训练参数配置
- `plan.md` v4.2 - 项目总计划

### 8.4 术语表

| 术语 | 说明 |
|------|------|
| CPCV | Combinatorial Purged Cross-Validation（组合清除交叉验证） |
| PBO | Probability of Backtest Overfitting（回测过拟合概率） |
| FracDiff | Fractional Differencing（分数阶差分） |
| ADF | Augmented Dickey-Fuller Test（平稳性检验） |
| Meta-Labeling | 元标签（预测信号质量的二阶模型） |
| Purge | 清除（删除训练集中与验证集有重叠的样本） |
| Embargo | 禁运（验证集结束后一段时间内的训练样本也删除） |
| OR5 | OR5 Anti-Kaggle Hardening（OR5 反 Kaggle 硬化规则） |

---

## 九、总结

### 9.1 Phase C 核心成果

**已实现：**
1. ✅ Base Model 信号生成（SMA + Momentum）
2. ✅ CPCV 零泄漏交叉验证（15 条 path）
3. ✅ FracDiff 平稳性特征工程
4. ✅ Meta-Labeling 完整训练管道
5. ✅ 过拟合检测机制（PBO + Dummy Sentinel）
6. ✅ 数据技术债拨备（CAGR -3%, MDD +10%）

**待完成：**
1. ❌ Meta Trainer 单元测试（C1）
2. ❌ 完整管道验证（C2-C3）
3. ❌ 验收报告文档（D1-D5）
4. ❌ 真实数据集成测试（I1-I3）

### 9.2 下一步行动

**立即执行（Day 1-2）：**
1. 补充 `tests/test_meta_trainer.py`
2. 跑通完整训练管道
3. 生成所有验收报告
4. 执行真实数据集成测试

**最终验收（Day 2 下午）：**
1. 全量测试通过（165+ 测试）
2. PBO 门控通过（< 0.3 或 Warning）
3. Dummy Feature 哨兵通过
4. 文档齐全（5 份报告）
5. 提交主子审批

**进入 Phase D 前置条件：**
- ✅ Phase C 所有代码完成
- ✅ 165+ 测试全通过
- ✅ PBO < 0.5（Pass 或 Warning）
- ✅ Dummy Feature 哨兵通过
- ✅ 5 份验收报告完成
- ✅ 主子审批通过

---

**文档状态**: ✅ 已完成，立即执行

**下一步**: 李得勤开始执行 C1（补充 `test_meta_trainer.py`）

---

*张得功 敬上*  
*2026-02-28*
