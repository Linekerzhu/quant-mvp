# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v4.5] - 2026-03-02

### R21 深度审计修复 (Commit: `a4c0f8f`)

**审计轮次**: R21 (2026-03-02)
**审计官**: Internal Audit (张德功)
**状态**: ✅ 已完成 - 165/165 测试通过

### Fixed - 修复

#### R20 [HIGH]: min_data_in_leaf 修复 (Commit: `f64fd2a`)
- **问题**: min_data_in_leaf=200过大，导致15/15 CPCV paths全部死模型
- **修复**: min_data_in_leaf从200降至100
- **结果**: 存活路径从0/15提升至10/15

#### R21-F1 [P2]: 死路径检测
- **问题**: 5/15 CPCV paths因bagging有效样本≈280，对min_data_in_leaf=100仍边缘，导致死模型
- **修复**: 在Gate检查前添加dead_ratio检测，>0.5时抛出F8错误
- **文件**: `src/models/meta_trainer.py`

#### R21-F2 [P1→Phase D]: DSR Majority-Class Baseline
- **问题**: DSR baseline=positive_ratio，当pos_ratio<0.5时可被trivial模型骗过
- **修复**: baseline改为max(pos_ratio, 1-pos_ratio)
- **文件**: `src/models/overfitting.py`

### Changed - 修改
- `meta_trainer.py`: 添加死路径检测逻辑
- `overfitting.py`: DSR baseline计算逻辑
- `check_overfitting()`返回值添加`dead_path_count`和`dead_path_ratio`

---

## [v4.4] - 2026-03-01

### 内审修复完成 + 外部审核整改 (R14-R18)

**审计轮次**: R14-R18 + EXT-Q1/Q2/Q5 (2026-03-01)
**审计官**: Internal Audit (赵连顺)
**状态**: ✅ 已完成 - 165/165 测试通过

### Fixed - 修复

#### R14 内审修复 (Commit: `140c8d6`)
- **A1 [HIGH]**: PBO 改用 AFML 排名方法 (Ch7 §7.4.2)
  - PBO = P(best IS path 的 OOS 排名 > 中位数)
- **A2 [MEDIUM]**: DSR 添加 skewness/kurtosis + 多重测试校正
- **A4 [MEDIUM]**: BaseModel .loc→.iloc 冷启动修复
- **A9 [LOW]**: Dummy sentinel per-fold 检查

#### R14-A3 样本权重 (Commit: `1928b85`)
- **A3 [HIGH]**: 样本权重 per-fold 重算
- 添加 SampleWeightCalculator 实例
- 在 _train_cpcv_fold 中对每个 fold 重算样本权重
- 避免训练权重受到测试集结构影响

#### R15 PBO 修正 (Commit: `2271bd7`)
- **N1 [NORMAL]**: PBO 计算逻辑修正
- 只检查 IS rank #1 的路径，而非所有路径

#### 外部审核修复 (Commit: `8fd8db4`)
- **EXT-Q2 [P0]**: FracDiff 全局预计算
  - 在 train() 中全局计算 FracDiff，避免每 fold 损失数据
- **EXT-Q1 [P0]**: Early Stopping 隔离 test set
  - 从训练集尾部切 20% 作为 validation set
  - test set 只用于最终评估，不参与 early stopping
- **Q5 [P1]**: find_min_d 预检查样本量

#### 剩余技术债 (Commit: `35dc287`)
- **A5 [P1]**: Forward-only purge (AFML Ch7 标准)
- **A6 [P1]**: FracDiff d 全局固定
- **A8 [P2]**: Feature lookback >= purge (10→60)

#### 回归修复 (2026-03-01)
- **Q5-REG**: 修复 `fracdiff.py` UnboundLocalError
  - 问题：函数内部重复定义 `MIN_ADF_SAMPLES` 导致局部变量冲突
  - 修复：删除函数内重复定义，直接使用模块级常量

### Changed - 修改
- **config/training.yaml**: purge_window 10 → 60
- **src/models/purged_kfold.py**: 实现 forward-only purge
- **src/models/meta_trainer.py**: 添加 per-fold 权重重算
- **src/models/overfitting.py**: PBO/DSR 算法修正

### Documentation - 文档
- **docs/audit/FIX_LOG.md**: 完整修复记录汇总
- 更新审计历史表格
- 添加修复流程规范

---

## [v4.3] - 2026-02-28

### Phase C 完成 - Meta-Labeling MVP

**状态**: ✅ Phase C 全部功能通过金融审计
**测试**: 165/165 passing

### Added - 新增
- **Phase C 完整实现**:
  - Base Models (SMA Cross + Momentum)
  - CPCV (Combinatorial Purged K-Fold, 15 paths)
  - FracDiff (Fractional Differentiation, d ≈ 0.4)
  - Meta-Labeling Pipeline
- **Deflated Sharpe Ratio**: z-score 实现
- **PBO (Probability of Backtest Overfitting)**: AFML 标准实现
- **技术债务拨备**: CAGR -3%, MDD +10%

### Changed - 修改
- **docs/PHASE_C_STATUS.md**: Phase C 完成状态报告
- **src/models/**: 完整的 Meta-Labeling 实现
- **tests/**: 165 个测试全部通过

---

## [v4.2] - 2026-02-26

### OR5 审计裁决 - Phase C 架构重构

**审计轮次**: OR5 (2026-02-26)
**审计官**: External Audit (规划组)
**状态**: ✅ 已完成

### Added - 新增

- **Meta-Labeling 强制架构**: Phase C 从"单模型直接预测"升级为 Base Model → Meta Model pipeline
- **FracDiff 特征**: 分数阶差分 (d ≈ 0.4)，保留时序记忆同时实现平稳性
- **手写 CPCV**: CombinatorialPurgedKFold (15 paths)，严禁 sklearn KFold
- **数据技术债拨备**: 回测报告硬编码 CAGR -3%, MDD +10%
- **docs/PHASE_C_IMPL_GUIDE.md**: Phase C 4步 SOP 实施指南
- **docs/OR5_CONTRACT.md**: OR5 审计契约文件
- **tests/test_universe.py**: 股票池模块冒烟测试
- **tests/test_feature_importance.py**: 特征重要性模块冒烟测试

### Changed - 修改

- **plan.md**: 升级到 v4.2，Phase C 完全重写为 4 步 SOP
- **config/training.yaml**: LightGBM 参数硬化（OR5 契约红线）
  - max_depth: 3 (LOCKED)
  - num_leaves: 7 (LOCKED)
  - min_data_in_leaf: 200 (LOCKED)
  - learning_rate: 0.01, lambda_l1: 1.0, feature_fraction: 0.5
- **src/labels/triple_barrier.py**: OR5 Hotfix
  - Maximum Pessimism Principle (HF-1)
  - Gap execution: 跳空穿越用实际开盘价结算
  - Collision detection: 同日双穿强制止损
  - 止损检查优先于止盈
- **config/features.yaml**: T6 整改 - 4个特征 requires_ohlc 从 true 改为 false
  - rsi_14, macd_line_pct, macd_histogram_pct, pv_correlation_5d
- **config/event_protocol.yaml**: T7 整改 - embargo/feature_lookback 20天缺口风险注释

### Fixed - 修复

- **T1**: Burn-in 预警 - 添加 FracDiff 与 CPCV 断层衔接警告
- **T2**: plan.md §6.5 公式示例 - gap=70d 更正为 gap=50d
- **T3**: PBO 三档门控对齐 - Warning 档 (0.3-0.5) 补充到 IMPL_GUIDE
- **T5**: early_stopping_rounds 位置 - 注释说明是 callback 参数

### Security - 安全

- OR4 误报确认：validate.py OHLC 校验、corporate_actions.py 退市检测均无问题
- OR5 验证通过：HF-1 Maximum Pessimism、HF-2 LightGBM 硬化、ATR 无前视泄漏

---

## [v4.1] - 2026-02-25

### 执行层迁移 Alpaca → Futu OpenAPI

### Changed

- 交易执行从 Alpaca API 切换到 Futu OpenAPI
- 成本模型参数调整（Futu 佣金结构 vs Alpaca 零佣金）
- Docker 部署适配 FutuOpenD 网关

---

## [v4.0] - 2026-02-24

### 最终打磨

### Added

- **测试确定性**: 单元测试强制使用静态 Mock Data，禁止网络依赖
- **Agent 上下文保护**: `.claudeignore` / `.cursorignore` 隔离大文件
- **每日流水线幂等性**: 中断重启不产生副作用
- **Dummy Feature 噪声哨兵**: 第二道过拟合检测
- **Hash 覆盖 RawClose + 隐含调整因子**: 堵住拆股识别链路审计盲区
- **漂移阈值 universe 自适应**: `max(10, 1% × UniverseSize)`
- **幸存者偏差区间估算**: 平稳期/危机期分别披露
- **信号不一致分类字段**: 加速 debug

### Fixed

- CPCV 配置：n_splits 从 10 降至 6，确保 fold_size > gap
- Embargo：从 60d 降至 40d（R27-B1），保证训练数据充足

---

## [v3.0] - 2026-02-23

### Hash 冻结 + 漂移检测

### Added

- **Hash 冻结机制**: 数据完整性验证
- **成本模型参数口径**: 统一定义
- **成分变更日规则**: 实际生效日期
- **Kelly 病态防护**: 独立 Kelly + 总量归一化
- **三条高价值验收项**
- **Agent 防御性原则**: 依赖图约束、禁止迎合测试

---

## [v2.0] - 2026-02-22

### 数据合约 + 事件协议

### Added

- **数据合约**: 复权、分红、公司行为、可交易性定义
- **事件生成协议**: Triple Barrier 事件定义 + 并发规则
- **单模型先行**: LightGBM MVP
- **Meta-Labeling 条件化**: 样本 ≥ 5000 启用
- **成本校准闭环**: 周度校准
- **模型换代门控**: Feature Stability + Turnover/Exposure

---

## [v1.0] - 2026-02-21

### 项目初始化

### Added

- 项目骨架搭建
- yfinance 数据采集
- 基础特征工程
- Triple Barrier 标注
- LightGBM 训练脚本
- CPCV 验证框架
- Docker 化部署

---

## 审计历史

| 轮次 | 日期 | 状态 | 主要发现 | 整改 |
|------|------|------|----------|------|
| Q5-REG | 2026-03-01 | ✅ 完成 | fracdiff.py 重复常量定义导致 UnboundLocalError | 已修复 |
| R18 | 2026-03-01 | ✅ 完成 | P0 全部修复、P1 冗余特征、P2 稳定性 | 全部完成 |
| R17 | 2026-03-01 | ✅ 完成 | 日历错配、噪声确定性、备份源闭环 | 全部完成 |
| R16 | 2026-03-01 | ✅ 完成 | R15 回归修复、CPCV 配置修正 | 全部完成 |
| R15 | 2026-03-01 | ✅ 完成 | PBO 计算逻辑修正 | N1 完成 |
| R14 | 2026-03-01 | ✅ 完成 | PBO/DSR 算法、样本权重、BaseModel | A1-A9 全部完成 |
| EXT-Q | 2026-03-01 | ✅ 完成 | FracDiff 预计算、Early Stopping 隔离 | Q1-Q5 全部完成 |
| OR9-13 | 2026-02-28 | ✅ 完成 | 内审修复汇总 (P0-P2) | 全部完成 |
| OR5 | 2026-02-26 | ✅ 完成 | Meta-Labeling 强制、FracDiff、CPCV 手写 | T1-T7 全部完成 |
| OR4 | 2026-02-25 | ✅ 完成 | Phase A/B 安全审计 | 2×误报、1×属实（已修） |
| OR3 | - | - | （跳过，合并至 OR4） | - |

---

## 版本规划

- **v4.4** (当前): R14-R18 内审修复完成，外部审核整改完成，165测试通过
- **v4.3** ✅: Phase C 完成，Meta-Labeling + FracDiff + CPCV 全部实现
- **v5.0** (计划): Phase C+ 双模型集成
- **v6.0** (计划): Phase D 风控系统
- **v7.0** (计划): Phase E 模拟盘
- **v8.0** (计划): Phase F 实盘

---

*Last updated: 2026-03-01*
