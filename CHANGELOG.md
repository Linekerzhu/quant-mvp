# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
| OR5 | 2026-02-26 | ✅ 完成 | Meta-Labeling 强制、FracDiff、CPCV 手写、Burn-in 陷阱 | T1-T7 全部完成 |
| OR4 | 2026-02-25 | ✅ 完成 | Phase A/B 安全审计 | 2×误报、1×属实（已修） |
| OR3 | - | - | （跳过，合并至 OR4） | - |

---

## 版本规划

- **v4.2** (当前): OR5 审计裁决，Phase C 准备就绪
- **v4.3** (计划): Phase C 完成，Meta-Labeling + FracDiff + CPCV 全部实现
- **v5.0** (计划): Phase C+ 双模型集成
- **v6.0** (计划): Phase D 风控系统
- **v7.0** (计划): Phase E 模拟盘
- **v8.0** (计划): Phase F 实盘

---

*Last updated: 2026-02-26*
