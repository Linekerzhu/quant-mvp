# Phase A-B 修复计划 - 已完成 ✅

## 修复完成清单

### Phase A 修复 ✅
- [x] **A19**: 实现 Tiingo 备源 (ingest.py) - 自动故障转移 + 功能降级日志
- [x] **A23**: WAP 全覆盖 (daily_job.py, wap_utils.py) - 所有 Parquet 写入使用原子写入
- [x] **A26**: 修复缺失值处理逻辑 (validate.py) - 正确识别连续 NaN
- [x] **A18**: 创建 data_sources.yaml - 双数据源配置

### Phase B 修复 ✅
- [x] **B16**: 事件不重叠约束 (triple_barrier.py) - 严格实现 §6.5 协议
- [x] **B18**: 特征 NaN 跳过机制 (build_features.py) - 添加 features_valid 标记
- [x] **B11**: 创建 feature_importance.py - IC 追踪 + 漂移检测
- [x] **B12**: 创建 feature_stability.py - 特征稳定性门控
- [x] **B13**: 创建 test_overfit_sentinels.py - 双哨兵测试框架
- [x] **B19**: 优化样本权重算法 (sample_weights.py) - O(n log n) 复杂度

### 新增文件
- ✅ src/data/wap_utils.py - WAP 工具函数
- ✅ src/features/feature_importance.py
- ✅ src/features/feature_stability.py
- ✅ config/data_sources.yaml
- ✅ tests/test_overfit_sentinels.py

## 提交信息
fix: Phase A-B 审计问题全面修复

- B16: 实现事件不重叠约束，确保同一标的持仓期内不开新事件
- B18: 特征 NaN 处理改为标记跳过机制，不填 0
- B11/B12: 新增 feature_importance.py 和 feature_stability.py
- B13: 新增双哨兵测试框架 (Dummy Feature + Time Shuffle)
- B19: 优化样本权重为向量化算法
- A19: 实现 Tiingo 备源 + 自动故障转移
- A23: 所有 Parquet 写入使用 WAP 模式
- A26: 修复缺失值连续检测逻辑
- A18: 新增 data_sources.yaml 配置
