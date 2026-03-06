# 量化交易项目 - 执行蓝图锁定

**项目**: 美股日频量化MVP系统  
**目标**: 构建可持续运行的量化交易系统，养活自己（覆盖运行成本）  
**启动时间**: 2026-02-23  
**执行者**: 李得勤  

---

## 📋 蓝图文件

| 文件 | 版本 | 路径 |
|------|------|------|
| 主蓝图 | v4 | `~/quant-mvp/plan.md` |
| 补丁 | v4 patch | `~/quant-mvp/plan_v4_patch.md` |

**约束**: 所有实现必须同时满足 plan.md 和 plan_v4_patch.md 的要求

---

## 🎯 项目定位

- **市场**: 美股 S&P 500
- **频率**: 日频（T+1调仓）
- **核心**: LightGBM ML选股 + Triple Barrier标签 + CPCV验证
- **风控**: 双重过拟合哨兵 + Fractional Kelly仓位 + 多层风控闸门
- **执行**: Alpaca Paper → 小资金实盘

---

## 📊 分阶段进度跟踪

### Phase A: 基础设施与数据管道 [✅ 审计通过]
- [x] 项目骨架建立
- [x] 配置文件编写
- [x] 静态Mock数据集生成
- [x] 双数据源采集实现
- [x] PIT数据管理
- [x] 公司行为处理模块
- [x] Hash冻结+漂移检测
- [x] 数据质量校验
- [x] 股票池管理
- [x] 事件日志系统
- [x] 测试套件
- [x] Git初始提交
- [x] **深度审计完成** (docs/AUDIT_PHASE_A.md)

### Phase B: 特征工程与标签 [✅ 审计通过]
- [x] 多时间尺度特征（动量/波动率/成交量/均线偏离/市场状态）
- [x] Dummy Noise Feature注入（Plan v4关键）
- [x] Regime Detector（波动率+ADX）
- [x] Triple Barrier标签（按事件生成协议）
- [x] 样本唯一性加权（Uniqueness）
- [x] 双重哨兵测试框架（时间打乱 + Dummy Feature）
- [x] 特征版本登记机制
- [x] **深度安全审计完成** (docs/AUDIT_PHASE_B_SECURITY.md)

### Phase A-B 统筹审计 [✅ 通过]
- [x] **集成审计完成** (docs/AUDIT_PHASE_AB_INTEGRATION.md)
  - [x] 数据流完整性验证
  - [x] 接口兼容性检查
  - [x] 配置一致性验证
  - [x] 端到端集成测试
  - [x] BusinessDay修复（交易日计算）

### Phase C: 建模与验证 [✅ 已完成]
- [x] LightGBM单模型
- [x] Dummy Feature哨兵
- [x] CPCV验证
- [x] PBO计算
- [x] Deflated Sharpe Ratio (z-score)
- [x] **审计通过** (docs/audit/RECHECK_AUDIT_BY_KOULIANCAI.md)

### Phase C+: 双模型集成 [条件扩展]
- [ ] CatBoost训练
- [ ] IC加权集成
- [ ] 模型换代门控

### Phase D: 信号过滤与风控 [✅已完成]
- [x] Meta-Labeling（条件启用 -> OR5强制）
- [x] Fractional Kelly仓位
- [x] 多层风控闸门
- [x] PDT合规守卫

### Phase E: 模拟盘运行 [4-6周，日历时间]
- [ ] Alpaca Paper Trading
- [ ] 显式滑点模型
- [ ] 每日流水线幂等性
- [ ] 成本模型周度校准
- [ ] 研究-交易信号一致性验证

### Phase F: 小资金实盘 [4-6周，日历时间]
- [ ] 极小资金上线（5%-10%）
- [ ] 每周复盘报告
- [ ] 月度重训
- [ ] 自动降级机制

---

## ⚠️ 关键约束（必须遵守）

1. **硬门控**: PBO >= 0.5 或 Deflated Sharpe <= 0 或 Dummy Feature哨兵触发 → 回退Phase B
2. **数据合约**: 违反 §3.5 等同于数据泄漏
3. **测试要求**: 零网络依赖，全部基于静态Mock数据 (当前: **165/165 通过** ✅)
4. **原子提交**: 每完成子任务必须Git Commit
5. **依赖图约束**: 禁止反向修改上游已冻结产出
6. **禁止迎合测试**: 测试不通过应修复实现，而非修改测试

---

## ✅ OR5 架构契约状态

| 契约 | 状态 |
|------|------|
| LightGBM参数锁死 | ✅ 已实现 |
| Meta-Labeling | ✅ 已实现 |
| FracDiff | ✅ 已实现 |
| CPCV手写 | ✅ 已实现 |
| 回测扣减 | ✅ 已实现 |

**契约文档**: `docs/OR5_CONTRACT.md`

---

## 📝 重要参数记忆

### 股票池配置
- min_history_days: 60
- min_adv_usd: 5,000,000
- min_listing_years: 2
- exit_limit_slippage: 0.005

### Triple Barrier参数
- atr_window: 20
- tp_mult: 2.0
- sl_mult: 2.0
- max_holding_days: 10

### Kelly仓位
- kelly_fraction: 0.25
- max_gross_leverage: 1.0
- target_annual_vol: 0.15

### 风控参数
- 单日亏损上限: 1.0%
- 组合最大回撤: 10%（降仓）/ 12%（Kill-Switch）
- 单票最大仓位: 10%
- 单行业暴露上限: 30%

---

## 🔧 技术栈

- Python 3.11+
- yfinance（主）+ Tiingo/Alpha Vantage（备）
- DuckDB + Parquet
- LightGBM + CatBoost（条件扩展）
- Alpaca Trade API
- Docker + docker-compose

---

*蓝图已锁定，即刻开工！*
