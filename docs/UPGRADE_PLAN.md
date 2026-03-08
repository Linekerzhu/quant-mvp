# Quant-MVP 升级计划 v1.0
## 从 MVP 到生产级量化系统

> 制定日期: 2026-03-08  
> 当前阶段: Phase F（Expert Oracle 已完成）  
> 下一阶段: Phase G（Walk-Forward 回测验证）

---

## 架构总览

```mermaid
graph LR
    G[Phase G 回测验证] --> H[Phase H 模型升级]
    H --> I[Phase I 信号扩展]
    I --> J[Phase J 实盘上线]
    
    style G fill:#ff6b6b,color:#fff
    style H fill:#feca57,color:#333
    style I fill:#48dbfb,color:#333
    style J fill:#0abde3,color:#fff
```

> [!IMPORTANT]
> **Phase G 是绝对门槛**。如果回测结果不达标，不得进入后续阶段。
> 这不是流程形式主义——是防止用真金白银验证一个不赚钱的系统。

---

## Phase G: Walk-Forward 回测验证（预计 1-2 周）

### 目标
用 2024-01 至 2025-12 的历史数据，模拟 daily_job.py 的完整流程，生成样本外净值曲线。

### 技术方案

**新建文件**: `src/backtest/walk_forward.py`

```
数据准备:
  下载 2023-01 至 2025-12 全量 S&P500 OHLCV (yfinance)
  存储: data/backtest/daily_2023_2025.parquet

回测流程 (逐日模拟):
  for each trading_day in 2024-01-02 ... 2025-12-31:
    1. 取 day 前 252 天数据 → features
    2. 生成 SMA + Momentum 信号
    3. Meta-Label 置信度 (用当前已训练模型)
    4. Kelly 仓位 + 风控
    5. 虚拟执行 (T+1 Open 价格)
    6. 扣除 Futu 真实费用结构
    7. 记录 NAV、持仓、交易

输出:
  data/backtest/equity_curve.parquet
  data/backtest/trade_log.parquet
  data/backtest/metrics.json
```

### 技术标准

| 指标 | 最低达标线 | 理想目标 | 一票否决线 |
|------|-----------|---------|-----------|
| 年化 Sharpe (扣费后) | > 0.5 | > 1.0 | < 0 |
| 年化收益率 | > 5% | > 15% | < 0% |
| 最大回撤 | < 25% | < 15% | > 35% |
| 胜率 | > 48% | > 52% | < 40% |
| 盈亏比 | > 1.0 | > 1.3 | < 0.8 |
| 日均换手率 | < 30% | < 15% | > 50% |
| Calmar Ratio | > 0.3 | > 1.0 | < 0 |

### 风险控制

- **禁止偷看未来数据**: 特征计算只用 T 日及之前的数据
- **存活者偏差**: 用动态的 S&P500 成分股（每季度更新），不能用当前成分股回测过去
- **成本真实性**: 必须用 `config/training.yaml` 中的 Futu 费率结构，不能用零成本
- **冷启动**: 前 60 天（模型暖机期）不计入绩效统计

### Go/No-Go 检查点

```
□ 样本外 Sharpe > 0 ?
  → YES: 进入 Phase H
  → NO:  分析原因，可能需要重做 Phase C (模型训练)
  
□ 最大回撤 < 35% ?
  → YES: 继续
  → NO:  检查风控参数是否合理，调整后重跑

□ 换手率 < 50% ?
  → YES: 继续  
  → NO:  信号太频繁，费用吃掉利润，需要调整信号频率
```

---

## Phase H: 模型升级与重训练（预计 2-3 周）

### 前置条件
Phase G 已通过 Go/No-Go。

### H1: 特征利用率提升

**问题**: 系统已计算 20+ 个特征，但模型只用了 3 个。

**方案**: 扩展元模型的特征集，利用已有的特征储备。

```python
# 当前模型输入 (3个)
current_features = ["volume_7d", "ret_1d", "fracdiff"]

# 升级后 (12-15个，从已有特征中选取)
upgraded_features = [
    # 动量族 (已有)
    "returns_5d", "returns_20d", "returns_60d",
    # 波动率族 (已有)
    "rv_5d", "rv_20d", "atr_20",
    # 均值回归族 (已有)
    "rsi_14", "price_vs_sma20_zscore", "price_vs_sma60_zscore",
    # 趋势族 (已有)
    "macd_line_pct", "adx_14",
    # 量价族 (已有)
    "relative_volume_20d", "pv_divergence_bull",
    # 市场环境 (已有)
    "regime_combined", "vix_change_5d",
    # 分数阶差分 (已有)
    "fracdiff",
]
```

### H2: 训练集扩大

**问题**: 当前只有 ~100 条训练样本。

**方案**:
```
1. 使用 2023-01 至 2025-06 数据重新生成 Triple Barrier 标签
2. 跨全部 517 只股票 × ~500 个交易日 = 潜在 ~258,000 条样本
3. 过滤 label=0 后，预计保留 ~150,000 条
4. 使用 Purged K-Fold CV (config 中已配置 n_splits=6)
5. 训练新的 meta_model_v2
```

### H3: 模型训练参数 (已有配置，直接使用)

```yaml
# config/training.yaml 中已锁定的参数 (不修改)
lightgbm:
  max_depth: 3        # LOCKED
  num_leaves: 7       # LOCKED  
  min_data_in_leaf: 100
  feature_fraction: 0.5
  learning_rate: 0.01
  n_estimators: 500
  early_stopping_rounds: 50
```

### 技术标准

| 指标 | 标准 |
|------|------|
| 训练样本量 | ≥ 50,000 条 |
| CPCV 平均 AUC | > 0.55 |
| Dummy 特征排名 | 不在 Top 25% |
| 时间乱序哨兵 AUC | < 0.55 |
| PBO (过拟合概率) | < 0.30 |

### 风险控制

- **过拟合防线**: 
  - `training.yaml` 中已有 OR5 红线参数 (max_depth=3 锁死)
  - Dummy noise 哨兵 + 时间乱序哨兵双重检测
  - PBO < 0.30 硬门槛
- **回归测试**: 新模型 v2 必须在 Phase G 回测框架上重跑，Sharpe 不得比 v1 差
- **A/B 切换**: 线上保留 v1 模型作为 fallback，v2 先跑虚拟盘验证 30 天

---

## Phase I: 信号源扩展（预计 2-4 周）

### 前置条件
Phase H 模型升级完成，新模型通过回测验证。

### I1: 新增均值回归信号

**文件**: `src/signals/base_models.py` 新增 `BaseModelMeanReversion`

```python
@SignalModelRegistry.register('mean_reversion')
class BaseModelMeanReversion(BaseSignalGenerator):
    """
    Bollinger Band Mean Reversion Signal
    - BUY:  price < lower band (超卖反弹)
    - SELL: price > upper band (超买回落)
    """
    def __init__(self, window=20, num_std=2.0): ...
```

**原理**: 与现有的 SMA + Momentum（趋势跟踪）信号正交。在震荡市中，趋势信号失效时均值回归能补位。

### I2: 信号合成策略改进

**问题**: 当前 `.groupby().sum()` 导致 76 只股票信号湮灭。

**改进方案**: 合并前先做投票过滤。

```python
# daily_job.py _step_risk_and_size 中
# 旧: positions = sizer.calculate_positions(signals)
#     positions = positions.groupby('symbol')['target_weight'].sum()

# 新: 先投票，再sizing
def merge_signals(signals):
    """投票制: 只保留所有模型方向一致的信号"""
    votes = signals.groupby('symbol')['side'].agg(['sum', 'count'])
    # 全票通过: |sum| == count (所有模型同向)
    unanimous = votes[votes['sum'].abs() == votes['count']]
    return signals[signals['symbol'].isin(unanimous.index)]
```

### I3: Kronos Oracle 阈值动态化

**问题**: 当前否决阈值固定 `< -1%`，否决率 70%。

**改进**: 根据 VIX 水平动态调整。

```python
def dynamic_veto_threshold(vix_level):
    """VIX 高时放宽否决标准，VIX 低时收紧"""
    if vix_level > 25:     # 恐慌市
        return -0.03       # 只否决预测跌 > 3%
    elif vix_level > 18:   # 正常市
        return -0.01       # 否决预测跌 > 1% (当前)
    else:                  # 低波动市
        return -0.005      # 否决预测跌 > 0.5% (更严格)
```

### 技术标准

| 指标 | 标准 |
|------|------|
| 新信号与现有信号相关性 | < 0.3 (正交性) |
| 加入新信号后回测 Sharpe | ≥ 原 Sharpe × 0.95 (不能变差) |
| 信号湮灭率 | 从 40% 降至 < 15% |

### 风险控制

- **渐进式**: 新信号先只参与投票，不独立产生仓位
- **回测验证**: 每加入一个信号源，重跑 Phase G 回测
- **可回滚**: 新信号有独立的开关（`config/features.yaml` 中配置）

---

## Phase J: 实盘上线（预计 2-4 周）

### 前置条件
- Phase G-I 全部完成
- 虚拟交易连续运行 ≥ 60 天
- 虚拟交易 Sharpe > 0.5

### J1: 安全检查清单

```
上线前必须满足:
□ 60+ 天虚拟交易记录
□ 样本外 Sharpe > 0.5 (扣费后)
□ 最大回撤 < 20%
□ Kronos 否决率在 30%-80% 之间 (极端值说明系统异常)
□ Futu OpenD 连接稳定 ≥ 7 天
□ 每笔订单设置硬止损 (-3% per position)
□ 单日最大损失熔断 (-2% NAV)
□ 总账户最大损失熔断 (-10% NAV)
```

### J2: 分级上线策略

```
Week 1-2: 侦察阶段
  资金: $5,000
  仓位: max_single = 5% ($250)
  交易: 每天最多 3 笔
  目标: 验证滑点模型准确性

Week 3-4: 验证阶段
  资金: $10,000  
  仓位: max_single = 8%
  交易: 每天最多 5 笔
  条件: Week 1-2 滑点 < 预期的 1.5 倍

Month 2-3: 放量阶段
  资金: $25,000
  仓位: max_single = 10% (正常配置)
  交易: 按系统计算
  条件: 累计 Sharpe > 0
```

### J3: 实盘风控硬件

```yaml
# config/risk_limits.yaml 新增
live_trading:
  # 单笔止损 (per position)
  stop_loss_pct: 0.03       # -3% 强制平仓
  
  # 单日最大损失 (portfolio level)
  daily_loss_limit_pct: 0.02  # -2% NAV 全部平仓并暂停
  
  # 累计最大损失 (kill switch)
  total_loss_limit_pct: 0.10  # -10% NAV 永久停机等待人工
  
  # 流动性检查
  min_adv_usd: 5000000      # 日均成交额 > $5M
  max_participation_pct: 0.01  # 单笔不超过日成交量 1%
  
  # 执行超时
  order_timeout_seconds: 60
  max_slippage_pct: 0.005    # 滑点 > 0.5% 取消订单
```

### 风险预算

```
总风险预算: 初始资金的 10% ($2,500 on $25K)
  └── 模型风险: 5%  (系统性预测失败)
  └── 执行风险: 2%  (滑点、延迟)
  └── 技术风险: 2%  (宕机、API 故障)
  └── 黑天鹅: 1%   (市场熔断等)
```

---

## 时间线总览

```
2026-03 W2  ┃ Phase G: 下载历史数据 + 搭建回测框架
2026-03 W3  ┃ Phase G: 运行回测 + 分析结果
            ┃ ── Go/No-Go 检查点 ──
2026-03 W4  ┃ Phase H: 扩展特征集 + 重新生成标签
2026-04 W1  ┃ Phase H: 训练 v2 模型 + CPCV 验证
2026-04 W2  ┃ Phase H: 回归回测验证
            ┃ ── Go/No-Go 检查点 ──
2026-04 W3  ┃ Phase I: 新增信号源 + 合成策略改进
2026-04 W4  ┃ Phase I: 回测验证新信号效果
2026-05 W1  ┃ Phase J: Futu 端到端测试
2026-05 W2  ┃ Phase J: $5K 侦察上线
2026-05 W3+ ┃ Phase J: 分级放量
```

---

## 每阶段交付物

| Phase | 代码交付 | 数据交付 | 文档交付 |
|-------|---------|---------|---------|
| G | `src/backtest/walk_forward.py` | equity_curve, trade_log, metrics | 回测报告 |
| H | 更新 `_step_signals`, 新 model_v2 | 新训练集, CV 结果 | 模型卡片 |
| I | 新 signal class, 改 merge 逻辑 | 对比回测结果 | 信号研报 |
| J | 实盘风控参数, 熔断逻辑 | 实盘交易日志 | 运维手册 |
