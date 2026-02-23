# plan.md v4 补丁备忘录

> 本文档收录两位外部专家对 v4 的最终微调建议。这些建议不涉及架构变更，均为执行层细节补强。
> Agent 在实现对应模块时，应将下列条款视为 plan.md 的补充约束。

---

## 补丁 1：数据源切换时 OHLC 特征降级（影响 §3.5 + Phase A）

**原文**："备源若不提供复权 OHLC，则仅使用 Adj Close 计算收益率，OHLC 相关特征标记为'近似'。"

**补丁**：标记"近似"不够安全——不同源对 Adj OHLC 的定义不一致，切换后会导致特征分布漂移。改为硬规则：

- 当发生数据源切换且备源缺 AdjOHLC 时，**禁用所有依赖 OHLC 的特征**（ATR、K 线形态等），仅保留基于 AdjClose 和 Volume 的特征。
- 写入事件日志（level=WARN，type=`feature_degradation`），标注被禁用的特征列表。
- 主源恢复后自动恢复全特征集，同样写入日志。

---

## 补丁 2：Dummy Feature 哨兵阈值收紧（影响 Phase B/C）

**原文**："Gain Importance 排名或 SHAP 排名进入前 50%"

**补丁**：改为双门槛，任一满足即阻塞：

- **排名门槛**：`dummy_noise` 进入前 **25%**（而非 50%）。
- **相对贡献门槛**：`dummy_gain / median_real_feature_gain > 1.0`（dummy 比真实特征中位数还重要）。

这样即使未来特征数扩展，哨兵灵敏度也不会被稀释。

---

## 补丁 3：时间打乱哨兵统计稳健化（影响 Phase B）

**原文**："打乱后 AUC > 0.55，判定泄漏。"

**补丁**：单次 AUC 在小样本下会抖动。改为：

- 运行 **n=5 次**不同随机种子。
- 阻塞条件：**均值 > 0.55** 或 **任一次 > 0.58**。

---

## 补丁 4：幂等执行的部分成交语义（影响 Phase E）

**原文**："若订单已提交，重启时先查询订单状态，不重复提交。"

**补丁**：需覆盖 `partially_filled` 边界。补充规则：

- 每个 `(trade_date, symbol)` 对应唯一的 `intent_id`（幂等键）。
- 订单状态为 `partially_filled` 时：允许 **至多一次** "撤单-重挂"操作，新订单继承原 `intent_id`，事件日志记录原 `order_id` → 新 `order_id` 的映射及原因。
- 任何重启都必须先查询 `intent_id` 对应的活跃订单，**禁止产生同一 intent 的重复订单**。

---

## 补丁 5：成分变更日志增加来源字段（影响 §3 + Phase A）

**原文**："手动录入并记录进事件日志。"

**补丁**：日志 payload 中增加：

```json
{
  "rebalance_date": "2025-06-20",
  "rebalance_date_source": "manual",
  "evidence": "S&P Dow Jones Indices press release 2025-06-06, URL: https://..."
}
```

---

## 补丁 6：Kelly 输入估计窗口与护栏（影响 Phase D）

**原文**：`f_i = p_i / a_i - (1-p_i) / b_i`

**补丁**：补充 p/a/b 的估计来源与护栏：

- p、a、b 仅基于 **最近 M 笔已平仓交易**（默认 M=50）或 **最近 6 个月**，取样本量更小者。
- 若已平仓交易 < 20 笔，退化为 **等权固定仓位**（每标的 = 目标波动率 / 标的波动率，再归一化），不使用 Kelly。
- 参数 `kelly_min_trades` 写入 `config/position_sizing.yaml`。

---

## 补丁 7：yfinance 防封禁（影响 Phase A ingest.py）

- `ingest.py` 必须实现 **指数退避重试**（Exponential Backoff）：初始等待 1s，最大等待 60s，最多重试 5 次。
- 开发/调试阶段使用 `requests-cache`（SQLite 后端）缓存 API 响应，避免重复请求。
- 批量请求间隔 ≥ 0.5s。

---

## 补丁 8：Parquet 原子写入（影响所有数据落盘）

- 所有 Parquet 写入采用 **Write-Audit-Publish (WAP)** 模式：
  1. 写入 `{filename}.tmp`
  2. 校验文件可读 + 行数/schema 正确
  3. `os.rename("{filename}.tmp", "{filename}")` 原子替换
- 若步骤 2 失败，删除 `.tmp` 并写入事件日志（level=ERROR）。

---

## 补丁 9：实盘账户类型强制约束（影响 §8 + Phase F）

- 实盘 **强制使用 Margin 账户**（Alpaca 最低 $2,000）。
- **禁止使用 Cash 账户**：Cash 账户虽无 PDT 限制，但受 T+1 交收约束，频繁周转易触发 Good Faith Violation (GFV)。
- PDT 风控由系统内置的 `pdt_guard.py` 负责拦截。
