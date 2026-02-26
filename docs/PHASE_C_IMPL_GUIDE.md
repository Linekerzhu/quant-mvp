# Phase C 工程实施指南 (Implementation Guide)

> **定位**：本文档是 `plan.md` Phase C 段的逐行实施手册，供工程 Agent 直接执行。
> 架构决策与验收标准以 `plan.md v4.2` 为准，本文档只负责 HOW（怎么写代码）。

## 你需要知道的一切（2 分钟版）

### 项目是什么
美股日频量化交易系统。用 LightGBM 预测交易信号的盈亏概率，决定仓位大小。

### 现在到哪了
- **Phase A（数据管道）**: ✅ 完成。yfinance 拉数据 → 校验 → 特征工程 → 20 个特征。
- **Phase B（标签系统）**: ✅ 完成。Triple Barrier 打标 → 样本权重 → sklearn-ready 数据。
- **Phase C（模型训练）**: 🔴 现在开始。本文档就是 Phase C 的施工图纸。

### 核心架构约束（违反即一票否决）
这些是外部审计官签署的强制契约，不是建议，是红线：

1. **LightGBM 不能直接预测涨跌** → 必须用 Meta-Labeling 架构
2. **max_depth ≤ 3, num_leaves ≤ 7** → 已锁死在 training.yaml
3. **必须用分数阶差分（FracDiff）** → 不能喂绝对价格，也不能只用一阶差分
4. **必须手写 CPCV Purge+Embargo** → 不能用标准 KFold，不能用第三方库糊弄
5. **回测报告必须扣减** → CAGR -3%, MDD +10%（数据技术债惩罚）

### 施工顺序（严禁颠倒）

```
Step 1: Base Model（炮灰信号源）
   ↓
Step 2: CPCV 隔离器（手撕 PurgedKFold）
   ↓
Step 3: FracDiff 特征重构
   ↓
Step 4: Meta-MVP 闭环
```

---

## 前置任务：Push OR5 Hotfix

工作目录中有未提交的 OR5 审计热修复。**在开始 Phase C 之前必须先提交。**

```bash
cd /path/to/quant-mvp
git add -A
git commit -m "hotfix(OR5): Maximum Pessimism Principle + LGB反Kaggle硬化 + PhaseC契约

- triple_barrier: Gap execution + Collision detection + 止损优先
- training.yaml: max_depth=3, num_leaves=7, min_data_in_leaf=200
- 新增 test_smoke_or5.py (29 tests) + OR5_CONTRACT.md
- 112/112 tests passing"
git push origin main
```

验证：`git log --oneline -1` 应显示 OR5 hotfix commit。

---

## Step 1: Base Model（炮灰信号源）

### 目标
写一个极简规则策略，为每个交易日的每只股票生成方向信号 `side ∈ {+1, -1, 0}`。
这个策略**不需要赚钱**——它只是 Meta Model 的输入信号源。

### 新建文件

**`src/signals/base_models.py`**

```
接口定义:

class BaseModelSMA:
    """双均线金叉/死叉信号"""
    
    def __init__(self, fast_window=20, slow_window=60):
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入: 含 symbol, date, adj_close 的 DataFrame
        输出: 同 DataFrame，新增 'side' 列
              side = +1: 快均线 > 慢均线（看多）
              side = -1: 快均线 < 慢均线（看空）
              side =  0: 数据不足（冷启动期）
        
        关键约束:
        - 均线必须用 .shift(1)，即 T 日信号只能用 T-1 及之前的数据
        - 不能偷看 T 日的收盘价来决定 T 日的信号
        """
```

### 需要实现的 Base Model（至少 2 个，后续可扩展）

| 模型 | 逻辑 | side=+1 条件 | side=-1 条件 |
|------|------|-------------|-------------|
| SMA Cross | 20/60 双均线 | SMA20 > SMA60 | SMA20 < SMA60 |
| Momentum | 20 日动量 | returns_20d > 0 | returns_20d < 0 |

### 核心防泄漏检查
```python
# ❌ 错误：T 日信号用了 T 日价格
sma_fast = df['adj_close'].rolling(20).mean()
df['side'] = np.where(sma_fast > sma_slow, 1, -1)

# ✅ 正确：T 日信号只能用 T-1 及之前
sma_fast = df['adj_close'].shift(1).rolling(20).mean()
df['side'] = np.where(sma_fast > sma_slow, 1, -1)
```

### 与 Triple Barrier 的对接

Base Model 产生 side 后，只有 `side != 0` 的日期才触发 Triple Barrier 打标。
在 `triple_barrier.py` 的 `_is_valid_event` 中增加一个检查：

```python
# 如果 df 有 'side' 列，只在 side != 0 时触发事件
if 'side' in symbol_df.columns:
    if symbol_df.loc[idx, 'side'] == 0:
        return False, 'no_signal'
```

打标后的 label 含义变化：
- 旧：label=1 表示"价格上涨到止盈"
- 新（Meta-Labeling）：label=1 表示"**Base Model 这次信号赚了钱**"，label=0 表示"亏了"

### 测试标准
```
tests/test_base_models.py:
  - test_sma_signal_no_lookahead: 验证 shift(1)
  - test_signal_values: side ∈ {-1, 0, +1}
  - test_cold_start: 前 slow_window 天 side=0
  - test_signal_with_pipeline: base_model → triple_barrier 能跑通
```

### 交付物
- `src/signals/base_models.py`（2 个 Base Model）
- `tests/test_base_models.py`（4+ 个测试）
- 全量测试通过

---

## Step 2: CPCV 隔离器（最硬的骨头）

### 目标
手写 `CombinatorialPurgedKFold`，确保训练集和验证集之间零信息泄漏。

### 为什么不能用标准 KFold
金融数据有时间依赖性。标准 KFold 会把 2024 年 3 月的数据放进训练集，
然后用 2024 年 2 月的数据做验证——模型偷看了未来。

### 新建文件

**`src/models/purged_kfold.py`**

```
核心接口:

class CombinatorialPurgedKFold:
    """
    AFML Ch7: Combinatorial Purged K-Fold Cross-Validation
    
    参数 (从 config/training.yaml 读取):
        n_splits: 6          # 将时间线切成 6 段
        n_test_splits: 2     # 每次选 2 段做测试
        purge_window: 10     # 天 (= max_holding_days)
        embargo_window: 40   # 天
    
    组合数: C(6,2) = 15 条 CPCV path
    """
    
    def __init__(self, config_path="config/training.yaml"):
        ...
    
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.array, np.array]]:
        """
        输入: 含 date, label_exit_date 的 DataFrame（必须已经过 triple_barrier 打标）
        
        产出: (train_indices, test_indices) 的迭代器，共 15 组
        
        每组的隔离规则:
        1. 时间线切成 6 段，选 2 段做 Test
        2. Purge: 从 Train 中删除所有满足以下条件的样本:
           样本的 [entry_date, label_exit_date] 区间
           与 Test 集的 [min_date - max_lookback, max_date] 有任何一天交集
        3. Embargo: Test 集 max_date 之后 40 天内的 Train 样本，全部删除
        
        关键:
        - 用 label_exit_date（精确的退出日期），不是用 max_holding_days 近似
        - 每个 path 的有效训练天数必须 ≥ 200 天，否则标记为 invalid
        """
    
    def get_n_paths(self) -> int:
        """返回 15"""
        return comb(self.n_splits, self.n_test_splits)
```

### Purge 的精确算法（逐样本）

```python
def _purge(self, train_indices, test_df, full_df):
    """
    对 train_indices 中的每个样本:
    
    1. 取该样本的 entry_date 和 exit_date = label_exit_date
    2. 取 test_df 的时间范围: test_start = test_df['date'].min()
                              test_end = test_df['date'].max()
    3. 计算 test 的特征回溯边界: test_lookback_start = test_start - max_lookback_days
    
    4. 如果样本区间 [entry_date, exit_date] 与 
       [test_lookback_start, test_end] 有任何交集:
       → 从 train 中删除该样本
    
    交集判定: entry_date <= test_end AND exit_date >= test_lookback_start
    """
```

### Embargo 的精确算法

```python
def _embargo(self, train_indices, test_end_date, full_df):
    """
    从 train_indices 中删除所有:
    full_df.loc[idx, 'date'] 在 (test_end_date, test_end_date + embargo_window] 内的样本
    """
```

### 自证方法（审计官会检查）

实现完成后，必须能输出以下日志：

```
CPCV Path 1/15: Test=[fold_2, fold_5]
  Test range: 2021-03-15 ~ 2021-09-30
  Train before purge: 2847 samples
  Purged: 312 samples (overlap with test label periods)
  Embargoed: 89 samples (within 40d after test end)
  Train after purge: 2446 samples (effective 487 days)
  ✅ Valid (>= 200 days)
```

### 测试标准
```
tests/test_cpcv.py:
  - test_no_temporal_overlap: 验证 train 和 test 无时间交集
  - test_purge_removes_overlapping_labels: 模拟已知重叠，确认被删除
  - test_embargo_gap: 验证 test_end 后 40 天内无 train 样本
  - test_all_paths_valid: 15 条 path 有效训练天数均 ≥ 200
  - test_purge_uses_real_exit_date: 确认使用 label_exit_date 而非 max_holding_days
```

### 交付物
- `src/models/purged_kfold.py`
- `tests/test_cpcv.py`（5+ 个测试）
- 全量测试通过

---

## Step 3: FracDiff 特征重构

### 目标
用分数阶差分替代粗暴的对数收益率，在保持平稳性的同时保留时序记忆。

### 背景知识（1 分钟版）
- 一阶差分 (d=1): 绝对平稳，但抹杀所有记忆（如 returns_5d）
- 零阶差分 (d=0): 保留全部记忆，但非平稳（如裸价格）
- 分数阶差分 (0<d<1): 折中——找到最小的 d 使得序列刚好平稳

美股日频数据的最优 d 通常在 **0.35 ~ 0.65** 之间。

### 新建文件

**`src/features/fracdiff.py`**

```
核心接口:

def fracdiff_fixed_window(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """
    固定窗口分数阶差分。
    
    参数:
        series: 价格序列（如 adj_close）
        d: 差分阶数，0 < d < 1
        window: 权重截断窗口（默认 100 天）
    
    返回:
        差分后的序列（前 window-1 个值为 NaN）
    
    算法:
        weights[0] = 1
        weights[k] = weights[k-1] * (d - k + 1) / k    (k = 1, 2, ..., window-1)
        fracdiff[t] = sum(weights[k] * series[t-k] for k in range(window))
    """

def find_optimal_d(
    series: pd.Series, 
    d_range: np.arange = np.arange(0.0, 1.05, 0.05),
    significance: float = 0.05
) -> float:
    """
    二分法 / 网格搜索找最小 d，使 ADF 检验 p < significance。
    
    关键约束:
    - 必须只在 TRAIN 集上运行（不能用 test 数据拟合 d）
    - 返回满足平稳性的最小 d（保留最大记忆）
    
    返回:
        最优 d 值（如 0.45）
    """
```

### 与现有特征工程的集成

在 `src/features/build_features.py` 中新增 FracDiff 特征：

```python
# 在 build_features() 中，对 adj_close 施加 FracDiff
# d 值在 CPCV 的每个 fold 内独立拟合（防止信息泄漏）
# 
# 新增特征列:
#   fracdiff_close: FracDiff(adj_close, d=optimal_d)
#
# 注意: d 的拟合属于 Phase C 训练循环的一部分，不是 build_features 的一部分
# build_features 只负责「给定 d 值，计算 fracdiff」
# d 的搜索在训练脚本中完成
```

### ADF 检验用法

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(fracdiff_series.dropna(), maxlag=1, regression='c')
p_value = result[1]
is_stationary = p_value < 0.05
```

### 测试标准
```
tests/test_fracdiff.py:
  - test_d_zero_is_original: FracDiff(d=0) ≈ 原序列
  - test_d_one_is_diff: FracDiff(d=1) ≈ 一阶差分
  - test_optimal_d_stationary: 找到的 d 使 ADF p < 0.05
  - test_memory_preserved: d < 1 时，与原序列的相关性 > 0（记忆保留）
  - test_no_future_leakage: FracDiff[t] 只使用 t 及之前的数据
```

### 交付物
- `src/features/fracdiff.py`
- `tests/test_fracdiff.py`（5+ 个测试）
- `pip install statsmodels --break-system-packages`（ADF 检验依赖）

---

## Step 4: Meta-MVP 闭环

### 目标
将前 3 步的所有组件串联成完整的训练-验证-输出管道。

### 新建文件

**`src/models/meta_trainer.py`**

```
核心流程:

class MetaTrainer:
    """
    Meta-Labeling 训练管道。
    
    完整流程:
    1. 加载 Phase A-B 产出的特征+标签数据
    2. Base Model 生成方向信号 side
    3. 过滤: 只保留 side != 0 的样本
    4. 标签转换: {profit → 1, loss → 0}（Meta-Label: 信号是否盈利）
    5. 对每个 CPCV fold:
       a. 在 train 集上用二分法找最优 FracDiff d
       b. 用该 d 值计算 train 和 test 的 FracDiff 特征
       c. 训练 LightGBM (从 training.yaml 读参数)
       d. 在 test 集上预测概率 p
    6. 汇总 15 条 path 的结果
    7. 输出: 概率校准曲线、AUC、PBO 估计
    """
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        返回:
        {
            'paths': [...],          # 15 条 path 的 AUC / Accuracy
            'mean_auc': float,       # 平均 AUC
            'pbo': float,            # Probability of Backtest Overfitting
            'feature_importance': {}, # SHAP 或 MDA
            'dummy_sentinel': {      # 哨兵检查
                'dummy_rank': int,
                'passed': bool
            }
        }
        """
```

### LightGBM 调用方式

```python
import lightgbm as lgb
import yaml

with open('config/training.yaml') as f:
    cfg = yaml.safe_load(f)

lgb_params = cfg['lightgbm']
# lgb_params 已经包含 max_depth=3, num_leaves=7 等硬化参数

train_data = lgb.Dataset(
    X_train, 
    label=y_train,           # Meta-Label: 1=信号盈利, 0=信号亏损
    weight=w_train            # AFML uniqueness 权重（已在 Phase B 计算好）
)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=lgb_params.pop('n_estimators', 500),
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(lgb_params.pop('early_stopping_rounds', 50))]
)

# 输出概率
proba = model.predict(X_test)  # P(base signal is profitable)
```

### 哨兵检查（必须通过才算有效）

```python
# 1. Dummy Feature Sentinel
# dummy_noise 在 Phase A 已注入特征，如果它排名进入 top 25%，说明过拟合
importance = model.feature_importance(importance_type='gain')
dummy_rank = ...  # dummy_noise 的排名
assert dummy_rank > len(features) * 0.25, "Overfitting detected!"

# 2. PBO (Probability of Backtest Overfitting)
# 15 条 CPCV path 中，如果超过 50% 的 path 在 test 上表现不如随机
# 则 PBO > 0.5，必须拒绝模型
pbo = calculate_pbo(path_results)
assert pbo < 0.5, f"PBO={pbo:.2f} >= 0.5, model rejected!"
```

### 回测报告扣减（硬编码）

```python
# 最终输出报告时:
SURVIVORSHIP_CAGR_PENALTY = 0.02
LOOKAHEAD_CAGR_PENALTY = 0.01
MDD_INFLATION = 0.10

report['adjusted_cagr'] = report['raw_cagr'] - SURVIVORSHIP_CAGR_PENALTY - LOOKAHEAD_CAGR_PENALTY
report['adjusted_mdd'] = report['raw_mdd'] + MDD_INFLATION
# 展示时必须用 adjusted 值
```

### 测试标准
```
tests/test_meta_trainer.py:
  - test_full_pipeline_runs: 合成数据跑通完整流程
  - test_meta_label_binary: 标签只有 0 和 1
  - test_sample_weight_passed: LGB 接收到 sample_weight
  - test_dummy_sentinel_catches_overfit: 人造过拟合场景触发哨兵
  - test_cpcv_15_paths: 确认产出 15 条 path
  - test_lgb_params_from_config: 参数从 YAML 读取，不是硬编码
```

### 交付物
- `src/models/meta_trainer.py`
- `tests/test_meta_trainer.py`（6+ 个测试）
- 全量测试通过

---

## 文件结构总览（Phase C 完成后）

```
src/
├── signals/
│   └── base_models.py          # Step 1: SMA Cross + Momentum
├── features/
│   ├── build_features.py       # 已有，Phase A
│   └── fracdiff.py             # Step 3: 分数阶差分
├── models/
│   ├── purged_kfold.py         # Step 2: CPCV 手写隔离器
│   └── meta_trainer.py         # Step 4: Meta-Labeling 训练管道
├── labels/
│   ├── triple_barrier.py       # 已有，Phase B (含 OR5 hotfix)
│   └── sample_weights.py       # 已有，Phase B
tests/
├── test_base_models.py         # Step 1 测试
├── test_cpcv.py                # Step 2 测试
├── test_fracdiff.py            # Step 3 测试
└── test_meta_trainer.py        # Step 4 测试
```

---

## 里程碑检查点

| 检查点 | 完成标志 | 阻断条件 |
|--------|---------|---------|
| Step 1 完成 | base_model → triple_barrier 跑通，side 无前视 | side 用了 T 日价格 |
| Step 2 完成 | 15 条 CPCV path，全部 ≥ 200 天训练数据 | train/test 有时间交集 |
| Step 3 完成 | ADF p < 0.05 且与原序列相关性 > 0 | d 在 test 集上拟合 |
| Step 4 完成 | 15 path AUC 汇总 + PBO < 0.5 + 哨兵通过 | PBO ≥ 0.5 |

**每个 Step 完成后都要跑全量测试确认不回归，然后单独 commit + push。**

---

## 如果遇到问题

| 问题 | 对策 |
|------|------|
| FracDiff 后 ADF 仍不平稳 | 增大 d 值或增大窗口 window |
| CPCV purge 后训练数据太少 | 检查 embargo_window 是否过大，或数据总量不足 |
| LightGBM AUC ≈ 0.50 | 正常。日频美股信噪比极低，0.52 就值得认真对待 |
| PBO > 0.5 | 模型过拟合，需要进一步压制（减 num_leaves、增 min_data_in_leaf） |
| dummy_noise 进入 top 25% | 过拟合确认，拒绝当前模型，回查特征 |
