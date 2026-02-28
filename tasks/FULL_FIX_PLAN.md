# Phase C 整改计划

> **制定人**: 张得功
> **制定日期**: 2026-02-27
> **来源**: 四角审计汇总 (FULL_AUDIT_SUMMARY.md)

---

## 一、整改概览

| 优先级 | 问题数 | 总工时 |
|--------|--------|--------|
| P0 高 | 4 | 9h |
| P1 中 | 4 | 6h |
| **合计** | **8** | **15h** |

---

## 二、高优先级整改 (P0)

### P0-1: meta_trainer.py 职责拆分

| 项目 | 内容 |
|------|------|
| **问题** | `meta_trainer.py` 249行，混合训练/检测/惩罚逻辑 |
| **来源** | 张得功架构审计 |
| **负责人** | 张得功（设计）→ 李得勤（实现） |
| **工时** | 4h |
| **目标** | 拆分为 3 个模块 |

**拆分方案**:
```
src/models/
├── meta_trainer.py      # 核心训练逻辑（~120行）
├── overfitting_check.py # PBO/Dummy检测（~60行）
└── label_converter.py   # Meta-Label转换（~40行）
```

**验收标准**:
- [ ] 各模块单一职责
- [ ] 单元测试全部通过
- [ ] `run_pipeline.py` 无需修改调用

---

### P0-2: 配置依赖倒置

| 项目 | 内容 |
|------|------|
| **问题** | `purged_kfold.py` 内部读取配置文件，违反DIP |
| **来源** | 张得功架构审计 |
| **负责人** | 李得勤 |
| **工时** | 3h |
| **目标** | 改为构造函数参数注入 |

**整改方案**:
```python
# Before (错误)
class CombinatorialPurgedKFold:
    def __init__(self, config_path: str = ...):
        self._load_from_config(config_path)  # 硬编码读取

# After (正确)
class CombinatorialPurgedKFold:
    def __init__(self, n_splits=6, n_test_splits=2, 
                 purge_window=10, embargo_window=40, ...):
        self.n_splits = n_splits
        # 纯参数注入

# 调用层组装
config = load_config("config/training.yaml")
cpcv = CombinatorialPurgedKFold(**config['validation']['cpcv'])
```

**验收标准**:
- [ ] 类内部无 `yaml.load` 调用
- [ ] 19个CPCV测试全部通过
- [ ] `run_pipeline.py` 负责组装

---

### P0-3: Base Model 抽象基类

| 项目 | 内容 |
|------|------|
| **问题** | SMA/Momentum 无统一接口，无法静态检查 |
| **来源** | 张得功架构审计 |
| **负责人** | 李得勤 |
| **工时** | 2h |
| **目标** | 添加 `BaseSignalGenerator` 抽象基类 |

**整改方案**:
```python
# src/signals/base.py
from abc import ABC, abstractmethod

class BaseSignalGenerator(ABC):
    """信号生成器抽象基类"""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Returns:
            DataFrame with 'side' column: +1 (long), -1 (short), 0 (hold)
        """
        pass

# src/signals/base_models.py
from .base import BaseSignalGenerator

class BaseModelSMA(BaseSignalGenerator):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
```

**验收标准**:
- [ ] `BaseSignalGenerator` 定义在 `src/signals/base.py`
- [ ] SMA/Momentum 继承并实现抽象方法
- [ ] 静态类型检查通过

---

### P0-4: README 安装步骤补充

| 项目 | 内容 |
|------|------|
| **问题** | README安装步骤过于简略，新人难以上手 |
| **来源** | 赵连顺文档审计 |
| **负责人** | 赵连顺 |
| **工时** | 1h |
| **目标** | 补充完整安装指南 |

**整改内容**:
```markdown
## 安装步骤

### 1. 环境要求
- Python 3.9+
- pip 或 poetry

### 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

### 3. 安装依赖
pip install -r requirements.txt

### 4. 验证安装
python -c "from src.signals.base_models import BaseModelSMA; print('OK')"

### 5. 运行测试
pytest tests/ -v
```

**验收标准**:
- [ ] 包含 Python 版本要求
- [ ] 包含虚拟环境创建步骤
- [ ] 包含验证命令
- [ ] 新人可在 10 分钟内完成环境搭建

---

## 三、中优先级整改 (P1)

### P1-1: 浮点数边界比较

| 项目 | 内容 |
|------|------|
| **问题** | `d` 值边界检查使用 `==` 比较浮点数 |
| **来源** | 李得勤代码审计 |
| **负责人** | 李得勤 |
| **工时** | 1h |
| **目标** | 使用 `math.isclose()` 替代 |

**整改位置**: `src/features/fracdiff.py`

```python
# Before
if d == 0:
    return df.copy()

# After
import math
if math.isclose(d, 0, rel_tol=1e-9):
    return df.copy()
```

---

### P1-2: 空 Series 检查

| 项目 | 内容 |
|------|------|
| **问题** | 部分函数缺少空 Series 边界检查 |
| **来源** | 李得勤代码审计 |
| **负责人** | 李得勤 |
| **工时** | 1h |
| **目标** | 添加防御式检查 |

**整改位置**: `src/features/fracdiff.py`, `src/signals/base_models.py`

```python
# 添加检查
if df is None or len(df) == 0:
    return df.copy() if df is not None else pd.DataFrame()
```

---

### P1-3: BDay 假日误差

| 项目 | 内容 |
|------|------|
| **问题** | `BDay` 不考虑交易所假日 |
| **来源** | 李得勤代码审计 |
| **负责人** | 李得勤 |
| **工时** | 2h |
| **目标** | 使用 `pd.tseries.offsets.CustomBusinessDay` |

**整改方案**:
```python
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

class ExchangeCalendar(AbstractHolidayCalendar):
    """交易所假日日历"""
    rules = [
        # 可添加交易所特定假日
    ]

EXCHANGE_BDAY = CustomBusinessDay(calendar=ExchangeCalendar())
```

**注意**: 这是渐进式改进，不影响当前功能

---

### P1-4: ingest.py 拆分

| 项目 | 内容 |
|------|------|
| **问题** | `ingest.py` 500+ 行，职责过多 |
| **来源** | 赵连顺文档审计 |
| **负责人** | 李得勤 |
| **工时** | 2h |
| **目标** | 拆分为多个模块 |

**拆分方案**:
```
src/data/
├── ingest.py          # 入口函数（~100行）
├── validators.py      # 数据验证（~150行）
├── transformers.py    # 数据转换（~150行）
└── fetchers.py        # 数据获取（~150行）
```

---

## 四、任务分配汇总

| 任务ID | 任务 | 负责人 | 工时 | 优先级 |
|--------|------|--------|------|--------|
| P0-1 | meta_trainer 拆分 | 张得功 + 李得勤 | 4h | P0 |
| P0-2 | 配置依赖倒置 | 李得勤 | 3h | P0 |
| P0-3 | Base Model 抽象 | 李得勤 | 2h | P0 |
| P0-4 | README 补充 | 赵连顺 | 1h | P0 |
| P1-1 | 浮点数比较 | 李得勤 | 1h | P1 |
| P1-2 | 空 Series 检查 | 李得勤 | 1h | P1 |
| P1-3 | BDay 假日 | 李得勤 | 2h | P1 |
| P1-4 | ingest.py 拆分 | 李得勤 | 2h | P1 |

### 工作量分布

| 人员 | 任务数 | 工时 |
|------|--------|------|
| 张得功 | 1（设计） | 2h |
| 李得勤 | 6（实现） | 12h |
| 赵连顺 | 1（文档） | 1h |
| **合计** | **8** | **15h** |

---

## 五、时间线

```
Week 1: P0 整改
├── Day 1-2: P0-1 meta_trainer 拆分
├── Day 2-3: P0-2 配置依赖倒置
├── Day 3:   P0-3 Base Model 抽象
└── Day 4:   P0-4 README 补充

Week 2: P1 整改（可选）
├── Day 1: P1-1 + P1-2 边界处理
├── Day 2-3: P1-3 BDay 假日
└── Day 3-4: P1-4 ingest.py 拆分
```

---

## 六、验收流程

1. **代码整改** → 李得勤提交 → 李得勤自测通过
2. **架构审核** → 张得功 Review → 确认符合设计
3. **文档更新** → 赵连顺检查 → 确认文档同步
4. **最终验收** → 李成荣审批 → 合并主分支

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 拆分后接口变化 | 高 | 保持对外接口不变，仅重构内部 |
| 配置注入遗漏 | 中 | 全局搜索 `yaml.load` 确保无遗漏 |
| 测试不通过 | 中 | 每个任务完成后立即运行测试 |

---

*张得功 敬呈*
*2026-02-27*
