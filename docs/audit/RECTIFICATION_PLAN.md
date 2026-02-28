# 量化交易系统整改计划

> **统筹人**: 张得功（八品领侍）
> **日期**: 2026-02-28
> **状态**: 待执行

---

## 一、整改依据

本计划基于以下四份审计报告整合：

| 审计人 | 角度 | 评分 | 结论 |
|--------|------|------|------|
| 张得功 | 架构/规划 | 3.4/5 ⭐ | 技术债约14小时 |
| 李得勤 | 代码实现 | - | ✅ 51测试全通过 |
| 寇连材 | 金融数学 | - | ✅ 通过 |
| 赵连顺 | 文档/维护 | 8/10 | 良好，需改进 |

**总评**: 系统核心功能完备，测试覆盖完整，架构层面存在技术债需偿还。

---

## 二、问题汇总清单

### 🔴 高优先级（P0-P1）

| # | 问题 | 来源 | 影响 | 工时 |
|---|------|------|------|------|
| 1 | `meta_trainer.py` 职责膨胀，混合训练+检测+惩罚逻辑 | 得功 | 可维护性↓ | 4h |
| 2 | 配置硬编码违反 DIP，`purged_kfold.py` 内部读取配置 | 得功 | 单元测试困难 | 3h |
| 3 | 缺少 Base Model 抽象基类，无法静态检查 | 得功 | 扩展成本高 | 2h |
| 4 | README 安装步骤简略，缺少环境配置说明 | 连顺 | 新人上手难 | 1h |

### 🟡 中优先级（P2）

| # | 问题 | 来源 | 影响 | 工时 |
|---|------|------|------|------|
| 5 | Momentum 模型 valid_mask 逻辑可提取提高可读性 | 得勤 | 代码可读性↓ | 0.5h |
| 6 | 部分函数缺少空 Series/NaN 边界检查 | 得勤 | 潜在运行时错误 | 1h |
| 7 | BDay 假日计算可能有误差（需验证） | 得勤 | 回测准确性 | 1h |
| 8 | `ingest.py` 过长（500+行），可拆分 | 连顺 | 可维护性↓ | 2h |

### 🟢 低优先级（P3）

| # | 问题 | 来源 | 影响 | 工时 |
|---|------|------|------|------|
| 9 | 缺少模型注册表机制 | 得功 | 添加新模型需改入口 | 2h |
| 10 | 缺少验证器抽象接口 | 得功 | 添加新验证方法需改多处 | 3h |
| 11 | 统一日志系统（部分用logging） | 连顺 | 一致性↓ | 1h |
| 12 | 补充 API 文档自动生成配置 | 连顺 | 文档完整性 | 0.5h |

**总工时估算**: 约 21 小时

---

## 三、任务分配

### 👤 张得功（统筹人）
- **职责**: 整体协调、架构设计审查、验收
- **任务**: 监督 P0-P1 执行，审批设计方案

### 👤 李得勤（代码执行）
- **职责**: 具体代码重构、Bug 修复
- **任务**: 
  - [x] 拆分 `meta_trainer.py`
  - [x] 重构配置依赖注入
  - [x] 添加抽象基类
  - [x] 边界检查补充
  - [x] 提取 Momentum valid_mask 逻辑
  - [x] 补充空 Series/NaN 检查
  - [x] 验证 BDay 假日逻辑
  - [x] 拆分 `ingest.py`

### 👤 赵连顺（文档维护）
- **职责**: 文档更新、README 完善
- **任务**:
  - [ ] 补充 README 安装步骤
  - [ ] 拆分并补充 `ingest.py` 文档
  - [ ] 配置文件注释补充

### 👤 寇连材（金融把控）
- **职责**: 金融逻辑审核、验收把关
- **任务**: 
  - [ ] 验证 BDay 假日计算逻辑
  - [ ] 审核代码修改不影响金融正确性

---

## 四、整改执行计划

### Phase 1: 核心架构优化（Week 1）

| 任务ID | 任务名称 | 负责人 | 工时 | 验收标准 |
|--------|----------|--------|------|----------|
| T1 | 拆分 `meta_trainer.py` | 得勤 | 4h | 拆分为 3 个文件，测试全过 |
| T2 | 配置依赖倒置 | 得勤 | 3h | 移除内部配置读取，构造函数注入 |
| T3 | 添加 Base Model 抽象基类 | 得勤 | 2h | 定义 ABC，现有模型继承 |
| T4 | 补充 README 安装步骤 | 连顺 | 1h | 包含 Python 版本、虚拟环境、依赖安装 |

**里程碑**: 架构符合 SOLID 原则 ≥ 80%

---

### Phase 2: 代码质量提升（Week 2）

| 任务ID | 任务名称 | 负责人 | 工时 | 验收标准 | 状态 |
|--------|----------|--------|------|----------|------|
| T5 | 提取 Momentum valid_mask 逻辑 | 得勤 | 0.5h | 独立方法，有单元测试 | ✅ |
| T6 | 补充空 Series/NaN 检查 | 得勤 | 1h | 关键函数有防御式检查 | ✅ |
| T7 | 验证 BDay 假日逻辑 | 得勤 | 1h | 确认使用label_exit_date，BDay仅fallback | ✅ |
| T8 | 拆分 `ingest.py` | 得勤 | 2h | 拆为 2-3 个文件，功能不变 | ✅ |

**里程碑**: 代码可维护性评分 ≥ 9/10

---

### Phase 3: 扩展性增强（Week 3+）

| 任务ID | 任务名称 | 负责人 | 工时 | 验收标准 | 状态 |
|--------|----------|--------|------|----------|------|
| T9 | 实现模型注册表 | 得勤 | 2h | 装饰器自动注册，入口无需修改 | ✅ |
| T10 | 抽象验证器接口 | 得勤 | 3h | BaseValidator + 多实现 | ✅ |
| T11 | 统一日志系统 | 得勤 | 1h | 全部使用 EventLogger | ✅ |
| T12 | 配置 API 文档生成 | 连顺 | 0.5h | pdoc 配置完成，可生成 | ⏳ |
| - | **Deflated Sharpe** | 得勤 | 3h | 过拟合检测增强 | ✅ |

**里程碑**: 架构扩展性评分 ≥ 4/5

---

## 五、详细任务描述

### T1: 拆分 meta_trainer.py

**当前状态**: 249 行，混合多个职责

**拆分方案**:
```
src/models/
├── meta_trainer.py       # 核心训练逻辑 (~100行)
├── overfitting.py        # PBO计算 + Dummy Feature哨兵 (~80行)
└── label_converter.py    # Meta-Label转换 + 数据惩罚 (~50行)
```

**验收标准**:
- [ ] 三个文件独立存在
- [ ] 原有 51 个测试全部通过
- [ ] `MetaTrainer` 导入路径不变

---

### T2: 配置依赖倒置

**当前问题**:
```python
# purged_kfold.py L56-73
if Path(config_path).exists():
    self._load_from_config(config_path)
```

**重构方案**:
```python
class CombinatorialPurgedKFold:
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 10,
        embargo_window: int = 40,
        ...
    ):
        # 纯参数注入
        pass

# 调用层组装
config = load_config("config/training.yaml")
cpcv = CombinatorialPurgedKFold(**config['validation']['cpcv'])
```

**验收标准**:
- [ ] 移除 `_load_from_config()` 方法
- [ ] 构造函数仅接受参数
- [ ] 测试可 mock 参数

---

### T3: 添加抽象基类

**新增文件**: `src/signals/base.py`

```python
from abc import ABC, abstractmethod

class BaseSignalGenerator(ABC):
    """所有 Base Model 的抽象基类"""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: OHLCV 数据
            
        Returns:
            包含 'side' 列的 DataFrame，值域 {-1, 0, +1}
        """
        pass
```

**修改现有模型**:
```python
from src.signals.base import BaseSignalGenerator

class BaseModelSMA(BaseSignalGenerator):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 现有实现
        pass
```

**验收标准**:
- [ ] 抽象基类定义完成
- [ ] SMA/Momentum 继承基类
- [ ] 可用 `isinstance(obj, BaseSignalGenerator)` 检查

---

### T4: 补充 README 安装步骤

**新增章节**:

```markdown
## 安装指南

### 环境要求

- Python 3.9+
- pip 或 conda

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repo-url>
   cd quant-mvp
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**
   ```bash
   pytest tests/
   ```

### 常见问题

**Q: pytest 找不到模块?**
A: 确保在项目根目录运行，并已激活虚拟环境

**Q: LightGBM 安装失败?**
A: Mac 用户需先 `brew install cmake libomp`
```

**验收标准**:
- [ ] 新人可按步骤完成安装
- [ ] 包含常见问题 FAQ

---

### T5-T8: 中优先级任务

**T5: 提取 valid_mask**
```python
# 在 BaseModelMomentum 中提取
def _validate_price_data(self, price_prev, price_curr) -> pd.Series:
    """检查价格数据有效性"""
    return (
        (price_prev > 0) & 
        (price_curr > 0) & 
        ~price_prev.isna() & 
        ~price_curr.isna()
    )
```

**T6: 边界检查**
- 关键函数添加 `assert not df.empty`
- Series 长度检查
- NaN/Inf 检查

**T7: BDay 验证**
- 编写测试用例覆盖节假日边界
- 验证 `pd.bdate_range()` 与实际交易日对照

**T8: 拆分 ingest.py**
```
src/data/
├── ingest.py          # 主入口 (~200行)
├── validators.py      # 数据验证逻辑 (~150行)
└── cache.py           # 缓存管理 (~100行)
```

---

## 六、验收标准

### 架构层面
- [ ] SOLID 符合度 ≥ 80%
- [ ] 无循环依赖
- [ ] 所有抽象基类有单元测试

### 代码层面
- [ ] 51 个现有测试全部通过
- [ ] 新增边界检查测试 ≥ 10 个
- [ ] 代码覆盖率 ≥ 85%

### 文档层面
- [ ] README 包含完整安装步骤
- [ ] 所有模块有 docstring
- [ ] 配置文件注释完整

### 金融层面
- [ ] BDay 计算准确
- [ ] 无 look-ahead bias
- [ ] 过度拟合防御机制有效

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 重构引入 Bug | 高 | 重构前后运行全量测试对比 |
| 工时超预期 | 中 | 优先完成 P0-P1，P3 可延后 |
| 接口变更影响调用方 | 中 | 保持对外接口不变，仅重构内部 |

---

## 八、进度跟踪

| Phase | 开始日期 | 预计完成 | 实际完成 | 状态 |
|-------|----------|----------|----------|------|
| Phase 1 | 2026-02-28 | - | 2026-02-28 | ✅ 已完成 |
| Phase 2 | 2026-02-28 | - | 2026-02-28 | ✅ 已完成 |
| Phase 3 | 2026-02-28 | - | 2026-02-28 | ✅ 已完成（T12待连顺） |

---

**批准人**: 张得功  
**批准日期**: 2026-02-28

---

*张得功 谨呈*
*2026-02-28*
