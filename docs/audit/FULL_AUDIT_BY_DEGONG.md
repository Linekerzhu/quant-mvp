# Phase C 全栈审计报告 - 规划层面

> 审计人：张得功（八品领侍）
> 审计日期：2026-02-27
> 审计角度：任务拆解、工作量估算、架构设计

---

## 一、审计范围

| Step | 文件 | 职责 |
|------|------|------|
| Step 1 | `src/signals/base_models.py` | 基础信号生成（SMA/Momentum） |
| Step 2 | `src/models/purged_kfold.py` | CPCV 交叉验证器 |
| Step 3 | `src/features/fracdiff.py` | 分数阶差分特征工程 |
| Step 4 | `src/models/meta_trainer.py` | Meta-Labeling 训练管道 |
| Step 4 | `run_pipeline.py` | 端到端入口脚本 |

---

## 二、架构合理性评估

### 2.1 模块划分 ✅ 良好

```
quant-mvp/
├── run_pipeline.py           # 入口层
├── config/training.yaml      # 配置层
├── src/
│   ├── signals/              # 信号生成
│   │   └── base_models.py
│   ├── features/             # 特征工程
│   │   └── fracdiff.py
│   └── models/               # 模型训练
│       ├── purged_kfold.py
│       └── meta_trainer.py
└── tests/                    # 测试层
```

**优点：**
- 清晰的 **分层架构**：入口 → 配置 → 核心模块 → 测试
- 符合 **单一职责**：每个模块功能单一明确
- 遵循 **AFML 结构**：信号 → 标签 → 验证 → 训练

**建议改进：**
- `meta_trainer.py` 249 行略显臃肿，可拆分为：
  - `meta_trainer.py` - 核心训练逻辑
  - `overfitting_check.py` - PBO/Dummy 检测
  - `label_converter.py` - 标签转换逻辑

### 2.2 依赖关系 ⚠️ 可优化

**当前依赖图：**
```
run_pipeline.py
    ├── src/signals/base_models.py
    ├── src/models/meta_trainer.py
    │       ├── src/models/purged_kfold.py
    │       └── (internal) LightGBM
    └── config/training.yaml
```

**问题发现：**

1. **循环风险**：`meta_trainer.py` 在 `train()` 方法内部动态导入 `purged_kfold.py`
   ```python
   # meta_trainer.py L219
   from src.models.purged_kfold import CombinatorialPurgedKFold
   ```
   - 这不是循环依赖，但 **延迟导入** 说明架构边界不够清晰

2. **配置耦合**：`purged_kfold.py` 直接读取 `config/training.yaml`
   - 违反 **依赖倒置**：高层模块不应依赖低层配置
   - 建议：通过构造函数参数传递，而非内部读取配置

3. **缺少抽象层**：Base Model 没有统一接口
   ```python
   # 当前：直接调用具体类
   from src.signals.base_models import BaseModelSMA, BaseModelMomentum
   
   # 建议：定义抽象基类
   class BaseSignalGenerator(ABC):
       @abstractmethod
       def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
           pass
   ```

---

## 三、SOLID 原则符合度

### 3.1 单一职责原则 (SRP) ✅ 75%

| 模块 | 评分 | 说明 |
|------|------|------|
| `base_models.py` | ⭐⭐⭐⭐⭐ | 每个类只负责一种信号 |
| `fracdiff.py` | ⭐⭐⭐⭐⭐ | 只负责分数差分计算 |
| `purged_kfold.py` | ⭐⭐⭐⭐ | 只负责交叉验证分割 |
| `meta_trainer.py` | ⭐⭐⭐ | 职责较多（训练+检测+惩罚） |

**扣分项：** `meta_trainer.py` 混合了：
- 训练逻辑 `_train_cpcv_fold()`
- 过拟合检测 `_calculate_pbo()`, `_dummy_feature_sentinel()`
- 数据惩罚 `apply_data_penalty()`

### 3.2 开闭原则 (OCP) ⚠️ 60%

**问题：** 添加新的 Base Model 需要修改 `run_pipeline.py` 的 `get_base_model()` 函数

```python
# run_pipeline.py L107-118
def get_base_model(model_type: str, config: dict):
    if model_type == 'sma':
        return BaseModelSMA(fast_window=20, slow_window=60)
    elif model_type == 'momentum':
        return BaseModelMomentum(window=20)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

**建议改进：** 使用 **工厂模式** 或 **注册表模式**

```python
# 推荐方案
class BaseModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs):
        return cls._models[name](**kwargs)

# 使用装饰器注册
@BaseModelRegistry.register('sma')
class BaseModelSMA:
    pass
```

### 3.3 里氏替换原则 (LSP) ✅ 85%

**现状：** 两个 Base Model（SMA/Momentum）接口一致，可互换

```python
# 两者都实现相同接口
def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    pass
```

**改进空间：** 缺少显式抽象基类，无法静态检查

### 3.4 接口隔离原则 (ISP) ✅ 90%

**优点：**
- `FracDiffTransformer` 实现了 sklearn 风格接口 `fit/transform/fit_transform`
- 各模块接口精简，没有冗余方法

### 3.5 依赖倒置原则 (DIP) ⚠️ 50%

**问题严重：**

1. **配置硬编码**
   ```python
   # purged_kfold.py L56-73
   if Path(config_path).exists():
       self._load_from_config(config_path)
   ```
   - 配置路径在类内部硬编码
   - 违反 DIP：高层业务逻辑依赖具体配置文件

2. **建议改进：**
   ```python
   # 推荐方案：通过构造函数注入
   class CombinatorialPurgedKFold:
       def __init__(
           self,
           n_splits: int = 6,
           n_test_splits: int = 2,
           ...
       ):
           # 纯参数注入，不读取配置文件
           self.n_splits = n_splits
           ...
   
   # 在调用层组装
   config = load_config("config/training.yaml")
   cpcv = CombinatorialPurgedKFold(**config['validation']['cpcv'])
   ```

---

## 四、扩展性评估

### 4.1 添加新 Base Model 🔴 困难

**当前流程：**
1. 在 `base_models.py` 添加新类
2. 修改 `run_pipeline.py` 的 `get_base_model()`
3. 修改 CLI 参数 `--model-type` choices

**改进建议：**
```python
# 1. 定义抽象基类
from abc import ABC, abstractmethod

class BaseSignalGenerator(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号 (+1, -1, 0)"""
        pass

# 2. 使用注册表
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register_model('sma')
class BaseModelSMA(BaseSignalGenerator):
    pass

# 3. 自动发现
def get_base_model(model_type: str, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)
```

### 4.2 添加新的验证方法 🟡 中等

**当前：** CPCV 已实现，但添加新方法（如 Walk-Forward）需要：
1. 在 `purged_kfold.py` 添加新类
2. 修改 `meta_trainer.py` 的 `train()` 方法

**建议：** 抽象出 `BaseValidator` 接口

```python
class BaseValidator(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """生成分割索引"""
        pass

class CPCVValidator(BaseValidator):
    pass

class WalkForwardValidator(BaseValidator):
    pass
```

### 4.3 添加新特征工程 🟢 良好

**优点：** `FracDiffTransformer` 已实现 sklearn 接口，可无缝集成到 Pipeline

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('fracdiff', FracDiffTransformer(d=0.5)),
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier())
])
```

### 4.4 多模型支持 🔴 未实现

**当前：** 只支持 LightGBM

**建议架构：**
```python
class MetaModel(ABC):
    @abstractmethod
    def fit(self, X, y, eval_set=None):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass

class LightGBMMetaModel(MetaModel):
    pass

class CatBoostMetaModel(MetaModel):
    pass
```

---

## 五、工作量估算

### 5.1 各模块复杂度评估

| 模块 | 代码行数 | 认知复杂度 | 测试覆盖 |
|------|---------|-----------|---------|
| `base_models.py` | 152 | 低 | ✅ 高 |
| `purged_kfold.py` | 260 | 中 | ✅ 高 |
| `fracdiff.py` | 289 | 中 | ✅ 高 |
| `meta_trainer.py` | 249 | **高** | 🟡 中 |
| `run_pipeline.py` | 310 | 中 | 🟡 中 |

### 5.2 技术债估算

| 债务项 | 修复工时 | 优先级 |
|--------|---------|--------|
| 抽象 Base Model 接口 | 2h | P1 |
| 重构 meta_trainer.py | 4h | P2 |
| 配置依赖倒置 | 3h | P2 |
| 添加模型注册表 | 2h | P3 |
| 抽象验证器接口 | 3h | P3 |

**总计：** 约 14 小时架构优化工作量

---

## 六、风险识别

### 6.1 高风险 🔴

1. **`meta_trainer.py` 职责膨胀**
   - 当前 249 行，未来可能继续增长
   - 建议：立即拆分

2. **配置硬编码路径**
   - `config_path` 在多处硬编码
   - 影响：单元测试困难，部署不灵活

### 6.2 中风险 🟡

1. **缺少抽象层**
   - Base Model、Validator、Meta Model 都缺少统一接口
   - 影响：扩展成本高

2. **错误处理不完善**
   - `meta_trainer.py` 中部分异常只是 log，未向上传播
   - 影响：生产环境可能隐藏问题

### 6.3 低风险 🟢

1. **测试覆盖良好**
   - `tests/` 目录有完整测试套件
   - 包括：`test_base_models.py`, `test_cpcv.py`, `test_fracdiff.py`

---

## 七、优化建议（按优先级）

### P0 - 立即修复

1. **拆分 `meta_trainer.py`**
   ```
   src/models/
   ├── meta_trainer.py      # 核心训练
   ├── overfitting.py       # PBO/Dummy 检测
   └── label_converter.py   # 标签转换
   ```

### P1 - 本周完成

2. **添加 Base Model 抽象基类**
   ```python
   # src/signals/base.py
   class BaseSignalGenerator(ABC):
       @abstractmethod
       def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
           pass
   ```

3. **配置依赖倒置**
   - 移除 `purged_kfold.py` 中的 `_load_from_config()`
   - 改为构造函数参数注入

### P2 - 下周完成

4. **添加模型注册表**
   - 实现 `BaseModelRegistry`
   - 自动发现和注册 Base Model

5. **抽象验证器接口**
   - 统一 CPCV 和 Walk-Forward 的调用方式

### P3 - 未来迭代

6. **添加 Meta Model 抽象**
   - 支持多模型切换（LightGBM/CatBoost/XGBoost）

7. **改进错误处理**
   - 统一异常类型
   - 添加重试机制

---

## 八、总结评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构合理性 | ⭐⭐⭐⭐ | 分层清晰，但有改进空间 |
| SOLID 符合度 | ⭐⭐⭐ | SRP/ISP 良好，DIP 需改进 |
| 扩展性 | ⭐⭐⭐ | 特征工程良好，模型扩展困难 |
| 可测试性 | ⭐⭐⭐⭐ | 测试覆盖良好，但配置耦合影响 |
| 文档完整性 | ⭐⭐⭐⭐ | 代码注释充分，配置有说明 |

**综合评分：3.4 / 5 ⭐**

---

## 九、附录：架构改进路线图

```
Phase 1 (Week 1): 偿还技术债
├── 拆分 meta_trainer.py
├── 添加抽象基类
└── 配置依赖倒置

Phase 2 (Week 2): 提升扩展性
├── 实现模型注册表
├── 抽象验证器接口
└── 添加更多 Base Model（RSI/MACD）

Phase 3 (Week 3+): 生产化
├── 多模型支持
├── 统一错误处理
└── 监控和日志增强
```

---

*张得功谨呈*
*2026-02-27*
