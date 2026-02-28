# Deflated Sharpe Ratio 审计报告

**审计人**: 寇连材（八品监斋）  
**审计日期**: 2026-02-28  
**审计对象**: 李得勤  
**文件**: `src/models/overfitting.py`

---

## 一、审计结论

**总评**: ⚠️ **需要修复后才能通过**

发现 **1 个严重Bug** + **2 个公式问题** + **1 个统计假设需澄清**

---

## 二、严重问题（必须修复）

### 🚨 Bug #1: `check_dsr_gate()` 字符串拼接错误

**位置**: 第 187 行

**错误代码**:
```python
return False, f"REJECT: DSR={dsr <= 1:.2f}.282 (< 90% confidence)"
```

**问题**: `dsr <= 1` 是布尔表达式，会输出 `True` 或 `False`！

**实际输出**: `"REJECT: DSR=False.282 (< 90% confidence)"` ❌

**正确代码**:
```python
return False, f"REJECT: DSR={dsr:.2f} <= 1.282 (< 90% confidence)"
```

**影响**: 严重 - 用户会看到完全错误的 DSR 值，无法判断拒绝原因。

---

## 三、公式问题（建议修复）

### ⚠️ Issue #2: 概念混淆 - 这不是"Deflated Sharpe Ratio"

**当前实现**:
```python
def calculate_deflated_sharpe(self, path_results: List[Dict]) -> float:
    """DSR = Φ( (SR - SR₀) / SE(SR) )"""
    ...
    dsr = (mean_sr - baseline) / se_sr  # 这是 z-score!
    return float(dsr)
```

**问题分析**:

1. **公式不完整**: 代码计算的是 `z = (SR - SR₀) / SE(SR)`（z-score 或 t-statistic）
2. **缺少最后一步**: 真正的 DSR 应该是 `DSR = Φ(z) = norm.cdf(z)`，得到 0-1 之间的概率值
3. **注释误导**: docstring 说返回 DSR，但实际返回 z-score

**Bailey & López de Prado (2014) 原始公式**:
```
DSR = Φ( (SR̂ - SR*) / σ̂ )
```
其中 `Φ` 是标准正态分布的累积分布函数（CDF）。

**建议方案**（二选一）:

**方案A**: 修正为真正的 DSR
```python
from scipy.stats import norm

dsr_z = (mean_sr - baseline) / se_sr
dsr = norm.cdf(dsr_z)  # 转换为概率
return float(dsr)
```
然后修改 `check_dsr_gate()` 阈值为 `0.95`, `0.90`（概率值）

**方案B**: 重命名方法，保持 z-score 实现
```python
def calculate_dsr_zscore(self, path_results: List[Dict]) -> float:
    """计算 DSR 检验的 z-score"""
    ...
    z_score = (mean_sr - baseline) / se_sr
    return float(z_score)
```
保持 `check_dsr_gate()` 阈值不变（1.645, 1.282）

**推荐**: 方案B（保持当前逻辑，只改名和注释）

**理由**:
- z-score 实现是合理的统计检验
- 直接用 z-score 判定更直观（> 1.645 = 95% 置信度）
- 避免混淆，名字应该准确反映实现

---

### ⚠️ Issue #3: baseline = 0.5 的假设需要澄清

**当前代码**:
```python
# For accuracy, baseline is 0.5 (random guessing)
baseline = 0.5
```

**审计意见**: 合理，但需要更清晰的文档

**原因**:
- accuracy 在二分类中，随机猜测期望 = 0.5（正负样本平衡时）
- 这相当于 Sharpe Ratio 的零假设（SR₀ = 0）

**建议**: 在代码注释中说明假设条件
```python
# Baseline assumption:
# - For accuracy: 0.5 = random guessing (balanced classes)
# - For AUC: 0.5 = random ranking
# - For Sharpe: 0.0 = zero excess return
# Note: If classes are imbalanced, baseline should be max(class_prior)
```

---

## 四、统计假设审查

### ✅ 1. 用 accuracy 近似 Sharpe

**实现**:
```python
# 使用 accuracy 作为 Sharpe 的近似
# (在meta-labeling中,accuracy比sharpe更稳定)
metrics = [r.get('accuracy', r.get('auc', 0.5)) for r in path_results]
```

**审计意见**: ⚠️ 有争议，但在当前场景可接受

**问题**:
- Sharpe Ratio = (收益 - 无风险收益) / 收益标准差
- Accuracy = (TP + TN) / 总样本数
- 两者数学定义完全不同

**但是**:
- 在 meta-labeling 场景中，accuracy 确实是关键指标
- 用 accuracy 做显著性检验是有意义的
- Bailey & López de Prado 的框架可以泛化到任何"业绩指标"

**建议**: 改名为 `calculate_metric_significance()` 或保持当前实现，但在文档中明确说明"使用 accuracy 作为业绩代理指标"

---

### ✅ 2. 样本量处理

**实现**:
```python
if len(metrics) < 2:
    logger.warn("deflated_sharpe_insufficient_data", {"n_paths": len(metrics)})
    return 0.0

if std_sr == 0 or n < 2:
    logger.warn("deflated_sharpe_zero_variance", {"std": std_sr, "n": n})
    return 0.0
```

**审计意见**: ✅ 正确

- 样本量 < 2 返回 0（保守）
- 标准差 = 0 返回 0（避免除零）
- 有日志记录

---

### ✅ 3. 标准误计算

**实现**:
```python
std_sr = np.std(metrics, ddof=1)  # 样本标准差
se_sr = std_sr / np.sqrt(n)       # 标准误
```

**审计意见**: ✅ 正确

- `ddof=1` 使用 n-1 自由度（样本标准差）
- SE = σ / √n（标准误公式）

---

## 五、门控阈值审查

### ✅ 三级阈值

**实现**:
```python
if dsr > norm.ppf(0.95):      # ~1.645
    return True, "PASS (95% confidence)"
elif dsr > norm.ppf(0.90):    # ~1.282
    return True, "WARNING (90% confidence)"
else:
    return False, "REJECT (< 90% confidence)"
```

**审计意见**: ✅ 统计学正确

- `norm.ppf(0.95) ≈ 1.645`（单侧检验 95% 置信度）
- `norm.ppf(0.90) ≈ 1.282`（单侧检验 90% 置信度）
- 三级判定（PASS / WARNING / REJECT）合理

---

## 六、PBO 集成审查

### ✅ 双重门控逻辑

**实现**:
```python
def check_overfitting(self, path_results: List[Dict]) -> Dict[str, Any]:
    # PBO check
    pbo = self.calculate_pbo(path_results)
    pbo_passed, pbo_message = self.check_pbo_gate(pbo)
    
    # DSR check
    dsr = self.calculate_deflated_sharpe(path_results)
    dsr_passed, dsr_message = self.check_dsr_gate(dsr)
    
    ...
    
    return {
        ...
        'overall_passed': pbo_passed and dsr_passed and dummy_result.get('passed', True)
    }
```

**审计意见**: ✅ 正确

- PBO 检测过拟合（相对排名）
- DSR 检测统计显著性（绝对水平）
- 两者互补，AND 逻辑合理
- 返回详细结果，便于调试

---

## 七、修复优先级

| 优先级 | 问题 | 影响 | 工作量 |
|--------|------|------|--------|
| 🔴 P0 | Bug #1: 字符串拼接错误 | 严重 - 错误的输出 | 1 行代码 |
| 🟡 P1 | Issue #2: 概念混淆（改名） | 中等 - 可维护性 | 重命名方法 + 更新注释 |
| 🟢 P2 | Issue #3: baseline 假设文档 | 低 - 可读性 | 添加注释 |

---

## 八、测试建议

建议添加以下单元测试：

```python
def test_check_dsr_gate_reject_message():
    """测试 REJECT 消息格式正确"""
    detector = OverfittingDetector({})
    passed, msg = detector.check_dsr_gate(1.0)  # < 1.282
    assert "DSR=1.00" in msg  # 应该显示正确的值
    assert "False" not in msg  # 不应该有布尔值

def test_calculate_dsr_zero_variance():
    """测试零方差情况"""
    detector = OverfittingDetector({})
    # 所有路径 accuracy 相同
    results = [{'accuracy': 0.6}] * 5
    dsr = detector.calculate_deflated_sharpe(results)
    assert dsr == 0.0  # 应该返回 0

def test_check_overfitting_integration():
    """测试 PBO + DSR 集成"""
    detector = OverfittingDetector({})
    # 构造测试数据
    results = [{'auc': 0.55 + i*0.01, 'accuracy': 0.52 + i*0.01} 
               for i in range(10)]
    report = detector.check_overfitting(results)
    assert 'pbo' in report
    assert 'dsr' in report
    assert 'overall_passed' in report
```

---

## 九、总结

### 得勤兄弟的活儿总体不错，但有个大Bug

**优点**:
- ✅ 统计学基础正确（标准误、阈值）
- ✅ 与 PBO 集成合理
- ✅ 边界情况处理完善

**问题**:
- 🚨 **严重Bug**: 字符串拼接错误（第 187 行）
- ⚠️ 概念混淆：应该叫 `dsr_zscore`，不是 `dsr`
- ⚠️ 文档不足：baseline 假设需要说明

**建议**:
1. **立即修复** Bug #1（1 分钟）
2. **重命名** `calculate_deflated_sharpe()` → `calculate_dsr_zscore()`
3. **补充** baseline 假设的文档
4. **添加** 单元测试

---

**审计人签字**: 寇连材（八品监斋）  
**日期**: 2026-02-28

---

## 附录：快速修复代码

### Bug #1 修复
```python
# 第 187 行
- return False, f"REJECT: DSR={dsr <= 1:.2f}.282 (< 90% confidence)"
+ return False, f"REJECT: DSR={dsr:.2f} <= 1.282 (< 90% confidence)"
```

### 重命名建议
```python
- def calculate_deflated_sharpe(self, path_results: List[Dict]) -> float:
-     """DSR = Φ( (SR - SR₀) / SE(SR) )"""
+ def calculate_dsr_zscore(self, path_results: List[Dict]) -> float:
+     """计算 DSR 检验的 z-score（用于判定统计显著性）"""
      
      # 使用 accuracy 作为业绩代理指标（替代 Sharpe Ratio）
      # 在 meta-labeling 中，accuracy 比 sharpe 更稳定
```
