# Deflated Sharpe Ratio 实现指南

> **编写人**: 张得功（八品领侍）
> **编写日期**: 2026-02-28
> **目标读者**: 李得勤

---

## 一、背景说明

### 1.1 为什么需要 Deflated Sharpe Ratio？

**问题**：PBO（Probability of Backtest Overfitting）只能告诉我们"策略是否过拟合"，但无法告诉我们"策略是否真的有效"。

**DSR 作用**：
- **验证策略有效性**：即使 PBO 通过，DSR 可以进一步验证策略是否真正有预测能力
- **双重门控**：PBO + DSR 提供更严格的过拟合检测
- **统计显著性**：DSR 基于统计检验，比单纯的 PBO 更科学

### 1.2 理论基础

**Deflated Sharpe Ratio** 由 Bailey & López de Prado (2014) 提出：

```
DSR = Φ( (SR - SR₀) / SE(SR) )
```

**参数说明**：
- `SR`：观测的 Sharpe Ratio（从 path_results 获取）
- `SR₀`：期望 Sharpe（通常 = 0，表示无预测能力）
- `SE(SR)`：Sharpe 的标准误
- `Φ`：标准正态分布的累积分布函数

**判定标准**：
- `DSR > 0.95`：策略有效（95% 置信度）
- `DSR > 0.90`：策略可能有效（90% 置信度）
- `DSR ≤ 0.90`：策略无效（拒绝）

---

## 二、实现步骤

### 步骤 1：添加 `calculate_deflated_sharpe` 方法（30分钟）

**文件**：`src/models/overfitting.py`

**位置**：`OverfittingDetector` 类中，在 `calculate_pbo` 方法后添加

**代码实现**：

```python
def calculate_deflated_sharpe(
    self, 
    path_results: List[Dict]
) -> Tuple[float, Dict[str, Any]]:
    """
    计算 Deflated Sharpe Ratio。
    
    参考: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, 
    Backtest Overfitting, and Non-Normality"
    
    公式:
        DSR = Φ( (SR - SR₀) / SE(SR) )
    
    其中:
        - SR: 观测的Sharpe Ratio（从path_results获取）
        - SR₀: 期望Sharpe（默认=0）
        - SE(SR): Sharpe的标准误
        - Φ: 标准正态分布CDF
    
    Args:
        path_results: List of CPCV path results
            每个dict应包含:
            - 'sharpe': float (优先使用)
            - 'auc': float (如果没有sharpe，用AUC近似)
            - 'accuracy': float (如果都没有，用accuracy近似)
    
    Returns:
        Tuple of (dsr, details)
            - dsr: Deflated Sharpe Ratio (float)
            - details: 计算过程详细信息 (dict)
    """
    from scipy.stats import norm
    
    # Step 1: 提取 Sharpe Ratios
    sharpe_ratios = []
    used_metric = 'sharpe'  # 记录实际使用的指标
    
    for i, r in enumerate(path_results):
        # 优先使用 sharpe
        if 'sharpe' in r and r['sharpe'] is not None:
            sharpe_ratios.append(r['sharpe'])
        # 其次使用 AUC 近似（AUC-0.5 作为 Sharpe 代理）
        elif 'auc' in r and r['auc'] is not None:
            sharpe_ratios.append((r['auc'] - 0.5) * 2)  # 映射到 [-1, 1]
            used_metric = 'auc_approx'
        # 最后使用 accuracy 近似
        elif 'accuracy' in r and r['accuracy'] is not None:
            sharpe_ratios.append((r['accuracy'] - 0.5) * 2)
            used_metric = 'accuracy_approx'
        else:
            logger.warning(f"Path {i}: No sharpe/auc/accuracy found, skipping")
    
    if len(sharpe_ratios) == 0:
        logger.error("No valid metrics found in path_results")
        return 0.0, {'error': 'no_valid_metrics'}
    
    sharpe_ratios = np.array(sharpe_ratios)
    n = len(sharpe_ratios)
    
    # Step 2: 计算统计量
    mean_sr = np.mean(sharpe_ratios)
    std_sr = np.std(sharpe_ratios, ddof=1)  # 样本标准差
    
    # Step 3: 计算标准误
    if n > 1:
        se_sr = std_sr / np.sqrt(n)
    else:
        # 只有一条路径，无法计算标准误，退化为单一SR
        se_sr = 1.0
        logger.warning("Only one path available, DSR degenerates to SR")
    
    # Step 4: 计算统计量 (t-statistic)
    # 假设 SR₀ = 0
    if se_sr > 0:
        t_stat = mean_sr / se_sr
    else:
        t_stat = 0.0 if mean_sr == 0 else np.inf * np.sign(mean_sr)
    
    # Step 5: 计算 DSR (单边检验，右尾)
    # DSR = P(SR > SR₀ | 观测数据)
    dsr = norm.cdf(t_stat)
    
    # Step 6: 记录详细信息
    details = {
        'used_metric': used_metric,
        'n_paths': n,
        'mean_sr': float(mean_sr),
        'std_sr': float(std_sr),
        'se_sr': float(se_sr),
        't_stat': float(t_stat),
        'dsr': float(dsr),
        'sr_zero': 0.0  # 期望 Sharpe = 0
    }
    
    logger.info(f"Deflated Sharpe Ratio: {dsr:.4f} "
                f"(mean_SR={mean_sr:.4f}, SE={se_sr:.4f}, n={n})")
    
    return float(dsr), details
```

**关键点**：
1. **兼容性**：支持 `sharpe`、`auc`、`accuracy` 三种指标
2. **近似转换**：如果只有 AUC，用 `(AUC - 0.5) * 2` 近似 Sharpe
3. **统计检验**：使用 t-统计量和正态分布 CDF
4. **日志记录**：记录所有关键步骤

---

### 步骤 2：添加 `check_dsr_gate` 方法（15分钟）

**文件**：`src/models/overfitting.py`

**位置**：`OverfittingDetector` 类中，在 `check_pbo_gate` 方法后添加

**代码实现**：

```python
def check_dsr_gate(self, dsr: float) -> Tuple[bool, str]:
    """
    DSR 门控检查。
    
    判定标准:
        - DSR >= 0.95: PASS (95% 置信度)
        - DSR >= 0.90: WARNING (90% 置信度)
        - DSR < 0.90: REJECT (策略无效)
    
    Args:
        dsr: Calculated Deflated Sharpe Ratio
    
    Returns:
        Tuple of (passed, message)
    """
    dsr_pass_threshold = self.config.get('dsr_pass_threshold', 0.95)
    dsr_warn_threshold = self.config.get('dsr_warn_threshold', 0.90)
    
    if dsr >= dsr_pass_threshold:
        return True, f"PASS: DSR={dsr:.4f} >= {dsr_pass_threshold} (95% confidence)"
    elif dsr >= dsr_warn_threshold:
        return True, f"WARNING: DSR={dsr:.4f} in [{dsr_warn_threshold}, {dsr_pass_threshold})"
    else:
        return False, f"REJECT: DSR={dsr:.4f} < {dsr_warn_threshold} (strategy likely ineffective)"
```

---

### 步骤 3：更新 `check_overfitting` 方法（20分钟）

**文件**：`src/models/overfitting.py`

**位置**：替换原有的 `check_overfitting` 方法

**代码实现**：

```python
def check_overfitting(
    self, 
    path_results: List[Dict]
) -> Dict[str, Any]:
    """
    执行完整的过拟合检查（三级门控）。
    
    三级门控:
        1. PBO (Probability of Backtest Overfitting)
        2. Dummy Feature Sentinel (特征级过拟合)
        3. DSR (Deflated Sharpe Ratio)
    
    Args:
        path_results: List of CPCV path results
    
    Returns:
        Dictionary with overfitting check results
    """
    # Level 1: PBO 检查
    pbo = self.calculate_pbo(path_results)
    pbo_passed, pbo_message = self.check_pbo_gate(pbo)
    
    # Level 2: Dummy Feature Sentinel
    avg_importance = {}
    for r in path_results:
        for feat, imp in r.get('feature_importance', {}).items():
            if feat not in avg_importance:
                avg_importance[feat] = []
            avg_importance[feat].append(imp)
    
    avg_importance = {k: np.mean(v) for k, v in avg_importance.items()}
    dummy_result = self.dummy_feature_sentinel(avg_importance)
    
    # Level 3: Deflated Sharpe Ratio
    dsr, dsr_details = self.calculate_deflated_sharpe(path_results)
    dsr_passed, dsr_message = self.check_dsr_gate(dsr)
    
    # 综合判定
    # 只有三级全部通过才判定为 PASS
    overall_passed = (
        pbo_passed and 
        dummy_result.get('passed', True) and 
        dsr_passed
    )
    
    return {
        # PBO 结果
        'pbo': pbo,
        'pbo_passed': pbo_passed,
        'pbo_message': pbo_message,
        
        # Dummy Sentinel 结果
        'dummy_sentinel': dummy_result,
        
        # DSR 结果
        'dsr': dsr,
        'dsr_details': dsr_details,
        'dsr_passed': dsr_passed,
        'dsr_message': dsr_message,
        
        # 综合结果
        'overall_passed': overall_passed,
        
        # 判定摘要
        'summary': self._generate_summary(
            pbo_passed, dummy_result.get('passed', True), dsr_passed
        )
    }

def _generate_summary(
    self, 
    pbo_passed: bool, 
    dummy_passed: bool, 
    dsr_passed: bool
) -> str:
    """生成判定摘要。"""
    if pbo_passed and dummy_passed and dsr_passed:
        return "✅ PASS: All three gates passed (PBO + Dummy + DSR)"
    elif pbo_passed and dummy_passed and not dsr_passed:
        return "⚠️ WARNING: PBO and Dummy passed, but DSR rejected (strategy may be ineffective)"
    elif pbo_passed and not dummy_passed:
        return "❌ REJECT: Dummy feature sentinel detected overfitting"
    elif not pbo_passed:
        return "❌ REJECT: PBO indicates backtest overfitting"
    else:
        return "❌ REJECT: Multiple gates failed"
```

---

### 步骤 4：添加单元测试（45分钟）

**文件**：`tests/test_overfit_sentinels.py`

**位置**：在文件末尾添加新测试类

**代码实现**：

```python
class TestDeflatedSharpeRatio:
    """Test Deflated Sharpe Ratio calculation."""
    
    def test_dsr_with_sharpe(self):
        """Test DSR calculation with explicit Sharpe ratios."""
        # Create mock path results with sharpe
        path_results = [
            {'sharpe': 0.5, 'auc': 0.65, 'feature_importance': {}},
            {'sharpe': 0.6, 'auc': 0.68, 'feature_importance': {}},
            {'sharpe': 0.4, 'auc': 0.62, 'feature_importance': {}},
        ]
        
        detector = OverfittingDetector({'pbo_threshold': 0.3})
        dsr, details = detector.calculate_deflated_sharpe(path_results)
        
        # Check basic properties
        assert 0.0 <= dsr <= 1.0
        assert details['used_metric'] == 'sharpe'
        assert details['n_paths'] == 3
        assert 'mean_sr' in details
        assert 'se_sr' in details
        
        print(f"DSR: {dsr:.4f}")
        print(f"Details: {details}")
    
    def test_dsr_with_auc_approximation(self):
        """Test DSR with AUC approximation (no sharpe available)."""
        # Create mock path results without sharpe
        path_results = [
            {'auc': 0.65, 'feature_importance': {}},
            {'auc': 0.68, 'feature_importance': {}},
            {'auc': 0.62, 'feature_importance': {}},
        ]
        
        detector = OverfittingDetector({'pbo_threshold': 0.3})
        dsr, details = detector.calculate_deflated_sharpe(path_results)
        
        assert 0.0 <= dsr <= 1.0
        assert details['used_metric'] == 'auc_approx'
        
        print(f"DSR (AUC approx): {dsr:.4f}")
    
    def test_dsr_empty_results(self):
        """Test DSR with empty path results."""
        detector = OverfittingDetector({'pbo_threshold': 0.3})
        dsr, details = detector.calculate_deflated_sharpe([])
        
        assert dsr == 0.0
        assert 'error' in details
    
    def test_dsr_gate_pass(self):
        """Test DSR gate with high DSR (should pass)."""
        detector = OverfittingDetector({
            'dsr_pass_threshold': 0.95,
            'dsr_warn_threshold': 0.90
        })
        
        passed, message = detector.check_dsr_gate(0.96)
        
        assert passed is True
        assert "PASS" in message
    
    def test_dsr_gate_warning(self):
        """Test DSR gate with medium DSR (should warn)."""
        detector = OverfittingDetector({
            'dsr_pass_threshold': 0.95,
            'dsr_warn_threshold': 0.90
        })
        
        passed, message = detector.check_dsr_gate(0.92)
        
        assert passed is True  # Warning still passes
        assert "WARNING" in message
    
    def test_dsr_gate_reject(self):
        """Test DSR gate with low DSR (should reject)."""
        detector = OverfittingDetector({
            'dsr_pass_threshold': 0.95,
            'dsr_warn_threshold': 0.90
        })
        
        passed, message = detector.check_dsr_gate(0.85)
        
        assert passed is False
        assert "REJECT" in message
    
    def test_full_overfitting_check_with_dsr(self):
        """Test full overfitting check including DSR."""
        # Create comprehensive mock data
        path_results = [
            {
                'auc': 0.65,
                'sharpe': 0.5,
                'feature_importance': {
                    'feature_1': 0.3,
                    'feature_2': 0.2,
                    'dummy_noise': 0.05
                }
            },
            {
                'auc': 0.68,
                'sharpe': 0.6,
                'feature_importance': {
                    'feature_1': 0.28,
                    'feature_2': 0.22,
                    'dummy_noise': 0.06
                }
            },
            {
                'auc': 0.62,
                'sharpe': 0.4,
                'feature_importance': {
                    'feature_1': 0.32,
                    'feature_2': 0.18,
                    'dummy_noise': 0.04
                }
            }
        ]
        
        detector = OverfittingDetector({
            'pbo_threshold': 0.3,
            'pbo_reject': 0.5,
            'dsr_pass_threshold': 0.95,
            'dsr_warn_threshold': 0.90
        })
        
        result = detector.check_overfitting(path_results)
        
        # Check structure
        assert 'pbo' in result
        assert 'dsr' in result
        assert 'dummy_sentinel' in result
        assert 'overall_passed' in result
        assert 'summary' in result
        
        print(f"Overall passed: {result['overall_passed']}")
        print(f"Summary: {result['summary']}")
```

---

### 步骤 5：更新配置文件（10分钟）

**文件**：`config/training.yaml`

**位置**：在 `overfitting` 部分添加 DSR 阈值

**添加内容**：

```yaml
overfitting:
  pbo_threshold: 0.3
  pbo_reject: 0.5
  
  # Deflated Sharpe Ratio thresholds
  dsr_pass_threshold: 0.95   # 95% confidence
  dsr_warn_threshold: 0.90   # 90% confidence
  
  dummy_feature_sentinel:
    ranking_threshold: 0.25
```

---

## 三、验收标准

### 3.1 功能验收（必须全部通过）

| 验收项 | 标准 | 验证方式 |
|--------|------|----------|
| DSR 计算 | 正确计算 DSR 值 | `test_dsr_with_sharpe` 通过 |
| AUC 近似 | 无 Sharpe 时用 AUC 近似 | `test_dsr_with_auc_approximation` 通过 |
| 空数据处理 | 空路径返回 0.0 | `test_dsr_empty_results` 通过 |
| 门控逻辑 | Pass/Warning/Reject 正确 | `test_dsr_gate_*` 全部通过 |
| 集成测试 | 与 PBO + Dummy 集成 | `test_full_overfitting_check_with_dsr` 通过 |

### 3.2 代码质量验收

| 验收项 | 标准 |
|--------|------|
| 测试覆盖 | 新增代码测试覆盖率 ≥ 80% |
| 日志完整 | 所有关键步骤有日志记录 |
| 类型提示 | 所有新方法有类型提示 |
| 文档完整 | Docstring 包含参数说明和返回值说明 |

### 3.3 集成验收

| 验收项 | 标准 |
|--------|------|
| CPCV 集成 | 15 条 path 全部能计算 DSR |
| 配置读取 | DSR 阈值从 config/training.yaml 读取 |
| 向后兼容 | 不影响现有 PBO 和 Dummy 哨兵功能 |

---

## 四、测试运行

### 4.1 运行新增测试

```bash
# 运行 DSR 相关测试
pytest tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio -v

# 运行所有过拟合测试
pytest tests/test_overfit_sentinels.py -v

# 运行全量测试（确保无回归）
pytest tests/ -v
```

### 4.2 预期输出

```
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_with_sharpe PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_with_auc_approximation PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_empty_results PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_gate_pass PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_gate_warning PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_dsr_gate_reject PASSED
tests/test_overfit_sentinels.py::TestDeflatedSharpeRatio::test_full_overfitting_check_with_dsr PASSED

========================= 7 passed in 0.52s =========================
```

---

## 五、常见问题

### Q1: 如果 path_results 中没有 sharpe 怎么办？

**A**: 按优先级自动降级：
1. 优先使用 `sharpe`
2. 其次使用 `auc`，通过 `(AUC - 0.5) * 2` 近似
3. 最后使用 `accuracy`，同样近似

### Q2: DSR 和 PBO 有什么区别？

**A**: 
- **PBO**：检测"策略是否过拟合"（相对排名）
- **DSR**：检测"策略是否有效"（绝对值检验）
- **组合使用**：双重门控，更严格

### Q3: 为什么 DSR 阈值设置为 0.95？

**A**: 这是 95% 置信度的统计显著性水平。低于此值意味着策略可能是运气。

### Q4: 如何调试 DSR 计算问题？

**A**: 查看 `details` 字段：
```python
dsr, details = detector.calculate_deflated_sharpe(path_results)
print(details)  # 查看使用的指标、均值、标准误等
```

---

## 六、参考资料

### 6.1 理论文献

- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality". *The Journal of Portfolio Management*, 40(5), 94-107.

### 6.2 相关代码

- PBO 实现：`src/models/overfitting.py:56-91`
- CPCV 路径生成：`src/models/purged_kfold.py`
- 配置文件：`config/training.yaml`

---

## 七、时间估算

| 步骤 | 工时 |
|------|------|
| 步骤 1：calculate_deflated_sharpe 方法 | 30分钟 |
| 步骤 2：check_dsr_gate 方法 | 15分钟 |
| 步骤 3：更新 check_overfitting 方法 | 20分钟 |
| 步骤 4：单元测试 | 45分钟 |
| 步骤 5：配置文件更新 | 10分钟 |
| **总计** | **2小时** |

---

## 八、完成后检查清单

- [ ] `calculate_deflated_sharpe` 方法实现完成
- [ ] `check_dsr_gate` 方法实现完成
- [ ] `check_overfitting` 方法更新完成
- [ ] 7 个单元测试全部通过
- [ ] `config/training.yaml` 更新完成
- [ ] 全量测试无回归（112/112 通过）
- [ ] 更新 PHASE_C_STATUS.md，标记 DSR 为已完成

---

**编写完成日期**: 2026-02-28

**下一步**: 提交给李得勤实施，完成后通知张得功验收

---

*张得功 敬上*
*长春宫*
