# Phase C Step 3 修复任务

> **发起人**: 李得勤 (审计发现)  
> **负责人**: 张得功 (修复任务制定)  
> **严重级别**: 🔴 P1 - 严重（影响核心算法正确性）

---

## 1. 问题概述

### 1.1 问题描述

**文件**: `src/features/fracdiff.py`（待实现）  
**函数**: `fracdiff_weights()`  
**问题类型**: 数学公式符号错误

### 1.2 问题详情

**当前错误公式**:
```python
weights[k] = weights[k-1] * (d - k + 1) / k
```

**正确公式**:
```python
weights[k] = weights[k-1] * (k - 1 - d) / k
```

**差异分析**:
```
错误: (d - k + 1) = -(k - 1 - d)
正确: (k - 1 - d)

两者相差一个负号！
```

### 1.3 影响范围

| 场景 | 错误行为 | 期望行为 |
|------|----------|----------|
| d = 1 | 计算价格"加和"而非"差分" | 应为一阶差分 |
| 0 < d < 1 | 部分权重符号错误 | 权重符号应符合分数阶差分数学定义 |
| 平稳性检验 | ADF 检验可能失效 | 应在合理 d 值下通过平稳性检验 |

---

## 2. 修复任务清单

### 任务 T1: 修复权重计算公式

**状态**: ⏳ 待完成  
**负责人**: 待分配  
**预估工时**: 0.5 小时

**具体修改**:
```python
# 文件: src/features/fracdiff.py
# 函数: fracdiff_weights() 或内嵌在 fracdiff_fixed_window() 中

# ❌ 错误代码
# weights[k] = weights[k-1] * (d - k + 1) / k

# ✅ 正确代码
weights[k] = weights[k-1] * (k - 1 - d) / k
```

**验证标准**:
- [ ] 代码中公式已修改为 `(k - 1 - d) / k`
- [ ] 代码注释已更新，说明公式来源

---

### 任务 T2: 验证修复后的权重正确性

**状态**: ⏳ 待完成  
**负责人**: 待分配  
**预估工时**: 1 小时

**验证方法**:

1. **d=0 边界验证**
   - 输入: 任意价格序列
   - 期望: FracDiff(d=0) ≈ 原序列（权重 [1, 0, 0, ...]）
   
2. **d=1 边界验证**
   - 输入: 价格序列 [p0, p1, p2, p3, ...]
   - 期望: FracDiff(d=1) = [NaN, p1-p0, p2-p1, p3-p2, ...]（一阶差分）
   - 权重序列应为 [1, -1, 0, 0, ...]

3. **权重符号验证**
   - 对于 0 < d < 1，所有权重应为正数
   - 数学依据: 分数阶差分权重在 0<d<1 时全为正

**测试代码框架**:
```python
def test_weights_formula_correctness():
    """验证权重计算公式符号正确"""
    # d=1 时，权重应为 [1, -1, 0, 0, ...]
    weights_d1 = calculate_weights(d=1.0, window=5)
    expected = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(weights_d1, expected)
    
    # d=0.5 时，所有非零权重应为正
    weights_d05 = calculate_weights(d=0.5, window=5)
    assert all(w >= 0 for w in weights_d05), "d=0.5 时权重应全为非负"
```

**验收标准**:
- [ ] d=0 时输出等于原序列
- [ ] d=1 时输出等于一阶差分
- [ ] 0<d<1 时所有权重为正

---

### 任务 T3: 回归测试

**状态**: ⏳ 待完成  
**负责人**: 待分配  
**预估工时**: 1.5 小时

**测试范围**:

1. **单元测试** (`tests/test_fracdiff.py`)
   - [ ] `test_d_zero_is_original`: FracDiff(d=0) ≈ 原序列
   - [ ] `test_d_one_is_diff`: FracDiff(d=1) ≈ 一阶差分
   - [ ] `test_weights_sign_correctness`: 权重符号验证
   - [ ] `test_optimal_d_stationary`: 最优 d 值使 ADF p < 0.05
   - [ ] `test_memory_preserved`: d < 1 时与原序列相关性 > 0
   - [ ] `test_no_future_leakage`: 无未来数据泄露

2. **集成测试**
   - [ ] FracDiff → Meta-Trainer 端到端测试
   - [ ] 与 build_features.py 集成验证

3. **全量回归测试**
   - [ ] 运行 `pytest tests/` 确保无其他模块受影响
   - [ ] 确认测试通过率 100%

**验收标准**:
- [ ] 所有 FracDiff 相关测试通过
- [ ] 全量测试通过（无回归）
- [ ] 测试覆盖率 ≥ 80%

---

## 3. 实施计划

### 3.1 依赖关系

```
T1 (修复公式) ─┬─→ T2 (验证权重)
               └─→ T3 (回归测试)
```

### 3.2 时间线

| 阶段 | 任务 | 预计完成时间 |
|------|------|--------------|
| Day 1 AM | T1: 修复公式 | 0.5 小时 |
| Day 1 PM | T2: 验证权重 | 1 小时 |
| Day 2 AM | T3: 回归测试 | 1.5 小时 |
| Day 2 PM | 代码审查 & 提交 | 1 小时 |

### 3.3 提交规范

```bash
# 提交信息模板
git commit -m "fix(fracdiff): 修正权重计算公式符号错误

- 公式修正: (d - k + 1) → (k - 1 - d)
- 影响: 修复 d=1 时计算错误，确保 0<d<1 权重符号正确
- 测试: 新增边界测试验证 d=0 和 d=1 行为
- 审计: CLOSES Phase C Step 3 公式符号问题"
```

---

## 4. 参考文档

- [PHASE_C_IMPL_GUIDE.md](/docs/PHASE_C_IMPL_GUIDE.md) - Step 3 FracDiff 规范
- [OR5_CODE_AUDIT.md](/docs/audit/OR5_CODE_AUDIT.md) - 审计记录
- [AFML Chapter 5](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises) - 分数阶差分参考实现

---

## 5. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 公式修正后 ADF 检验不通过 | 高 | 重新校准 optimal_d 搜索范围 |
| 与其他特征工程代码耦合 | 中 | 全面回归测试 |
| 历史回测结果变化 | 中 | 记录变更，更新基线 |

---

## 6. 审批记录

| 角色 | 姓名 | 审批意见 | 日期 |
|------|------|----------|------|
| 审计发现 | 寇连材 | 公式符号错误确认 | 2026-02-27 |
| 任务制定 | 张得功 | 修复任务已制定 | 2026-02-27 |
| 修复实施 | 待定 | | |
| 代码审查 | 待定 | | |

---

*张得功 制定*  
*长春宫量化金融团队*
