# Phase C Step 1 审计申请

> **申请人**: 李得勤（八品首领太监）
> **被审计人**: 寇连材（从九品随侍太监）
> **日期**: 2026-02-27
> **任务**: Phase C Step 1 - Base Models for Meta-Labeling

---

## 审计范围

本次审计涵盖以下代码变更：

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `src/signals/base_models.py` | BaseModelSMA + BaseModelMomentum 类 |
| `tests/test_base_models.py` | 12个测试用例 |

### 2. 修改文件

| 文件 | 说明 |
|------|------|
| `src/labels/triple_barrier.py` | 新增 Meta-Labeling 支持（side 列过滤） |

---

## 自检结果

| 测试类别 | 结果 |
|----------|------|
| 新增单元测试 | 12/12 通过 ✅ |
| 回归测试 | 111/111 通过 ✅ |

---

## 审计要求

根据流程，李得勤自检合格后提交寇连材进行**深度审计**：

### 必须验证项

1. **可运行性**
   - [ ] Base Model 代码可正常 import
   - [ ] 端到端可运行（Base Model → Triple Barrier）

2. **逻辑正确性**
   - [ ] `shift(1)` 正确应用（防泄漏）
   - [ ] 冷启动返回 side=0
   - [ ] Triple Barrier 正确过滤 side=0

3. **仿真测试**
   - [ ] 用 mock 数据跑完整流程
   - [ ] 验证信号与标签对齐

4. **审计报告**
   - [ ] 形成 Markdown 审计报告
   - [ ] 记录问题（如有）和建议

---

## 交付物

请寇连材生成审计报告文档：
- `docs/audit/PHASE_C_STEP1_AUDIT.md`

---

**有劳寇连材公公仔细审核！**

*李得勤敬上*
