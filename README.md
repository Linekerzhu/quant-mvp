# Quant MVP - AI-Powered Quantitative Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📊 Project Status & Navigation

> **当前阶段**: Phase A-B-C 全部完成 ✅ | Phase D 准备就绪 🚦
> 
> **最新版本**: v4.4 (R14-R18 内审修复完成)
> 
> **测试基线**: **165/165 passing** ✅
> 
> **代码覆盖率**: 语句 87% | 分支 78% | 函数 92%

### 核心文档导航

| 文档 | 位置 | 说明 |
|------|------|------|
| **📋 项目总规划** | [`plan.md`](plan.md) | 完整的6阶段执行计划（A→B→C→D→E→F） |
| **✅ Phase C 状态** | [`docs/PHASE_C_STATUS.md`](docs/PHASE_C_STATUS.md) | 实施完成报告 |
| **📊 修复记录汇总** | [`docs/audit/FIX_LOG.md`](docs/audit/FIX_LOG.md) | 所有内审修复记录 |
| **🤝 OR5 审计契约** | [`docs/OR5_CONTRACT.md`](docs/OR5_CONTRACT.md) | 审计官签署的5项红线契约 |
| **🔍 审计历史** | [`docs/audit/`](docs/audit/) | 所有审计轮次的完整记录 |
| **📝 变更日志** | [`CHANGELOG.md`](CHANGELOG.md) | 版本演进和重要变更 |

### 当前进度

```
Phase A: 数据管道        ████████████████ 100% ✅
Phase B: 特征与标签      ████████████████ 100% ✅
Phase C: Meta-Labeling   ████████████████ 100% ✅
Phase D: 风控系统        ░░░░░░░░░░░░░░░░   0% ⏸️  (待启动)
Phase E: 模拟盘          ░░░░░░░░░░░░░░░░   0% ⏸️  (待启动)
Phase F: 实盘            ░░░░░░░░░░░░░░░░   0% ⏸️  (待启动)
```

### 最新审计记录

| 轮次 | 日期 | 状态 | 关键发现 |
|------|------|------|----------|
| **R18** | 2026-03-01 | ✅ 已完成 | P0 全部修复、P1 冗余特征、P2 稳定性 |
| **R17** | 2026-03-01 | ✅ 已完成 | 日历错配、噪声确定性、备份源闭环 |
| **R16** | 2026-03-01 | ✅ 已完成 | R15 回归修复、CPCV 配置修正 |
| **R15** | 2026-03-01 | ✅ 已完成 | PBO 计算逻辑修正 |
| **R14** | 2026-03-01 | ✅ 已完成 | PBO/DSR 算法、样本权重、BaseModel |
| **EXT-Q** | 2026-03-01 | ✅ 已完成 | FracDiff 预计算、Early Stopping 隔离 |
| **OR9-13** | 2026-02-28 | ✅ 已完成 | 内审修复汇总 (P0-P2) |
| **OR5** | 2026-02-26 | ✅ 已完成 | Meta-Labeling 强制架构、FracDiff、CPCV 手写 |

---

## Overview

A production-grade quantitative trading system for US equities (S&P 500) with:
- **Meta-Labeling Architecture**: Base Model → Meta Model pipeline (OR5 审计强制)
- **Machine Learning**: LightGBM with hardened anti-overfitting parameters
- **Feature Engineering**: FracDiff (Fractional Differentiation) for memory preservation
- **Rigor**: Hand-written CPCV (Combinatorial Purged K-Fold), Deflated Sharpe, PBO detection
- **Risk Control**: Fractional Kelly sizing, multi-layer circuit breakers
- **Live Trading**: Futu OpenAPI (Moomoo) integration with simulate → real progression

## Architecture (v4.4)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Base Model    │────▶│  Triple Barrier  │────▶│   Meta Model    │
│  (SMA/Momentum) │     │    Labeling      │     │   (LightGBM)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   side ∈ {+1,-1,0}     Meta-Label: 1=profit     Probability p
                         Meta-Label: 0=loss      → Kelly Sizing
```

**Key Insight**: LightGBM does NOT predict price direction. It predicts **whether the Base Model's signal will be profitable**.

### v4.4 修复亮点

- **R14-R18 内审完成**: 所有 HIGH/MEDIUM 问题修复
- **PBO 算法修正**: 严格遵循 AFML Ch7 排名方法
- **DSR 完善**: 添加 skewness/kurtosis + 多重测试校正
- **样本权重优化**: per-fold 重算，避免测试集泄漏
- **FracDiff 全局预计算**: 避免 CPCV 每 fold 数据损失
- **Early Stopping 隔离**: 从训练集切分 validation，test set 不参与
- **Forward-only Purge**: 符合 AFML Ch7 标准

## Quick Start

### 1. Setup Environment
```bash
# Clone and enter directory
cd quant-mvp

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt
```

### 2. Run with Docker
```bash
docker-compose up -d
```

### 3. Run Tests
```bash
pytest tests/ -v
```

## Project Structure

```
quant-mvp/
├── plan.md                      # 📋 项目总规划 (v4.4)
├── CHANGELOG.md                 # 📝 版本变更日志
├── AUDIT_RECORDS_SUMMARY.md     # 📊 Phase A+B 审计汇总
├── config/                      # 配置文件 (YAML)
│   ├── data_contract.yaml       # 数据合约
│   ├── event_protocol.yaml      # Triple Barrier 参数
│   ├── universe.yaml            # 股票池定义
│   ├── features.yaml            # 特征注册表
│   ├── training.yaml            # LightGBM 硬化参数
│   ├── risk_limits.yaml         # 风控阈值
│   └── position_sizing.yaml     # Kelly 参数
├── docs/                        # 📚 文档中心
│   ├── OR5_CONTRACT.md          # OR5 审计契约
│   ├── PHASE_C_STATUS.md        # Phase C 完成状态
│   ├── PHASE_C_IMPL_GUIDE.md    # Phase C 4步实施指南
│   └── audit/                   # 🔍 审计记录
│       ├── FIX_LOG.md           # 完整修复记录汇总
│       ├── OR5_CODE_AUDIT.md    # OR5 代码级审计
│       ├── FULL_AUDIT_BY_LIANSHUN.md      # 完整审计
│       ├── AUDIT_PHASE_A.md     # Phase A 审计
│       ├── AUDIT_PHASE_B.md     # Phase B 审计
│       └── ...                  # 其他审计记录
├── src/                         # 源代码
│   ├── data/                    # 数据采集 & 验证
│   ├── features/                # 特征工程 + FracDiff
│   ├── labels/                  # Triple Barrier 标注
│   ├── signals/                 # Base Models
│   ├── models/                  # Meta-Labeling + CPCV + PBO/DSR
│   ├── backtest/                # 回测引擎
│   ├── risk/                    # 风险管理
│   ├── execution/               # 交易执行 (Futu OpenAPI)
│   └── ops/                     # 运维 & 调度
├── tests/                       # 测试套件 (165 tests)
├── data/                        # 数据存储
├── models/                      # 训练好的模型
└── reports/                     # 输出报告
```

## Configuration

See `config/` directory for all configuration files:
- `data_contract.yaml` - Data definitions & corporate actions
- `event_protocol.yaml` - Triple Barrier parameters
- `universe.yaml` - Stock universe & filters
- `features.yaml` - Feature registry
- `training.yaml` - Model training parameters (**LightGBM hardened**)
- `risk_limits.yaml` - Risk thresholds
- `position_sizing.yaml` - Kelly parameters

### LightGBM Hardened Parameters (OR5 Contract)

```yaml
lightgbm:
  max_depth: 3              # LOCKED - 严禁超过 3
  num_leaves: 7             # LOCKED - <= 2^3 - 1
  min_data_in_leaf: 200     # LOCKED - 强制统计显著性
  learning_rate: 0.01       # LOCKED - 降速学习
  lambda_l1: 1.0            # LOCKED - 特征稀疏化
  feature_fraction: 0.5     # LOCKED - 双重随机化
  n_estimators: 500         # Increased to compensate for low learning_rate
```

## Phase C Implementation ✅ 已完成

Phase C 已完成全部4步 SOP 实施 (详见 `docs/PHASE_C_IMPL_GUIDE.md`):

| Step | Component | File | Status | Description |
|------|-----------|------|--------|-------------|
| 1 | Base Model | `src/signals/base_models.py` | ✅ | SMA Cross + Momentum signals |
| 2 | CPCV | `src/models/purged_kfold.py` | ✅ | Hand-written PurgedKFold (15 paths) |
| 3 | FracDiff | `src/features/fracdiff.py` | ✅ | Fractional Differentiation (d ≈ 0.4) |
| 4 | Meta-MVP | `src/models/meta_trainer.py` | ✅ | Full Meta-Labeling pipeline |

### 核心算法实现

**PBO (Probability of Backtest Overfitting)**
- 实现: `src/models/overfitting.py`
- 算法: AFML Ch7 §7.4.2 rank-based 方法
- 定义: PBO = P(IS最优路径在OOS表现差于中位数)

**DSR (Deflated Sharpe Ratio)**
- 实现: `src/models/overfitting.py`
- 特性: 包含 skewness/kurtosis + 多重测试校正

**CPCV (Combinatorial Purged K-Fold)**
- 实现: `src/models/purged_kfold.py`
- 特性: Forward-only purge, 严格 embargo

### 修复亮点 (v4.4)

- ✅ **R14**: PBO/DSR 算法严格对齐 AFML
- ✅ **R14-A3**: per-fold 样本权重重算
- ✅ **EXT-Q2**: FracDiff 全局预计算
- ✅ **EXT-Q1**: Early Stopping 隔离 test set
- ✅ **A5**: Forward-only purge (AFML 标准)
- ✅ **A8**: purge_window 10→60

## OR5 Audit Contract

This project follows the OR5 Audit Contract (`docs/OR5_CONTRACT.md`):

1. **LightGBM Parameter Lock** - max_depth=3, num_leaves=7, min_data_in_leaf=200
2. **Meta-Labeling Mandatory** - No direct price prediction
3. **FracDiff Required** - Preserve memory while achieving stationarity
4. **CPCV Hand-written** - No sklearn KFold, proper Purge+Embargo
5. **Data Tech Debt Provision** - CAGR -3%, MDD +10%

## Development

### Code Style
```bash
black src/ tests/
isort src/ tests/
```

### Testing
All tests use static mock data in `tests/fixtures/` - no network calls.

```bash
# Run all tests
pytest tests/ -v

# Run specific Phase C tests
pytest tests/test_base_models.py tests/test_cpcv.py tests/test_fracdiff.py tests/test_meta_trainer.py -v
```

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `plan.md` | 完整项目规划（v4.4） |
| `docs/OR5_CONTRACT.md` | OR5 审计契约 |
| `docs/PHASE_C_STATUS.md` | Phase C 完成状态 |
| `docs/PHASE_C_IMPL_GUIDE.md` | Phase C 4步实施指南 |
| `docs/audit/FIX_LOG.md` | 完整修复记录汇总 |
| `config/training.yaml` | LightGBM 硬化参数 |
| `src/labels/triple_barrier.py` | Maximum Pessimism Principle |
| `src/models/overfitting.py` | PBO/DSR 实现 |
| `src/models/meta_trainer.py` | Meta-Labeling Pipeline |

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk.

---

Quant MVP v4.4 - Phase A-B-C Complete ✅ | R14-R18 Audits Complete ✅ | 165/165 Tests Passing ✅
