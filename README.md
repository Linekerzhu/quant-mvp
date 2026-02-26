# Quant MVP - AI-Powered Quantitative Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📊 Project Status & Navigation

> **当前阶段**: OR5 审计已完成 ✅，准备进入 Phase C 实施
> 
> **最新版本**: v4.2 (OR5 审计裁决版)
> 
> **测试基线**: 97/97 passing，覆盖率 57%

### 核心文档导航

| 文档 | 位置 | 说明 |
|------|------|------|
| **📋 项目总规划** | [`plan.md`](plan.md) | 完整的6阶段执行计划（A→B→C→D→E→F） |
| **🚧 Phase C 施工指南** | [`docs/PHASE_C_IMPL_GUIDE.md`](docs/PHASE_C_IMPL_GUIDE.md) | 4步SOP详细实施手册 |
| **🤝 OR5 审计契约** | [`docs/OR5_CONTRACT.md`](docs/OR5_CONTRACT.md) | 审计官签署的5项红线契约 |
| **🔍 审计历史** | [`docs/audit/`](docs/audit/) | 所有审计轮次的完整记录 |
| **📝 变更日志** | [`CHANGELOG.md`](CHANGELOG.md) | 版本演进和重要变更 |

### 当前进度

```
Phase A: 数据管道        ████████████████ 100% ✅
Phase B: 特征与标签      ████████████████ 100% ✅
Phase C: Meta-Labeling   ░░░░░░░░░░░░░░░░   0% 🔴 待开工
Phase D: 风控系统        ░░░░░░░░░░░░░░░░   0% ⏸️
Phase E: 模拟盘          ░░░░░░░░░░░░░░░░   0% ⏸️
Phase F: 实盘            ░░░░░░░░░░░░░░░░   0% ⏸️
```

### 审计历史快览

| 轮次 | 日期 | 状态 | 关键发现 |
|------|------|------|----------|
| **OR5** | 2026-02-26 | ✅ 已完成 | Meta-Labeling 强制架构、FracDiff、CPCV 手写 |
| OR4 | 2026-02-25 | ✅ 已完成 | Phase A/B 数据管道安全审计 |
| OR3 | - | - | （跳过，合并至 OR4） |

**OR5 最新整改**: commit `5c35141` - 7项整改全部完成（Burn-in预警、features.yaml修正、embargo缺口文档化等）

---

## Overview

A production-grade quantitative trading system for US equities (S&P 500) with:
- **Meta-Labeling Architecture**: Base Model → Meta Model pipeline (OR5 审计强制)
- **Machine Learning**: LightGBM with hardened anti-overfitting parameters
- **Feature Engineering**: FracDiff (Fractional Differentiation) for memory preservation
- **Rigor**: Hand-written CPCV (Combinatorial Purged K-Fold), Deflated Sharpe, PBO detection
- **Risk Control**: Fractional Kelly sizing, multi-layer circuit breakers
- **Live Trading**: Futu OpenAPI (Moomoo) integration with simulate → real progression

## Architecture (v4.2)

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
├── plan.md                      # 📋 项目总规划 (v4.2)
├── CHANGELOG.md                 # 📝 版本变更日志
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
│   ├── PHASE_C_IMPL_GUIDE.md    # Phase C 4步实施指南
│   └── audit/                   # 🔍 审计记录
│       ├── OR5_CODE_AUDIT.md    # OR5 代码级审计 + 整改计划
│       ├── AUDIT_PHASE_A.md     # OR4: Phase A 审计
│       └── AUDIT_PHASE_B.md     # OR4: Phase B 审计
├── src/                         # 源代码
│   ├── data/                    # 数据采集 & 验证
│   ├── features/                # 特征工程 + FracDiff
│   ├── labels/                  # Triple Barrier 标注
│   ├── signals/                 # Base Models (待建)
│   ├── models/                  # Meta-Labeling + CPCV (待建)
│   ├── backtest/                # 回测引擎
│   ├── risk/                    # 风险管理
│   ├── execution/               # 交易执行 (Futu OpenAPI)
│   └── ops/                     # 运维 & 调度
├── tests/                       # 测试套件 (静态 mock 数据)
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

## Phase C Implementation (Next Steps)

Phase C follows a strict 4-step SOP (see `docs/PHASE_C_IMPL_GUIDE.md`):

| Step | Component | File to Create | Description |
|------|-----------|----------------|-------------|
| 1 | Base Model | `src/signals/base_models.py` | SMA Cross + Momentum signals (炮灰信号源) |
| 2 | CPCV | `src/models/purged_kfold.py` | Hand-written PurgedKFold (15 paths) |
| 3 | FracDiff | `src/features/fracdiff.py` | Fractional Differentiation (d ≈ 0.4) |
| 4 | Meta-MVP | `src/models/meta_trainer.py` | Full Meta-Labeling pipeline |

**⚠️ 重要约束**:
- Burn-in 衔接：FracDiff 必须在全量时间轴预计算（见 IMPL_GUIDE Step 3）
- Embargo 缺口：当前 embargo=40 < feature_lookback=60，需要补偿（见 IMPL_GUIDE Step 2）

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
| `plan.md` | 完整项目规划（v4.2） |
| `docs/OR5_CONTRACT.md` | OR5 审计契约 |
| `docs/PHASE_C_IMPL_GUIDE.md` | Phase C 4步实施指南 |
| `docs/audit/OR5_CODE_AUDIT.md` | OR5 代码级审计记录 |
| `config/training.yaml` | LightGBM 硬化参数 |
| `src/labels/triple_barrier.py` | Maximum Pessimism Principle |

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk.

---

*Quant MVP v4.2 - OR5 Audit Complete, Phase C Ready* 🚦
