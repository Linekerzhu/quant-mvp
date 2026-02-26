# Quant MVP - AI-Powered Quantitative Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

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
├── config/           # Configuration files (YAML)
├── docs/             # Documentation
│   ├── OR5_CONTRACT.md         # OR5 Audit Contract (审计契约)
│   └── PHASE_C_IMPL_GUIDE.md   # Phase C Implementation Guide
├── src/              # Source code
│   ├── data/         # Data ingestion & validation
│   ├── features/     # Feature engineering + FracDiff
│   ├── labels/       # Triple Barrier labeling (Maximum Pessimism)
│   ├── signals/      # Base Models (SMA Cross, Momentum)
│   ├── models/       # Meta-Labeling + CPCV + PurgedKFold
│   ├── backtest/     # Backtesting engine
│   ├── risk/         # Risk management
│   ├── execution/    # Trading execution (Futu OpenAPI)
│   └── ops/          # Operations & scheduling
├── tests/            # Test suite (static mock data)
├── data/             # Data storage
├── models/           # Trained models
└── reports/          # Output reports
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
  feature_fraction: 0.5     # LOCKED - 双重随机化
  learning_rate: 0.01       # LOCKED - 降速学习
  lambda_l1: 1.0            # LOCKED - 特征稀疏化
```

## Phase C Implementation

Phase C follows a strict 4-step SOP (see `docs/PHASE_C_IMPL_GUIDE.md`):

| Step | Component | Description |
|------|-----------|-------------|
| 1 | Base Model | SMA Cross + Momentum signals (炮灰信号源) |
| 2 | CPCV | Hand-written `PurgedKFold` (15 paths) |
| 3 | FracDiff | Fractional Differentiation (d ≈ 0.4) |
| 4 | Meta-MVP | Full Meta-Labeling pipeline |

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

## Key Files

| File | Purpose |
|------|---------|
| `plan.md` | Full project plan (v4.2) |
| `docs/OR5_CONTRACT.md` | Audit contract |
| `docs/PHASE_C_IMPL_GUIDE.md` | Step-by-step implementation guide |
| `config/training.yaml` | LightGBM parameters (locked) |
| `src/labels/triple_barrier.py` | Maximum Pessimism Principle |

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk.

---

*Quant MVP v4.2 - OR5 Audit Ruling: Meta-Labeling Architecture*
