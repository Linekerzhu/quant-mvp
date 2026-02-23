# Quant MVP - AI-Powered Quantitative Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Overview

A production-grade quantitative trading system for US equities (S&P 500) with:
- **Machine Learning**: LightGBM/CatBoost with Triple Barrier labeling
- **Rigor**: CPCV validation, Deflated Sharpe, PBO detection
- **Risk Control**: Fractional Kelly sizing, multi-layer circuit breakers
- **Live Trading**: Alpaca integration with paper → live progression

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
├── src/              # Source code
│   ├── data/         # Data ingestion & validation
│   ├── features/     # Feature engineering
│   ├── labels/       # Triple Barrier labeling
│   ├── models/       # ML models
│   ├── backtest/     # Backtesting engine
│   ├── risk/         # Risk management
│   ├── execution/    # Trading execution
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
- `training.yaml` - Model training parameters
- `risk_limits.yaml` - Risk thresholds
- `position_sizing.yaml` - Kelly parameters

## Development

### Code Style
```bash
black src/ tests/
isort src/ tests/
```

### Testing
All tests use static mock data in `tests/fixtures/` - no network calls.

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk.
