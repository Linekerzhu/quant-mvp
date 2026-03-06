# 🤖 AI Agent Deployment & Operations Guide

**Target Audience:** AI Agents (Coding Assistants, Ops Agents, Auto-Deployment bots)
**System:** Quant-MVP US Equities Trading System
**Current Status:** Phase E (Simulated Paper Trading) Ready

This document provides explicit instructions for an AI Agent to deploy, maintain, and monitor the `quant-mvp` trading system autonomously.

---

## 🚀 1. Deployment SOP (Standard Operating Procedure)

When instructed to deploy this system to a new server/container, follow these exact steps:

### 1.1 Environment Setup
1. **Clone/Copy Project**: Ensure the project is in the working directory.
2. **Install Python**: Verify Python 3.11+ is available.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Ensure lightgbm and catboost are also installed explicitly if missing
   pip install lightgbm catboost
   ```

### 1.2 Configuration
1. **Environment Variables**: Use `replace_file_content` to copy `.env.example` to `.env`.
   - Ensure `TIINGO_API_KEY`, `FUTU_HOST`, `FUTU_PORT` (usually 11111), and `FUTU_TRD_ENV` (SIMULATE or REAL) are set.
   - Configure SMTP credentials for `ALERT_*` variables.
2. **Configuration Files**: Verify `config/*.yaml`. (Leave defaults unless user requests changes).

### 1.3 FutuOpenD Gateway (Critical!)
1. Futu OpenAPI requires a local gateway.
2. Read `opend/README.md` for gateway details.
3. Start the gateway corresponding to the OS.
4. Verify connection using the test script:
   ```bash
   PYTHONPATH=. python3 tests/check_futu_connection.py
   ```
5. Setup the gateway monitor as a background daemon (or run it periodically):
   ```bash
   PYTHONPATH=. python3 src/ops/opend_monitor.py
   ```

### 1.4 Initial Model Generation (Cold Start)
The daily job (`src/ops/daily_job.py`) **requires** a trained Meta-Model bundle.
- **For full training**: Execute `run_pipeline.py`.
- **For immediate sandbox testing**: Execute the mock model builder:
  ```bash
  PYTHONPATH=. python3 tests/mock_model_builder.py
  ```
  *(This places a mock LightGBM bundle inside `models/`).*

### 1.5 Schedule the Daily Job
The system is entirely orchestrated by `src/ops/daily_job.py`. It is idempotent and safe to restart.
Setup a cron job to run this at Market Close (e.g., 16:30 EST) or according to user preference.
```bash
# Example cron entry for 16:30 EST (adjust timezone)
30 16 * * * cd /path/to/quant-mvp && PYTHONPATH=. python3 src/ops/daily_job.py >> logs/cron.log 2>&1
```

---

## 📈 2. Daily Operations & Monitoring (Agent Instructions)

When asked to "Check system status" or "Monitor the pipeline", perform these checks:

### 2.1 Check Event Logs
Logs are structured as JSONL in `logs/events/`. 
**Action:** Use `grep_search` or `run_command` (tail) to read today's log.
```bash
tail -n 50 logs/events/events_$(date +%Y-%m-%d).jsonl
```
- **Success Criteria:** Look for `{"type": "daily_job_complete"}`.
- **Error Handling:** If you find `"level": "ERROR"` or `"type": "step_failed"`, extract the payload, read the stack trace if available, and report exactly which step (ingest, features, models, execute) failed.

### 2.2 Idempotency and Resumption
If `daily_job.py` failed halfway, it saves its state in `data/checkpoints/YYYY-MM-DD.json`.
- **Action:** If the user asks you to retry, simply run `python3 src/ops/daily_job.py` again. It will automatically skip completed steps (e.g., if data was ingested, it starts from feature building).
- If you need a completely fresh run, delete the checkpoint first: `rm data/checkpoints/$(date +%Y-%m-%d).json`.

### 2.3 Verify Data Artifacts
Data flows through Parquet files in `data/processed/`.
- Ensure files like `features_YYYY-MM-DD.parquet` or `signals_YYYY-MM-DD.parquet` exist and are updating.

---

## 🛠 3. Weekly Maintenance tasks

When the user asks for a "Weekly Review" or it's scheduled on a weekend, trigger the reporting and calibration scripts:

### 3.1 Slippage Calibration
```bash
PYTHONPATH=. python3 src/backtest/cost_calibration.py
```
This will read recent trades, calculate actual spread vs expected, and dynamically update the `spread_bps` in `config/training.yaml`.

### 3.2 Generate Weekly Report
```bash
PYTHONPATH=. python3 src/ops/weekly_report.py
```
This generates a markdown report in `reports/weekly_report_YYYY_WW.md`.
**Action:** Use `view_file` to read the generated markdown and summarize it to the user conceptually (Alpha vs SPY, Signal consistency, Cost metrics).

---

## 🚨 4. Emergency & Alert Handling

If `src/ops/opend_monitor.py` or `daily_job.py` fires an alert to the user's email, the user might paste it to you.
1. **"Futu OpenD Disconnected"**: The gateway is down. Advise the user to restart the Futu client physically (it might require graphical 2FA or captcha).
2. **"PDT Guard Triggered"**: The system blocked a trade because the account would violate Pattern Day Trader rules. **Do not override this manually unless instructed.**
3. **"Risk Limits Exceeded"**: The Risk Engine halted execution. Check `config/risk_limits.yaml` and historical PnL to diagnose.

---

## 💡 5. Meta-Labeling Model Retraining
Typically done monthly.
1. Trigger `run_pipeline.py`.
2. Ensure the resulting `meta_training_results.yaml` shows **PBO < 0.3** and **DSR > 1.282**. If these fail mathematical simulation checks, **DO NOT deploy the new model**. Inform the user the alpha is decaying.
