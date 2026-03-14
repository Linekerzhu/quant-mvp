#!/bin/bash
# Daily Paper Trading Runner
# Runs at US market close (17:00 ET = 05:00 CST next day)
# This script is called by crontab

set -euo pipefail

PROJECT_DIR="/Users/zjz/quant-mvp"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

echo "=== Daily Paper Trading Run: $(date) ===" | tee "${LOG_FILE}"

# Run with PYTHONPATH set
export PYTHONPATH="${PROJECT_DIR}"
python3 run_daily_with_notify.py >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
echo "=== Finished: $(date), exit code: ${EXIT_CODE} ===" | tee -a "${LOG_FILE}"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "daily_*.log" -mtime +30 -delete 2>/dev/null || true

exit ${EXIT_CODE}
