import pytest
import os
import json
from src.ops.daily_job import DailyJob

def test_daily_job_idempotency(tmp_path):
    job = DailyJob()
    job.ckpt_dir = str(tmp_path)
    
    # Run load/save checkpoints
    trade_date = "2026-03-06"
    
    # Initially empty
    state = job._load_checkpoint(trade_date)
    assert not state
    
    # Save dummy state
    state["ingest"] = True
    state["validate"] = True
    job._save_checkpoint(trade_date, state)
    
    # Reload
    new_state = job._load_checkpoint(trade_date)
    assert new_state["ingest"] is True
    assert new_state.get("features", False) is False

