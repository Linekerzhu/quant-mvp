"""
Daily Job - Main Pipeline Orchestration

Coordinates the daily data pipeline with idempotency guarantees.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import yaml
import pandas as pd

from src.data.ingest import DualSourceIngest
from src.data.validate import DataValidator
from src.data.integrity import IntegrityManager
from src.data.corporate_actions import CorporateActionsHandler
from src.data.universe import UniverseManager
from src.ops.event_logger import get_logger

logger = get_logger()


class DailyJob:
    """Idempotent daily data pipeline."""
    
    def __init__(self):
        self.ingest = DualSourceIngest()
        self.validator = DataValidator()
        self.integrity = IntegrityManager()
        self.corp_actions = CorporateActionsHandler()
        self.universe = UniverseManager()
        
        # Load config
        with open("config/universe.yaml") as f:
            self.config = yaml.safe_load(f)
    
    def run(
        self,
        trade_date: Optional[str] = None,
        resume_from: Optional[str] = None
    ) -> bool:
        """
        Run daily pipeline.
        
        Args:
            trade_date: Date to run for (default: today)
            resume_from: Step to resume from (for idempotency)
            
        Returns:
            Success flag
        """
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("daily_job_start", {"trade_date": trade_date, "resume_from": resume_from})
        
        steps = [
            ("ingest", self._step_ingest),
            ("validate", self._step_validate),
            ("integrity", self._step_integrity),
            ("corporate_actions", self._step_corp_actions),
            ("universe", self._step_universe),
        ]
        
        # Skip completed steps if resuming
        skip = resume_from is not None
        
        for step_name, step_func in steps:
            if skip and step_name != resume_from:
                logger.info("step_skipped", {"step": step_name, "reason": "already_complete"})
                continue
            
            skip = False
            
            try:
                result = step_func(trade_date)
                logger.info("step_complete", {"step": step_name, "result": result})
            except Exception as e:
                logger.error("step_failed", {"step": step_name, "error": str(e)})
                return False
        
        logger.info("daily_job_complete", {"trade_date": trade_date})
        return True
    
    def _step_ingest(self, trade_date: str):
        """Step 1: Data ingestion."""
        # Get universe symbols
        universe_info = self.universe.build_universe(
            pd.DataFrame()  # Would load existing data
        )
        
        # Ingest last 5 days (for feature calculation)
        end = pd.Timestamp(trade_date)
        start = end - timedelta(days=10)
        
        data = self.ingest.ingest(
            symbols=universe_info['symbols'][:10],  # Limit for MVP
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d')
        )
        
        # Save to raw
        data.to_parquet(f"data/raw/daily_{trade_date}.parquet", index=False)
        
        return {"rows": len(data), "symbols": data['symbol'].nunique()}
    
    def _step_validate(self, trade_date: str):
        """Step 2: Data validation."""
        import pandas as pd
        
        data = pd.read_parquet(f"data/raw/daily_{trade_date}.parquet")
        
        passed, cleaned, report = self.validator.validate(data)
        
        # Save validated data
        cleaned.to_parquet(f"data/processed/validated_{trade_date}.parquet", index=False)
        
        return {"passed": passed, "pass_rate": report.get('pass_rate', 0)}
    
    def _step_integrity(self, trade_date: str):
        """Step 3: Integrity check and hash freezing."""
        import pandas as pd
        
        data = pd.read_parquet(f"data/processed/validated_{trade_date}.parquet")
        
        # Check for drift
        universe_size = data['symbol'].nunique()
        should_freeze, drifts = self.integrity.detect_drift(data, universe_size)
        
        if should_freeze:
            raise RuntimeError(f"Data drift detected: {len(drifts)} events")
        
        # Freeze hashes
        self.integrity.freeze_data(data)
        
        # Create snapshot
        snapshot = self.integrity.create_snapshot(data, f"{trade_date}_v1")
        
        return {"snapshot": str(snapshot), "drifts": len(drifts)}
    
    def _step_corp_actions(self, trade_date: str):
        """Step 4: Corporate actions processing."""
        import pandas as pd
        
        data = pd.read_parquet(f"data/processed/validated_{trade_date}.parquet")
        
        processed, info = self.corp_actions.apply_all(data)
        
        # Save processed data
        processed.to_parquet(f"data/processed/corp_actions_{trade_date}.parquet", index=False)
        
        return info
    
    def _step_universe(self, trade_date: str):
        """Step 5: Universe management."""
        import pandas as pd
        
        data = pd.read_parquet(f"data/processed/corp_actions_{trade_date}.parquet")
        
        universe_info = self.universe.build_universe(data)
        
        # Save universe info
        import json
        with open(f"data/processed/universe_{trade_date}.json", 'w') as f:
            json.dump(universe_info, f, indent=2, default=str)
        
        return universe_info['metadata']


if __name__ == '__main__':
    job = DailyJob()
    success = job.run()
    exit(0 if success else 1)
