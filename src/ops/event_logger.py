"""
Event Logger - Structured append-only event logging

All system events are logged in JSON Lines format for auditability.
"""

import json
import os
from datetime import datetime, timezone  # P2-C3: Add timezone for UTC
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class EventLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class EventLogger:
    """Append-only structured event logger."""
    
    def __init__(self, log_dir: str = "logs/events"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file (daily rotation)
        self.current_file = self._get_current_file()
    
    def _get_current_file(self) -> Path:
        """Get current day's log file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # P2-C3: UTC
        return self.log_dir / f"events_{date_str}.jsonl"
    
    def _ensure_file(self):
        """Ensure we're writing to the correct file (handles date rollover)."""
        expected = self._get_current_file()
        if expected != self.current_file:
            self.current_file = expected
    
    def log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        level: EventLevel = EventLevel.INFO,
        symbol: Optional[str] = None
    ) -> None:
        """
        Log an event.
        
        Args:
            event_type: Type of event (e.g., 'data_ingest', 'model_train')
            payload: Event-specific data
            level: Event severity level
            symbol: Optional symbol for stock-specific events
        """
        self._ensure_file()
        
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # P2-C3: UTC
            "level": level.value,
            "type": event_type,
            "symbol": symbol,
            "payload": payload
        }
        
        # Append to file
        with open(self.current_file, 'a') as f:
            f.write(json.dumps(event, default=str) + '\n')
    
    # Convenience methods
    def debug(self, event_type: str, payload: Dict[str, Any], symbol: Optional[str] = None):
        self.log(event_type, payload, EventLevel.DEBUG, symbol)
    
    def info(self, event_type: str, payload: Dict[str, Any], symbol: Optional[str] = None):
        self.log(event_type, payload, EventLevel.INFO, symbol)
    
    def warn(self, event_type: str, payload: Dict[str, Any], symbol: Optional[str] = None):
        self.log(event_type, payload, EventLevel.WARN, symbol)
    
    def error(self, event_type: str, payload: Dict[str, Any], symbol: Optional[str] = None):
        self.log(event_type, payload, EventLevel.ERROR, symbol)


# Global logger instance
_logger: Optional[EventLogger] = None


def get_logger() -> EventLogger:
    """Get or create global event logger."""
    global _logger
    if _logger is None:
        _logger = EventLogger()
    return _logger
