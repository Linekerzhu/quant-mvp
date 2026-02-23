"""
Tests for event logger module.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.ops.event_logger import EventLogger, EventLevel, get_logger


class TestEventLogger:
    """Test event logging functionality."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_logger_creation(self, temp_log_dir):
        """Test logger can be created."""
        logger = EventLogger(log_dir=temp_log_dir)
        assert logger.log_dir.exists()
    
    def test_log_event(self, temp_log_dir):
        """Test logging an event."""
        logger = EventLogger(log_dir=temp_log_dir)
        
        logger.info("test_event", {"key": "value"}, symbol="AAPL")
        
        # Check log file was created
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 1
        
        # Check content
        with open(log_files[0]) as f:
            event = json.loads(f.readline())
        
        assert event['type'] == 'test_event'
        assert event['level'] == 'INFO'
        assert event['symbol'] == 'AAPL'
        assert event['payload']['key'] == 'value'
        assert 'timestamp' in event
    
    def test_log_levels(self, temp_log_dir):
        """Test different log levels."""
        logger = EventLogger(log_dir=temp_log_dir)
        
        logger.debug("debug_event", {})
        logger.info("info_event", {})
        logger.warn("warn_event", {})
        logger.error("error_event", {})
        
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        assert len(log_files) == 1
        
        with open(log_files[0]) as f:
            events = [json.loads(line) for line in f]
        
        levels = [e['level'] for e in events]
        assert 'DEBUG' in levels
        assert 'INFO' in levels
        assert 'WARN' in levels
        assert 'ERROR' in levels
    
    def test_append_only(self, temp_log_dir):
        """Test that logging is append-only."""
        logger = EventLogger(log_dir=temp_log_dir)
        
        logger.info("first", {})
        logger.info("second", {})
        
        log_files = list(Path(temp_log_dir).glob("*.jsonl"))
        
        with open(log_files[0]) as f:
            events = [json.loads(line) for line in f]
        
        assert len(events) == 2
        assert events[0]['type'] == 'first'
        assert events[1]['type'] == 'second'
    
    def test_global_logger(self):
        """Test global logger singleton."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2
