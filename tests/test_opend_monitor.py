import pytest
from unittest.mock import patch, MagicMock
import socket
from src.ops.opend_monitor import OpenDMonitor

@patch("socket.create_connection")
@patch("src.ops.alerts.AlertManager.send_alert")
def test_opend_monitor_success(mock_alert, mock_socket):
    # Simulate successful connection
    mock_socket.return_value = MagicMock()
    
    monitor = OpenDMonitor(max_failures=3)
    res = monitor.check_connection()
    
    assert res is True
    assert monitor.consecutive_failures == 0
    assert not mock_alert.called

@patch("socket.create_connection")
@patch("src.ops.alerts.AlertManager.send_alert")
def test_opend_monitor_failure_alert(mock_alert, mock_socket):
    # Simulate connection refused
    mock_socket.side_effect = ConnectionRefusedError("Connection refused")
    
    monitor = OpenDMonitor(max_failures=3)
    
    # Fail 1
    assert monitor.check_connection() is False
    assert monitor.consecutive_failures == 1
    assert not mock_alert.called
    
    # Fail 2
    assert monitor.check_connection() is False
    assert monitor.consecutive_failures == 2
    assert not mock_alert.called
    
    # Fail 3 (Triggers alert)
    assert monitor.check_connection() is False
    assert monitor.consecutive_failures == 3
    assert mock_alert.called
    
    # Verify alert content
    args, kwargs = mock_alert.call_args
    assert args[0] == "CRITICAL"
    assert "Offline" in args[1]
