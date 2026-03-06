import pytest
import os
from unittest.mock import patch, MagicMock
from src.ops.alerts import AlertManager

def test_alert_manager_no_config():
    with patch.dict(os.environ, {"ALERT_EMAIL_TO": "", "SMTP_HOST": ""}):
        manager = AlertManager()
        assert manager.send_alert("INFO", "Test", "Msg") is False

@patch("smtplib.SMTP")
def test_alert_manager_with_config(mock_smtp):
    manager = AlertManager()
    manager.email_to = "test@example.com"
    manager.smtp_host = "localhost"
    manager.smtp_port = 587
    manager.smtp_user = "user"
    manager.smtp_pass = "pass"
    
    assert manager.send_alert("WARNING", "Test", "Msg") is True
    assert mock_smtp.called
