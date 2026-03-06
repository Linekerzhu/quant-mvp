import os
import smtplib
from email.message import EmailMessage
from typing import Dict, Any

from src.ops.event_logger import get_logger

logger = get_logger()

class AlertManager:
    """
    Handles critical system alerts via email / Telegram.
    """
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.email_to = os.getenv("ALERT_EMAIL_TO")
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = os.getenv("SMTP_PORT", 587)
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        
    def send_alert(self, level: str, title: str, message: str) -> bool:
        """
        Send an alert message.
        Level: INFO, WARNING, CRITICAL
        """
        if not self.email_to or not self.smtp_host:
            logger.warn("alert_skipped_no_config", {"title": title})
            return False
            
        try:
            msg = EmailMessage()
            msg.set_content(message)
            msg['Subject'] = f"[{level}] Quant-MVP: {title}"
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to
            
            with smtplib.SMTP(self.smtp_host, int(self.smtp_port)) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
                
            logger.info("alert_sent", {"level": level, "title": title})
            return True
            
        except Exception as e:
            logger.error("alert_failed", {"error": str(e), "title": title})
            return False
