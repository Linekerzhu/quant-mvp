import os
import smtplib
from email.message import EmailMessage
import requests
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
        
        # Email settings
        self.email_to = os.getenv("EMAIL_TO") or os.getenv("ALERT_EMAIL_TO")
        self.smtp_host = os.getenv("EMAIL_SMTP_SERVER") or os.getenv("SMTP_HOST")
        self.smtp_port = os.getenv("EMAIL_PORT") or os.getenv("SMTP_PORT", 587)
        self.smtp_user = os.getenv("EMAIL_USERNAME") or os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("EMAIL_PASSWORD") or os.getenv("SMTP_PASS")
        
        # Telegram settings
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def send_alert(self, level: str, title: str, message: str) -> bool:
        """
        Send an alert message via configured channels.
        Level: INFO, WARNING, CRITICAL
        """
        success = False
        
        # 1. Telegram Alert
        if self.telegram_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"[{level}] {title}\n\n{message}"
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    logger.info("telegram_alert_sent", {"level": level, "title": title})
                    success = True
                else:
                    logger.error("telegram_alert_failed", {"status": resp.status_code, "text": resp.text})
            except Exception as e:
                logger.error("telegram_alert_exception", {"error": str(e)})
                
        # 2. Email Alert
        if self.email_to and self.smtp_host and self.smtp_user and self.smtp_pass:
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
                    
                logger.info("email_alert_sent", {"level": level, "title": title})
                success = True
            except Exception as e:
                logger.error("email_alert_exception", {"error": str(e)})
                
        if not (self.telegram_token and self.telegram_chat_id) and not (self.email_to and self.smtp_host):
            logger.warn("alert_skipped_no_config", {"title": title})
            
        return success
