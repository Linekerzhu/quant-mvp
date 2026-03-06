import socket
import time
from typing import Optional

from src.ops.event_logger import get_logger
from src.ops.alerts import AlertManager

logger = get_logger()

class OpenDMonitor:
    """
    Monitors the FutuOpenD gateway process.
    Typically run as a background service or cron job.
    """
    def __init__(self, host: str = '127.0.0.1', port: int = 11111, max_failures: int = 3):
        self.host = host
        self.port = port
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.alerts = AlertManager()
        
    def check_connection(self) -> bool:
        """
        Attempts to open a TCP socket to the OpenD gateway.
        Returns True if successful, False otherwise.
        """
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0):
                self.consecutive_failures = 0
                logger.info("opend_monitor_ok", {"host": self.host, "port": self.port})
                return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            self.consecutive_failures += 1
            logger.warn("opend_monitor_failed", {
                "host": self.host, 
                "port": self.port, 
                "failures": self.consecutive_failures,
                "error": str(e)
            })
            
            if self.consecutive_failures == self.max_failures:
                self._trigger_alert(str(e))
                
            return False
            
    def _trigger_alert(self, error_str: str):
        title = "FutuOpenD Gateway Offline"
        message = (
            f"CRITICAL: The FutuOpenD gateway at {self.host}:{self.port} is unreachable.\n"
            f"Consecutive failures: {self.consecutive_failures}\n"
            f"Error: {error_str}\n\n"
            "Action required: Check host machine, unlock via verification code and restart."
        )
        self.alerts.send_alert("CRITICAL", title, message)
        
    def run_loop(self, interval_seconds: int = 60):
        """
        Runs a continuous monitoring loop.
        """
        logger.info("opend_monitor_started", {"interval": interval_seconds})
        try:
            while True:
                self.check_connection()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("opend_monitor_stopped", {})
            
if __name__ == '__main__':
    # When run stand-alone
    monitor = OpenDMonitor()
    monitor.run_loop()
