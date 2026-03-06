import yaml
from typing import Dict, List, Any
import pandas as pd

from src.ops.event_logger import get_logger

logger = get_logger()

class RiskEngine:
    """
    Multi-layer Risk Management Engine (L1 to L4).
    """
    def __init__(self, config_path: str = 'config/risk_limits.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.single_stock_max = self.config['position_level']['single_stock_max']
        self.industry_max = self.config['position_level']['industry_max']
        self.daily_loss_limit = self.config['portfolio_level']['daily_loss_limit']
        
        self.dd_warning = self.config['portfolio_level']['max_drawdown']['warning']
        self.dd_kill = self.config['portfolio_level']['max_drawdown']['kill_switch']
        self.max_consecutive_losses = self.config['portfolio_level']['consecutive_losses']['threshold']
        
        # System state
        self.is_kill_switched = False
        self.is_auto_degraded = False

    def validate_positions(
        self, 
        target_positions: pd.DataFrame, 
        sector_map: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Validate and clip positions against L2 Risk Limits.
        
        Args:
            target_positions: DataFrame with 'symbol' and 'target_weight'
            sector_map: Mapping of symbol to sector/industry
        """
        if target_positions.empty:
            return target_positions
            
        df = target_positions.copy()
        
        # Cap single stock exposure
        df['target_weight'] = df['target_weight'].clip(
            lower=-self.single_stock_max, 
            upper=self.single_stock_max
        )
        
        # Cap industry exposure if sector map provided
        if sector_map and self.config['position_level']['track_sectors']:
            df['sector'] = df['symbol'].map(sector_map)
            sector_exposure = df.groupby('sector')['target_weight'].sum().abs()
            
            for sector, exposure in sector_exposure.items():
                if exposure > self.industry_max:
                    scale = self.industry_max / exposure
                    mask = df['sector'] == sector
                    df.loc[mask, 'target_weight'] *= scale
                    logger.warn("sector_exposure_exceeded", {"sector": sector, "exposure": float(exposure), "limit": self.industry_max})
                    
        return df[['symbol', 'target_weight']]

    def check_portfolio_health(
        self, 
        daily_pnl: float, 
        current_drawdown: float, 
        consecutive_loss_days: int
    ) -> Dict[str, Any]:
        """
        L3 and L4 Risk checks. Triggered dynamically.
        
        Returns:
            Dict containing actions to take (e.g., reduce_risk, kill_switch, degrade)
        """
        action = "NORMAL"
        msg = ""
        multiplier = 1.0
        
        if current_drawdown >= self.dd_kill:
            self.is_kill_switched = True
            action = "KILL_SWITCH"
            msg = f"Drawdown {current_drawdown:.2%} >= Kill Switch limit {self.dd_kill:.2%}"
            multiplier = 0.0
        elif current_drawdown >= self.dd_warning:
            action = "REDUCE_RISK"
            msg = f"Drawdown {current_drawdown:.2%} >= Warning limit {self.dd_warning:.2%}"
            # Further reduction logic handled by PositionSizer, but RiskEngine can override
            
        if daily_pnl <= -self.daily_loss_limit:
            if action != "KILL_SWITCH":
                action = "HALT_TRADING_TODAY"
                msg = f"Daily loss {daily_pnl:.2%} breached limit {self.daily_loss_limit:.2%}"
                multiplier = 0.0
                
        if consecutive_loss_days >= self.max_consecutive_losses:
            if action not in ["KILL_SWITCH", "HALT_TRADING_TODAY"]:
                action = "REDUCE_RISK"
                msg = f"{consecutive_loss_days} consecutive losses"
                multiplier = min(multiplier, 0.5)
                
        # L4 Auto Degrade
        if current_drawdown >= self.config['system_level']['auto_degrade']['triggers']['drawdown']['threshold']:
            self.is_auto_degraded = True
            if action != "KILL_SWITCH":
                msg += " | AUTO_DEGRADE triggered"
        
        if action != "NORMAL":
            logger.warn("risk_event", {"action": action, "message": msg})
            
        return {
            "status": action,
            "message": msg,
            "multiplier": multiplier,
            "is_kill_switched": self.is_kill_switched,
            "is_auto_degraded": self.is_auto_degraded
        }
