import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import yaml

from src.ops.event_logger import get_logger

logger = get_logger()

class IndependentKellySizer:
    """
    Independent Kelly Position Sizer with Fractional constraints and normalization.
    Implements Phase D requirements for sizing logic.
    """
    def __init__(self, config_path: str = 'config/position_sizing.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.kelly_config = self.config.get('kelly', {})
        self.fraction = self.kelly_config.get('fraction', 0.25)
        
        position_config = self.config.get('position', {})
        self.scales = position_config.get('scales', {})
        self.vol_target = position_config.get('volatility_target', {}).get('annual_target', 0.15)
        self.dd_scaling = position_config.get('drawdown_scaling', {})
        
        portfolio_config = self.config.get('portfolio', {})
        self.max_gross_leverage = portfolio_config.get('max_gross_leverage', 1.0)
        self.concentration = portfolio_config.get('concentration', {})
        self.max_single = self.concentration.get('max_single_position', 0.10)
        self.min_single = self.concentration.get('min_position_size', 0.01)

    def calculate_kelly_fraction(self, p: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate independent Kelly fraction for a single asset.
        f = p/a - (1-p)/b
        where:
        a = average loss (positive)
        b = average win (positive)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        q = 1.0 - p
        f = (p / avg_loss) - (q / avg_win)
        return max(0.0, f)

    def calculate_positions(
        self, 
        signals: pd.DataFrame, 
        current_drawdown: float = 0.0
    ) -> pd.DataFrame:
        """
        Calculate target positions based on Meta-Labeling signals and Kelly formula.
        
        Args:
            signals: DataFrame containing ['symbol', 'side', 'prob', 'avg_win', 'avg_loss', 'realized_vol']
            current_drawdown: Current portfolio drawdown (positive float, e.g., 0.05 for 5% DD)
        
        Returns:
            DataFrame with target positions ('symbol', 'target_weight')
        """
        if signals.empty:
            return pd.DataFrame({'symbol': [], 'target_weight': []})
            
        results = []
        for _, row in signals.iterrows():
            symbol = row['symbol']
            side = row.get('side', 0)
            if side == 0:
                results.append({'symbol': symbol, 'target_weight': 0.0})
                continue
                
            p = float(row.get('prob', 0.5))
            # Apply Kelly formula
            avg_win_val = float(row.get('avg_win', 0.0)) if 'avg_win' in row.index else 0.0
            avg_loss_val = float(row.get('avg_loss', 0.0)) if 'avg_loss' in row.index else 0.0
            
            if avg_win_val > 0 and avg_loss_val > 0:
                raw_kelly = self.calculate_kelly_fraction(p, avg_win_val, avg_loss_val)
            else:
                # If historical win/loss stats are missing, fallback to simple probability mapping
                logger.debug(f"Missing win/loss stats for {symbol}, falling back to half Kelly of simple prob logic")
                raw_kelly = max(0.0, p - (1-p)) # assuming a=1 b=1
            
            # 1. Fractional Kelly
            f_weight = raw_kelly * self.fraction
            
            # 2. Confidence scaling (optional based on config)
            if self.scales.get('confidence', True):
                # Scale from [0.5, 1.0] prob to [0, 1] weight multiplier
                conf_multiplier = max(0.0, (p - 0.5) * 2.0)
                f_weight *= conf_multiplier
                
            # 3. Volatility scaling
            if self.scales.get('volatility', True) and 'realized_vol' in row.index:
                r_vol = float(row['realized_vol'])
                if r_vol > 0:
                    vol_multiplier = self.vol_target / r_vol
                    f_weight *= vol_multiplier
                    
            results.append({'symbol': symbol, 'raw_weight': f_weight * side})

        df_pos = pd.DataFrame(results)
        
        # 4. Drawdown Scaling (Portfolio level modifier)
        dd_multiplier = 1.0
        if self.scales.get('drawdown', True) and current_drawdown > 0:
            threshold = self.dd_scaling.get('threshold', 0.05)
            ceiling = self.dd_scaling.get('ceiling', 0.10)
            floor = self.dd_scaling.get('floor', 0.25)
            
            if current_drawdown >= ceiling:
                dd_multiplier = floor
            elif current_drawdown > threshold:
                # Linear interpolation
                ratio = (current_drawdown - threshold) / (ceiling - threshold)
                dd_multiplier = 1.0 - ratio * (1.0 - floor)
                
            df_pos['raw_weight'] *= dd_multiplier

        # 5. Total Gross Leverage Normalization
        gross_exposure = df_pos['raw_weight'].abs().sum()
        if gross_exposure > self.max_gross_leverage:
            scale_factor = self.max_gross_leverage / gross_exposure
            df_pos['raw_weight'] *= scale_factor
            
        # 6. Apply individual concentration limits
        def apply_limits(w):
            sign = 1 if w > 0 else -1 if w < 0 else 0
            abs_w = abs(w)
            if abs_w < self.min_single:
                return 0.0
            if abs_w > self.max_single:
                return self.max_single * sign
            return w
            
        df_pos['target_weight'] = df_pos['raw_weight'].apply(apply_limits)
        
        # Ensure final check on gross limits after capping
        final_gross = df_pos['target_weight'].abs().sum()
        if final_gross > self.max_gross_leverage:
            final_scale = self.max_gross_leverage / final_gross
            df_pos['target_weight'] *= final_scale

        return df_pos[['symbol', 'target_weight']]
