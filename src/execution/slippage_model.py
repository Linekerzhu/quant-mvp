import yaml
from pathlib import Path
from typing import Dict, Any

class SlippageModel:
    """
    Explicit Slippage and Cost Model based on Futu US Stock fee structure.
    Calculates expected fill price and associated fees for mock trading.
    """
    def __init__(self, config_path: str = "config/training.yaml"):
        path = Path(config_path)
        if not path.exists():
            path = Path(__file__).parent.parent.parent / config_path
            
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)["cost_model"]
            
        self.comm = self.config["commission"]
        # Use default Tier 3 for backtest/paper
        tier_idx = self.config["platform_fee"]["backtest_assumed_tier"] - 1
        self.pf = self.config["platform_fee"]["tiers"][tier_idx]
        self.reg = self.config["regulatory_fees"]
        
    def estimate_cost(self, qty: float, price: float, side: str, adv_usd: float = 0.0) -> Dict[str, float]:
        """
        Estimate exact slippage and commission costs.
        
        Args:
            qty: Order quantity
            price: Order price (mid price)
            side: 'buy' or 'sell'
            adv_usd: Average Daily Volume in USD
        """
        side = side.lower()
        trade_value = qty * price
        
        # 1. Commission
        commission = qty * self.comm["per_share_usd"]
        commission = max(commission, self.comm["min_per_order_usd"])
        max_comm = trade_value * self.comm["max_per_order_pct"]
        if commission > max_comm:
            commission = max_comm
            
        # 2. Platform fee
        platform_fee = qty * self.pf["per_share_usd"]
        platform_fee = max(platform_fee, self.pf["min_per_order_usd"])
        
        # 3. Regulatory fees
        sec_fee = 0.0
        taf_fee = 0.0
        finra_fee = trade_value * self.reg["finra_fee"]["rate"]
        
        if side == "sell":
            sec_fee = trade_value * self.reg["sec_fee"]["rate"]
            taf = qty * self.reg["taf_fee"]["per_share_usd"]
            taf = max(taf, self.reg["taf_fee"]["min_per_trade_usd"])
            taf = min(taf, self.reg["taf_fee"]["max_per_trade_usd"])
            taf_fee = taf
            
        # 4. Spread and Market Impact
        spread_bps = self.config["spread_bps"]["default"]
        if adv_usd > 0:
            if adv_usd < 20_000_000:
                spread_bps = self.config["spread_bps"]["by_adv_bucket"]["low"]
            elif adv_usd < 100_000_000:
                spread_bps = self.config["spread_bps"]["by_adv_bucket"]["mid"]
            else:
                spread_bps = self.config["spread_bps"]["by_adv_bucket"]["high"]
                
        spread_cost = trade_value * (spread_bps / 10000.0)
        
        impact_bps = 0.0
        if adv_usd > 0:
            coeff = self.config["impact_bps"]["coeff"]
            impact_bps = coeff * (trade_value / adv_usd) * 10000
            
        impact_cost = trade_value * (impact_bps / 10000.0)
        
        total_slippage = spread_cost + impact_cost
        total_fees = commission + platform_fee + sec_fee + taf_fee + finra_fee
        
        # Calculate execution price (worse than mid)
        price_impact = total_slippage / qty if qty > 0 else 0
        fill_price = price + price_impact if side == "buy" else price - price_impact
        
        return {
            "commission": commission,
            "platform_fee": platform_fee,
            "sec_fee": sec_fee,
            "taf_fee": taf_fee,
            "finra_fee": finra_fee,
            "total_fees": total_fees,
            "slippage_cost": total_slippage,
            "total_cost": total_fees + total_slippage,
            "fill_price": fill_price,
            "original_price": price
        }
