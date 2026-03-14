"""
Phase J: RL-based Dynamic Position Sizer

Replaces IndependentKellySizer with a PPO-trained agent that learns
adaptive position sizing based on market state, model confidence, 
and portfolio risk.

AUDIT J-A1: Training uses time-split (no random shuffle). 
AUDIT J-A2: Action space constraints are enforced in step(), not learned.
AUDIT J-A3: Reward shaping includes activity penalty to prevent "do nothing" strategy.
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, Tuple, List

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_RL = True
except ImportError:
    HAS_RL = False

from src.ops.event_logger import get_logger
logger = get_logger()


class TradingEnv(gym.Env if HAS_RL else object):
    """
    Gymnasium environment for RL-based portfolio sizing.
    
    State (dim ~20):
        - LTR rank scores for Top-K stocks (normalized to [0, 1])
        - Portfolio-level features: current drawdown, holding days, realized vol
        - Market features: VIX proxy, market breadth
        - Current position weights
    
    Action (dim K):
        - Continuous weight vector ∈ [0, max_single_position]
        - Auto-normalized to sum ≤ max_total_exposure
    
    Reward:
        r = daily_return - λ₁ * drawdown_penalty - λ₂ * turnover_cost + λ₃ * activity_bonus
    
    AUDIT J-A2: Hard constraints on position sizes are enforced in step(),
    NOT delegated to agent learning.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        returns_matrix: np.ndarray,
        rank_scores: np.ndarray,
        market_features: np.ndarray,
        max_k: int = 10,
        max_single_pos: float = 0.08,  # HOLX regression: no single stock > 8%
        max_total_exposure: float = 1.0,
        lambda_dd: float = 2.0,
        lambda_turnover: float = 0.5,
        lambda_activity: float = 0.01,
        transaction_cost: float = 0.001,
    ):
        """
        Args:
            returns_matrix: (n_days, n_stocks) daily returns for Top-K stocks
            rank_scores: (n_days, n_stocks) LTR rank scores
            market_features: (n_days, n_market_features) market-level state
            max_k: Number of stocks in the portfolio
            max_single_pos: Maximum weight per stock (OR5 compliance)
            max_total_exposure: Maximum total weight
            lambda_dd: Drawdown penalty coefficient
            lambda_turnover: Turnover cost coefficient  
            lambda_activity: Bonus for being invested (prevents "do nothing")
            transaction_cost: Round-trip cost per unit of turnover
        """
        super().__init__()
        
        self.returns = returns_matrix
        self.rank_scores = rank_scores
        self.market_features = market_features
        self.n_days = returns_matrix.shape[0]
        self.n_stocks = returns_matrix.shape[1]
        self.max_k = min(max_k, self.n_stocks)
        self.max_single_pos = max_single_pos
        self.max_total_exposure = max_total_exposure
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.lambda_activity = lambda_activity
        self.transaction_cost = transaction_cost
        
        # State: rank_scores(K) + market_features(M) + current_weights(K) + [drawdown, holding_days, nav]
        n_market = market_features.shape[1] if market_features.ndim > 1 else 1
        self.state_dim = self.max_k + n_market + self.max_k + 3
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        # Continuous weights for each position
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.max_k,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        """Reset environment to start of episode."""
        super().reset(seed=seed) if HAS_RL else None
        self.current_step = 0
        self.weights = np.zeros(self.max_k)
        self.nav = 1.0
        self.peak_nav = 1.0
        self.holding_days = 0
        self.total_turnover = 0.0
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """Build state vector from current market state."""
        t = min(self.current_step, self.n_days - 1)
        
        # Rank scores for Top-K stocks
        ranks = self.rank_scores[t, :self.max_k]
        
        # Market features
        mkt = self.market_features[t] if self.market_features.ndim > 1 else np.array([self.market_features[t]])
        
        # Current portfolio state
        drawdown = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-6)
        hold_norm = min(self.holding_days / 20.0, 1.0)  # normalize to ~1 month
        
        state = np.concatenate([
            ranks,
            mkt.flatten(),
            self.weights,
            [drawdown, hold_norm, self.nav],
        ]).astype(np.float32)
        
        # Clip to observation space bounds
        state = np.clip(state, -10.0, 10.0)
        
        # Pad or truncate to exact state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return state
    
    def step(self, action: np.ndarray):
        """
        Execute one trading day.
        
        AUDIT J-A2: Hard constraints enforced here, not learned.
        """
        # Enforce action constraints (AUDIT J-A2)
        action = np.clip(action, 0, self.max_single_pos)
        total_weight = np.sum(action)
        if total_weight > self.max_total_exposure:
            action = action * (self.max_total_exposure / total_weight)
        
        # Calculate turnover
        turnover = np.sum(np.abs(action - self.weights))
        self.total_turnover += turnover
        
        # Update weights
        old_weights = self.weights.copy()
        self.weights = action
        
        # Calculate portfolio return
        t = min(self.current_step, self.n_days - 1)
        daily_returns = self.returns[t, :self.max_k]
        portfolio_return = np.dot(self.weights, daily_returns)
        
        # Transaction costs
        tc_cost = turnover * self.transaction_cost
        net_return = portfolio_return - tc_cost
        
        # Update NAV
        self.nav *= (1 + net_return)
        self.peak_nav = max(self.peak_nav, self.nav)
        
        # Update holding days
        if np.sum(self.weights) > 0.01:
            self.holding_days += 1
        else:
            self.holding_days = 0
        
        # Reward calculation (AUDIT J-A3: includes activity bonus)
        drawdown = (self.peak_nav - self.nav) / max(self.peak_nav, 1e-6)
        dd_penalty = self.lambda_dd * max(drawdown - 0.05, 0)  # only penalize DD > 5%
        turnover_cost = self.lambda_turnover * turnover
        
        # Activity bonus: reward for being invested (prevents "never trade" strategy)
        activity = np.sum(self.weights)
        activity_bonus = self.lambda_activity * activity
        
        reward = net_return - dd_penalty - turnover_cost + activity_bonus
        
        # Episode termination
        self.current_step += 1
        terminated = self.current_step >= self.n_days
        truncated = self.nav < 0.5  # portfolio lost > 50% → stop
        
        obs = self._get_observation()
        info = {
            'nav': self.nav,
            'drawdown': drawdown,
            'turnover': turnover,
            'portfolio_return': portfolio_return,
            'total_weight': np.sum(self.weights),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass


class RLSizer:
    """
    RL-based Dynamic Position Sizer using PPO.
    
    Workflow:
        1. train(): Train PPO agent on historical data
        2. allocate(): Use trained agent to size positions
        3. Falls back to equal-weight if RL model fails
    """
    
    def __init__(
        self,
        max_k: int = 10,
        max_single_pos: float = 0.08,  # HOLX regression: no single stock > 8%
        max_total_exposure: float = 1.0,
        model_path: str = 'data/checkpoints/rl_sizer.zip',
    ):
        self.max_k = max_k
        self.max_single_pos = max_single_pos
        self.max_total_exposure = max_total_exposure
        self.model_path = model_path
        self.agent = None
    
    def train(
        self,
        returns_matrix: np.ndarray,
        rank_scores: np.ndarray,
        market_features: np.ndarray,
        total_timesteps: int = 50000,
        train_ratio: float = 0.8,
    ) -> Dict:
        """
        Train PPO agent on historical data.
        
        AUDIT J-A1: Time-split training. No random shuffle.
        
        Args:
            returns_matrix: (n_days, n_stocks) daily returns
            rank_scores: (n_days, n_stocks) LTR scores
            market_features: (n_days, n_features) market state
            total_timesteps: PPO training budget
            train_ratio: Fraction for training (rest for OOS validation)
            
        Returns:
            Dict with training and validation metrics
        """
        if not HAS_RL:
            raise ImportError("gymnasium and stable-baselines3 required")
        
        # Time split (AUDIT J-A1)
        n_days = returns_matrix.shape[0]
        n_train = int(n_days * train_ratio)
        
        train_env = TradingEnv(
            returns_matrix=returns_matrix[:n_train],
            rank_scores=rank_scores[:n_train],
            market_features=market_features[:n_train],
            max_k=self.max_k,
            max_single_pos=self.max_single_pos,
            max_total_exposure=self.max_total_exposure,
        )
        
        # Train PPO
        vec_env = DummyVecEnv([lambda: train_env])
        self.agent = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=42,
        )
        
        print(f"[RL] Training PPO on {n_train} days...")
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.agent.save(self.model_path)
        print(f"[RL] Model saved to {self.model_path}")
        
        # Validate on OOS data
        val_env = TradingEnv(
            returns_matrix=returns_matrix[n_train:],
            rank_scores=rank_scores[n_train:],
            market_features=market_features[n_train:],
            max_k=self.max_k,
            max_single_pos=self.max_single_pos,
            max_total_exposure=self.max_total_exposure,
        )
        
        val_metrics = self._evaluate(val_env, "OOS")
        
        # Also evaluate equal-weight baseline on same OOS period
        baseline_metrics = self._evaluate_baseline(
            returns_matrix[n_train:], self.max_k, self.max_single_pos
        )
        
        return {
            'train_days': n_train,
            'val_days': n_days - n_train,
            'val_nav': val_metrics['final_nav'],
            'val_sharpe': val_metrics['sharpe'],
            'val_max_dd': val_metrics['max_drawdown'],
            'val_avg_positions': val_metrics['avg_n_positions'],
            'baseline_nav': baseline_metrics['final_nav'],
            'baseline_sharpe': baseline_metrics['sharpe'],
            'baseline_max_dd': baseline_metrics['max_drawdown'],
        }
    
    def allocate(
        self,
        rank_scores: np.ndarray,
        market_features: np.ndarray,
        current_weights: np.ndarray,
        drawdown: float = 0.0,
        holding_days: int = 0,
        nav: float = 1.0,
    ) -> np.ndarray:
        """
        Use trained agent to allocate portfolio weights.
        
        Falls back to equal-weight if agent not loaded.
        
        Args:
            rank_scores: LTR scores for Top-K stocks
            market_features: Current market state
            current_weights: Current position weights
            drawdown: Current portfolio drawdown
            holding_days: Days with current positions
            nav: Current NAV
            
        Returns:
            numpy array of portfolio weights for Top-K stocks
        """
        if self.agent is None:
            return self._fallback_allocation(rank_scores)
        
        # Build observation vector
        ranks = np.zeros(self.max_k)
        ranks[:min(len(rank_scores), self.max_k)] = rank_scores[:self.max_k]
        
        mkt = market_features.flatten() if hasattr(market_features, 'flatten') else np.array([market_features])
        
        weights = np.zeros(self.max_k)
        weights[:min(len(current_weights), self.max_k)] = current_weights[:self.max_k]
        
        state = np.concatenate([
            ranks, mkt, weights,
            [drawdown, min(holding_days / 20.0, 1.0), nav],
        ]).astype(np.float32)
        
        # Pad to match observation space
        obs_dim = self.agent.observation_space.shape[0]
        if len(state) < obs_dim:
            state = np.pad(state, (0, obs_dim - len(state)))
        elif len(state) > obs_dim:
            state = state[:obs_dim]
        
        state = np.clip(state, -10.0, 10.0)
        
        # Get action from agent
        action, _ = self.agent.predict(state, deterministic=True)
        
        # Enforce constraints (AUDIT J-A2)
        action = np.clip(action, 0, self.max_single_pos)
        total = np.sum(action)
        if total > self.max_total_exposure:
            action = action * (self.max_total_exposure / total)
        
        return action
    
    def load(self, path: Optional[str] = None):
        """Load pre-trained model."""
        path = path or self.model_path
        if os.path.exists(path):
            self.agent = PPO.load(path)
            print(f"[RL] Loaded model from {path}")
        else:
            print(f"[RL] No model found at {path}, using fallback")
    
    def _fallback_allocation(self, rank_scores: np.ndarray) -> np.ndarray:
        """Equal-weight fallback when RL model unavailable."""
        n = min(len(rank_scores), self.max_k)
        weights = np.zeros(self.max_k)
        if n > 0:
            w = min(self.max_single_pos, self.max_total_exposure / n)
            weights[:n] = w
        return weights
    
    def _evaluate(self, env: 'TradingEnv', label: str = "") -> Dict:
        """Run evaluation episode and collect metrics."""
        obs, _ = env.reset()
        nav_history = [1.0]
        daily_rets = []
        n_positions = []
        
        done = False
        while not done:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            nav_history.append(info['nav'])
            if len(nav_history) >= 2:
                daily_rets.append(nav_history[-1] / nav_history[-2] - 1)
            n_positions.append(info['total_weight'] / self.max_single_pos)
        
        rets = np.array(daily_rets)
        sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
        peak = nav_history[0]
        max_dd = 0
        for n in nav_history:
            peak = max(peak, n)
            dd = (peak - n) / peak
            max_dd = max(max_dd, dd)
        
        logger.info(f"[RL] {label}: NAV={nav_history[-1]:.4f}, Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")
        
        return {
            'final_nav': nav_history[-1],
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'avg_n_positions': np.mean(n_positions),
            'total_return': nav_history[-1] - 1.0,
        }
    
    @staticmethod
    def _evaluate_baseline(returns: np.ndarray, max_k: int, max_pos: float) -> Dict:
        """Equal-weight baseline for comparison."""
        n_stocks = min(returns.shape[1], max_k)
        w = min(max_pos, 1.0 / max(n_stocks, 1))
        weights = np.array([w] * n_stocks + [0] * (returns.shape[1] - n_stocks))
        
        nav = 1.0
        nav_history = [1.0]
        for t in range(returns.shape[0]):
            port_ret = np.dot(weights[:returns.shape[1]], returns[t])
            nav *= (1 + port_ret)
            nav_history.append(nav)
        
        rets = np.diff(nav_history) / np.array(nav_history[:-1])
        sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
        peak = nav_history[0]
        max_dd = 0
        for n in nav_history:
            peak = max(peak, n)
            dd = (peak - n) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'final_nav': nav,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }
