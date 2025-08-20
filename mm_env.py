import gymnasium as gym
import numpy as np
from market_simulator import MarketSimulator
from market_maker import MarketMaker

class MarketMakingEnv(gym.Env):
    def __init__(self, data_path, config):
        self.sim = MarketSimulator(data_path, **config)
        self.df = self.sim.load_data()
        self.df['volatility'] = self.sim.compute_volatility(self.df['mid_price'])
        self.max_steps = len(self.df)
        self.action_space = gym.spaces.Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.mm = MarketMaker(
            base_edge=self.sim.mm.base_edge,
            vol_multiplier=self.sim.mm.vol_multiplier,
            inventory_limit=self.sim.mm.inventory_limit
        )
        self.step_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            self.sim.mm.inventory,
            row['volatility'],
            self.df['mid_price'].pct_change().fillna(0).iloc[self.step_idx]
        ], dtype=np.float32)

    def step(self, action):
        edge, skew = action
        row = self.df.iloc[self.step_idx]
        mid = row['mid_price']
        vol = row['volatility']
        ts = row['timestamp']

        bid, ask = self.sim.mm.quote(mid, edge, skew)
        side, price = self.sim.executor.simulate_fill(bid, ask, mid)

        prev_pnl = self.sim.mm.total_pnl
        if side and price:
            self.sim.mm.update(side, price, mid)

        reward = self.sim.mm.total_pnl - prev_pnl - 0.01 * abs(self.sim.mm.inventory)

        self.sim.log.append({
            'timestamp': ts,
            'mid_price': mid,
            'bid': bid,
            'ask': ask,
            'executed_price': price if price else None,
            'trade_side': side if side else 'none',
            'inventory': self.sim.mm.inventory,
            'cash': self.sim.mm.cash,
            'pnl': self.sim.mm.total_pnl,
            'spread_pnl': self.sim.mm.spread_pnl,
            'inventory_pnl': self.sim.mm.inventory_pnl
        })

        self.step_idx += 1
        done = self.step_idx >= self.max_steps

        if done:
            obs = np.zeros(3, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, done, False, {}
