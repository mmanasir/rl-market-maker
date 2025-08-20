import numpy as np
from scipy.special import expit

class ExecutionSimulator:
    def __init__(self, rng_seed=42, latency=1, fee_per_trade=0.01):
        np.random.seed(rng_seed)
        self.latency = latency
        self.fee = fee_per_trade

    def simulate_fill(self, bid, ask, mid_price):
        spread = ask - bid
        edge = (ask - mid_price)
        side = np.random.choice(['buy', 'sell'])  
        fill_prob = expit(-2.0 * edge)
        filled = np.random.rand() < fill_prob

        if filled:
            if side == 'buy':
                return 'buy', ask
            else:
                return 'sell', bid
        else:
            return None, None