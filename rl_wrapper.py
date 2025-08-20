from market_maker import MarketMaker

class RLDrivenMarketMaker(MarketMaker):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prev_obs = [0.0, 0.2, 0.0]

    def quote(self, mid_price, volatility):
        obs = [self.inventory, volatility, self.prev_obs[2]]
        action, _ = self.model.predict(obs, deterministic=True)
        edge, skew = action
        self.prev_obs = obs
        return super().quote(mid_price, edge, skew)