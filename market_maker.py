import numpy as np

class MarketMaker:
    def __init__(self, base_edge=0.1, vol_multiplier=0.5, inventory_limit=10):
        self.base_edge = base_edge
        self.vol_multiplier = vol_multiplier
        self.inventory_limit = inventory_limit

        self.inventory = 0
        self.cash = 0.0

        self.spread_pnl = 0.0
        self.inventory_pnl = 0.0
        self.total_pnl = 0.0

    def quote(self, mid_price, edge, skew):
        bid = mid_price - edge + skew
        ask = mid_price + edge + skew
        return bid, ask

    def update(self, trade_side, executed_price, mid_price):
        if trade_side == 'buy':
            self.inventory -= 1
            self.cash += executed_price
            spread_pnl = executed_price - mid_price
        elif trade_side == 'sell':
            self.inventory += 1
            self.cash -= executed_price
            spread_pnl = mid_price - executed_price
        else:
            spread_pnl = 0.0

        self.spread_pnl += spread_pnl
        self.inventory_pnl = self.inventory * mid_price
        self.total_pnl = self.cash + self.inventory_pnl