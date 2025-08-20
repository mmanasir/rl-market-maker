import pandas as pd
import numpy as np
from market_maker import MarketMaker
from execution_simulator import ExecutionSimulator

class MarketSimulator:
    def __init__(self, data_path, base_edge=0.1, vol_multiplier=0.5, inventory_limit=10, vol_window=20, latency=1, fee_per_trade=0.01, mm=None):
        self.data_path = data_path
        self.mm = mm or MarketMaker(base_edge, vol_multiplier, inventory_limit)
        self.executor = ExecutionSimulator(latency=latency, fee_per_trade=fee_per_trade)
        self.vol_window = vol_window
        self.log = []

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.columns = [col.lower() for col in df.columns]
        df['mid_price'] = (df['high'] + df['low']) / 2
        return df

    def compute_volatility(self, prices):
        return prices.rolling(self.vol_window).std().fillna(0.2)

    def run(self):
        df = self.load_data()
        df['volatility'] = self.compute_volatility(df['mid_price'])

        for i, row in df.iterrows():
            time = row['timestamp']
            mid = row['mid_price']
            vol = row['volatility']

            bid, ask = self.mm.quote(mid, vol, 0.0)  
            side, price = self.executor.simulate_fill(bid, ask, mid)

            if side and price:
                self.mm.update(side, price, mid)

            self.log.append({
                'timestamp': time,
                'mid_price': mid,
                'bid': bid,
                'ask': ask,
                'executed_price': price if price else np.nan,
                'trade_side': side if side else 'none',
                'inventory': self.mm.inventory,
                'cash': self.mm.cash,
                'pnl': self.mm.total_pnl,
                'spread_pnl': self.mm.spread_pnl,
                'inventory_pnl': self.mm.inventory_pnl
            })

        return pd.DataFrame(self.log)