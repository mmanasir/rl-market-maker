import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, results_df, output_dir='output'):
        self.df = results_df.copy()
        self.output_dir = output_dir

    def compute_metrics(self):
        pnl = self.df['pnl']
        returns = pnl.diff().fillna(0)

        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 390)
        max_dd = (pnl.cummax() - pnl).max()
        total_return = pnl.iloc[-1]
        final_inventory = self.df['inventory'].iloc[-1]
        trade_count = self.df['executed_price'].notna().sum() if 'executed_price' in self.df else 'N/A'

        print('--- Results ---')
        print(f'Total return:        ${total_return:.2f}')
        print(f'Sharpe ratio:        {sharpe:.2f}')
        print(f'Max drawdown:        {max_dd:.2f}')
        print(f'Final inventory:     {final_inventory}')
        print(f'Trades executed:     {trade_count}')

    def plot_pnl(self):
        df = self.df.dropna(subset=['timestamp']).copy()
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['pnl'], label='Total PnL', color='blue')
        plt.plot(df['timestamp'], df['spread_pnl'], label='Spread PnL', linestyle='--', color='green')
        plt.plot(df['timestamp'], df['inventory_pnl'], label='Inventory PnL', linestyle='--', color='red')
        plt.title('Market Maker PnL Breakdown')
        plt.xlabel('Time')
        plt.ylabel('PnL ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pnl_plot.png')
        plt.close()

    def plot_inventory(self):
        df = self.df.dropna(subset=['timestamp']).copy()
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        plt.figure(figsize=(12, 4))
        plt.plot(df['timestamp'], df['inventory'], label='Inventory', color='orange')
        plt.title('Inventory Over Time')
        plt.xlabel('Time')
        plt.ylabel('Inventory (shares)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/inventory_plot.png')
        plt.close()

    def save_log(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.df.to_csv(f'{self.output_dir}/mm_log.csv', index=False)

    def run_all(self):
        self.compute_metrics()
        self.plot_pnl()
        self.plot_inventory()
        self.save_log()