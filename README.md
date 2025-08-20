# RL Market Maker

Market making simulator with a PPO reinforcement learning agent.

## File Overview

- `config.yaml` — defines data and simulation parameters  
- `fetch_data.py` — downloads market data from Yahoo Finance and saves it to `data.csv`  
- `market_maker.py` — core market maker logic: quoting, inventory, PnL updates  
- `execution_simulator.py` — simulates order fills with latency and fees  
- `market_simulator.py` — runs historical backtest using market maker + execution  
- `mm_env.py` — Gymnasium environment wrapper for RL training  
- `rl_wrapper.py` — wraps PPO model into a market maker agent  
- `train_rl_agent.py` — trains a PPO RL agent, saves checkpoints  
- `run_simulation.py` — loads latest checkpoint, evaluates agent, produces plots/logs  
- `evaluation.py` — computes metrics, saves PnL and inventory plots, writes log  

## Setup

Requires Python 3.10+

```bash
pip install -r requirements.txt
```

## Configuration

All parameters are defined in `config.yaml`.

Example:

```yaml
data:
  ticker: "NVDA"          # Stock ticker to fetch from Yahoo Finance
  interval: "1m"          # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
  period: "5d"            # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
  output_path: "data.csv" # CSV file where fetched data is stored

  # NOTE:
  # intervals 1m & 2m only work with 1d & 5d periods
  # intervals 5m to 90m can only use 1d, 5d, 1mo periods
  # intervals 1d or higher can use any period

simulation:
  base_edge: 0.1          # Baseline bid/ask spread around mid-price
  vol_multiplier: 0.5     # Spread sensitivity to volatility
  inventory_limit: 10     # Max long/short inventory before no further trades
  vol_window: 20          # Rolling window size (ticks) for volatility calc
  latency: 1              # Execution delay (ticks) simulating order latency
  fee_per_trade: 0.01     # Transaction fee per trade, deducted from PnL
```

## Fetch Data

```bash
python fetch_data.py
```

Downloads historical OHLCV data from Yahoo Finance and saves it as `data.csv`.

## Train Agent

```bash
python train_rl_agent.py
```

- On startup, the script checks the `output/` folder for existing models.  
- If previous checkpoints (`ppo_mm_agent_step_*.zip`) exist, it automatically loads the latest step and continues training.  
- If none exist, it starts training a new PPO model from scratch.  
- During training, pressing Ctrl+C saves the current checkpoint before exiting.  
- At the end of training, the model is also saved automatically.  
- All checkpoints are written to `output/` as `ppo_mm_agent_step_{timesteps}.zip`

## Evaluate Agent

```bash
python run_simulation.py
```

- Automatically finds the most recent model checkpoint in `output/` and loads it.  
- Runs a full backtest using the stored config and environment.  
- Results are written into a subfolder based on the model step, e.g. `output/step_{timesteps}/`

## Outputs

Each evaluation produces the following files inside the `output/step_{timesteps}/` folder:

- `pnl_plot.png` — PnL breakdown over time  
- `inventory_plot.png` — Inventory levels over time  
- `mm_log.csv` — Full simulation log (timestamps, trades, PnL components)  

## Reset Training

1. Delete all existing model checkpoints `ppo_mm_agent_step_\*.zip` from the `output/` folder.
2. Run training again.
3. A fresh PPO agent will be trained and new checkpoints saved.
