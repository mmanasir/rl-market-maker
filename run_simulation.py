import os
import re
import glob
import yaml
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from mm_env import MarketMakingEnv
from evaluation import Evaluator

def get_latest_checkpoint():
    checkpoints = glob.glob("output/ppo_mm_agent_step_*.zip")
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in output/")
    steps = [(int(re.search(r"step_(\d+)", f).group(1)), f) for f in checkpoints]
    return max(steps)[1]

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["output_path"]
env = MarketMakingEnv(data_path=data_path, config=config["simulation"])
model_path = get_latest_checkpoint()
model = PPO.load(model_path)

obs, _ = env.reset()
done = False

timestamps = []
rewards = []
inventories = []
pnls = []
spread_pnls = []
inventory_pnls = []
executed_prices = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    state = env.sim.mm
    ts = env.df.iloc[env.step_idx]['timestamp'] if env.step_idx < len(env.df) else None
    price = env.sim.log[-1]["executed_price"] if env.sim.log else None

    timestamps.append(ts)
    rewards.append(reward)
    inventories.append(state.inventory)
    pnls.append(state.total_pnl)
    spread_pnls.append(state.spread_pnl)
    inventory_pnls.append(state.inventory_pnl)
    executed_prices.append(price)

log = pd.DataFrame({
    "timestamp": timestamps,
    "reward": rewards,
    "inventory": inventories,
    "pnl": pnls,
    "spread_pnl": spread_pnls,
    "inventory_pnl": inventory_pnls,
    "executed_price": executed_prices,
})

step_count = int(re.search(r"step_(\d+)", model_path).group(1))
output_dir = f"output/step_{step_count}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

evaluator = Evaluator(log, output_dir=output_dir)
evaluator.run_all()

print(f"Loaded model: {model_path}")
print(f"Results saved to: {output_dir}")