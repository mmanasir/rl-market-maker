import os
import re
import glob
import signal
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from mm_env import MarketMakingEnv

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["output_path"]
os.makedirs("output", exist_ok=True)

def make_env():
    return Monitor(MarketMakingEnv(data_path=data_path, config=config["simulation"]))

env = DummyVecEnv([make_env])

def get_latest_checkpoint():
    checkpoints = glob.glob("output/ppo_mm_agent_step_*.zip")
    if not checkpoints:
        return None
    steps = [(int(re.search(r"step_(\d+)", f).group(1)), f) for f in checkpoints]
    return max(steps)[1]

latest_path = get_latest_checkpoint()
if latest_path:
    model = PPO.load(latest_path, env=env)
    model.set_env(env)
    print(f"Loaded model from: {latest_path}")
else:
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    print("Initialized new model.")

def save_model(*args):
    steps = int(model.num_timesteps)
    path = f"output/ppo_mm_agent_step_{steps}.zip"
    model.save(path)
    print(f"\nModel saved to {path}")
    exit(0)

signal.signal(signal.SIGINT, save_model)

timesteps = 1_000_000
model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

save_model()