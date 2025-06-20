import pandas as pd, numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from src.models.ppo_env import AssetBalancingEnv

ROOT = Path(__file__).resolve().parents[2]
PPO_PATH = ROOT/'models'/'ppo.zip'

def recommend(df: pd.DataFrame, n_steps=10):
    env = AssetBalancingEnv(df['util_ratio'].values,
                            df['total_capacity_kg'].values)
    model = PPO.load(PPO_PATH, env=env)
    obs, _ = env.reset()
    moves = []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        i, j = divmod(action, len(df))
        moves.append(f"{df.loc[i,'Service Point Name']} → {df.loc[j,'Service Point Name']}")
        obs, _, _, _, _ = env.step(action)
    return moves