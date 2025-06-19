import pandas as pd, numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from src.models.ppo_env import AssetBalancingEnv

ROOT = Path(__file__).resolve().parents[2]
df   = pd.read_csv(ROOT/'data/sp_feature_table.csv')
env  = AssetBalancingEnv(df['util_ratio'].values,
                         df['total_capacity_kg'].values)

model = PPO("MlpPolicy", env, verbose=1).learn(20_000)
model.save(ROOT/'models'/'ppo.zip')
