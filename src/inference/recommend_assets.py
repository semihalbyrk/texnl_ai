import pandas as pd
import joblib
from stable_baselines3 import PPO
from src.models.ppo_env import AssetBalancingEnv

def recommend_asset_reallocation():
    df = pd.read_csv("data/sp_feature_table.csv")
    util = df["utilization"].values
    cap = df["total_capacity_kg"].values
    names = df["Service Point"].tolist()

    env = AssetBalancingEnv(util, cap, names=names)

    try:
        model = PPO.load("models/ppo.zip", env=env)
    except:
        return ["❌ PPO modeli yüklenemedi"]

    obs = env.reset()
    recommendations = []

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        move = env.get_last_move_str()
        if move:
            recommendations.append(move)

    return recommendations
