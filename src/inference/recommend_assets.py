import joblib
import pandas as pd
from stable_baselines3 import PPO
from src.models.ppo_env import AssetBalancingEnv
from collections import Counter
from pathlib import Path

def recommend_asset_reallocation():
    # Load feature table
    df = pd.read_csv("data/sp_feature_table.csv")

    # PPO ortamı için gerekli veriler
    names = df["Service Point"].tolist()
    util = df["utilization"].values
    cap = df["total_capacity_kg"].values
    env = AssetBalancingEnv(util, cap, names=names)

    # Model yükle
    model_path = Path("models/ppo.zip")
    if not model_path.exists():
        return ["❌ PPO modeli bulunamadı. Lütfen önce eğitimi tamamlayın."]
    
    model = PPO.load(model_path, env=env)

    # Öneri üret
    obs = env.reset()
    recommendations = []

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        move_str = env.get_last_move_str()
        if move_str:
            recommendations.append(move_str)
        if done:
            break

    # Aynı önerileri gruplandır
    grouped = Counter(recommendations)
    summary = [f"{move} × {count}" for move, count in grouped.items()]

    return summary if summary else ["ℹ️ DRL önerisi üretilemedi."]
