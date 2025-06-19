import joblib
import pandas as pd
from stable_baselines3 import PPO
from src.models.ppo_env import AssetBalancingEnv
from collections import Counter
from pathlib import Path

def recommend_asset_reallocation():
    # Load feature table
    df = pd.read_csv("data/sp_feature_table.csv")

    # Prepare environment
    env = AssetBalancingEnv(df)

    # Load trained PPO model
    model_path = Path("models/ppo.zip")
    if not model_path.exists():
        return ["❌ PPO modeli bulunamadı. Lütfen önce eğitimi tamamlayın."]
    
    model = PPO.load(model_path, env=env)

    # Rollout to get recommendations
    obs = env.reset()
    recommendations = []

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        move_str = env.get_last_move_str()  # örn: "SP1 → SP2"
        if move_str:
            recommendations.append(move_str)
        if done:
            break

    # Önerileri grupla (Counter)
    grouped = Counter(recommendations)
    summary = [f"{move} × {count}" for move, count in grouped.items()]

    return summary if summary else ["ℹ️ DRL önerisi üretilmedi."]
