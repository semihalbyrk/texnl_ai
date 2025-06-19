import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

def run_anomaly_detection():
    df = pd.read_csv("data/sp_feature_table.csv")
    X = df[['total_kg','total_capacity_kg','tasks_per_week','utilization']]

    model_path = "models/ae_anomaly_model.pkl"
    if not joblib.os.path.exists(model_path):
        return ["⚠️ Anomali modeli bulunamadı."]

    model = joblib.load(model_path)
    preds = model.predict(X)
    df['anomaly'] = preds

    output = []
    for _, row in df[df['anomaly'] == -1].iterrows():
        output.append(f"{row['Service Point']} → anomalik davranış tespit edildi")

    return output
