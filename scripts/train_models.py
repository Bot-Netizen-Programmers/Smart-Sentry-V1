from sklearn.ensemble import IsolationForest
import pickle
import sys
sys.path.append('.')  # To import from data_preprocessing.py
from data_preprocessing import load_and_preprocess

def train_models(df, sensor_cols):
    phases = ['startup', 'steady', 'shutdown']
    models = {}
    for phase in phases:
        phase_data = df[df['phase'] == phase][sensor_cols]
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(phase_data)
        models[phase] = model
        print(f"Trained model for phase: {phase}")
    return models

if __name__ == "__main__":
    train_file = '../data/train_FD001.txt'
    df, scaler, sensor_cols = load_and_preprocess(train_file)
    models = train_models(df, sensor_cols)
    
    # Save models and scaler for later use
    with open('../models/isolation_forest_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Models and scaler saved.")