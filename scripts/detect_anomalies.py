import pandas as pd
import pickle
import numpy as np
import sys
sys.path.append('.')
from data_preprocessing import assign_phase

def load_test_data(test_file, scaler, sensor_cols):
    col_names = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + sensor_cols
    df = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
    df = assign_phase(df)
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    return df

def detect_anomaly(sample, phase, models):
    model = models[phase]
    # Instead of sample.reshape(1, -1), create a DataFrame with column names
    import pandas as pd
    sample_df = pd.DataFrame([sample], columns=models[phase].feature_names_in_)
    pred = model.predict(sample_df)
    score = model.decision_function(sample_df)[0]
    return pred[0] == -1, score


if __name__ == "__main__":
    test_file = '../data/test_FD001.txt'
    with open('../models/isolation_forest_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    test_df = load_test_data(test_file, scaler, sensor_cols)
    
    results = []
    for idx, row in test_df.iterrows():
        sample = row[sensor_cols].values
        phase = row['phase']
        is_anomaly, score = detect_anomaly(sample, phase, models)
        results.append({'unit': row['unit'], 'time': row['time'], 'phase': phase, 'is_anomaly': is_anomaly, 'score': score})
    
    results_df = pd.DataFrame(results)
    print(results_df.head())
    output_df = pd.DataFrame({
        'timestamp': test_df['time'].reset_index(drop=True),
        'value': test_df['sensor_1'].reset_index(drop=True),
        'anomalyScore': results_df['score'],
        'label': results_df['is_anomaly'].astype(int)
    })
    output_df.to_csv('anomaly_results.csv', index=False)
    print("Anomaly results saved to anomaly_results.csv")