import pandas as pd
import numpy as np

# List of sensor columns
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

def assign_phase(df, startup_ratio=0.1, shutdown_ratio=0.1):
    """
    Assign operational phase labels ('startup', 'steady', 'shutdown') based on unit and time.
    """
    df = df.copy()
    df['phase'] = 'steady'
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        max_cycle = unit_df['time'].max()
        startup_threshold = int(max_cycle * startup_ratio)
        shutdown_threshold = int(max_cycle * (1 - shutdown_ratio))
        df.loc[(df['unit'] == unit) & (df['time'] <= startup_threshold), 'phase'] = 'startup'
        df.loc[(df['unit'] == unit) & (df['time'] >= shutdown_threshold), 'phase'] = 'shutdown'
    return df

def load_test_data(test_file, scaler):
    """
    Load test data, assign phases, scale sensor columns.
    """
    col_names = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + sensor_cols
    df = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
    df = assign_phase(df)
    # Scale sensor columns
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    df[sensor_cols] = df[sensor_cols].astype('float32')
    return df

def detect_anomalies(df, models):
    """
    Detect anomalies per phase using corresponding Isolation Forest models.
    Returns a DataFrame with timestamp, sensor_1 value, anomalyScore, and label.
    """
    results = []
    for phase in ['startup', 'steady', 'shutdown']:
        phase_data = df[df['phase'] == phase]
        if phase_data.empty:
            continue
        samples = phase_data[sensor_cols]
        model = models[phase]
        preds = model.predict(samples)
        scores = model.decision_function(samples)
        phase_results = pd.DataFrame({
            'timestamp': phase_data['time'].values,
            'value': phase_data['sensor_1'].values,
            'anomalyScore': scores,
            'label': (preds == -1).astype(int)
        })
        results.append(phase_results)
    if results:
        return pd.concat(results).sort_values('timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['timestamp', 'value', 'anomalyScore', 'label'])
