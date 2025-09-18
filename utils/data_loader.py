import pandas as pd
import numpy as np

def load_cmapss_data(file_path):
    col_names = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    return df

def label_phases(df, startup_cycles=10, shutdown_cycles=10):
    df['phase'] = 'steady_state'
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        df.loc[(df['engine_id'] == engine_id) & (df['cycle'] <= startup_cycles), 'phase'] = 'startup'
        df.loc[(df['engine_id'] == engine_id) & (df['cycle'] > max_cycle - shutdown_cycles), 'phase'] = 'shutdown'
    return df

# Load training data
train_df = load_cmapss_data('data/train_FD001.txt')
train_df = label_phases(train_df)`
