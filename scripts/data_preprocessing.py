import pandas as pd
from sklearn.preprocessing import StandardScaler

def assign_phase(df, startup_ratio=0.1, shutdown_ratio=0.1):
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

def load_and_preprocess(train_file):
    col_names = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)
    
    df = assign_phase(df)
    
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    
    return df, scaler, sensor_cols

if __name__ == "__main__":
    train_file = '../data/train_FD001.txt'
    df, scaler, sensor_cols = load_and_preprocess(train_file)
    print(df.head())
