import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from cnn_autoencoder import create_cnn_autoencoder  # Assume this is your model definition file

# Utility functions (can be moved to utils/)
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

def create_sequences(data, seq_length=30):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def main(args):
    # Load and preprocess data
    print(f"Loading data from {args.data_path} ...")
    df = load_cmapss_data(args.data_path)
    df = label_phases(df)
    sensor_cols = [col for col in df.columns if 'sensor' in col]

    print(f"Filtering data for phase: {args.phase}")
    phase_data = df[df['phase'] == args.phase]
    if phase_data.empty:
        raise ValueError(f"No data found for phase '{args.phase}'")

    # Scale sensor data
    scaler = MinMaxScaler()
    phase_scaled = scaler.fit_transform(phase_data[sensor_cols])

    # Create sequences
    X_train = create_sequences(phase_scaled, args.seq_length)
    print(f"Training data shape: {X_train.shape}")

    # Create model
    autoencoder = create_cnn_autoencoder(args.seq_length, len(sensor_cols))
    autoencoder.summary()

    # Train model
    autoencoder.fit(X_train, X_train,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validation_split=0.1)

    # Save model and scaler
    save_dir = 'models/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'autoencoder_{args.phase}.h5')
    scaler_path = os.path.join(save_dir, f'scaler_{args.phase}.pkl')

    autoencoder.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D CNN Autoencoder for a given phase")
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data file')
    parser.add_argument('--phase', type=str, required=True, choices=['startup', 'steady_state', 'shutdown'], help='Operational phase to train on')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')

    args = parser.parse_args()
    main(args)
