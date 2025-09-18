import os
import numpy as np
import joblib
import tensorflow as tf

def load_model_and_scaler(phase):
    model_path = f'models/saved_models/autoencoder_{phase}.h5'
    scaler_path = f'models/saved_models/scaler_{phase}.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler for phase '{phase}' not found.")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def create_sequences(data, seq_length=30):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def calculate_reconstruction_error(model, sequences):
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1,2))
    return mse

def detect_anomalies(mse, threshold):
    return mse > threshold

def preprocess_and_detect(model, scaler, raw_data, seq_length=30, threshold=0.01):
    # raw_data: numpy array of sensor readings (num_samples, num_features)
    scaled_data = scaler.transform(raw_data)
    sequences = create_sequences(scaled_data, seq_length)
    errors = calculate_reconstruction_error(model, sequences)
    anomalies = detect_anomalies(errors, threshold)
    return anomalies, errors

# Example usage:
if __name__ == "__main__":
    phase = 'steady_state'  # or 'startup', 'shutdown'
    model, scaler = load_model_and_scaler(phase)

    # raw_data should be a numpy array of sensor readings for the phase
    # For demo, load some test data and preprocess accordingly

    # threshold can be tuned based on training reconstruction errors
    threshold = 0.01

    # anomalies, errors = preprocess_and_detect(model, scaler, raw_data, seq_length=30, threshold=threshold)
    # print(anomalies, errors)
