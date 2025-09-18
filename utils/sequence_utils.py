from sklearn.preprocessing import MinMaxScaler

sensor_cols = [col for col in train_df.columns if 'sensor' in col]

def create_sequences(data, seq_length=30):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Prepare data for one phase, e.g., steady_state
phase = 'steady_state'
phase_data = train_df[train_df['phase'] == phase]

# Scale sensor data
scaler = MinMaxScaler()
phase_scaled = scaler.fit_transform(phase_data[sensor_cols])

seq_length = 30
X_train = create_sequences(phase_scaled, seq_length)
print(f"Training sequences shape: {X_train.shape}")  # (num_sequences, seq_length, num_sensors)
