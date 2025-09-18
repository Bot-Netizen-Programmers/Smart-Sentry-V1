import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
def create_cnn_autoencoder(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(n_features, 3, activation='sigmoid', padding='same')(x)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
autoencoder = create_cnn_autoencoder(seq_length, len(sensor_cols))
autoencoder.summary()
# Train model
autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, validation_split=0.1)
