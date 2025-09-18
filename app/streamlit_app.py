import streamlit as st
import time

st.title("Smart Sentry: Real-Time Anomaly Detection")

# Simulate streaming
buffer = []
seq_length = 30
threshold = 0.01

sensor_to_plot = sensor_cols[0]  # Example sensor

placeholder = st.empty()

for i in range(len(test_scaled)):
    buffer.append(test_scaled[i])
    if len(buffer) == seq_length:
        seq = np.array(buffer).reshape(1, seq_length, len(sensor_cols))
        anomaly, error = detect_anomalies(autoencoder, seq, threshold)
        current_phase = test_df.iloc[i]['phase']
        sensor_value = test_df.iloc[i][sensor_to_plot]

        with placeholder.container():
            st.write(f"Cycle: {test_df.iloc[i]['cycle']}, Phase: {current_phase}")
            st.line_chart(test_df[sensor_to_plot].iloc[max(0, i-100):i+1])
            if anomaly[0]:
                st.error(f"Anomaly detected! Reconstruction error: {error[0]:.4f}")
            else:
                st.success("No anomaly detected.")

        buffer.pop(0)
        time.sleep(0.1)  # Simulate real-time delay
