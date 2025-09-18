Smart Sentry: Detailed Design Document
1. Introduction
Smart Sentry is an intelligent anomaly detection system designed to monitor aircraft engine sensor data and detect early signs of faults or failures. It leverages a phase-aware 1D Convolutional Neural Network (CNN) Autoencoder trained on NASA’s CMAPSS dataset to identify anomalies in different operational phases (startup, steady_state, shutdown).

2. Objectives
Detect anomalies in engine sensor data in real-time.
Incorporate operational phase awareness to improve detection accuracy.
Provide a scalable and modular architecture for training, inference, and deployment.
Support visualization and alerting via a dashboard.
Enable edge deployment for on-device inference.
3. System Overview
3.1 Components
Component

Description

Data Loader

Loads and preprocesses NASA CMAPSS dataset files.

Phase Labeler

Labels each data point as startup, steady_state, or shutdown.

Sequence Generator

Converts sensor data into fixed-length sequences for training.

CNN Autoencoder

1D CNN model that learns to reconstruct normal sensor sequences.

Trainer

Trains the autoencoder per operational phase.

Inference Engine

Performs anomaly detection using reconstruction error.

Threshold Calculator

Determines anomaly detection thresholds from training errors.

Dashboard (Streamlit)

Visualizes sensor data, phases, and anomaly alerts in real-time.

Edge Deployment

Converts and runs models on edge devices for low-latency inference.

4. Data Design
4.1 Dataset
Source: NASA CMAPSS dataset (FD001 to FD004)
Format: Text files with columns:
engine_id, cycle, setting_1, setting_2, setting_3
21 sensor measurements (sensor_1 to sensor_21)
4.2 Phase Labeling
Startup: First 10 cycles of each engine.
Shutdown: Last 10 cycles of each engine.
Steady State: All cycles in between.
4.3 Preprocessing
Select sensor columns.
Scale sensor data using MinMaxScaler (fit on training data).
Create overlapping sequences of fixed length (e.g., 30 cycles).
5. Model Design
5.1 Architecture: 1D CNN Autoencoder
Input: Sequence of shape (sequence_length, num_sensors)
Encoder:
Conv1D layers with ReLU activation
MaxPooling1D layers for downsampling
Latent Space: Compressed representation of input sequence
Decoder:
Conv1D + UpSampling1D layers to reconstruct input
Final Conv1D with sigmoid activation to output normalized sensor values
5.2 Training
Loss: Mean Squared Error (MSE) between input and reconstruction
Optimizer: Adam
Epochs: 20 (adjustable)
Batch size: 64 (adjustable)
Validation split: 10%
5.3 Phase-Aware Training
Train separate models for each phase (startup, steady_state, shutdown)
Save models and scalers per phase
6. Anomaly Detection
6.1 Reconstruction Error
Compute MSE between input sequence and reconstructed output.
Higher error indicates potential anomaly.
6.2 Thresholding
Calculate threshold per phase from training reconstruction errors (e.g., 95th percentile).
Flag sequences with error above threshold as anomalies.
7. System Architecture

Run
Copy code
+-------------------+       +-------------------+       +-------------------+
|   Data Loader     | ----> | Phase Labeler     | ----> | Sequence Generator |
+-------------------+       +-------------------+       +-------------------+
                                                                |
                                                                v
+-------------------+       +-------------------+       +-------------------+
|   CNN Autoencoder  | <---- |    Trainer        |       | Threshold Calculator|
+-------------------+       +-------------------+       +-------------------+
                                                                |
                                                                v
+-------------------+       +-------------------+       +-------------------+
|   Inference Engine | <----|   Saved Models    |       |   Scalers          |
+-------------------+       +-------------------+       +-------------------+
                                                                |
                                                                v
+-------------------+
|   Dashboard (UI)  |
+-------------------+
8. Implementation Details
8.1 File Structure

Run
Copy code
SmartSentry/
├── data/
├── models/
│   ├── cnn_autoencoder.py
│   ├── train_model.py
│   ├── inference.py
│   └── saved_models/
├── utils/
│   ├── data_loader.py
│   ├── phase_labeler.py
│   └── sequence_utils.py
├── app/
│   └── streamlit_app.py
├── edge_deployment/
├── notebooks/
├── requirements.txt
└── README.md
8.2 Key Scripts
train_model.py: Loads data, preprocesses, trains CNN autoencoder per phase, saves model and scaler.
inference.py: Loads saved model and scaler, preprocesses new data, computes reconstruction error, flags anomalies.
streamlit_app.py: Real-time dashboard for visualization and alerting.
Utility scripts for modular code reuse.
9. Deployment
9.1 Local Deployment
Run training scripts on local or cloud GPU-enabled machines.
Use Streamlit app for monitoring.
9.2 Edge Deployment
Convert trained models to TensorFlow Lite or ONNX.
Deploy lightweight inference code on edge devices.
Perform real-time anomaly detection with low latency.
10. Testing & Validation
Validate model reconstruction error on test datasets.
Tune thresholds to balance false positives and false negatives.
Test phase-aware detection improves accuracy over phase-agnostic models.
Perform end-to-end tests with streaming data simulation.
11. Future Enhancements
Incorporate additional sensor fusion or feature engineering.
Use more advanced architectures (e.g., Transformer-based autoencoders).
Implement adaptive thresholding or online learning.
Add alerting and notification integration (email, SMS).
Extend dashboard with historical analytics and root cause analysis.
12. References
NASA CMAPSS Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
TensorFlow 2.x Documentation: https://www.tensorflow.org/
Streamlit Documentation: https://docs.streamlit.io/
Appendix: Sample Model Architecture (Keras)
python
19 lines
Copy code
Download code
Click to expand
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
...
