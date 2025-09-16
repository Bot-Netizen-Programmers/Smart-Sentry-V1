import os
import pickle
from detect import load_test_data, detect_anomalies

def main():
    # Paths
    test_file = 'data/test_FD001.txt'
    models_path = 'models/isolation_forest_models.pkl'
    scaler_path = 'models/scaler.pkl'
    results_dir = 'results/smartsentry'
    results_filename = 'realKnownCause_test.csv'  # Must match NAB dataset name

    # Load models and scaler
    with open(models_path, 'rb') as f:
        models = pickle.load(f)  # Expecting dict with keys: 'startup', 'steady', 'shutdown'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load and preprocess test data
    test_df = load_test_data(test_file, scaler)

    # Detect anomalies
    results_df = detect_anomalies(test_df, models)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save results in NAB-compatible format
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Anomaly detection results saved to {results_path}")

if __name__ == "__main__":
    main()
