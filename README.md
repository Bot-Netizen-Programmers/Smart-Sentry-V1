
---

# Anomaly Detection Project with Isolation Forest and NAB Integration

This project implements an anomaly detection pipeline using Isolation Forest models trained for different operational phases (startup, steady, shutdown). It supports real-time and batch anomaly detection on sensor data and integrates with the Numenta Anomaly Benchmark (NAB) for standardized evaluation.

---

## Features

- Phase-aware anomaly detection using separate Isolation Forest models.
- Efficient preprocessing with sensor scaling and phase assignment.
- Real-time capable detection functions for single or batch samples.
- NAB-compatible results output and scoring integration.
- Threshold optimization support via NAB.
- Clear folder structure for models, data, results, and configs.

---

## Folder Structure

```
.
├── data/
│   └── test_FD001.txt                 # Raw test data file
├── models/
│   ├── isolation_forest_models.pkl   # Pre-trained Isolation Forest models per phase
│   └── scaler.pkl                    # Scaler for sensor data
├── results/
│   └── smartsentry/                  # Detector results folder (NAB expects detector subfolders here)
│       └── realKnownCause_test.csv   # Detection results CSV (named after dataset)
├── config/
│   ├── profiles.json                 # NAB profiles config
│   └── thresholds.json              # NAB thresholds config (updated with smartsentry detector)
├── labels/
│   └── combined_windows.json        # NAB ground truth anomaly windows
├── detect.py                       # Core anomaly detection functions
├── run_detection.py                # Script to run detection and save results
├── README.md                       # This file
└── ...
```

---

## Setup Instructions

1. **Install dependencies**

```bash
pip install pandas scikit-learn numpy
```

2. **Prepare models and scaler**

- Place your trained Isolation Forest models in `models/isolation_forest_models.pkl`.
- Place your fitted scaler in `models/scaler.pkl`.

3. **Prepare NAB**

- Download or clone NAB repository: https://github.com/numenta/NAB
- Copy NAB's `config/`, `labels/`, and `data/` folders into your project root or adjust paths accordingly.

4. **Update NAB thresholds config**

Add your detector `"smartsentry"` to `config/thresholds.json` with all required profiles:

```json
{
  "smartsentry": {
    "realKnownCause": { "threshold": 0.5 },
    "realKnownCause2": { "threshold": 0.5 },
    "artificialWithAnomaly": { "threshold": 0.5 },
    "artificialNoAnomaly": { "threshold": 0.5 },
    "standard": { "threshold": 0.5 }
  }
}
```

---

## Usage

### 1. Run anomaly detection on test data

Run the detection script to process test data, assign phases, scale sensors, predict anomalies, and save results in NAB-compatible format:

```bash
python run_detection.py
```

This will generate:

```
results/smartsentry/realKnownCause_test.csv
```

with columns:

- `timestamp` (time)
- `value` (sensor_1 value)
- `anomalyScore` (model decision function score)
- `label` (0/1 anomaly prediction)

---

### 2. Run NAB scoring

Run NAB scoring on your results:

```bash
python run.py --score --resultsDir=results --dataDir=data -d smartsentry --normalize --optimize
```

- `--resultsDir=results` points to the parent folder containing detector folders.
- `-d smartsentry` specifies your detector folder.
- `--normalize --optimize` enables threshold optimization.

Confirm prompt with `y`.

---

## Code Overview

### detect.py

Contains functions to:

- Assign operational phases based on unit and time.
- Load and preprocess test data (scaling sensors).
- Detect anomalies per phase using pre-loaded Isolation Forest models.

### run_detection.py

Main script to:

- Load models and scaler.
- Load and preprocess test data.
- Run anomaly detection.
- Save results in NAB-compatible CSV format.

---

## Example Code Snippets

### Phase assignment and preprocessing

```python
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
```

### Anomaly detection per phase

```python
def detect_anomalies(df, models):
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
    return pd.concat(results).sort_values('timestamp').reset_index(drop=True)
```

---

## Notes

- Ensure your results CSV filename matches the NAB dataset name (e.g., `realKnownCause_test.csv`).
- Adjust thresholds in `config/thresholds.json` or use `--optimize` flag for automatic tuning.
- For real-time detection, adapt detection functions to process one sample at a time with pre-loaded models and scaler.

---

## Troubleshooting

- **OSError about missing directories**: Make sure your `results` folder contains a subfolder named exactly as your detector (`smartsentry`).
- **KeyError in thresholds**: Add your detector and all required profiles to `config/thresholds.json`.
- **NAB scoring errors**: Use `--resultsDir` as the parent folder containing detector folders, not the detector folder itself.

---

## Contact

For questions or help with adapting this project, feel free to open an issue or contact me.

---

# End of README

---