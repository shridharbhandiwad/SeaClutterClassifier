# 🎯 Maritime Radar Target Classification System

A comprehensive machine learning system for classifying maritime radar detections as either **sea clutter** or **actual targets** (vessels). This project includes synthetic data generation using physical radar models, feature engineering, and state-of-the-art ML models for binary classification.

![Radar Classification](https://img.shields.io/badge/Radar-Classification-blue)
![ML Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20RandomForest-green)
![Dataset Size](https://img.shields.io/badge/Dataset-1GB%2B-orange)
![Interactive](https://img.shields.io/badge/Interface-Streamlit-red)

## 🌟 Features

### 🔬 **Realistic Radar Data Generation**
- **Physical Models**: K-distribution and Weibull models for sea clutter
- **Vessel Simulation**: Realistic vessel tracks with different ship types
- **Environmental Conditions**: Multiple sea states (calm, moderate, rough)
- **Radar Physics**: SNR calculation based on radar equation
- **Large Scale**: Generate datasets >= 1GB with millions of detections

### 🤖 **Advanced Machine Learning**
- **Dual Models**: XGBoost and Random Forest classifiers
- **Feature Engineering**: 20+ engineered features from raw radar parameters
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Performance Metrics**: Comprehensive evaluation with ROC, precision-recall curves
- **Model Persistence**: Save/load trained models

### 🖥️ **Interactive Testing Interface**
- **Streamlit Web App**: User-friendly interface for model testing
- **Real-time Prediction**: Input radar parameters and get instant classification
- **Visualization**: Radar plots, parameter analysis, probability breakdowns
- **Batch Processing**: Upload CSV files for bulk classification
- **Preset Scenarios**: Quick testing with predefined vessel/clutter examples

### 📊 **Comprehensive Dataset**
- **Required Fields**: TrackID, Range, Azimuth, Elevation, Doppler, RCS, SNR, Timestamp
- **Labeled Data**: Binary classification (target/clutter) for supervised learning
- **Track Continuity**: Sequential detections forming realistic vessel tracks
- **Multiple Sea States**: Datasets for various environmental conditions
- **Train/Val/Test Splits**: Proper data splitting for model evaluation

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone repository (if applicable) or ensure all files are in the same directory
pip install -r requirements.txt
```

### 2. **Generate Dataset and Train Models**

```bash
# Quick training (small dataset for testing)
python train_radar_models.py --quick

# Full pipeline (complete dataset generation and training)
python train_radar_models.py --full

# Custom training
python train_radar_models.py --generate-main --main-size-gb 2.0 --cross-validation
```

### 3. **Interactive Testing**

```bash
# Launch the web interface
streamlit run interactive_radar_tester.py
```

Then open your browser to `http://localhost:8501` and start testing!

## 📁 Project Structure

```
maritime_radar_classification/
├── 📊 Data Generation
│   ├── radar_data_generator.py      # Synthetic radar data generation
│   └── train_radar_models.py        # Training pipeline script
├── 🤖 Machine Learning
│   ├── radar_ml_models.py           # XGBoost & Random Forest models
│   └── interactive_radar_tester.py  # Streamlit testing interface
├── 📋 Configuration
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This file
└── 📈 Generated Files (after training)
    ├── maritime_radar_dataset_*.parquet  # Generated datasets
    ├── radar_classifier_*.joblib         # Trained models
    ├── training_results.json             # Training metrics
    ├── training_summary.txt              # Human-readable summary
    └── data_splits/                      # Train/val/test splits
```

## 🔧 Detailed Usage

### **Dataset Generation**

```python
from radar_data_generator import RadarDatasetGenerator, EnvironmentConfig

# Create generator for specific sea state
env_config = EnvironmentConfig(sea_state=3)  # Moderate sea
generator = RadarDatasetGenerator(env_config=env_config)

# Generate 1GB dataset
dataset = generator.generate_complete_dataset(target_size_gb=1.0)
generator.save_dataset(dataset, "my_radar_dataset")
```

### **Model Training**

```python
from radar_ml_models import RadarTargetClassifier
import pandas as pd

# Load dataset
df = pd.read_parquet("maritime_radar_dataset_main.parquet")

# Train models
classifier = RadarTargetClassifier()
results = classifier.train_models(df)

# Save models
classifier.save_models("my_classifier")
```

### **Single Prediction**

```python
# Load trained models
classifier = RadarTargetClassifier()
classifier.load_models("radar_classifier")

# Make prediction
detection_data = {
    'Range_m': 15000.0,
    'Azimuth_deg': 45.0,
    'Elevation_deg': 0.2,
    'Doppler_ms': 8.5,
    'RCS_dBsm': 35.0,
    'SNR_dB': 25.0,
    'TrackID': 'TEST_001',
    'Timestamp': '2024-01-01T12:00:00Z'
}

result = classifier.predict_single(detection_data, model_name='xgboost')
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")
```

## 📋 Dataset Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `TrackID` | string | Unique track identifier | `"TARGET_000123"` |
| `Range_m` | float | Distance from radar (meters) | `15000.0` |
| `Azimuth_deg` | float | Bearing angle (degrees) | `45.0` |
| `Elevation_deg` | float | Elevation angle (degrees) | `0.2` |
| `Doppler_ms` | float | Radial velocity (m/s) | `8.5` |
| `RCS_dBsm` | float | Radar Cross Section (dBsm) | `35.0` |
| `SNR_dB` | float | Signal-to-Noise Ratio (dB) | `25.0` |
| `Timestamp` | string | ISO-8601 timestamp | `"2024-01-01T12:00:00Z"` |
| `Label` | string | Classification label | `"target"` or `"clutter"` |

## 🎯 Command Line Options

### **Training Script Options**

```bash
python train_radar_models.py [OPTIONS]

Dataset Generation:
  --generate-main              Generate main dataset
  --generate-sea-states        Generate sea state datasets
  --main-size-gb FLOAT         Main dataset size in GB (default: 1.0)
  --sea-state-size-gb FLOAT    Sea state dataset size in GB (default: 0.3)

Training Configuration:
  --dataset-path PATH          Use existing dataset file
  --train-ratio FLOAT          Training set ratio (default: 0.7)
  --val-ratio FLOAT            Validation set ratio (default: 0.15)
  --test-ratio FLOAT           Test set ratio (default: 0.15)
  --random-state INT           Random seed (default: 42)

Model Options:
  --cross-validation           Perform cross-validation
  --cv-folds INT              Number of CV folds (default: 5)
  --save-models               Save trained models (default: True)
  --save-splits               Save train/val/test splits

Quick Options:
  --quick                     Quick training (small dataset, no CV)
  --full                      Full pipeline (generate + train + CV)
```

### **Interactive Interface**

The Streamlit interface provides:

- 🎛️ **Parameter Input**: Interactive forms for radar detection parameters
- 📊 **Real-time Results**: Instant classification with confidence scores
- 📈 **Visualizations**: Radar plots, parameter analysis, probability charts
- 🚢 **Preset Scenarios**: Quick testing with typical vessel/clutter examples
- 📁 **Batch Processing**: Upload CSV files for bulk classification
- 💾 **Result Export**: Download classification results

## 🔬 Technical Details

### **Physical Models**

1. **Sea Clutter Generation**:
   - K-distribution for amplitude fluctuations
   - Weibull distribution for different sea states
   - Range-dependent RCS modeling
   - Angular dependence simulation

2. **Target Simulation**:
   - Realistic vessel types (fishing boats, cargo ships, tankers, etc.)
   - Physics-based RCS calculations
   - Kinematic vessel movement models
   - Doppler velocity from radial motion

3. **Radar Equation**:
   - SNR calculation based on radar parameters
   - Range loss (1/R⁴) modeling
   - Antenna gain and noise figure effects

### **Feature Engineering**

The system creates 20+ engineered features:

- **Spatial**: Range bins, azimuth sectors, elevation categories
- **Temporal**: Time of day, day of week patterns
- **Signal**: RCS normalization, power ratios, linear transformations
- **Kinematic**: Doppler categorization, velocity analysis
- **Statistical**: Track-level aggregations, consistency metrics

### **Model Architecture**

1. **XGBoost Classifier**:
   - Gradient boosting with hyperparameter tuning
   - Optimized for structured/tabular data
   - Grid search over learning rate, depth, estimators

2. **Random Forest**:
   - Ensemble of decision trees
   - Feature importance analysis
   - Robust to overfitting

3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC and Precision-Recall curves
   - Confusion matrices
   - Cross-validation scores

## 📊 Expected Performance

Based on the synthetic data with realistic physics modeling:

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| **XGBoost** | ~95-98% | ~95-97% | ~98-99% |
| **Random Forest** | ~93-96% | ~93-96% | ~96-98% |

*Performance varies with sea state and dataset composition*

## 🛠️ Customization

### **Custom Radar Configuration**

```python
from radar_data_generator import RadarConfig

custom_radar = RadarConfig(
    frequency=9.4e9,        # X-band frequency
    transmit_power=50e3,    # 50 kW transmitter
    antenna_gain=40,        # 40 dB antenna gain
    range_resolution=75     # 75m range resolution
)
```

### **Custom Environment**

```python
from radar_data_generator import EnvironmentConfig

rough_sea = EnvironmentConfig(
    sea_state=7,           # Very rough sea
    wind_speed=25,         # 25 m/s wind
    wave_height=6.0        # 6m significant wave height
)
```

### **Custom Model Parameters**

```python
from radar_ml_models import RadarTargetClassifier

classifier = RadarTargetClassifier()
# Modify hyperparameter grids in radar_ml_models.py
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Memory Issues**: Reduce dataset size with `--main-size-gb 0.5`
2. **Training Time**: Use `--quick` for fast testing
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Streamlit Issues**: Ensure port 8501 is available

### **Performance Optimization**

- Use smaller datasets for development/testing
- Enable multiprocessing with `n_jobs=-1` in models
- Use parquet format for faster I/O
- Consider GPU acceleration for XGBoost if available

## 🤝 Contributing

To extend this project:

1. **Add New Features**: Modify `RadarFeatureEngineering` class
2. **New Models**: Extend `RadarTargetClassifier` with additional algorithms
3. **Custom Scenarios**: Add new vessel types or environmental conditions
4. **UI Improvements**: Enhance the Streamlit interface

## 📝 License

This project is designed for educational and research purposes. Feel free to modify and distribute according to your needs.

## 🎯 Use Cases

- **Maritime Surveillance**: Coast guard and naval applications
- **Research**: Radar signal processing and ML algorithm development
- **Education**: Teaching radar principles and ML classification
- **Benchmarking**: Testing new algorithms against realistic datasets
- **System Design**: Evaluating radar performance under different conditions

---

**🚀 Ready to classify some radar targets? Start with `python train_radar_models.py --quick` and then launch the interactive interface with `streamlit run interactive_radar_tester.py`!**