# 🚀 Quick Start: Run Maritime Radar Classification on Google Colab

## 📋 What You'll Get

- **🔬 Synthetic radar dataset** (0.5 GB of realistic maritime data)
- **🤖 Trained ML models** (XGBoost & Random Forest with ~95%+ accuracy)
- **📊 Comprehensive visualizations** (radar plots, performance charts)
- **💾 All results saved to Google Drive** automatically

**Total time: ~25-35 minutes**

## 🎯 Step-by-Step Instructions

### 1. Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com)
- Create a new notebook

### 2. Upload Files
Upload these 3 files to your Colab session:
- ✅ `radar_data_generator.py`
- ✅ `radar_ml_models.py`
- ✅ `colab_setup.py`

### 3. Run the Pipeline
Copy and paste this code into a Colab cell and run it:

```python
# Load and execute the complete pipeline
exec(open('colab_setup.py').read())
report = run_complete_pipeline()
```

### 4. Access Your Results
Everything will be saved to your Google Drive at:
```
/MyDrive/Maritime_Radar_Classification/
```

## 📁 What Gets Generated

```
📁 Maritime_Radar_Classification/
├── 📊 datasets/
│   └── maritime_radar_dataset_main.parquet  # 500MB of radar data
├── 🤖 models/
│   ├── radar_classifier_xgboost.joblib      # Trained XGBoost model
│   └── radar_classifier_random_forest.joblib# Trained Random Forest model
├── 📈 results/
│   ├── training_results.json               # Performance metrics
│   └── prediction_tests.json               # Test scenarios
├── 📊 visualizations/
│   ├── dataset_analysis.png                # Data distribution plots
│   ├── radar_polar_plot.html               # Interactive radar plot
│   ├── model_comparison.png                # Model performance
│   └── confusion_matrices.png              # Classification matrices
└── 📋 Maritime_Radar_Classification_Report.json
```

## ⚙️ Customization Options

Before running `run_complete_pipeline()`, you can modify:

```python
DATASET_SIZE_GB = 0.5    # Dataset size (0.1-2.0 GB)
SEA_STATE = 3            # Sea conditions (1=calm, 7=rough)
TEST_SIZE = 0.2          # Test set ratio
CROSS_VALIDATION = True  # Enable cross-validation
```

## 🎯 Expected Results

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| **XGBoost** | ~95-98% | ~95-97% | ~8-12 minutes |
| **Random Forest** | ~93-96% | ~93-96% | ~5-8 minutes |

## 🔧 Troubleshooting

**Memory Issues?**
```python
DATASET_SIZE_GB = 0.1  # Use smaller dataset
```

**Import Errors?**
- Check all 3 files are uploaded
- File names must match exactly
- Restart runtime if needed

**Slow Performance?**
- Use GPU runtime: Runtime → Change runtime type → GPU
- Disable cross-validation for faster training

## 📱 Using Your Models

After completion, download your models from Google Drive and use them:

```python
import joblib
import pandas as pd

# Load trained models
xgb_model = joblib.load('radar_classifier_xgboost.joblib')
rf_model = joblib.load('radar_classifier_random_forest.joblib')

# Make predictions
new_detection = {
    'Range_m': 10000.0,
    'Azimuth_deg': 45.0,
    'Elevation_deg': 0.1,
    'Doppler_ms': 8.5,
    'RCS_dBsm': 25.0,
    'SNR_dB': 18.0,
    'TrackID': 'TEST_001',
    'Timestamp': '2024-01-01T12:00:00Z'
}

# Convert to DataFrame and predict
df = pd.DataFrame([new_detection])
prediction = xgb_model.predict(df)[0]  # 'target' or 'clutter'
confidence = xgb_model.predict_proba(df)[0].max()

print(f"Prediction: {prediction} ({confidence:.1%} confidence)")
```

## 🎉 That's It!

You'll have a complete maritime radar classification system with:
- Realistic synthetic data
- Production-ready ML models
- Comprehensive evaluation metrics
- Ready-to-use trained models

Perfect for research, education, or building real radar surveillance systems!

---

**Need the full documentation?** See `Google_Colab_Instructions.md` for detailed explanations and advanced usage.