# 🎯 Maritime Radar Classification - Google Colab Setup Guide

This guide will help you run the Maritime Radar Target Classification System on Google Colab with automatic Google Drive integration for data storage.

## 🚀 Quick Start

### Step 1: Upload Files to Google Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Create a new notebook** or open an existing one
3. **Upload the required files** using the file upload button in Colab:

**Required Files:**
- `radar_data_generator.py`
- `radar_ml_models.py` 
- `colab_setup.py` (the main script)

**Optional Files:**
- `train_radar_models.py`
- `demo.py`
- `requirements.txt`

### Step 2: Run the Setup Script

In your first cell, copy and paste the entire content of `colab_setup.py` or upload it and run:

```python
# If you uploaded colab_setup.py, run this:
exec(open('colab_setup.py').read())

# Or paste the entire colab_setup.py content directly into the cell
```

### Step 3: Execute the Pipeline

After the setup is complete, run the main pipeline:

```python
# Run the complete pipeline
report = run_complete_pipeline()
```

## 📋 What Happens During Execution

### 1. Environment Setup (~ 2-3 minutes)
- ✅ Installs required Python packages
- ✅ Mounts Google Drive
- ✅ Creates project directories
- ✅ Imports libraries

### 2. Data Generation (~ 5-10 minutes)
- 🔬 Generates synthetic radar dataset (default: 0.5 GB)
- 🌊 Simulates realistic sea clutter and vessel targets
- 📊 Creates comprehensive radar detection data

### 3. Data Analysis (~ 2-3 minutes)
- 📈 Statistical analysis of generated data
- 🎨 Creates visualization plots
- 🎯 Interactive radar polar plots

### 4. Model Training (~ 10-15 minutes)
- 🤖 Trains XGBoost and Random Forest models
- 🔄 Performs hyperparameter optimization
- 📊 Evaluates model performance

### 5. Model Evaluation (~ 2-3 minutes)
- 📈 Performance comparison charts
- 🔥 Confusion matrices
- 🎯 Test scenario predictions

### 6. Save to Google Drive (~ 1-2 minutes)
- 💾 Saves all results to your Google Drive
- 📁 Organizes files in structured folders
- 📋 Creates comprehensive report

## 📁 Output File Structure

After completion, your Google Drive will contain:

```
📁 Maritime_Radar_Classification/
├── 📊 datasets/
│   ├── maritime_radar_dataset_main.parquet  # Generated radar data
│   └── dataset_info.json                    # Dataset metadata
├── 🤖 models/
│   ├── radar_classifier_xgboost.joblib      # Trained XGBoost model
│   ├── radar_classifier_random_forest.joblib # Trained Random Forest model
│   └── feature_engineering.joblib           # Feature engineering pipeline
├── 📈 results/
│   ├── training_results.json               # Model performance metrics
│   ├── training_summary.json               # Training summary
│   └── prediction_tests.json               # Test scenario results
├── 📊 visualizations/
│   ├── dataset_analysis.png                # Data distribution plots
│   ├── radar_polar_plot.html               # Interactive radar plot
│   ├── model_comparison.png                # Model performance comparison
│   └── confusion_matrices.png              # Model confusion matrices
├── 📋 source_code/
│   ├── radar_data_generator.py             # Backup of source files
│   ├── radar_ml_models.py
│   └── colab_setup.py
└── 📋 Maritime_Radar_Classification_Report.json  # Comprehensive report
```

## ⚙️ Configuration Options

You can modify these variables in the `colab_setup.py` script:

```python
# Dataset configuration
DATASET_SIZE_GB = 0.5    # Dataset size (0.1 - 2.0 GB recommended for Colab)
SEA_STATE = 3            # Sea conditions (1=calm, 7=very rough)
RANDOM_STATE = 42        # For reproducible results

# Model configuration
TEST_SIZE = 0.2          # Test set ratio
CROSS_VALIDATION = True  # Enable cross-validation
CV_FOLDS = 5            # Number of cross-validation folds
```

## 🎯 Expected Performance

With the default configuration, you can expect:

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| **XGBoost** | ~95-98% | ~95-97% | ~8-12 minutes |
| **Random Forest** | ~93-96% | ~93-96% | ~5-8 minutes |

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. Memory Issues**
```python
# Reduce dataset size
DATASET_SIZE_GB = 0.1  # Start smaller
```

**2. Import Errors**
- Ensure all required files are uploaded to Colab
- Check file names match exactly
- Restart runtime if needed

**3. Google Drive Mount Issues**
```python
# Re-mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**4. Long Training Times**
```python
# Use smaller dataset for testing
DATASET_SIZE_GB = 0.1
CROSS_VALIDATION = False  # Disable for faster training
```

### Performance Optimization

**For Faster Execution:**
- Use smaller datasets (0.1-0.3 GB)
- Disable cross-validation initially
- Use GPU runtime (Runtime → Change runtime type → GPU)

**For Better Model Performance:**
- Increase dataset size (1.0+ GB)
- Enable cross-validation
- Try different sea states

## 📱 Using Your Trained Models

### Download Models from Google Drive

1. Navigate to `/MyDrive/Maritime_Radar_Classification/models/` in Google Drive
2. Download the `.joblib` files:
   - `radar_classifier_xgboost.joblib`
   - `radar_classifier_random_forest.joblib`
   - `feature_engineering.joblib`

### Load and Use Models in Python

```python
import joblib
import pandas as pd

# Load trained models
xgb_model = joblib.load('radar_classifier_xgboost.joblib')
rf_model = joblib.load('radar_classifier_random_forest.joblib')
feature_pipeline = joblib.load('feature_engineering.joblib')

# Make predictions on new data
def predict_radar_detection(detection_data):
    """
    detection_data: dict with keys:
    - Range_m, Azimuth_deg, Elevation_deg, Doppler_ms, 
    - RCS_dBsm, SNR_dB, TrackID, Timestamp
    """
    df = pd.DataFrame([detection_data])
    features = feature_pipeline.transform(df)
    
    # Get predictions
    xgb_pred = xgb_model.predict(features)[0]
    xgb_prob = xgb_model.predict_proba(features)[0].max()
    
    rf_pred = rf_model.predict(features)[0]
    rf_prob = rf_model.predict_proba(features)[0].max()
    
    return {
        'xgboost': {'prediction': xgb_pred, 'confidence': xgb_prob},
        'random_forest': {'prediction': rf_pred, 'confidence': rf_prob}
    }

# Example usage
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

result = predict_radar_detection(new_detection)
print(f"XGBoost: {result['xgboost']['prediction']} ({result['xgboost']['confidence']:.1%})")
print(f"Random Forest: {result['random_forest']['prediction']} ({result['random_forest']['confidence']:.1%})")
```

## 🚀 Advanced Usage

### Run Individual Components

```python
# Generate dataset only
dataset, generator = generate_radar_dataset()

# Analyze existing dataset
analyze_dataset(dataset)

# Train models only
classifier, results = train_models(dataset)

# Test specific scenarios
test_results = test_predictions(classifier)

# Save specific components
save_to_drive(dataset, classifier, training_results, test_results)
```

### Custom Configurations

```python
# Different sea states
SEA_STATE = 1  # Calm sea
SEA_STATE = 7  # Very rough sea

# Larger datasets (if you have Colab Pro)
DATASET_SIZE_GB = 2.0

# Different train/test splits
TEST_SIZE = 0.3  # 30% test set
```

## 📞 Support and Next Steps

### Scaling Up
- **Colab Pro**: For larger datasets and faster execution
- **Local Deployment**: Download models for real-time applications
- **Cloud Deployment**: Use cloud platforms for production systems

### Customization
- **New Vessel Types**: Modify radar signature models
- **Different Environments**: Adjust sea state and weather conditions
- **Additional Features**: Extend feature engineering pipeline
- **New Models**: Add different ML algorithms

### Real-World Applications
- **Maritime Surveillance**: Coast guard and naval operations
- **Port Security**: Harbor monitoring systems
- **Research**: Radar signal processing studies
- **Education**: Teaching radar and ML concepts

---

## 🎉 Ready to Start?

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Upload the files**: `radar_data_generator.py`, `radar_ml_models.py`, `colab_setup.py`
3. **Run the setup**: Copy and execute the `colab_setup.py` content
4. **Execute the pipeline**: `report = run_complete_pipeline()`
5. **Access your results**: Check your Google Drive!

**Estimated total time: 25-35 minutes**

Good luck with your maritime radar classification project! 🎯🌊