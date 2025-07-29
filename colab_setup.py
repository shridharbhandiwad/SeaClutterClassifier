#!/usr/bin/env python3
"""
Maritime Radar Target Classification System - Google Colab Setup

This script contains all the code needed to run the maritime radar classification
system on Google Colab with automatic Google Drive integration.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Upload this file along with the radar project files to Colab
2. Run each section in sequence
3. All outputs will be automatically saved to Google Drive

Required files to upload to Colab:
- radar_data_generator.py
- radar_ml_models.py
- train_radar_models.py (optional)
- demo.py (optional)
"""

# =============================================================================
# SECTION 1: ENVIRONMENT SETUP
# =============================================================================

print("🚀 MARITIME RADAR CLASSIFICATION - GOOGLE COLAB SETUP")
print("=" * 60)

# Install required packages
print("📦 Installing required packages...")
import subprocess
import sys

packages = [
    "numpy>=1.21.0",
    "pandas>=1.5.0", 
    "scikit-learn>=1.2.0",
    "xgboost>=1.7.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "joblib>=1.2.0",
    "scipy>=1.9.0",
    "pyarrow>=10.0.0",
    "fastparquet>=0.8.0",
    "tqdm>=4.64.0",
    "python-dateutil>=2.8.0",
    "pytz>=2022.0"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ Packages installed successfully!")

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully!")
except ImportError:
    print("ℹ️  Not running in Google Colab - skipping Drive mount")
    # For local testing, create a local directory
    import os
    os.makedirs('/tmp/drive/MyDrive', exist_ok=True)
    print("✅ Local directory created for testing")

# Create project directories
import os
PROJECT_DIR = '/content/drive/MyDrive/Maritime_Radar_Classification'

# Create all necessary subdirectories
directories = [
    PROJECT_DIR,
    f'{PROJECT_DIR}/datasets',
    f'{PROJECT_DIR}/models', 
    f'{PROJECT_DIR}/results',
    f'{PROJECT_DIR}/visualizations',
    f'{PROJECT_DIR}/source_code'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

print(f"\n📁 Project directories created:")
print(f"   Main: {PROJECT_DIR}")
for subdir in directories[1:]:
    print(f"   - {os.path.basename(subdir)}/")

# =============================================================================
# SECTION 2: IMPORT LIBRARIES
# =============================================================================

print("\n📚 Importing libraries...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import pickle
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("✅ Libraries imported successfully!")
print(f"🐍 Python version: {sys.version}")
print(f"🐼 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")

# =============================================================================
# SECTION 3: LOAD PROJECT MODULES
# =============================================================================

print("\n🔧 Loading project modules...")

# This section assumes you've uploaded the radar project files to Colab
try:
    from radar_data_generator import RadarDatasetGenerator, RadarConfig, EnvironmentConfig
    from radar_ml_models import RadarTargetClassifier
    print("✅ Project modules loaded successfully!")
except ImportError as e:
    print(f"❌ Error loading project modules: {e}")
    print("\n📋 REQUIRED FILES:")
    print("Please upload these files to your Colab session:")
    print("1. radar_data_generator.py")
    print("2. radar_ml_models.py")
    print("3. (Optional) train_radar_models.py")
    print("4. (Optional) demo.py")
    print("\nUse the file upload button in Colab or run:")
    print("from google.colab import files")
    print("uploaded = files.upload()")
    raise

# Copy source files to Google Drive for backup
import shutil
source_files = ['radar_data_generator.py', 'radar_ml_models.py', 'colab_setup.py']
for file in source_files:
    if os.path.exists(f'/content/{file}'):
        shutil.copy(f'/content/{file}', f'{PROJECT_DIR}/source_code/')
        print(f"📋 Backed up {file} to Google Drive")

# =============================================================================
# SECTION 4: CONFIGURATION
# =============================================================================

print("\n⚙️  Configuration...")

# Dataset configuration
DATASET_SIZE_GB = 0.5  # Adjust based on your needs (Colab has memory limits)
SEA_STATE = 3  # Moderate sea conditions (1=calm, 7=very rough)
RANDOM_STATE = 42

# Model configuration  
TEST_SIZE = 0.2
CROSS_VALIDATION = True
CV_FOLDS = 5

print(f"📊 Dataset size: {DATASET_SIZE_GB} GB")
print(f"🌊 Sea state: {SEA_STATE} (moderate)")
print(f"🎯 Test split: {TEST_SIZE}")
print(f"🔄 Cross validation: {CROSS_VALIDATION} ({CV_FOLDS} folds)")

# =============================================================================
# SECTION 5: DATA GENERATION
# =============================================================================

def generate_radar_dataset():
    """Generate synthetic radar dataset"""
    print("\n🔬 GENERATING RADAR DATASET")
    print("=" * 40)
    
    print(f"🌊 Generating maritime radar dataset...")
    print(f"📊 Target size: {DATASET_SIZE_GB} GB")
    print(f"🌊 Sea state: {SEA_STATE} (moderate conditions)")
    
    # Create environment configuration
    env_config = EnvironmentConfig(sea_state=SEA_STATE)
    
    # Initialize data generator
    generator = RadarDatasetGenerator(env_config=env_config)
    
    # Generate dataset
    print("🔄 Starting data generation...")
    start_time = datetime.now()
    
    dataset = generator.generate_complete_dataset(target_size_gb=DATASET_SIZE_GB)
    
    generation_time = datetime.now() - start_time
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"⏱️  Generation time: {generation_time}")
    print(f"📊 Total samples: {len(dataset):,}")
    print(f"🎯 Targets: {len(dataset[dataset['Label'] == 'target']):,}")
    print(f"🌊 Clutter: {len(dataset[dataset['Label'] == 'clutter']):,}")
    print(f"💾 Memory usage: {dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return dataset, generator

# =============================================================================
# SECTION 6: DATA ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_dataset(dataset):
    """Analyze and visualize the dataset"""
    print("\n📊 DATASET ANALYSIS")
    print("=" * 30)
    
    # Basic info
    print("📋 Dataset Overview:")
    print(f"   Shape: {dataset.shape}")
    print(f"   Columns: {list(dataset.columns)}")
    print(f"   Memory: {dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print("\n📈 Statistical Summary:")
    print(dataset.describe())
    
    print("\n🎯 Label Distribution:")
    label_counts = dataset['Label'].value_counts()
    print(label_counts)
    print(f"   Target ratio: {label_counts['target'] / len(dataset):.2%}")
    
    # Create comprehensive visualizations
    print("\n🎨 Creating visualizations...")
    
    # Main analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Maritime Radar Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Label distribution pie chart
    axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Target vs Clutter Distribution')
    
    # 2. Range distribution by label
    for label in dataset['Label'].unique():
        subset = dataset[dataset['Label'] == label]
        axes[0, 1].hist(subset['Range_m'], alpha=0.7, label=label, bins=50)
    axes[0, 1].set_xlabel('Range (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Range Distribution')
    axes[0, 1].legend()
    
    # 3. RCS distribution by label
    for label in dataset['Label'].unique():
        subset = dataset[dataset['Label'] == label]
        axes[0, 2].hist(subset['RCS_dBsm'], alpha=0.7, label=label, bins=50)
    axes[0, 2].set_xlabel('RCS (dBsm)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Radar Cross Section Distribution')
    axes[0, 2].legend()
    
    # 4. SNR distribution by label
    for label in dataset['Label'].unique():
        subset = dataset[dataset['Label'] == label]
        axes[1, 0].hist(subset['SNR_dB'], alpha=0.7, label=label, bins=50)
    axes[1, 0].set_xlabel('SNR (dB)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Signal-to-Noise Ratio Distribution')
    axes[1, 0].legend()
    
    # 5. Doppler distribution by label
    for label in dataset['Label'].unique():
        subset = dataset[dataset['Label'] == label]
        axes[1, 1].hist(subset['Doppler_ms'], alpha=0.7, label=label, bins=50)
    axes[1, 1].set_xlabel('Doppler Velocity (m/s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Doppler Velocity Distribution')
    axes[1, 1].legend()
    
    # 6. Azimuth distribution
    axes[1, 2].hist(dataset['Azimuth_deg'], bins=36, alpha=0.7, color='skyblue')
    axes[1, 2].set_xlabel('Azimuth (degrees)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Azimuth Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_DIR}/visualizations/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create interactive radar plot
    print("🎯 Creating interactive radar plot...")
    sample_data = dataset.sample(n=min(5000, len(dataset)))  # Sample for performance
    
    fig_polar = px.scatter_polar(
        sample_data,
        r='Range_m',
        theta='Azimuth_deg', 
        color='Label',
        size='RCS_dBsm',
        hover_data=['SNR_dB', 'Doppler_ms'],
        title='Radar Detections - Polar Plot (Sample Data)',
        color_discrete_map={'target': 'red', 'clutter': 'blue'}
    )
    
    fig_polar.update_layout(
        polar=dict(
            radialaxis=dict(title='Range (m)'),
            angularaxis=dict(title='Azimuth (degrees)')
        ),
        width=800,
        height=600
    )
    
    fig_polar.show()
    fig_polar.write_html(f'{PROJECT_DIR}/visualizations/radar_polar_plot.html')
    
    print("✅ Dataset analysis complete!")
    print(f"📊 Visualizations saved to: {PROJECT_DIR}/visualizations/")

# =============================================================================
# SECTION 7: MODEL TRAINING
# =============================================================================

def train_models(dataset):
    """Train machine learning models"""
    print("\n🤖 MODEL TRAINING")
    print("=" * 25)
    
    # Initialize classifier
    print("🔧 Initializing radar target classifier...")
    classifier = RadarTargetClassifier()
    
    # Train models
    print("🔄 Training models (this may take several minutes)...")
    start_time = datetime.now()
    
    training_results = classifier.train_models(
        dataset,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    training_time = datetime.now() - start_time
    
    print(f"\n✅ Training completed in {training_time}!")
    print("\n📊 Training Results:")
    print("=" * 50)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, results in training_results.items():
        print(f"\n🎯 {model_name.upper()} Model:")
        print(f"   Accuracy:  {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall:    {results['recall']:.4f}")
        print(f"   F1 Score:  {results['f1_score']:.4f}")
        print(f"   ROC AUC:   {results['roc_auc']:.4f}")
        
        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_model = model_name
    
    print(f"\n🏆 Best model: {best_model} (accuracy: {best_accuracy:.4f})")
    
    return classifier, training_results

# =============================================================================
# SECTION 8: MODEL EVALUATION
# =============================================================================

def evaluate_models(classifier, training_results):
    """Create model evaluation visualizations"""
    print("\n📈 MODEL EVALUATION")
    print("=" * 25)
    
    # Model comparison visualization
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(training_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of metrics
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        values = [training_results[model_name][metric] for metric in metrics]
        axes[0].bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x + width/2)
    axes[0].set_xticklabels(metrics, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Radar chart of metrics
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax_polar = plt.subplot(1, 2, 2, projection='polar')
    for model_name in model_names:
        values = [training_results[model_name][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax_polar.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax_polar.fill(angles, values, alpha=0.25)
    
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(metrics)
    ax_polar.set_ylim(0, 1)
    ax_polar.set_title('Model Performance Radar Chart')
    ax_polar.legend()
    ax_polar.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_DIR}/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    
    X_test = classifier.X_test
    y_test = classifier.y_test
    
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        model = classifier.models[model_name]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Clutter', 'Target'],
                    yticklabels=['Clutter', 'Target'],
                    ax=axes[i])
        
        axes[i].set_title(f'{model_name.title()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_DIR}/visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model evaluation complete!")

# =============================================================================
# SECTION 9: PREDICTION TESTING
# =============================================================================

def test_predictions(classifier):
    """Test models with example scenarios"""
    print("\n🎯 PREDICTION TESTING")
    print("=" * 30)
    
    test_scenarios = [
        {
            'name': 'Large Cargo Ship',
            'data': {
                'Range_m': 12000.0,
                'Azimuth_deg': 45.0,
                'Elevation_deg': 0.1,
                'Doppler_ms': 8.2,
                'RCS_dBsm': 38.0,
                'SNR_dB': 22.0,
                'TrackID': 'TEST_CARGO',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'expected': 'target'
        },
        {
            'name': 'Sea Clutter',
            'data': {
                'Range_m': 8000.0,
                'Azimuth_deg': 180.0,
                'Elevation_deg': 0.0,
                'Doppler_ms': 0.5,
                'RCS_dBsm': -28.0,
                'SNR_dB': 4.0,
                'TrackID': 'TEST_CLUTTER',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'expected': 'clutter'
        },
        {
            'name': 'Fast Patrol Boat',
            'data': {
                'Range_m': 6000.0,
                'Azimuth_deg': 315.0,
                'Elevation_deg': -0.1,
                'Doppler_ms': 15.3,
                'RCS_dBsm': 12.0,
                'SNR_dB': 18.0,
                'TrackID': 'TEST_PATROL',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'expected': 'target'
        },
        {
            'name': 'Small Fishing Boat',
            'data': {
                'Range_m': 4000.0,
                'Azimuth_deg': 120.0,
                'Elevation_deg': 0.05,
                'Doppler_ms': 3.2,
                'RCS_dBsm': 8.0,
                'SNR_dB': 12.0,
                'TrackID': 'TEST_FISHING',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'expected': 'target'
        }
    ]
    
    print("🧪 Testing with example scenarios:")
    print("=" * 40)
    
    test_results = []
    
    for scenario in test_scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Expected: {scenario['expected']}")
        print(f"   Parameters: Range={scenario['data']['Range_m']:.0f}m, "
              f"RCS={scenario['data']['RCS_dBsm']:.1f}dBsm, "
              f"SNR={scenario['data']['SNR_dB']:.1f}dB")
        
        scenario_results = {'name': scenario['name'], 'expected': scenario['expected']}
        
        # Test with both models
        for model_name in classifier.models.keys():
            result = classifier.predict_single(scenario['data'], model_name)
            
            # Check if prediction matches expected
            correct = "✅" if result['prediction'] == scenario['expected'] else "❌"
            
            print(f"   {model_name.title()}: {result['prediction']} "
                  f"({result['confidence']:.1%} confidence) {correct}")
            
            scenario_results[f'{model_name}_prediction'] = result['prediction']
            scenario_results[f'{model_name}_confidence'] = result['confidence']
            scenario_results[f'{model_name}_correct'] = result['prediction'] == scenario['expected']
        
        test_results.append(scenario_results)
    
    return test_results

# =============================================================================
# SECTION 10: SAVE RESULTS TO GOOGLE DRIVE
# =============================================================================

def save_to_drive(dataset, classifier, training_results, test_results):
    """Save all results to Google Drive"""
    print("\n💾 SAVING TO GOOGLE DRIVE")
    print("=" * 35)
    
    # Save dataset
    print("📊 Saving dataset...")
    dataset_path = f'{PROJECT_DIR}/datasets/maritime_radar_dataset_main.parquet'
    dataset.to_parquet(dataset_path, compression='snappy')
    
    dataset_info = {
        'generation_time': datetime.now().isoformat(),
        'total_samples': len(dataset),
        'target_samples': len(dataset[dataset['Label'] == 'target']),
        'clutter_samples': len(dataset[dataset['Label'] == 'clutter']),
        'sea_state': SEA_STATE,
        'target_size_gb': DATASET_SIZE_GB,
        'columns': list(dataset.columns),
        'file_path': dataset_path
    }
    
    with open(f'{PROJECT_DIR}/datasets/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save models
    print("🤖 Saving models...")
    model_path = f'{PROJECT_DIR}/models/radar_classifier'
    classifier.save_models(model_path)
    
    # Save training results
    print("📈 Saving training results...")
    with open(f'{PROJECT_DIR}/results/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Save test results
    with open(f'{PROJECT_DIR}/results/prediction_tests.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Create comprehensive report
    print("📋 Creating comprehensive report...")
    report = {
        'project_info': {
            'name': 'Maritime Radar Target Classification System',
            'execution_time': datetime.now().isoformat(),
            'environment': 'Google Colab',
            'save_location': PROJECT_DIR
        },
        'dataset_info': dataset_info,
        'model_performance': training_results,
        'test_scenarios': test_results,
        'configuration': {
            'dataset_size_gb': DATASET_SIZE_GB,
            'sea_state': SEA_STATE,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE
        },
        'files_generated': {
            'dataset': dataset_path,
            'models': [f'{model_path}_{name}.joblib' for name in classifier.models.keys()],
            'visualizations': [
                f'{PROJECT_DIR}/visualizations/dataset_analysis.png',
                f'{PROJECT_DIR}/visualizations/radar_polar_plot.html',
                f'{PROJECT_DIR}/visualizations/model_comparison.png',
                f'{PROJECT_DIR}/visualizations/confusion_matrices.png'
            ],
            'results': [
                f'{PROJECT_DIR}/results/training_results.json',
                f'{PROJECT_DIR}/results/prediction_tests.json'
            ]
        }
    }
    
    with open(f'{PROJECT_DIR}/Maritime_Radar_Classification_Report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # File size summary
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(PROJECT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1
    
    print(f"\n✅ ALL FILES SAVED TO GOOGLE DRIVE!")
    print("=" * 50)
    print(f"📁 Location: {PROJECT_DIR}")
    print(f"📊 Dataset: {len(dataset):,} samples")
    print(f"🤖 Models: {len(classifier.models)} trained")
    print(f"📈 Visualizations: 4 files")
    print(f"📋 Total files: {file_count}")
    print(f"💾 Total size: {total_size / 1024**2:.1f} MB")
    
    return report

# =============================================================================
# SECTION 11: MAIN EXECUTION FUNCTION
# =============================================================================

def run_complete_pipeline():
    """Run the complete maritime radar classification pipeline"""
    print("\n🚀 RUNNING COMPLETE PIPELINE")
    print("=" * 45)
    
    try:
        # Step 1: Generate dataset
        dataset, generator = generate_radar_dataset()
        
        # Step 2: Analyze dataset
        analyze_dataset(dataset)
        
        # Step 3: Train models
        classifier, training_results = train_models(dataset)
        
        # Step 4: Evaluate models
        evaluate_models(classifier, training_results)
        
        # Step 5: Test predictions
        test_results = test_predictions(classifier)
        
        # Step 6: Save everything to Google Drive
        report = save_to_drive(dataset, classifier, training_results, test_results)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"🔗 Access your results at: {PROJECT_DIR}")
        print("\n📋 Next Steps:")
        print("1. 📱 Download models from Google Drive for deployment")
        print("2. 📊 Review visualizations and performance metrics")
        print("3. 🔬 Experiment with different configurations")
        print("4. 🚀 Scale up dataset size for better performance")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

# =============================================================================
# SECTION 12: USAGE INSTRUCTIONS
# =============================================================================

def print_usage_instructions():
    """Print instructions for using this script"""
    print("\n📖 USAGE INSTRUCTIONS")
    print("=" * 30)
    print("To run this script in Google Colab:")
    print("\n1. 📤 Upload required files:")
    print("   - radar_data_generator.py")
    print("   - radar_ml_models.py")
    print("   - colab_setup.py (this file)")
    print("\n2. 🔧 Modify configuration (optional):")
    print("   - DATASET_SIZE_GB: Dataset size (default: 0.5 GB)")
    print("   - SEA_STATE: Sea conditions 1-7 (default: 3)")
    print("\n3. ▶️  Run the pipeline:")
    print("   report = run_complete_pipeline()")
    print("\n4. 📁 Access results in Google Drive:")
    print("   /MyDrive/Maritime_Radar_Classification/")
    print("\n🔧 For custom execution, run individual functions:")
    print("   dataset, generator = generate_radar_dataset()")
    print("   analyze_dataset(dataset)")
    print("   classifier, results = train_models(dataset)")
    print("   # ... etc")

if __name__ == "__main__":
    print_usage_instructions()
    
    # Uncomment the line below to run the complete pipeline automatically
    # report = run_complete_pipeline()