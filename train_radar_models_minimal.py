#!/usr/bin/env python3
"""
Minimal Maritime Radar Target Classification Training Script

This ultra-lightweight script is designed for memory and storage constrained environments.
Features:
- Fixed hyperparameters (no grid search)
- Minimal memory usage
- Single model training
- Basic dataset generation
"""

import os
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

# Try to import required modules with fallbacks
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    import joblib
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Install with: pip install scikit-learn joblib")
    sys.exit(1)

try:
    from radar_data_generator import RadarDatasetGenerator
    from radar_feature_engineering import RadarFeatureEngineer
except ImportError:
    print("❌ Missing custom modules. Ensure all project files are present.")
    sys.exit(1)


def create_tiny_dataset(target_samples=5000):
    """Create a very small dataset for testing"""
    print(f"🔬 Generating tiny dataset ({target_samples:,} samples)...")
    
    # Create minimal generator
    generator = RadarDatasetGenerator()
    
    # Generate minimal data (override the normal generation)
    clutter_samples = target_samples // 5  # 20% clutter
    target_samples_actual = target_samples - clutter_samples
    
    # Generate clutter detections
    clutter_data = []
    for i in range(clutter_samples):
        clutter_data.append({
            'Range': np.random.uniform(1000, 50000),
            'Azimuth': np.random.uniform(0, 360),
            'RCS': np.random.uniform(-40, -10),
            'Doppler': np.random.uniform(-2, 2),
            'SNR': np.random.uniform(5, 20),
            'Label': 'clutter',
            'TrackID': f'C_{i}',
            'Timestamp': i
        })
    
    # Generate target detections  
    target_data = []
    for i in range(target_samples_actual):
        target_data.append({
            'Range': np.random.uniform(1000, 50000),
            'Azimuth': np.random.uniform(0, 360),
            'RCS': np.random.uniform(-20, 30),
            'Doppler': np.random.uniform(-50, 50),
            'SNR': np.random.uniform(10, 40),
            'Label': 'target',
            'TrackID': f'T_{i}',
            'Timestamp': i + clutter_samples
        })
    
    # Combine and shuffle
    all_data = clutter_data + target_data
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✅ Dataset created: {len(df):,} samples")
    print(f"   Clutter: {len(df[df['Label']=='clutter']):,}")
    print(f"   Target:  {len(df[df['Label']=='target']):,}")
    
    return df


def train_minimal_model(df):
    """Train a single model with fixed hyperparameters"""
    print("\n🤖 Training minimal model...")
    
    # Feature engineering
    feature_engineer = RadarFeatureEngineer()
    X, y, feature_names = feature_engineer.prepare_features(df)
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {X.shape[0]:,}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with fixed hyperparameters
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=1  # Single core
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   F1 Score:  {results['f1']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")
    
    return model, scaler, feature_names, results


def main():
    parser = argparse.ArgumentParser(description="Minimal Radar Classification Training")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Number of samples to generate (default: 5000)")
    parser.add_argument("--save-model", action="store_true",
                       help="Save trained model")
    
    args = parser.parse_args()
    
    print("🎯 MINIMAL RADAR CLASSIFIER TRAINING")
    print("=" * 50)
    print(f"Target samples: {args.samples:,}")
    
    # Clear memory
    gc.collect()
    
    # Generate or load dataset
    if os.path.exists("minimal_radar_dataset.parquet"):
        print("📁 Loading existing minimal dataset...")
        df = pd.read_parquet("minimal_radar_dataset.parquet")
    else:
        df = create_tiny_dataset(args.samples)
        # Save with maximum compression
        df.to_parquet("minimal_radar_dataset.parquet", compression='brotli')
        print("💾 Dataset saved to minimal_radar_dataset.parquet")
    
    # Train model
    model, scaler, feature_names, results = train_minimal_model(df)
    
    # Save model if requested
    if args.save_model:
        print("\n💾 Saving model...")
        joblib.dump({
            'model': model,
            'scaler': scaler, 
            'feature_names': feature_names,
            'results': results
        }, 'minimal_radar_classifier.joblib')
        print("✅ Model saved to minimal_radar_classifier.joblib")
    
    # Save results
    with open('minimal_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🎉 Training complete!")
    print("Files created:")
    print("  📊 minimal_radar_dataset.parquet")
    print("  📋 minimal_training_results.json")
    if args.save_model:
        print("  🤖 minimal_radar_classifier.joblib")


if __name__ == "__main__":
    main()