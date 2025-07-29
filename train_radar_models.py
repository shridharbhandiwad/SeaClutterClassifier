#!/usr/bin/env python3
"""
Maritime Radar Target Classification Training Script

This script:
1. Generates synthetic radar datasets with various sea states
2. Splits data into train/validation/test sets
3. Trains XGBoost and Random Forest models
4. Evaluates model performance
5. Saves trained models and results
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from radar_data_generator import RadarDatasetGenerator, RadarConfig, EnvironmentConfig
from radar_ml_models import RadarTargetClassifier, cross_validate_models


class DatasetSplitter:
    """Utility class for creating proper train/validation/test splits"""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_dataset(self, df: pd.DataFrame, stratify_col: str = 'Label', 
                     random_state: int = 42) -> tuple:
        """Split dataset into train/validation/test sets"""
        
        print(f"Original dataset size: {len(df):,} samples")
        print(f"Class distribution: {df[stratify_col].value_counts().to_dict()}")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(self.val_ratio + self.test_ratio),
            random_state=random_state,
            stratify=df[stratify_col]
        )
        
        # Second split: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=temp_df[stratify_col]
        )
        
        print(f"\nDataset splits:")
        print(f"Training: {len(train_df):,} samples ({len(train_df)/len(df):.1%})")
        print(f"Validation: {len(val_df):,} samples ({len(val_df)/len(df):.1%})")
        print(f"Test: {len(test_df):,} samples ({len(test_df)/len(df):.1%})")
        
        # Verify class distributions are maintained
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            class_dist = split_df[stratify_col].value_counts(normalize=True)
            print(f"{split_name} class distribution: {class_dist.to_dict()}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, output_dir: str = "data_splits"):
        """Save dataset splits to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        train_df.to_parquet(f"{output_dir}/train_dataset.parquet", compression='gzip')
        val_df.to_parquet(f"{output_dir}/val_dataset.parquet", compression='gzip')
        test_df.to_parquet(f"{output_dir}/test_dataset.parquet", compression='gzip')
        
        # Save split metadata
        metadata = {
            'split_timestamp': datetime.utcnow().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'class_distributions': {
                'train': train_df['Label'].value_counts().to_dict(),
                'val': val_df['Label'].value_counts().to_dict(),
                'test': test_df['Label'].value_counts().to_dict()
            }
        }
        
        with open(f"{output_dir}/split_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset splits saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - train_dataset.parquet ({len(train_df):,} samples)")
        print(f"  - val_dataset.parquet ({len(val_df):,} samples)")
        print(f"  - test_dataset.parquet ({len(test_df):,} samples)")
        print(f"  - split_metadata.json")


def generate_datasets(config):
    """Generate radar datasets with different configurations"""
    
    print("="*60)
    print("GENERATING RADAR DATASETS")
    print("="*60)
    
    datasets = {}
    
    if config.generate_main:
        print(f"\n📡 Generating main dataset ({config.main_size_gb:.1f} GB)...")
        main_generator = RadarDatasetGenerator()
        main_dataset = main_generator.generate_complete_dataset(target_size_gb=config.main_size_gb)
        main_generator.save_dataset(main_dataset, "maritime_radar_dataset_main")
        datasets['main'] = main_dataset
        print(f"✅ Main dataset generated: {len(main_dataset):,} samples")
    
    if config.generate_sea_states:
        print(f"\n🌊 Generating sea state datasets...")
        sea_states = [1, 3, 6]  # Calm, moderate, rough
        
        for sea_state in sea_states:
            print(f"\n  Sea State {sea_state}...")
            env_config = EnvironmentConfig(sea_state=sea_state)
            generator = RadarDatasetGenerator(env_config=env_config)
            dataset = generator.generate_complete_dataset(target_size_gb=config.sea_state_size_gb)
            generator.save_dataset(dataset, f"maritime_radar_dataset_sea_state_{sea_state}")
            datasets[f'sea_state_{sea_state}'] = dataset
            print(f"  ✅ Sea state {sea_state} dataset: {len(dataset):,} samples")
    
    return datasets


def train_models(config):
    """Train classification models"""
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    # Load or use generated dataset
    if config.dataset_path:
        print(f"\n📁 Loading dataset from {config.dataset_path}")
        if config.dataset_path.endswith('.parquet'):
            df = pd.read_parquet(config.dataset_path)
        else:
            df = pd.read_csv(config.dataset_path)
    else:
        print(f"\n📁 Loading main generated dataset...")
        df = pd.read_parquet("maritime_radar_dataset_main.parquet")
    
    print(f"Dataset loaded: {len(df):,} samples")
    print(f"Class distribution: {df['Label'].value_counts().to_dict()}")
    
    # Create dataset splits
    splitter = DatasetSplitter(
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    train_df, val_df, test_df = splitter.split_dataset(df, random_state=config.random_state)
    
    if config.save_splits:
        splitter.save_splits(train_df, val_df, test_df)
    
    # Combine train and validation for model training (we'll use test for final evaluation)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Initialize and train classifier
    print(f"\n🤖 Training models...")
    classifier = RadarTargetClassifier()
    
    # Train models
    start_time = time.time()
    results = classifier.train_models(
        train_val_df, 
        test_size=len(test_df)/(len(train_val_df) + len(test_df)),  # Use test_df size as test proportion
        random_state=config.random_state
    )
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.1f} seconds")
    
    # Final evaluation on test set
    print(f"\n📊 Final evaluation on test set...")
    test_results = evaluate_on_test_set(classifier, test_df)
    
    # Cross-validation
    if config.cross_validation:
        print(f"\n🔄 Performing {config.cv_folds}-fold cross-validation...")
        cv_results = cross_validate_models(train_val_df, cv_folds=config.cv_folds)
        
        # Add CV results to main results
        for model_name in results.keys():
            if model_name in cv_results:
                results[model_name]['cv_scores'] = cv_results[model_name]
    
    # Save models
    if config.save_models:
        print(f"\n💾 Saving trained models...")
        classifier.save_models("radar_classifier")
        
        # Save training results
        training_results = {
            'training_time_seconds': training_time,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'features_used': classifier.feature_names
            },
            'model_results': results,
            'test_results': test_results,
            'config': vars(config)
        }
        
        with open('training_results.json', 'w') as f:
            # Convert numpy types to JSON serializable types
            json_results = convert_numpy_types(training_results)
            json.dump(json_results, f, indent=2)
        
        print("✅ Models and results saved!")
    
    return classifier, results, test_results


def evaluate_on_test_set(classifier, test_df):
    """Evaluate trained models on test set"""
    
    # Prepare test features
    X_test, y_test, _ = classifier.feature_engineer.prepare_features(test_df)
    
    results = {}
    
    for model_name, model in classifier.models.items():
        print(f"\n  Evaluating {model_name}...")
        
        # Scale features if needed
        if model_name == 'xgboost' and hasattr(classifier, 'scaler'):
            X_test_processed = classifier.scaler.transform(X_test)
        else:
            X_test_processed = X_test
        
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['Clutter', 'Target'],
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'test_roc_auc': roc_auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    ROC AUC: {roc_auc:.4f}")
    
    return results


def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def create_training_summary(results, test_results, output_file="training_summary.txt"):
    """Create a human-readable training summary"""
    
    with open(output_file, 'w') as f:
        f.write("MARITIME RADAR TARGET CLASSIFICATION TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training completed: {datetime.utcnow().isoformat()}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n\n")
        
        for model_name in results.keys():
            if model_name in test_results:
                test_res = test_results[model_name]
                f.write(f"{model_name.upper().replace('_', ' ')} MODEL:\n")
                f.write(f"  Test Accuracy:  {test_res['test_accuracy']:.4f}\n")
                f.write(f"  Test F1 Score:  {test_res['test_f1']:.4f}\n")
                f.write(f"  Test Precision: {test_res['test_precision']:.4f}\n")
                f.write(f"  Test Recall:    {test_res['test_recall']:.4f}\n")
                f.write(f"  Test ROC AUC:   {test_res['test_roc_auc']:.4f}\n")
                
                # Cross-validation if available
                if 'cv_scores' in results[model_name]:
                    cv_scores = results[model_name]['cv_scores']
                    f.write(f"  CV F1 Score:    {cv_scores['mean_f1']:.4f} ± {cv_scores['std_f1']:.4f}\n")
                
                f.write("\n")
        
        f.write("CONFUSION MATRICES\n")
        f.write("-" * 20 + "\n\n")
        
        for model_name, test_res in test_results.items():
            cm = test_res['confusion_matrix']
            f.write(f"{model_name.upper()} Confusion Matrix:\n")
            f.write("                 Predicted\n")
            f.write("                Clutter Target\n")
            f.write(f"Actual Clutter  {cm[0][0]:6d}  {cm[0][1]:6d}\n")
            f.write(f"       Target   {cm[1][0]:6d}  {cm[1][1]:6d}\n\n")
    
    print(f"Training summary saved to {output_file}")


def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description="Train Maritime Radar Classification Models")
    
    # Dataset generation options
    parser.add_argument("--generate-main", action="store_true", 
                       help="Generate main dataset")
    parser.add_argument("--generate-sea-states", action="store_true",
                       help="Generate datasets for different sea states")
    parser.add_argument("--main-size-gb", type=float, default=1.0,
                       help="Size of main dataset in GB (default: 1.0)")
    parser.add_argument("--sea-state-size-gb", type=float, default=0.3,
                       help="Size of each sea state dataset in GB (default: 0.3)")
    
    # Training options
    parser.add_argument("--dataset-path", type=str,
                       help="Path to existing dataset file (skip generation)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio (default: 0.15)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility (default: 42)")
    
    # Model options
    parser.add_argument("--cross-validation", action="store_true",
                       help="Perform cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of CV folds (default: 5)")
    parser.add_argument("--save-models", action="store_true", default=True,
                       help="Save trained models (default: True)")
    parser.add_argument("--save-splits", action="store_true",
                       help="Save train/val/test splits")
    
    # Quick options
    parser.add_argument("--quick", action="store_true",
                       help="Quick training (small dataset, no CV)")
    parser.add_argument("--full", action="store_true",
                       help="Full training pipeline (generate data + train)")
    
    args = parser.parse_args()
    
    # Quick mode settings
    if args.quick:
        args.generate_main = True
        args.main_size_gb = 0.1  # Small dataset for quick testing
        args.cross_validation = False
        print("🚀 Quick mode activated: small dataset, no cross-validation")
    
    # Full mode settings
    if args.full:
        args.generate_main = True
        args.generate_sea_states = True
        args.cross_validation = True
        args.save_splits = True
        print("🎯 Full mode activated: complete pipeline")
    
    # Generate datasets if requested
    if args.generate_main or args.generate_sea_states:
        datasets = generate_datasets(args)
    
    # Train models
    classifier, results, test_results = train_models(args)
    
    # Create summary
    create_training_summary(results, test_results)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Files generated:")
    if args.generate_main:
        print("  📊 maritime_radar_dataset_main.parquet")
    if args.generate_sea_states:
        print("  📊 maritime_radar_dataset_sea_state_*.parquet")
    if args.save_models:
        print("  🤖 radar_classifier_*.joblib (trained models)")
        print("  📋 training_results.json")
        print("  📄 training_summary.txt")
    if args.save_splits:
        print("  🗂️ data_splits/ (train/val/test splits)")
    
    print("\n🎯 Ready for inference! Use interactive_radar_tester.py to test models.")


if __name__ == "__main__":
    main()