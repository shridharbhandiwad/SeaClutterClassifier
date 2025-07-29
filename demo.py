#!/usr/bin/env python3
"""
Quick Demo Script for Maritime Radar Classification

This script demonstrates basic usage of the radar classification system:
1. Generate a small sample dataset
2. Train basic models
3. Test with example predictions
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

from radar_data_generator import RadarDatasetGenerator, RadarConfig, EnvironmentConfig
from radar_ml_models import RadarTargetClassifier


def create_demo_dataset(size_mb: float = 50) -> pd.DataFrame:
    """Create a small demo dataset for quick testing"""
    print(f"🔄 Generating demo dataset ({size_mb} MB)...")
    
    # Use moderate sea state for demo
    env_config = EnvironmentConfig(sea_state=3)
    generator = RadarDatasetGenerator(env_config=env_config)
    
    # Generate small dataset
    dataset = generator.generate_complete_dataset(target_size_gb=size_mb/1000)
    
    print(f"✅ Demo dataset created: {len(dataset):,} samples")
    print(f"   - Clutter: {len(dataset[dataset['Label'] == 'clutter']):,}")
    print(f"   - Targets: {len(dataset[dataset['Label'] == 'target']):,}")
    
    return dataset


def train_demo_models(dataset: pd.DataFrame) -> RadarTargetClassifier:
    """Train models on demo dataset"""
    print(f"\n🤖 Training demo models...")
    
    classifier = RadarTargetClassifier()
    results = classifier.train_models(dataset, test_size=0.2)
    
    print(f"✅ Training complete!")
    
    # Display results
    for model_name, result in results.items():
        print(f"\n📊 {model_name.upper()} Results:")
        print(f"   Accuracy: {result['accuracy']:.3f}")
        print(f"   F1 Score: {result['f1_score']:.3f}")
        print(f"   ROC AUC:  {result['roc_auc']:.3f}")
    
    return classifier


def test_predictions(classifier: RadarTargetClassifier):
    """Test predictions with example scenarios"""
    print(f"\n🎯 Testing Predictions...")
    
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
                'TrackID': 'DEMO_CARGO',
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
                'TrackID': 'DEMO_CLUTTER',
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
                'TrackID': 'DEMO_PATROL',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'expected': 'target'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n  📋 {scenario['name']}:")
        
        # Test with both models
        for model_name in classifier.models.keys():
            result = classifier.predict_single(scenario['data'], model_name)
            
            # Check if prediction matches expected
            correct = "✅" if result['prediction'] == scenario['expected'] else "❌"
            
            print(f"    {model_name}: {result['prediction']} "
                  f"({result['confidence']:.1%} confidence) {correct}")


def save_demo_results(classifier: RadarTargetClassifier):
    """Save demo models for interactive testing"""
    print(f"\n💾 Saving demo models...")
    
    try:
        classifier.save_models("demo_radar_classifier")
        print(f"✅ Demo models saved as 'demo_radar_classifier_*'")
        print(f"   You can now use the interactive interface!")
    except Exception as e:
        print(f"❌ Error saving models: {e}")


def main():
    """Main demo function"""
    print("🎯 Maritime Radar Classification Demo")
    print("=" * 50)
    
    try:
        # Step 1: Create demo dataset
        dataset = create_demo_dataset(size_mb=50)
        
        # Step 2: Train models
        classifier = train_demo_models(dataset)
        
        # Step 3: Test predictions
        test_predictions(classifier)
        
        # Step 4: Save models
        save_demo_results(classifier)
        
        print(f"\n🎉 Demo Complete!")
        print(f"\nNext steps:")
        print(f"1. Run full training: python train_radar_models.py --full")
        print(f"2. Launch interactive interface: streamlit run interactive_radar_tester.py")
        print(f"3. Load demo models in the interface for testing")
        
    except KeyboardInterrupt:
        print(f"\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()