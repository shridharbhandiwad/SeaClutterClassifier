# Memory Optimization Guide for SeaClutterClassifier

This guide explains how to use the memory-optimized features to train radar classification models on resource-constrained systems.

## Quick Solutions for Common Issues

### ❌ "No space left on device" Error
```bash
# Try the low-memory mode with maximum compression
python train_radar_models.py --low-memory --compression brotli

# Or use the minimal training script
python train_radar_models_minimal.py --samples 5000 --save-model
```

### ❌ "TerminatedWorkerError" / Memory Issues
```bash
# Use single-core processing with small dataset
python train_radar_models.py --quick --low-memory --n-jobs 1

# Or train with minimal script
python train_radar_models_minimal.py --samples 1000
```

## Training Modes

### 1. Standard Mode (Default)
- Full hyperparameter tuning
- Multiple models (XGBoost + Random Forest)
- Cross-validation
- Large datasets (1GB+)

```bash
python train_radar_models.py --full
```

### 2. Quick Mode
- Small dataset (0.1 GB)
- No cross-validation
- Fast training

```bash
python train_radar_models.py --quick
```

### 3. Low Memory Mode 💾
- Very small datasets (0.05 GB)
- Single-core processing
- Minimal hyperparameter tuning
- Memory monitoring

```bash
python train_radar_models.py --low-memory
```

### 4. Minimal Mode 🔬
- Ultra-lightweight (5,000 samples)
- Single model only
- Fixed hyperparameters
- Maximum compression

```bash
python train_radar_models_minimal.py --samples 5000 --save-model
```

## Memory Optimization Options

### Dataset Size Control
```bash
# Control dataset sizes
--main-size-gb 0.1          # Main dataset size (default: 1.0 GB)
--sea-state-size-gb 0.05    # Sea state dataset size (default: 0.3 GB)
```

### Parallel Processing Control
```bash
# Control CPU usage
--n-jobs 1                  # Single core (prevents memory overload)
--n-jobs 4                  # Use 4 cores
--n-jobs -1                 # Use all cores (default)
```

### Data Compression
```bash
# Choose compression method
--compression gzip          # Default compression
--compression brotli        # Maximum compression (smallest files)
--compression snappy        # Fastest compression
--compression lz4          # Balanced speed/size
```

### Chunked Processing
```bash
# Process data in smaller chunks
--chunk-size 10000          # Process 10,000 samples at a time
```

## Memory Usage Examples

### For Very Limited Systems (< 4GB RAM)
```bash
python train_radar_models_minimal.py --samples 1000
```

### For Limited Systems (4-8GB RAM)
```bash
python train_radar_models.py --low-memory --main-size-gb 0.02 --n-jobs 1
```

### For Moderate Systems (8-16GB RAM)
```bash
python train_radar_models.py --quick --n-jobs 2
```

### For Powerful Systems (16GB+ RAM)
```bash
python train_radar_models.py --full
```

## Disk Space Optimization

### Compression Comparison
| Method | Speed | Compression Ratio | Use Case |
|--------|-------|------------------|----------|
| gzip   | Medium| Good (default)   | General use |
| brotli | Slow  | Excellent        | Disk space limited |
| snappy | Fast  | Fair             | Speed priority |
| lz4    | Fast  | Good             | Balanced |

### File Size Estimates
| Dataset Size | Gzip | Brotli | Savings |
|-------------|------|--------|---------|
| 1 GB raw   | ~300 MB | ~200 MB | 80% |
| 100 MB raw | ~30 MB  | ~20 MB  | 80% |
| 10 MB raw  | ~3 MB   | ~2 MB   | 80% |

## Troubleshooting

### Memory Monitoring
The optimized scripts now show memory usage:
```
💾 Memory usage before loading: 0.15 GB
💾 Memory usage after loading: 2.34 GB
💾 Memory usage after cleanup: 1.89 GB
```

### Common Error Solutions

#### "OSError: No space left on device"
1. Use `--low-memory` mode
2. Try `--compression brotli`
3. Use `train_radar_models_minimal.py`
4. Free up disk space

#### "TerminatedWorkerError" 
1. Use `--n-jobs 1`
2. Use `--low-memory`
3. Reduce `--main-size-gb`
4. Use minimal script

#### High Memory Usage
1. Monitor with `htop` or `top`
2. Use `--chunk-size 5000`
3. Close other applications
4. Use swap if available

## Performance Comparison

### Hyperparameter Grid Sizes

| Mode | XGBoost Combinations | RF Combinations | Training Time |
|------|---------------------|-----------------|---------------|
| Full | 243 (3³×3²)        | 360 (3×4×3×3×3) | Hours |
| Low Memory | 8 (2×1×2×1×1)  | 16 (2×2×2×2×1)  | Minutes |
| Minimal | 0 (fixed params)   | 0 (fixed params)| Seconds |

### Resource Requirements

| Mode | RAM Usage | Disk Space | CPU Cores | Time |
|------|-----------|------------|-----------|------|
| Full | 8-16 GB   | 2-4 GB     | All       | 2-6 hours |
| Quick | 2-4 GB    | 200-400 MB | All       | 30-60 min |
| Low Memory | 1-2 GB | 50-100 MB | 1         | 10-30 min |
| Minimal | 0.5-1 GB | 10-20 MB  | 1         | 1-5 min |

## Best Practices

1. **Start Small**: Always try `--quick` or `--low-memory` first
2. **Monitor Resources**: Watch memory and disk usage during training
3. **Use Compression**: Always use compression for large datasets
4. **Single Core**: Use `--n-jobs 1` on memory-limited systems
5. **Clean Up**: Scripts automatically clean up memory in low-memory mode
6. **Progressive Scaling**: Start with minimal, then increase if resources allow

## Example Commands

### Ultra-Minimal Training
```bash
# Absolute minimum - works on almost any system
python train_radar_models_minimal.py --samples 1000 --save-model
```

### Production Training on Limited Hardware
```bash
# Best balance for limited systems
python train_radar_models.py --low-memory --compression brotli --n-jobs 1
```

### Development Testing
```bash
# Quick testing during development
python train_radar_models.py --quick --n-jobs 2
```

### Full Production Training
```bash
# Full pipeline for production systems
python train_radar_models.py --full --compression gzip
```

## File Outputs

### Standard Training
- `maritime_radar_dataset_main.parquet` - Main training dataset
- `radar_classifier_*.joblib` - Trained models
- `training_results.json` - Performance metrics
- `training_summary.txt` - Human-readable summary

### Minimal Training
- `minimal_radar_dataset.parquet` - Small dataset
- `minimal_radar_classifier.joblib` - Single trained model
- `minimal_training_results.json` - Performance metrics

Remember: The goal is to get a working model first, then optimize for better performance as resources allow!