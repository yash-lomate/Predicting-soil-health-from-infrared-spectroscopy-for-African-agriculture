# Quick Start Guide

## Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yash-lomate/Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture.git
cd Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from Kaggle (optional - or use synthetic data)
# Visit: https://www.kaggle.com/c/afsis-soil-properties/data
# Place training.csv in data/ directory
```

## Quick Test (2 minutes)

```bash
# Generate synthetic data
python scripts/generate_synthetic_data.py --n_samples 500

# Run tests
python scripts/test_implementation.py

# Train a model
python scripts/train.py --train_path data/synthetic_training.csv --model ridge
```

## Common Commands

### Train Different Models

```bash
# Ridge Regression (fastest, good baseline)
python scripts/train.py --train_path data/training.csv --model ridge

# PLS (chemometrics standard)
python scripts/train.py --train_path data/training.csv --model pls

# Random Forest (handles non-linearity)
python scripts/train.py --train_path data/training.csv --model rf

# 1D CNN (requires PyTorch, best performance)
python scripts/train.py --train_path data/training.csv --model conv1d --epochs 100

# Multi-Task Network (requires PyTorch)
python scripts/train.py --train_path data/training.csv --model multitask --epochs 100
```

### Customize Training

```bash
# Change validation split
python scripts/train.py --train_path data/training.csv --model ridge --test_size 0.3

# Change preprocessing
python scripts/train.py --train_path data/training.csv --model pls --preprocessing snv

# Change output directory
python scripts/train.py --train_path data/training.csv --model rf --output_dir my_results

# Adjust deep learning settings
python scripts/train.py --train_path data/training.csv --model conv1d \
    --epochs 200 --batch_size 64
```

## Python API Usage

### Basic Example

```python
import sys
sys.path.insert(0, 'src')

from preprocessing import preprocess_pipeline
from utils import load_afsis_data, standardize_targets, evaluate_predictions
from models import RidgeRegressionCV

# Load data
X_train, y_train = load_afsis_data('data/training.csv')

# Split (in practice, use geographic splitting)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Preprocess
X_train = preprocess_pipeline(X_train, methods=['snv', 'savgol'])
X_val = preprocess_pipeline(X_val, methods=['snv', 'savgol'])

# Standardize targets
y_train_scaled, y_val_scaled, scalers = standardize_targets(y_train, y_val)

# Train model
model = RidgeRegressionCV(cv=5)
model.fit(X_train, y_train_scaled)

# Predict
y_pred_scaled = model.predict(X_val)

# Inverse transform
from utils import inverse_transform_targets
y_pred = inverse_transform_targets(y_pred_scaled, scalers)

# Evaluate
results = evaluate_predictions(y_val, y_pred, ['Ca', 'P', 'pH', 'SOC', 'Sand'])
print(results)
```

### Using Deep Learning

```python
from models import DeepLearningModel

# Create model
model = DeepLearningModel(
    model_type='conv1d',  # or 'multitask'
    input_size=3578,
    n_targets=5,
    epochs=100,
    batch_size=32
)

# Train
model.fit(X_train, y_train_scaled, X_val, y_val_scaled)

# Predict
y_pred_scaled = model.predict(X_val)

# Save model
model.save('models/my_cnn_model.pth')

# Load model
model.load('models/my_cnn_model.pth')
```

## Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/soil_property_prediction.ipynb
# Follow the interactive tutorial
```

## Common Issues

### PyTorch Not Available
```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or use baseline models only (Ridge, PLS, RF)
# They work without PyTorch
```

### Data File Not Found
```bash
# Create data directory
mkdir -p data

# Use synthetic data for testing
python scripts/generate_synthetic_data.py --output data/training.csv
```

### Out of Memory
```bash
# Reduce batch size for deep learning
python scripts/train.py --model conv1d --batch_size 16

# Use fewer PCA components for RF
# Edit baseline_models.py: n_components=20
```

## Understanding Results

### RPD (Residual Prediction Deviation)
- **RPD ≥ 3.0**: Excellent - Use for quantitative predictions
- **RPD ≥ 2.0**: Good - Suitable for screening
- **RPD ≥ 1.4**: Fair - Rough estimates only
- **RPD < 1.4**: Poor - Not reliable

### RMSE (Root Mean Squared Error)
- Lower is better
- Compare with standard deviation of targets
- RMSE < 0.5 × SD is generally good

### Example Output
```
Target          RMSE         MAE          R2           RPD         
================================================================================
Ca              3000.0000    2000.0000    0.8500       2.5000      # Good
P               15.0000      10.0000      0.7500       2.2000      # Good
pH              0.5000       0.3500       0.8000       2.6000      # Good
SOC             0.4000       0.3000       0.8200       2.8000      # Good
Sand            8.0000       6.0000       0.8500       2.5000      # Good
Mean            605.1000     403.3300     0.8100       2.5200      # Good
```

## File Locations

- **Results**: `results/` (CSV files, model predictions)
- **Models**: `results/*.pth` (PyTorch models)
- **Logs**: Console output
- **Data**: `data/` (training/test CSV files)

## Tips for Best Results

1. **Always preprocess**: SNV + Savitzky-Golay is a good default
2. **Start simple**: Ridge Regression is fast and interpretable
3. **Use geographic splitting**: Avoid data leakage from nearby samples
4. **Standardize targets**: Essential for multi-target regression
5. **Try ensembles**: Average predictions from multiple models
6. **Tune hyperparameters**: Use cross-validation

## Getting Help

- Check `README.md` for detailed documentation
- See `IMPLEMENTATION.md` for technical details
- Open an issue on GitHub for bugs
- Review example notebook for usage patterns

## Next Steps

1. ✓ Run quick test with synthetic data
2. ✓ Download real AfSIS data
3. ✓ Train baseline models
4. ✓ Evaluate results with RPD metric
5. ✓ Try advanced models if baseline RPD < 2.0
6. ✓ Experiment with preprocessing methods
7. ✓ Tune hyperparameters
8. ✓ Consider ensemble methods

Happy soil mapping! 🌍
