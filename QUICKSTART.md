# Quick Start Guide

This guide will help you get started with the Soil Property Prediction system.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yash-lomate/Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture.git
cd Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 2. Get the Data

1. Visit the Kaggle competition: https://www.kaggle.com/c/afsis-soil-properties/data
2. Download `train.csv` and `test.csv`
3. Place them in the `data/` directory

## 3. Train Your First Model

### Basic Training
```bash
python train.py --train_data data/train.csv
```

This will:
- Load and preprocess the data
- Train a Random Forest model (default)
- Evaluate on a validation set
- Save the model to `models/random_forest_model.pkl`

### Advanced Training Options

**Train with XGBoost:**
```bash
python train.py --train_data data/train.csv --model xgboost --n_estimators 200
```

**Use PCA for faster training:**
```bash
python train.py --train_data data/train.csv --use_pca --n_components 100
```

**Train an ensemble:**
```bash
python train.py --train_data data/train.csv --model ensemble
```

## 4. Make Predictions

```bash
python predict.py \
    --input data/test.csv \
    --model models/random_forest_model.pkl \
    --preprocessor models/preprocessor.pkl \
    --output predictions.csv
```

## 5. Explore with Jupyter

```bash
jupyter notebook notebooks/example_usage.ipynb
```

## Expected Output

After training, you'll see:
- Overall metrics (RMSE, MAE, R²) across all targets
- Per-target metrics for each soil property (Ca, P, pH, SOC, Sand)
- Saved model files in `models/` directory

Example output:
```
============================================================
Model Evaluation: random_forest
============================================================

Overall Metrics:
  RMSE: 2.4532
  MAE:  1.8234
  R2:   0.7845

Per-Target Metrics:

  Ca:
    RMSE: 3.2145
    MAE:  2.4532
    R2:   0.7234
...
```

## Tips

1. **Start Simple**: Use Random Forest with default parameters first
2. **Use PCA**: If training is slow, enable PCA with 50-100 components
3. **Ensemble**: For best results, use ensemble mode (slower but more accurate)
4. **Scaling**: Feature scaling is enabled by default (recommended)

## Troubleshooting

**Data not found?**
- Make sure `train.csv` is in the `data/` directory
- Check file permissions

**Out of memory?**
- Use PCA: `--use_pca --n_components 50`
- Reduce `n_estimators` for tree-based models

**Poor performance?**
- Try different models: `--model xgboost` or `--model ensemble`
- Increase `n_estimators`: `--n_estimators 200`

## Next Steps

- Experiment with different models and hyperparameters
- Try feature engineering techniques in `src/feature_engineering.py`
- Explore the Jupyter notebook for visualizations
- Submit predictions to Kaggle!

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review code in `src/` directory
- Open an issue on GitHub
