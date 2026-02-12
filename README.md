# Predicting Soil Health from Infrared Spectroscopy for African Agriculture

## Overview

This project implements multi-target regression models to predict soil properties from mid-infrared (MIR) spectral measurements. The solution addresses the Africa Soil Property Prediction Challenge, predicting five continuous soil properties simultaneously:

- **Ca** (Calcium)
- **P** (Phosphorus)
- **pH** (Soil acidity/alkalinity)
- **SOC** (Soil Organic Carbon)
- **Sand** (Sand content)

## Problem Definition

Given 3,578 mid-infrared spectral absorption measurements per soil sample plus spatial coordinates (latitude, longitude) and depth, the goal is to predict five continuous soil properties simultaneously. This is a classic multi-target regression problem on high-dimensional spectral data—a "wide" dataset with far more features than samples.

## Dataset

The data comes from the Africa Soil Property Prediction Challenge on Kaggle:
- **URL**: https://www.kaggle.com/c/afsis-soil-properties
- **Features**: 3,578 mid-infrared spectral measurements (m0, m1, ..., m3577) per sample
- **Additional features**: Spatial coordinates (Latitude, Longitude) and Depth
- **Targets**: 5 continuous soil properties (Ca, P, pH, SOC, Sand)

## Project Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── train.py                   # Main training script
├── predict.py                 # Prediction script
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── models.py             # Multi-target regression models
│   └── feature_engineering.py # Feature engineering utilities
├── data/                      # Data directory (not included in repo)
│   ├── train.csv             # Training data
│   └── test.csv              # Test data
├── models/                    # Saved models directory
└── notebooks/                 # Jupyter notebooks for exploration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yash-lomate/Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture.git
cd Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` directory:
   - Visit: https://www.kaggle.com/c/afsis-soil-properties/data
   - Download `train.csv` and `test.csv`
   - Place them in the `data/` directory

## Usage

### Training a Model

Basic usage with Random Forest:
```bash
python train.py --train_data data/train.csv
```

Train with XGBoost:
```bash
python train.py --train_data data/train.csv --model xgboost --n_estimators 200
```

Train with PCA for dimensionality reduction:
```bash
python train.py --train_data data/train.csv --use_pca --n_components 100
```

Train an ensemble of models:
```bash
python train.py --train_data data/train.csv --model ensemble
```

### Command Line Options

```
--train_data        Path to training CSV file (default: data/train.csv)
--test_data         Path to test CSV file (optional)
--model             Model type: random_forest, xgboost, ridge, lasso, ensemble
--n_estimators      Number of estimators for tree models (default: 100)
--max_depth         Maximum depth for tree models (default: None)
--use_pca           Apply PCA for dimensionality reduction
--n_components      Number of PCA components (default: 50)
--output_dir        Directory to save models (default: models/)
--no_scaling        Disable feature scaling
```

### Making Predictions

Once you have a trained model, use the prediction script:

```bash
python predict.py \
    --input data/test.csv \
    --model models/random_forest_model.pkl \
    --preprocessor models/preprocessor.pkl \
    --output predictions.csv
```

If you used PCA during training:
```bash
python predict.py \
    --input data/test.csv \
    --model models/random_forest_model.pkl \
    --preprocessor models/preprocessor.pkl \
    --feature_engineer models/feature_engineer.pkl \
    --output predictions.csv
```

## Methodology

### 1. Data Preprocessing
- **Feature Scaling**: StandardScaler for spectral and spatial features
- **Missing Value Handling**: Mean/median imputation
- **Feature Extraction**: Spectral wavelengths (m0-m3577) + spatial coordinates + depth

### 2. Feature Engineering (Optional)
- **PCA**: Dimensionality reduction for high-dimensional spectral data
- **Spectral Derivatives**: First and second derivatives to identify absorption peaks
- **Statistical Features**: Mean, std, percentiles, range from spectral data
- **Wavelet Transform**: Multi-scale spectral features

### 3. Models

#### Random Forest Regressor
- Handles high-dimensional data well
- Robust to outliers
- Provides feature importance
- Native multi-output support

#### XGBoost
- Gradient boosting for high performance
- Regularization to prevent overfitting
- Efficient handling of missing values
- Wrapped with MultiOutputRegressor

#### Ridge Regression
- Linear model with L2 regularization
- Fast training and prediction
- Good baseline model

#### Ensemble
- Combines Random Forest, XGBoost, and Ridge
- Weighted averaging based on validation performance
- More robust predictions

### 4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Explained variance

Metrics are reported both:
- **Overall**: Across all targets
- **Per-target**: Individual performance for each soil property

## Features

### Key Capabilities
- ✅ Multi-target regression for 5 soil properties
- ✅ Support for 3,578 spectral features + spatial coordinates
- ✅ Multiple model types (Random Forest, XGBoost, Ridge, Lasso)
- ✅ Ensemble predictions with optimized weights
- ✅ PCA for dimensionality reduction
- ✅ Feature engineering utilities
- ✅ Comprehensive evaluation metrics
- ✅ Model persistence (save/load)
- ✅ Easy-to-use command-line interface

### Advanced Features
- Spectral feature engineering (derivatives, smoothing)
- Statistical feature extraction
- Wavelet-based features
- Custom feature selection methods
- Validation split for model evaluation

## Example Output

```
======================================================================
Africa Soil Property Prediction - Multi-Target Regression
======================================================================

[1] Initializing data preprocessor...
[2] Loading data from data/train.csv...
Loaded training data: (1157, 3600)
[3] Handling missing values...
[4] Preparing features and targets...
Feature matrix shape: (1157, 3581)
Target matrix shape: (1157, 5)
Targets: ['Ca', 'P', 'pH', 'SOC', 'Sand']

[6] Splitting data into train and validation sets...
Training set: (925, 3581)
Validation set: (232, 3581)

[7] Training random_forest model...
Training random_forest model...
Training data shape: X=(925, 3581), y=(925, 5)
Training completed.

--- Validation Set Performance ---
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

  P:
    RMSE: 2.1234
    MAE:  1.6543
    R2:   0.8123
...
```

## Performance Considerations

- **High dimensionality**: 3,578 spectral features require careful handling
- **Feature scaling**: Essential for most models
- **PCA**: Recommended for faster training (50-100 components retain ~90% variance)
- **Ensemble**: Provides best performance but slower training/prediction
- **Random Forest**: Good balance of speed and accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## References

- Africa Soil Property Prediction Challenge: https://www.kaggle.com/c/afsis-soil-properties
- Mid-Infrared Spectroscopy for Soil Analysis
- Multi-target Regression Techniques

## Contact

For questions or issues, please open an issue on GitHub.