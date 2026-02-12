# Predicting Soil Health from Infrared Spectroscopy for African Agriculture

A comprehensive machine learning toolkit for predicting soil properties from mid-infrared spectral data. This project addresses the Africa Soil Information Service (AfSIS) challenge to enable large-scale, low-cost soil mapping in developing regions.

## 🌍 Problem Overview

**Goal**: Predict five continuous soil properties from 3,578 mid-infrared spectral measurements:
- **Ca** (Calcium): 0-40,000 ppm
- **P** (Phosphorus): 0-200 ppm
- **pH**: 4-9
- **SOC** (Soil Organic Carbon): 0-10%
- **Sand**: 0-100%

**Significance**: 
- Conventional soil testing: $20-50 per sample, takes days
- Infrared spectroscopy: 30 seconds per sample, negligible cost
- Impact: Enables sustainable agriculture for 60% of Sub-Saharan Africa's farming population

**Challenge**: Multi-target regression on high-dimensional data (3,578 features, ~1,157 samples)

## 📊 Dataset

### Kaggle Competition Dataset
- **Training**: 1,157 samples with labels
- **Test**: 728 samples
- **Features**: 3,578 mid-infrared spectral absorption measurements
- **Source**: [Kaggle AfSIS Competition](https://www.kaggle.com/c/afsis-soil-properties)

### iSDA AfSIS Full Release (Optional)
- **Samples**: 50,000+ from 15 African countries
- **Source**: [iSDA Africa](https://www.isda-africa.com/)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yash-lomate/Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture.git
cd Predicting-soil-health-from-infrared-spectroscopy-for-African-agriculture

# Install dependencies
pip install -r requirements.txt
```

### Download Data

1. Download data from [Kaggle Competition](https://www.kaggle.com/c/afsis-soil-properties/data)
2. Create a `data/` directory and place CSV files there

### Training Models

```bash
# Train Ridge Regression baseline
python scripts/train.py --train_path data/training.csv --model ridge

# Train PLS baseline
python scripts/train.py --train_path data/training.csv --model pls

# Train Random Forest with PCA
python scripts/train.py --train_path data/training.csv --model rf

# Train 1D CNN
python scripts/train.py --train_path data/training.csv --model conv1d --epochs 100

# Train Multi-Task Network
python scripts/train.py --train_path data/training.csv --model multitask --epochs 100
```

### Using Jupyter Notebook

```bash
jupyter notebook notebooks/soil_property_prediction.ipynb
```

## 📁 Project Structure

```
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── spectral_preprocessing.py    # SNV, Savitzky-Golay, MSC, etc.
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py               # Data loading and splitting
│   │   └── metrics.py                   # RMSE, RPD, evaluation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py           # Ridge, PLS, Random Forest
│   │   └── deep_learning_models.py      # 1D CNN, Multi-Task Network
│   └── __init__.py
├── scripts/
│   └── train.py                         # Training script
├── notebooks/
│   └── soil_property_prediction.ipynb   # Example notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 Models Implemented

### Baseline Models

1. **Ridge Regression with CV**
   - Regularized linear regression
   - Cross-validation for hyperparameter tuning
   - Standard baseline for high-dimensional data

2. **Partial Least Squares (PLS)**
   - Dimensionality reduction + regression
   - Standard chemometrics baseline
   - Effective for spectral data

3. **Random Forest with PCA**
   - PCA for dimensionality reduction (50 components)
   - Ensemble of decision trees
   - Captures non-linear relationships

### Advanced Models

4. **1D Convolutional Neural Network**
   - Treats spectrum as 1D signal
   - Captures local absorption patterns
   - Multiple conv layers with batch normalization

5. **Multi-Task Neural Network**
   - Shared layers for all soil properties
   - Task-specific heads for each property
   - Leverages correlations between targets

## 📈 Evaluation Metrics

### RMSE (Root Mean Squared Error)
- Lower is better
- Kaggle metric: Mean Column-wise RMSE

### RPD (Residual Prediction Deviation)
RPD = SD(actual) / RMSE(predictions)

**Interpretation**:
- RPD < 1.4: Poor model
- 1.4 ≤ RPD < 2.0: Fair model
- 2.0 ≤ RPD < 3.0: Good model ✓
- RPD ≥ 3.0: Excellent model ✓✓

### R² (Coefficient of Determination)
- Proportion of variance explained
- Ranges from -∞ to 1 (1 is perfect)

## 🔧 Spectral Preprocessing

Critical preprocessing methods implemented:

1. **Standard Normal Variate (SNV)**
   - Removes multiplicative scatter effects
   - Centers and scales each spectrum

2. **Savitzky-Golay Filter**
   - Smooths spectra while preserving features
   - Reduces noise from instrumentation

3. **Multiplicative Scatter Correction (MSC)**
   - Corrects for light scattering
   - Normalizes to reference spectrum

4. **Detrending**
   - Removes baseline drift
   - Linear or constant detrending

5. **Normalization**
   - Min-max, max-abs, or L2 normalization
   - Scales spectral intensities

## 💡 Key Features

- ✅ **Spectral Preprocessing**: SNV, Savitzky-Golay, MSC, normalization
- ✅ **Regularization**: Essential for high-dimensional data
- ✅ **Multi-Target Regression**: Simultaneous prediction of 5 soil properties
- ✅ **Geographic Splitting**: Avoid data leakage by splitting on sites
- ✅ **Target Standardization**: Independent scaling for different ranges
- ✅ **Comprehensive Evaluation**: RMSE, RPD, R², MAE metrics
- ✅ **Multiple Model Types**: From simple baselines to deep learning
- ✅ **Modular Design**: Easy to extend and customize

## 📚 Usage Examples

### Load and Preprocess Data

```python
from src.preprocessing import preprocess_pipeline
from src.utils import load_afsis_data, standardize_targets

# Load data
X_train, y_train = load_afsis_data('data/training.csv')

# Preprocess spectral data
X_train_prep = preprocess_pipeline(X_train, methods=['snv', 'savgol'])

# Standardize targets
y_train_scaled, scalers = standardize_targets(y_train)
```

### Train a Model

```python
from src.models import RidgeRegressionCV

# Train Ridge Regression
model = RidgeRegressionCV(cv=5)
model.fit(X_train_prep, y_train_scaled)

# Make predictions
y_pred_scaled = model.predict(X_val_prep)
y_pred = inverse_transform_targets(y_pred_scaled, scalers)
```

### Evaluate Predictions

```python
from src.utils import evaluate_predictions, print_evaluation_results

# Evaluate
target_names = ['Ca', 'P', 'pH', 'SOC', 'Sand']
results = evaluate_predictions(y_val, y_pred, target_names)
print_evaluation_results(results)
```

## 🎯 Tips for Best Results

1. **Always preprocess spectra**: Use SNV + Savitzky-Golay as a starting point
2. **Use regularization**: Ridge, PLS, or dropout prevent overfitting
3. **Split by geography**: Avoid data leakage from nearby samples
4. **Standardize targets**: Properties have vastly different scales
5. **Start simple**: Test single-property prediction before multi-target
6. **Try ensembles**: Combine multiple models for better results
7. **Tune hyperparameters**: Use cross-validation for optimization

## 📖 References

### Dataset & Competition
- [AfSIS Soil Property Prediction Challenge - Kaggle](https://www.kaggle.com/c/afsis-soil-properties)
- [iSDA Africa - Full Dataset](https://www.isda-africa.com/)

### Background
- Africa Soil Information Service (AfSIS)
- Funded by Bill & Melinda Gates Foundation
- Supporting sustainable agriculture in Sub-Saharan Africa

### Related Work
- Chemometrics and spectroscopy in soil science
- Mid-infrared spectroscopy for soil analysis
- Machine learning for agricultural applications

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional preprocessing methods
- New model architectures
- Ensemble methods
- Hyperparameter optimization
- Visualization tools
- Documentation improvements

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Africa Soil Information Service (AfSIS)
- Bill & Melinda Gates Foundation
- Kaggle competition organizers
- Contributors to scikit-learn, PyTorch, and other libraries

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is an educational and research project. For production use in agricultural contexts, please validate models thoroughly with domain experts and local soil testing laboratories.