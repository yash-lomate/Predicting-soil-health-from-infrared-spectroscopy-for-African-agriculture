# Data Directory

This directory should contain the dataset from the Africa Soil Property Prediction Challenge.

## Download the Data

1. Visit the Kaggle competition page:
   https://www.kaggle.com/c/afsis-soil-properties/data

2. Download the following files:
   - `train.csv` - Training dataset
   - `test.csv` - Test dataset (optional)

3. Place the files in this directory

## Data Format

### Training Data (train.csv)

Expected columns:
- `PIDN` - Sample identifier
- `m0` to `m3577` - 3,578 mid-infrared spectral measurements
- `Depth` - Soil depth (optional)
- `Latitude` - Spatial coordinate (optional)
- `Longitude` - Spatial coordinate (optional)
- `Ca` - Calcium content (target variable)
- `P` - Phosphorus content (target variable)
- `pH` - Soil pH (target variable)
- `SOC` - Soil Organic Carbon (target variable)
- `Sand` - Sand content (target variable)

### Test Data (test.csv)

Expected columns:
- `PIDN` - Sample identifier
- `m0` to `m3577` - 3,578 mid-infrared spectral measurements
- `Depth` - Soil depth (optional)
- `Latitude` - Spatial coordinate (optional)
- `Longitude` - Spatial coordinate (optional)

Note: Test data does not include target variables (Ca, P, pH, SOC, Sand).

## Example Structure

```
data/
├── README.md (this file)
├── train.csv
└── test.csv
```

## Data Statistics

- Number of samples: ~1,157 (training)
- Number of features: 3,578 spectral measurements + 3 spatial features
- Number of targets: 5 soil properties
- Data type: Continuous numerical values

## Citation

If you use this dataset, please cite:

Africa Soil Information Service (AfSIS)
Kaggle Competition: Africa Soil Property Prediction Challenge
https://www.kaggle.com/c/afsis-soil-properties
