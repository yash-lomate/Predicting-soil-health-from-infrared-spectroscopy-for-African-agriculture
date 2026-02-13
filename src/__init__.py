"""
Africa Soil Property Prediction Package

This package provides tools for predicting soil properties from mid-infrared
spectral measurements. It includes:
- Data preprocessing and handling
- Multi-target regression models
- Feature engineering for spectral data
"""

__version__ = '1.0.0'

from .data_preprocessing import SoilDataPreprocessor, handle_missing_values
from .models import MultiTargetSoilPredictor, EnsemblePredictor
from .feature_engineering import SpectralFeatureEngineer, select_important_wavelengths

__all__ = [
    'SoilDataPreprocessor',
    'handle_missing_values',
    'MultiTargetSoilPredictor',
    'EnsemblePredictor',
    'SpectralFeatureEngineer',
    'select_important_wavelengths'
]
