"""
Data preprocessing utilities for Africa Soil Property Prediction Challenge.

This module handles loading and preprocessing of mid-infrared spectral data
along with spatial coordinates and depth information.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SoilDataPreprocessor:
    """Preprocessor for soil spectral data."""
    
    def __init__(self, spectral_scaler=True, spatial_scaler=True):
        """
        Initialize the preprocessor.
        
        Args:
            spectral_scaler: Whether to apply standard scaling to spectral features
            spatial_scaler: Whether to apply standard scaling to spatial features
        """
        self.spectral_scaler = StandardScaler() if spectral_scaler else None
        self.spatial_scaler = StandardScaler() if spatial_scaler else None
        self.target_columns = ['Ca', 'P', 'pH', 'SOC', 'Sand']
        self.spatial_columns = ['Latitude', 'Longitude', 'Depth']
        
    def load_data(self, train_path, test_path=None):
        """
        Load training and optional test data.
        
        Args:
            train_path: Path to training CSV file
            test_path: Optional path to test CSV file
            
        Returns:
            train_df, test_df (if test_path provided) or train_df only
        """
        train_df = pd.read_csv(train_path)
        
        if test_path:
            test_df = pd.read_csv(test_path)
            return train_df, test_df
        
        return train_df
    
    def identify_feature_columns(self, df):
        """
        Identify spectral feature columns (typically named m0, m1, m2, ..., m3577).
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of spectral column names
        """
        # Get all columns that start with 'm' followed by digits
        spectral_cols = [col for col in df.columns if col.startswith('m') and col[1:].isdigit()]
        return sorted(spectral_cols, key=lambda x: int(x[1:]))
    
    def prepare_features(self, df, fit=False):
        """
        Prepare features by separating spectral and spatial data.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers (True for training data)
            
        Returns:
            Dictionary with 'spectral' and 'spatial' feature arrays
        """
        spectral_cols = self.identify_feature_columns(df)
        
        # Extract spectral features (mid-infrared measurements)
        X_spectral = df[spectral_cols].values
        
        # Extract spatial features if available
        available_spatial = [col for col in self.spatial_columns if col in df.columns]
        X_spatial = df[available_spatial].values if available_spatial else None
        
        # Apply scaling
        if self.spectral_scaler is not None:
            if fit:
                X_spectral = self.spectral_scaler.fit_transform(X_spectral)
            else:
                X_spectral = self.spectral_scaler.transform(X_spectral)
        
        if X_spatial is not None and self.spatial_scaler is not None:
            if fit:
                X_spatial = self.spatial_scaler.fit_transform(X_spatial)
            else:
                X_spatial = self.spatial_scaler.transform(X_spatial)
        
        # Combine features
        if X_spatial is not None:
            X_combined = np.hstack([X_spectral, X_spatial])
        else:
            X_combined = X_spectral
        
        return X_combined
    
    def prepare_targets(self, df):
        """
        Extract target variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of target values (Ca, P, pH, SOC, Sand)
        """
        available_targets = [col for col in self.target_columns if col in df.columns]
        
        if not available_targets:
            return None
        
        return df[available_targets].values
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and validation sets.
        
        Args:
            X: Feature matrix
            y: Target matrix
            test_size: Proportion of data for validation
            random_state: Random seed
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
