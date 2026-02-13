"""
Data loading and splitting utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def load_afsis_data(train_path, test_path=None):
    """
    Load AFSIS soil property prediction data from CSV files.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    test_path : str, optional
        Path to test data CSV file
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) or (X_train, y_train) if test_path is None
    """
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Target columns
    target_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    
    # Identify spectral columns (typically m followed by wavenumber)
    spectral_cols = [col for col in train_df.columns if col.startswith('m')]
    
    # Extract features and targets
    X_train = train_df[spectral_cols].values
    
    # Check if targets exist in training data
    available_targets = [col for col in target_cols if col in train_df.columns]
    
    if available_targets:
        y_train = train_df[available_targets].values
    else:
        y_train = None
    
    # Load test data if provided
    if test_path:
        test_df = pd.read_csv(test_path)
        X_test = test_df[spectral_cols].values
        
        # Check if targets exist in test data
        if all(col in test_df.columns for col in available_targets):
            y_test = test_df[available_targets].values
        else:
            y_test = None
        
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train


def geographic_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Split data based on geographic groups to avoid data leakage.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target matrix
    groups : np.ndarray
        Group labels (e.g., site IDs)
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, y_train, y_val)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    return X_train, X_val, y_train, y_val


def standardize_targets(y_train, y_val=None):
    """
    Standardize target variables independently.
    
    Parameters:
    -----------
    y_train : np.ndarray
        Training targets of shape (n_samples, n_targets)
    y_val : np.ndarray, optional
        Validation targets of shape (n_samples, n_targets)
        
    Returns:
    --------
    tuple
        (y_train_scaled, y_val_scaled, scalers) if y_val provided
        (y_train_scaled, scalers) otherwise
    """
    n_targets = y_train.shape[1] if y_train.ndim > 1 else 1
    
    if n_targets == 1:
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        if y_val is not None:
            y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).ravel()
            return y_train_scaled, y_val_scaled, [scaler]
        
        return y_train_scaled, [scaler]
    
    # Multiple targets - standardize each independently
    scalers = []
    y_train_scaled = np.zeros_like(y_train, dtype=np.float64)
    
    for i in range(n_targets):
        scaler = StandardScaler()
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).ravel()
        scalers.append(scaler)
    
    if y_val is not None:
        y_val_scaled = np.zeros_like(y_val, dtype=np.float64)
        for i in range(n_targets):
            y_val_scaled[:, i] = scalers[i].transform(y_val[:, i].reshape(-1, 1)).ravel()
        
        return y_train_scaled, y_val_scaled, scalers
    
    return y_train_scaled, scalers


def inverse_transform_targets(y_scaled, scalers):
    """
    Inverse transform standardized targets back to original scale.
    
    Parameters:
    -----------
    y_scaled : np.ndarray
        Scaled targets
    scalers : list
        List of StandardScaler objects
        
    Returns:
    --------
    np.ndarray
        Targets in original scale
    """
    if len(scalers) == 1:
        return scalers[0].inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    y_original = np.zeros_like(y_scaled, dtype=np.float64)
    for i, scaler in enumerate(scalers):
        y_original[:, i] = scaler.inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
    
    return y_original
