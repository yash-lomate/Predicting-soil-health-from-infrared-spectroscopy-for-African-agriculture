"""
Preprocessing utilities for spectral data.
Includes Savitzky-Golay smoothing, SNV correction, and other spectral preprocessing methods.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def savitzky_golay_filter(spectra, window_length=11, polyorder=2, deriv=0):
    """
    Apply Savitzky-Golay filter to spectral data.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
    window_length : int, default=11
        Length of the filter window (must be odd)
    polyorder : int, default=2
        Order of the polynomial used to fit the samples
    deriv : int, default=0
        Order of the derivative to compute (0 means only smoothing)
        
    Returns:
    --------
    np.ndarray
        Smoothed spectral data
    """
    if spectra.ndim == 1:
        return savgol_filter(spectra, window_length, polyorder, deriv=deriv)
    
    # Apply to each sample
    smoothed = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        smoothed[i] = savgol_filter(spectra[i], window_length, polyorder, deriv=deriv)
    
    return smoothed


def snv_correction(spectra):
    """
    Apply Standard Normal Variate (SNV) correction to spectral data.
    
    SNV removes multiplicative interferences of scatter and particle size.
    Each spectrum is centered and scaled to unit variance.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
        
    Returns:
    --------
    np.ndarray
        SNV-corrected spectral data
    """
    if spectra.ndim == 1:
        mean = np.mean(spectra)
        std = np.std(spectra)
        return (spectra - mean) / (std + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Apply to each sample
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)
    
    return (spectra - means) / (stds + 1e-10)


def msc_correction(spectra, reference=None):
    """
    Apply Multiplicative Scatter Correction (MSC) to spectral data.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
    reference : np.ndarray, optional
        Reference spectrum. If None, uses mean of all spectra.
        
    Returns:
    --------
    np.ndarray
        MSC-corrected spectral data
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        # Fit linear regression
        fit = np.polyfit(reference, spectra[i], 1)
        # Correct spectrum
        corrected[i] = (spectra[i] - fit[1]) / fit[0]
    
    return corrected


def normalize_spectra(spectra, method='minmax'):
    """
    Normalize spectral data.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
    method : str, default='minmax'
        Normalization method: 'minmax', 'maxabs', or 'l2'
        
    Returns:
    --------
    np.ndarray
        Normalized spectral data
    """
    if method == 'minmax':
        min_vals = np.min(spectra, axis=1, keepdims=True)
        max_vals = np.max(spectra, axis=1, keepdims=True)
        return (spectra - min_vals) / (max_vals - min_vals + 1e-10)
    
    elif method == 'maxabs':
        max_abs = np.max(np.abs(spectra), axis=1, keepdims=True)
        return spectra / (max_abs + 1e-10)
    
    elif method == 'l2':
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        return spectra / (norms + 1e-10)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detrend_spectra(spectra, method='linear'):
    """
    Remove baseline trends from spectral data.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
    method : str, default='linear'
        Detrending method: 'linear' or 'constant'
        
    Returns:
    --------
    np.ndarray
        Detrended spectral data
    """
    from scipy.signal import detrend
    
    if spectra.ndim == 1:
        return detrend(spectra, type=method)
    
    detrended = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        detrended[i] = detrend(spectra[i], type=method)
    
    return detrended


def preprocess_pipeline(spectra, methods=['snv', 'savgol']):
    """
    Apply a pipeline of preprocessing methods to spectral data.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Spectral data of shape (n_samples, n_features)
    methods : list, default=['snv', 'savgol']
        List of preprocessing methods to apply in order.
        Options: 'snv', 'msc', 'savgol', 'normalize', 'detrend'
        
    Returns:
    --------
    np.ndarray
        Preprocessed spectral data
    """
    processed = spectra.copy()
    
    for method in methods:
        if method == 'snv':
            processed = snv_correction(processed)
        elif method == 'msc':
            processed = msc_correction(processed)
        elif method == 'savgol':
            processed = savitzky_golay_filter(processed)
        elif method == 'normalize':
            processed = normalize_spectra(processed)
        elif method == 'detrend':
            processed = detrend_spectra(processed)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    return processed
