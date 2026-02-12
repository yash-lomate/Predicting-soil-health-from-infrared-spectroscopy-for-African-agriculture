"""
Feature engineering utilities for spectral data analysis.

This module provides tools for extracting and engineering features
from mid-infrared spectral measurements.
"""

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA


class SpectralFeatureEngineer:
    """Feature engineering for spectral data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.pca = None
        self.n_components = None
    
    def apply_pca(self, X, n_components=50, fit=False):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Input features
            n_components: Number of principal components
            fit: Whether to fit PCA (True for training data)
            
        Returns:
            Transformed features
        """
        if fit:
            self.n_components = n_components
            self.pca = PCA(n_components=n_components)
            return self.pca.fit_transform(X)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            return self.pca.transform(X)
    
    def compute_derivatives(self, spectral_data):
        """
        Compute first and second derivatives of spectral data.
        
        Derivatives can help identify absorption peaks and troughs.
        
        Args:
            spectral_data: Spectral measurements (n_samples, n_features)
            
        Returns:
            Dictionary with original, first derivative, and second derivative.
            Note: First derivative has n_features-1 columns, second derivative 
            has n_features-2 columns due to the nature of differentiation.
        """
        first_derivative = np.diff(spectral_data, axis=1)
        second_derivative = np.diff(first_derivative, axis=1)
        
        return {
            'original': spectral_data,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative
        }
    
    def smooth_spectra(self, spectral_data, window_size=5):
        """
        Apply smoothing to spectral data using Savitzky-Golay filter.
        
        Args:
            spectral_data: Spectral measurements
            window_size: Size of smoothing window (must be odd)
            
        Returns:
            Smoothed spectral data
        """
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed = np.apply_along_axis(
            lambda x: signal.savgol_filter(x, window_size, 2),
            axis=1,
            arr=spectral_data
        )
        
        return smoothed
    
    def extract_statistical_features(self, spectral_data):
        """
        Extract statistical features from spectral data.
        
        Args:
            spectral_data: Spectral measurements
            
        Returns:
            Array of statistical features
        """
        features = []
        
        # Mean, std, min, max
        features.append(np.mean(spectral_data, axis=1))
        features.append(np.std(spectral_data, axis=1))
        features.append(np.min(spectral_data, axis=1))
        features.append(np.max(spectral_data, axis=1))
        
        # Percentiles
        features.append(np.percentile(spectral_data, 25, axis=1))
        features.append(np.percentile(spectral_data, 50, axis=1))
        features.append(np.percentile(spectral_data, 75, axis=1))
        
        # Range and IQR
        features.append(np.max(spectral_data, axis=1) - np.min(spectral_data, axis=1))
        features.append(np.percentile(spectral_data, 75, axis=1) - 
                       np.percentile(spectral_data, 25, axis=1))
        
        return np.column_stack(features)
    
    def create_wavelet_features(self, spectral_data, scales=None):
        """
        Create features using wavelet transform.
        
        Args:
            spectral_data: Spectral measurements
            scales: Scales for wavelet transform
            
        Returns:
            Wavelet-transformed features
        """
        if scales is None:
            scales = [2, 4, 8, 16, 32]
        
        from scipy.signal import cwt, ricker
        
        wavelet_features = []
        for i in range(spectral_data.shape[0]):
            coefficients = cwt(spectral_data[i], ricker, scales)
            # Use statistics of coefficients as features
            wavelet_features.append([
                np.mean(coefficients),
                np.std(coefficients),
                np.max(coefficients),
                np.min(coefficients)
            ])
        
        return np.array(wavelet_features)
    
    def get_explained_variance_ratio(self):
        """
        Get explained variance ratio from PCA.
        
        Returns:
            Array of explained variance ratios
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet.")
        
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance(self):
        """
        Get cumulative explained variance from PCA.
        
        Returns:
            Array of cumulative variance
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet.")
        
        return np.cumsum(self.pca.explained_variance_ratio_)


def select_important_wavelengths(X, y, n_select=100, method='variance'):
    """
    Select important wavelengths/features from spectral data.
    
    Args:
        X: Spectral features
        y: Target values
        n_select: Number of features to select
        method: Selection method ('variance', 'correlation')
        
    Returns:
        Indices of selected features
    """
    if method == 'variance':
        # Select features with highest variance
        variances = np.var(X, axis=0)
        selected_indices = np.argsort(variances)[-n_select:]
    
    elif method == 'correlation':
        # Select features with highest absolute correlation with targets
        correlations = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            # Average correlation across all targets
            corr_sum = 0
            for j in range(y.shape[1]):
                corr_sum += abs(np.corrcoef(X[:, i], y[:, j])[0, 1])
            correlations[i] = corr_sum / y.shape[1]
        
        selected_indices = np.argsort(correlations)[-n_select:]
    
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    return selected_indices
