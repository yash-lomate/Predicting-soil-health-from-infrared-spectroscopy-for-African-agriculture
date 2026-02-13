"""
Simple test script to validate the soil property prediction implementation.
Tests basic functionality with synthetic data.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_preprocessing():
    """Test spectral preprocessing functions."""
    print("Testing preprocessing functions...")
    
    from preprocessing import (
        snv_correction,
        savitzky_golay_filter,
        preprocess_pipeline
    )
    
    # Create synthetic spectral data
    n_samples = 100
    n_features = 1000
    X = np.random.randn(n_samples, n_features) * 0.1 + 1.0
    
    # Test SNV
    X_snv = snv_correction(X)
    assert X_snv.shape == X.shape, "SNV output shape mismatch"
    assert np.allclose(np.mean(X_snv, axis=1), 0, atol=1e-10), "SNV mean not zero"
    assert np.allclose(np.std(X_snv, axis=1), 1, atol=1e-10), "SNV std not one"
    
    # Test Savitzky-Golay
    X_savgol = savitzky_golay_filter(X)
    assert X_savgol.shape == X.shape, "Savitzky-Golay output shape mismatch"
    
    # Test preprocessing pipeline
    X_prep = preprocess_pipeline(X, methods=['snv', 'savgol'])
    assert X_prep.shape == X.shape, "Pipeline output shape mismatch"
    
    print("✓ Preprocessing tests passed")


def test_data_utils():
    """Test data loading and processing utilities."""
    print("\nTesting data utilities...")
    
    from utils import standardize_targets, inverse_transform_targets
    
    # Create synthetic target data
    n_samples = 100
    n_targets = 5
    y = np.random.randn(n_samples, n_targets) * 100 + np.array([5000, 30, 6.5, 2.0, 50])
    
    # Test standardization
    y_scaled, scalers = standardize_targets(y)
    assert y_scaled.shape == y.shape, "Standardization output shape mismatch"
    assert len(scalers) == n_targets, "Wrong number of scalers"
    
    # Check that each target is standardized
    for i in range(n_targets):
        assert np.abs(np.mean(y_scaled[:, i])) < 1e-10, f"Target {i} mean not zero"
        assert np.abs(np.std(y_scaled[:, i]) - 1.0) < 1e-10, f"Target {i} std not one"
    
    # Test inverse transform
    y_inverse = inverse_transform_targets(y_scaled, scalers)
    assert np.allclose(y, y_inverse), "Inverse transform failed"
    
    print("✓ Data utilities tests passed")


def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    from utils import rmse, rpd, evaluate_predictions
    
    # Create synthetic predictions
    n_samples = 100
    n_targets = 5
    y_true = np.random.randn(n_samples, n_targets) * 10 + 50
    y_pred = y_true + np.random.randn(n_samples, n_targets) * 2  # Add noise
    
    # Test RMSE
    rmse_val = rmse(y_true[:, 0], y_pred[:, 0])
    assert rmse_val > 0, "RMSE should be positive"
    
    # Test RPD
    rpd_val = rpd(y_true[:, 0], y_pred[:, 0])
    assert rpd_val > 0, "RPD should be positive"
    
    # Test evaluate_predictions
    target_names = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    results = evaluate_predictions(y_true, y_pred, target_names)
    assert 'Mean' in results, "Results should contain Mean"
    assert all(name in results for name in target_names), "Missing target results"
    
    print("✓ Metrics tests passed")


def test_baseline_models():
    """Test baseline models."""
    print("\nTesting baseline models...")
    
    from models import RidgeRegressionBaseline, PLSBaseline, RandomForestPCA
    from sklearn.model_selection import train_test_split
    
    # Create synthetic data
    n_samples = 200
    n_features = 500
    n_targets = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_targets)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test Ridge Regression
    ridge = RidgeRegressionBaseline(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    assert y_pred_ridge.shape == y_test.shape, "Ridge prediction shape mismatch"
    
    # Test PLS
    pls = PLSBaseline(n_components=20)
    pls.fit(X_train, y_train)
    y_pred_pls = pls.predict(X_test)
    assert y_pred_pls.shape == y_test.shape, "PLS prediction shape mismatch"
    
    # Test Random Forest with PCA
    rf = RandomForestPCA(n_components=50, n_estimators=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    assert y_pred_rf.shape == y_test.shape, "RF prediction shape mismatch"
    
    print("✓ Baseline models tests passed")


def test_deep_learning_models():
    """Test deep learning models."""
    print("\nTesting deep learning models...")
    
    try:
        import torch
        from models import DeepLearningModel
        
        # Create synthetic data (smaller for quick testing)
        n_samples = 100
        n_features = 500
        n_targets = 5
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randn(n_samples, n_targets).astype(np.float32)
        X_val = np.random.randn(20, n_features).astype(np.float32)
        y_val = np.random.randn(20, n_targets).astype(np.float32)
        
        # Test 1D CNN (just a few epochs for testing)
        cnn = DeepLearningModel(
            model_type='conv1d',
            input_size=n_features,
            n_targets=n_targets,
            epochs=2,  # Very few epochs for testing
            batch_size=32
        )
        cnn.fit(X_train, y_train, X_val, y_val)
        y_pred_cnn = cnn.predict(X_val)
        assert y_pred_cnn.shape == y_val.shape, "CNN prediction shape mismatch"
        
        # Test Multi-Task Network
        mt = DeepLearningModel(
            model_type='multitask',
            input_size=n_features,
            n_targets=n_targets,
            epochs=2,  # Very few epochs for testing
            batch_size=32
        )
        mt.fit(X_train, y_train, X_val, y_val)
        y_pred_mt = mt.predict(X_val)
        assert y_pred_mt.shape == y_val.shape, "Multi-task prediction shape mismatch"
        
        print("✓ Deep learning models tests passed")
        
    except ImportError:
        print("⚠ PyTorch not available, skipping deep learning tests")


def main():
    """Run all tests."""
    print("="*80)
    print("Running Soil Property Prediction Tests")
    print("="*80)
    
    try:
        test_preprocessing()
        test_data_utils()
        test_metrics()
        test_baseline_models()
        test_deep_learning_models()
        
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
