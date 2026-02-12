"""
Test script to validate the soil property prediction implementation.

This script creates synthetic data and tests the complete pipeline.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import SoilDataPreprocessor, handle_missing_values
from src.models import MultiTargetSoilPredictor, EnsemblePredictor
from src.feature_engineering import SpectralFeatureEngineer


def create_synthetic_data(n_samples=100, n_features=3578, n_targets=5):
    """
    Create synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of spectral features (default 3578)
        n_targets: Number of target variables (default 5)
        
    Returns:
        DataFrame with synthetic data
    """
    print(f"Creating synthetic data: {n_samples} samples, {n_features} features...")
    
    # Create spectral features (m0, m1, ..., m3577)
    spectral_data = np.random.randn(n_samples, n_features)
    spectral_cols = [f'm{i}' for i in range(n_features)]
    
    df = pd.DataFrame(spectral_data, columns=spectral_cols)
    
    # Add spatial features
    df['Depth'] = np.random.uniform(0, 30, n_samples)
    df['Latitude'] = np.random.uniform(-35, 40, n_samples)
    df['Longitude'] = np.random.uniform(-20, 52, n_samples)
    
    # Add sample ID
    df['PIDN'] = [f'sample_{i}' for i in range(n_samples)]
    
    # Create synthetic targets (based on some features for realism)
    df['Ca'] = 2.5 * spectral_data[:, :10].mean(axis=1) + np.random.randn(n_samples) * 0.5 + 5
    df['P'] = 1.8 * spectral_data[:, 10:20].mean(axis=1) + np.random.randn(n_samples) * 0.3 + 3
    df['pH'] = 0.5 * spectral_data[:, 20:30].mean(axis=1) + np.random.randn(n_samples) * 0.2 + 6.5
    df['SOC'] = 1.2 * spectral_data[:, 30:40].mean(axis=1) + np.random.randn(n_samples) * 0.4 + 2
    df['Sand'] = 15 * spectral_data[:, 40:50].mean(axis=1) + np.random.randn(n_samples) * 5 + 50
    
    return df


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\n" + "="*70)
    print("TEST 1: Data Preprocessing")
    print("="*70)
    
    # Create synthetic data
    df = create_synthetic_data(n_samples=100, n_features=100)
    
    # Initialize preprocessor
    preprocessor = SoilDataPreprocessor()
    
    # Test feature preparation
    X = preprocessor.prepare_features(df, fit=True)
    y = preprocessor.prepare_targets(df)
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target matrix shape: {y.shape}")
    
    assert X.shape[0] == 100, "Feature matrix should have 100 samples"
    assert X.shape[1] == 103, "Feature matrix should have 100 spectral + 3 spatial features"
    assert y.shape[0] == 100, "Target matrix should have 100 samples"
    assert y.shape[1] == 5, "Target matrix should have 5 targets"
    
    print("✓ Data preprocessing test PASSED")
    return preprocessor, X, y


def test_model_training(X, y):
    """Test model training functionality."""
    print("\n" + "="*70)
    print("TEST 2: Model Training")
    print("="*70)
    
    # Split data
    n_train = 80
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Test Random Forest
    print("\nTesting Random Forest...")
    rf_model = MultiTargetSoilPredictor(
        model_type='random_forest',
        n_estimators=10,
        max_depth=5
    )
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_val, y_val)
    print(f"✓ Random Forest R² score: {rf_metrics['overall']['R2']:.4f}")
    
    # Test predictions
    y_pred = rf_model.predict(X_val)
    assert y_pred.shape == y_val.shape, "Prediction shape should match target shape"
    print(f"✓ Prediction shape: {y_pred.shape}")
    
    # Test Ridge
    print("\nTesting Ridge Regression...")
    ridge_model = MultiTargetSoilPredictor(
        model_type='ridge',
        alpha=1.0
    )
    ridge_model.train(X_train, y_train)
    ridge_metrics = ridge_model.evaluate(X_val, y_val)
    print(f"✓ Ridge R² score: {ridge_metrics['overall']['R2']:.4f}")
    
    print("\n✓ Model training test PASSED")
    return rf_model


def test_feature_engineering(X):
    """Test feature engineering functionality."""
    print("\n" + "="*70)
    print("TEST 3: Feature Engineering")
    print("="*70)
    
    feature_engineer = SpectralFeatureEngineer()
    
    # Test PCA
    print("\nTesting PCA...")
    X_pca = feature_engineer.apply_pca(X, n_components=20, fit=True)
    print(f"✓ Original shape: {X.shape}")
    print(f"✓ PCA shape: {X_pca.shape}")
    
    assert X_pca.shape[0] == X.shape[0], "PCA should preserve number of samples"
    assert X_pca.shape[1] == 20, "PCA should reduce to 20 components"
    
    # Test explained variance
    variance_ratio = feature_engineer.get_explained_variance_ratio()
    cumulative_variance = feature_engineer.get_cumulative_variance()
    print(f"✓ Explained variance ratio sum: {variance_ratio.sum():.4f}")
    print(f"✓ Cumulative variance: {cumulative_variance[-1]:.4f}")
    
    print("\n✓ Feature engineering test PASSED")


def test_ensemble():
    """Test ensemble functionality."""
    print("\n" + "="*70)
    print("TEST 4: Ensemble Model")
    print("="*70)
    
    # Create small dataset
    df = create_synthetic_data(n_samples=60, n_features=50)
    preprocessor = SoilDataPreprocessor()
    X = preprocessor.prepare_features(df, fit=True)
    y = preprocessor.prepare_targets(df)
    
    # Split data
    n_train = 40
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble = EnsemblePredictor()
    
    # Add models
    rf_model = MultiTargetSoilPredictor(model_type='random_forest', n_estimators=10)
    ridge_model = MultiTargetSoilPredictor(model_type='ridge', alpha=1.0)
    
    ensemble.add_model(rf_model)
    ensemble.add_model(ridge_model)
    
    print(f"✓ Ensemble has {len(ensemble.models)} models")
    
    # Train ensemble
    ensemble.train_all(X_train, y_train)
    
    # Make predictions
    y_pred = ensemble.predict(X_val, method='average')
    print(f"✓ Ensemble prediction shape: {y_pred.shape}")
    
    assert y_pred.shape == y_val.shape, "Ensemble prediction shape should match target shape"
    
    print("\n✓ Ensemble test PASSED")


def test_model_persistence():
    """Test model save/load functionality."""
    print("\n" + "="*70)
    print("TEST 5: Model Persistence")
    print("="*70)
    
    import tempfile
    import joblib
    
    # Create and train a simple model
    df = create_synthetic_data(n_samples=50, n_features=50)
    preprocessor = SoilDataPreprocessor()
    X = preprocessor.prepare_features(df, fit=True)
    y = preprocessor.prepare_targets(df)
    
    model = MultiTargetSoilPredictor(model_type='ridge')
    model.train(X, y)
    
    # Make predictions before saving
    y_pred_before = model.predict(X[:10])
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    
    model.save_model(tmp_path)
    print(f"✓ Model saved to {tmp_path}")
    
    # Load model
    loaded_model = MultiTargetSoilPredictor(model_type='ridge')
    loaded_model.load_model(tmp_path)
    print(f"✓ Model loaded from {tmp_path}")
    
    # Make predictions after loading
    y_pred_after = loaded_model.predict(X[:10])
    
    # Check if predictions are the same
    assert np.allclose(y_pred_before, y_pred_after), "Predictions should be the same after loading"
    print(f"✓ Predictions match after save/load")
    
    # Clean up
    os.remove(tmp_path)
    
    print("\n✓ Model persistence test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# Running All Tests for Soil Property Prediction")
    print("#"*70)
    
    try:
        # Test 1: Data preprocessing
        preprocessor, X, y = test_data_preprocessing()
        
        # Test 2: Model training
        model = test_model_training(X, y)
        
        # Test 3: Feature engineering
        test_feature_engineering(X)
        
        # Test 4: Ensemble
        test_ensemble()
        
        # Test 5: Model persistence
        test_model_persistence()
        
        # Summary
        print("\n" + "#"*70)
        print("# ALL TESTS PASSED ✓")
        print("#"*70)
        print("\nThe implementation is working correctly!")
        print("You can now use the system with real data from Kaggle.")
        
        return True
        
    except Exception as e:
        print("\n" + "#"*70)
        print("# TEST FAILED ✗")
        print("#"*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
