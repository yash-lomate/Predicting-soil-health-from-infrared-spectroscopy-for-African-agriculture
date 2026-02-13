"""
Main training script for Africa Soil Property Prediction Challenge.

This script provides a complete pipeline for training multi-target regression models
to predict soil properties from mid-infrared spectral data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import SoilDataPreprocessor, handle_missing_values
from src.models import MultiTargetSoilPredictor, EnsemblePredictor
from src.feature_engineering import SpectralFeatureEngineer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train soil property prediction models'
    )
    
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/train.csv',
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        default=None,
        help='Path to test data CSV file (optional)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'xgboost', 'ridge', 'lasso', 'ensemble'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of estimators for tree-based models'
    )
    
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Maximum depth for tree-based models'
    )
    
    parser.add_argument(
        '--use_pca',
        action='store_true',
        help='Apply PCA for dimensionality reduction'
    )
    
    parser.add_argument(
        '--n_components',
        type=int,
        default=50,
        help='Number of PCA components'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--no_scaling',
        action='store_true',
        help='Disable feature scaling'
    )
    
    return parser.parse_args()


def train_single_model(X_train, y_train, X_val, y_val, args):
    """
    Train a single model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        args: Command line arguments
        
    Returns:
        Trained model
    """
    # Initialize model
    model = MultiTargetSoilPredictor(
        model_type=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate on training and validation sets
    print("\n--- Training Set Performance ---")
    train_metrics = model.evaluate(X_train, y_train)
    model.print_evaluation(train_metrics)
    
    print("\n--- Validation Set Performance ---")
    val_metrics = model.evaluate(X_val, y_val)
    model.print_evaluation(val_metrics)
    
    return model


def train_ensemble(X_train, y_train, X_val, y_val, args):
    """
    Train an ensemble of models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        args: Command line arguments
        
    Returns:
        Trained ensemble
    """
    print("Training ensemble of models...")
    
    ensemble = EnsemblePredictor()
    
    # Add Random Forest
    rf_model = MultiTargetSoilPredictor(
        model_type='random_forest',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    ensemble.add_model(rf_model)
    
    # Add XGBoost
    xgb_model = MultiTargetSoilPredictor(
        model_type='xgboost',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth else 6
    )
    ensemble.add_model(xgb_model)
    
    # Add Ridge
    ridge_model = MultiTargetSoilPredictor(
        model_type='ridge',
        alpha=1.0
    )
    ensemble.add_model(ridge_model)
    
    # Train all models
    ensemble.train_all(X_train, y_train)
    
    # Optimize ensemble weights
    ensemble.optimize_weights(X_val, y_val)
    
    return ensemble


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*70)
    print("Africa Soil Property Prediction - Multi-Target Regression")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize preprocessor
    print("\n[1] Initializing data preprocessor...")
    preprocessor = SoilDataPreprocessor(
        spectral_scaler=not args.no_scaling,
        spatial_scaler=not args.no_scaling
    )
    
    # Load data
    print(f"\n[2] Loading data from {args.train_data}...")
    if not os.path.exists(args.train_data):
        print(f"ERROR: Training data file not found: {args.train_data}")
        print("\nExpected data format:")
        print("  - CSV file with columns: PIDN, m0, m1, ..., m3577, Ca, P, pH, SOC, Sand")
        print("  - Optional: Depth, Latitude, Longitude")
        print("\nTo download the data:")
        print("  Visit: https://www.kaggle.com/c/afsis-soil-properties/data")
        return
    
    train_df = preprocessor.load_data(args.train_data)
    print(f"Loaded training data: {train_df.shape}")
    
    # Handle missing values
    print("\n[3] Handling missing values...")
    train_df = handle_missing_values(train_df, strategy='mean')
    
    # Prepare features and targets
    print("\n[4] Preparing features and targets...")
    X = preprocessor.prepare_features(train_df, fit=True)
    y = preprocessor.prepare_targets(train_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Targets: {preprocessor.target_columns}")
    
    # Apply PCA if requested
    if args.use_pca:
        print(f"\n[5] Applying PCA with {args.n_components} components...")
        feature_engineer = SpectralFeatureEngineer()
        X = feature_engineer.apply_pca(X, n_components=args.n_components, fit=True)
        print(f"Reduced feature matrix shape: {X.shape}")
        
        # Print explained variance
        variance_ratio = feature_engineer.get_explained_variance_ratio()
        cumulative_variance = feature_engineer.get_cumulative_variance()
        print(f"Explained variance ratio (first 10): {variance_ratio[:10]}")
        print(f"Cumulative variance: {cumulative_variance[-1]:.4f}")
    
    # Split data
    print("\n[6] Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train model
    print(f"\n[7] Training {args.model} model...")
    
    if args.model == 'ensemble':
        model = train_ensemble(X_train, y_train, X_val, y_val, args)
    else:
        model = train_single_model(X_train, y_train, X_val, y_val, args)
    
    # Save model
    print(f"\n[8] Saving model to {args.output_dir}...")
    model_path = os.path.join(args.output_dir, f'{args.model}_model.pkl')
    model.save_model(model_path)
    
    # Save preprocessor
    import joblib
    preprocessor_path = os.path.join(args.output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    if args.use_pca:
        pca_path = os.path.join(args.output_dir, 'feature_engineer.pkl')
        joblib.dump(feature_engineer, pca_path)
        print(f"Feature engineer saved to {pca_path}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    
    # Process test data if provided
    if args.test_data and os.path.exists(args.test_data):
        print(f"\n[9] Processing test data from {args.test_data}...")
        test_df = preprocessor.load_data(args.test_data)
        X_test = preprocessor.prepare_features(test_df, fit=False)
        
        if args.use_pca:
            X_test = feature_engineer.apply_pca(X_test, fit=False)
        
        print("Making predictions on test data...")
        if args.model == 'ensemble':
            y_pred = model.predict(X_test, method='weighted')
        else:
            y_pred = model.predict(X_test)
        
        # Save predictions
        predictions_df = pd.DataFrame(
            y_pred,
            columns=preprocessor.target_columns
        )
        
        # Add sample ID if available
        if 'PIDN' in test_df.columns:
            predictions_df.insert(0, 'PIDN', test_df['PIDN'].values)
        
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    main()
