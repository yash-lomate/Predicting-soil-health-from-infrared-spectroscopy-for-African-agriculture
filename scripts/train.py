"""
Training script for soil property prediction models.

This script demonstrates how to train baseline and advanced models
for predicting soil properties from infrared spectroscopy data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import preprocess_pipeline
from utils import (
    load_afsis_data,
    standardize_targets,
    evaluate_predictions,
    print_evaluation_results
)
from models import (
    RidgeRegressionCV,
    PLSBaseline,
    RandomForestPCA,
    DeepLearningModel
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train soil property prediction models'
    )
    parser.add_argument(
        '--train_path',
        type=str,
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='Path to test data CSV (optional)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ridge',
        choices=['ridge', 'pls', 'rf', 'conv1d', 'multitask'],
        help='Model type to train'
    )
    parser.add_argument(
        '--preprocessing',
        type=str,
        nargs='+',
        default=['snv', 'savgol'],
        help='Preprocessing methods to apply'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Validation set size'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs for deep learning models'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for deep learning models'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Soil Property Prediction from Infrared Spectroscopy")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    X, y = load_afsis_data(args.train_path)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Preprocess spectral data
    print(f"\n3. Preprocessing spectral data with methods: {args.preprocessing}")
    X_train_prep = preprocess_pipeline(X_train, methods=args.preprocessing)
    X_val_prep = preprocess_pipeline(X_val, methods=args.preprocessing)
    
    # Standardize targets
    print("\n4. Standardizing targets...")
    y_train_scaled, y_val_scaled, scalers = standardize_targets(y_train, y_val)
    
    # Train model
    print(f"\n5. Training {args.model.upper()} model...")
    
    if args.model == 'ridge':
        model = RidgeRegressionCV(cv=5)
        model.fit(X_train_prep, y_train_scaled)
        y_pred_scaled = model.predict(X_val_prep)
        
    elif args.model == 'pls':
        model = PLSBaseline(n_components=50)
        model.fit(X_train_prep, y_train_scaled)
        y_pred_scaled = model.predict(X_val_prep)
        
    elif args.model == 'rf':
        model = RandomForestPCA(n_components=50, n_estimators=100)
        model.fit(X_train_prep, y_train_scaled)
        y_pred_scaled = model.predict(X_val_prep)
        
    elif args.model in ['conv1d', 'multitask']:
        model = DeepLearningModel(
            model_type=args.model,
            input_size=X_train_prep.shape[1],
            n_targets=y_train_scaled.shape[1],
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        model.fit(X_train_prep, y_train_scaled, X_val_prep, y_val_scaled)
        y_pred_scaled = model.predict(X_val_prep)
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{args.model}_model.pth')
        model.save(model_path)
        print(f"   Model saved to {model_path}")
    
    # Inverse transform predictions
    print("\n6. Inverse transforming predictions...")
    from utils import inverse_transform_targets
    y_pred = inverse_transform_targets(y_pred_scaled, scalers)
    
    # Evaluate
    print("\n7. Evaluation Results:")
    target_names = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    results = evaluate_predictions(y_val, y_pred, target_names)
    print_evaluation_results(results)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(args.output_dir, f'{args.model}_results.csv')
    results_df.to_csv(results_path)
    print(f"\nResults saved to {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame(y_pred, columns=target_names)
    pred_path = os.path.join(args.output_dir, f'{args.model}_predictions.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == '__main__':
    main()
