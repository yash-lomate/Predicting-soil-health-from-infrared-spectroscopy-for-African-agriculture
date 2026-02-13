"""
Model comparison script for comparing different models and configurations.

This script trains multiple models with different configurations and compares
their performance to help select the best model for your data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import SoilDataPreprocessor, handle_missing_values
from src.models import MultiTargetSoilPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare different models for soil property prediction'
    )
    
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/train.csv',
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='comparison_results',
        help='Directory to save comparison results'
    )
    
    return parser.parse_args()


def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, **kwargs):
    """
    Train and evaluate a single model.
    
    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional model parameters
        
    Returns:
        Dictionary with model name and metrics
    """
    print(f"\nTraining {model_type}...")
    
    model = MultiTargetSoilPredictor(model_type=model_type, **kwargs)
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    
    return {
        'model_type': model_type,
        'config': kwargs,
        'metrics': metrics,
        'model': model
    }


def main():
    """Main comparison function."""
    args = parse_args()
    
    print("="*70)
    print("Model Comparison for Soil Property Prediction")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = SoilDataPreprocessor()
    train_df = preprocessor.load_data(args.train_data)
    train_df = handle_missing_values(train_df, strategy='mean')
    
    X = preprocessor.prepare_features(train_df, fit=True)
    y = preprocessor.prepare_targets(train_df)
    
    # Split data
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y, test_size=0.2)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Define models to compare
    models_config = [
        {
            'model_type': 'random_forest',
            'n_estimators': 50,
            'max_depth': None,
            'name': 'Random Forest (50 trees)'
        },
        {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': None,
            'name': 'Random Forest (100 trees)'
        },
        {
            'model_type': 'xgboost',
            'n_estimators': 50,
            'max_depth': 6,
            'name': 'XGBoost (50 trees)'
        },
        {
            'model_type': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'name': 'XGBoost (100 trees)'
        },
        {
            'model_type': 'ridge',
            'alpha': 1.0,
            'name': 'Ridge Regression (α=1.0)'
        },
        {
            'model_type': 'ridge',
            'alpha': 10.0,
            'name': 'Ridge Regression (α=10.0)'
        },
    ]
    
    # Train and evaluate all models
    results = []
    for config in models_config:
        model_name = config.pop('name')
        result = train_and_evaluate_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **config
        )
        result['name'] = model_name
        results.append(result)
        
        # Print summary
        metrics = result['metrics']
        print(f"  Overall RMSE: {metrics['overall']['RMSE']:.4f}")
        print(f"  Overall R²:   {metrics['overall']['R2']:.4f}")
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results:
        metrics = result['metrics']
        row = {
            'Model': result['name'],
            'Overall_RMSE': metrics['overall']['RMSE'],
            'Overall_MAE': metrics['overall']['MAE'],
            'Overall_R2': metrics['overall']['R2']
        }
        
        # Add per-target metrics
        for target, target_metrics in metrics['per_target'].items():
            row[f'{target}_RMSE'] = target_metrics['RMSE']
            row[f'{target}_R2'] = target_metrics['R2']
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    csv_path = os.path.join(args.output_dir, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("Model Comparison Results")
    print("="*70)
    print("\nOverall Metrics:")
    print(comparison_df[['Model', 'Overall_RMSE', 'Overall_MAE', 'Overall_R2']].to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Overall RMSE comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall RMSE
    ax = axes[0, 0]
    models = comparison_df['Model']
    rmse = comparison_df['Overall_RMSE']
    ax.barh(models, rmse)
    ax.set_xlabel('RMSE')
    ax.set_title('Overall RMSE Comparison')
    ax.invert_yaxis()
    
    # Overall R²
    ax = axes[0, 1]
    r2 = comparison_df['Overall_R2']
    ax.barh(models, r2)
    ax.set_xlabel('R² Score')
    ax.set_title('Overall R² Comparison')
    ax.invert_yaxis()
    
    # Per-target RMSE
    ax = axes[1, 0]
    target_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    rmse_cols = [f'{t}_RMSE' for t in target_cols]
    comparison_df[rmse_cols].plot(kind='bar', ax=ax, rot=45)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('Per-Target RMSE Comparison')
    ax.legend(target_cols, loc='best')
    
    # Per-target R²
    ax = axes[1, 1]
    r2_cols = [f'{t}_R2' for t in target_cols]
    comparison_df[r2_cols].plot(kind='bar', ax=ax, rot=45)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Per-Target R² Comparison')
    ax.legend(target_cols, loc='best')
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    
    # Find best model
    best_idx = comparison_df['Overall_R2'].idxmax()
    best_model = comparison_df.iloc[best_idx]
    
    print("\n" + "="*70)
    print("Best Model (by R²):")
    print("="*70)
    print(f"Model: {best_model['Model']}")
    print(f"Overall RMSE: {best_model['Overall_RMSE']:.4f}")
    print(f"Overall MAE:  {best_model['Overall_MAE']:.4f}")
    print(f"Overall R²:   {best_model['Overall_R2']:.4f}")
    
    print("\n" + "="*70)
    print("Comparison completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
