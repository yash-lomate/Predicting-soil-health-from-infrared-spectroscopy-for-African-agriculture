"""
Prediction script for soil property prediction.

This script loads a trained model and makes predictions on new data.
"""

import os
import sys
import argparse
import pandas as pd
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions using trained soil property model'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data CSV file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--preprocessor',
        type=str,
        required=True,
        help='Path to preprocessor file'
    )
    
    parser.add_argument(
        '--feature_engineer',
        type=str,
        default=None,
        help='Path to feature engineer file (if PCA was used)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("="*70)
    print("Africa Soil Property Prediction - Making Predictions")
    print("="*70)
    
    # Load preprocessor
    print(f"\n[1] Loading preprocessor from {args.preprocessor}...")
    preprocessor = joblib.load(args.preprocessor)
    
    # Load feature engineer if provided
    feature_engineer = None
    if args.feature_engineer:
        print(f"[2] Loading feature engineer from {args.feature_engineer}...")
        feature_engineer = joblib.load(args.feature_engineer)
    
    # Load model
    print(f"\n[3] Loading model from {args.model}...")
    from src.models import MultiTargetSoilPredictor
    model = MultiTargetSoilPredictor()
    model.load_model(args.model)
    
    # Load input data
    print(f"\n[4] Loading input data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    # Preprocess features
    print("\n[5] Preprocessing features...")
    X = preprocessor.prepare_features(df, fit=False)
    print(f"Feature matrix shape: {X.shape}")
    
    # Apply feature engineering if used during training
    if feature_engineer:
        print("[6] Applying feature engineering...")
        X = feature_engineer.apply_pca(X, fit=False)
        print(f"Transformed feature matrix shape: {X.shape}")
    
    # Make predictions
    print("\n[7] Making predictions...")
    predictions = model.predict(X)
    print(f"Prediction matrix shape: {predictions.shape}")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(
        predictions,
        columns=preprocessor.target_columns
    )
    
    # Add sample ID if available
    if 'PIDN' in df.columns:
        predictions_df.insert(0, 'PIDN', df['PIDN'].values)
    
    # Save predictions
    print(f"\n[8] Saving predictions to {args.output}...")
    predictions_df.to_csv(args.output, index=False)
    
    # Display first few predictions
    print("\nFirst 5 predictions:")
    print(predictions_df.head())
    
    print("\n" + "="*70)
    print("Predictions completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
