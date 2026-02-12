"""
Script to generate synthetic sample data for testing and demonstration.

This script creates a small sample dataset that mimics the structure of the
Africa Soil Property Prediction Challenge data.
"""

import numpy as np
import pandas as pd
import os


def generate_sample_data(n_samples=100, n_features=3578, output_dir='data'):
    """
    Generate synthetic sample data.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of spectral features (default 3578)
        output_dir: Directory to save the sample data
    """
    print(f"Generating sample data with {n_samples} samples and {n_features} features...")
    
    np.random.seed(42)  # For reproducibility
    
    # Create spectral features (m0, m1, ..., m3577)
    # Simulate realistic spectral data with some structure
    base_spectrum = np.random.randn(n_features) * 0.5
    spectral_data = np.random.randn(n_samples, n_features) * 0.2 + base_spectrum
    
    # Add some absorption peaks (typical in IR spectroscopy)
    peak_positions = [500, 1000, 1500, 2000, 2500, 3000]
    for peak in peak_positions:
        if peak < n_features:
            # Create Gaussian peak
            x = np.arange(n_features)
            peak_shape = np.exp(-((x - peak) ** 2) / (2 * 100 ** 2))
            spectral_data += np.random.randn(n_samples, 1) * peak_shape * 2
    
    # Create DataFrame
    spectral_cols = [f'm{i}' for i in range(n_features)]
    df = pd.DataFrame(spectral_data, columns=spectral_cols)
    
    # Add sample IDs
    df.insert(0, 'PIDN', [f'sample_{i:04d}' for i in range(n_samples)])
    
    # Add spatial features
    df['Depth'] = np.random.uniform(0, 30, n_samples)
    df['Latitude'] = np.random.uniform(-35, 40, n_samples)
    df['Longitude'] = np.random.uniform(-20, 52, n_samples)
    
    # Create realistic target variables with correlations
    # Define feature ranges that adapt to available features
    chunk_size = max(1, n_features // 5)
    
    # Ca (Calcium) - typically 2-10 meq/100g
    ca_features = spectral_data[:, :min(chunk_size, n_features)]
    df['Ca'] = (
        2.5 * ca_features.mean(axis=1) +
        0.5 * df['Depth'] / 30 +
        np.random.randn(n_samples) * 1.5 + 5.0
    ).clip(0.5, 15)
    
    # P (Phosphorus) - typically 1-50 mg/kg
    p_features = spectral_data[:, min(chunk_size, n_features-1):min(2*chunk_size, n_features)]
    df['P'] = (
        15 * (p_features.mean(axis=1) if p_features.size > 0 else 0) +
        np.random.randn(n_samples) * 8 + 20
    ).clip(1, 100)
    
    # pH - typically 4.5-8.5
    ph_features = spectral_data[:, min(2*chunk_size, n_features-1):min(3*chunk_size, n_features)]
    df['pH'] = (
        0.8 * (ph_features.mean(axis=1) if ph_features.size > 0 else 0) +
        0.3 * df['Latitude'] / 40 +
        np.random.randn(n_samples) * 0.5 + 6.5
    ).clip(4.5, 8.5)
    
    # SOC (Soil Organic Carbon) - typically 0.1-5%
    soc_features = spectral_data[:, min(3*chunk_size, n_features-1):min(4*chunk_size, n_features)]
    df['SOC'] = (
        1.5 * (soc_features.mean(axis=1) if soc_features.size > 0 else 0) +
        0.1 * (30 - df['Depth']) / 30 +
        np.random.randn(n_samples) * 0.8 + 1.5
    ).clip(0.1, 8)
    
    # Sand - typically 0-100%
    sand_features = spectral_data[:, min(4*chunk_size, n_features-1):]
    df['Sand'] = (
        25 * (sand_features.mean(axis=1) if sand_features.size > 0 else 0) +
        5 * df['Latitude'] / 40 +
        np.random.randn(n_samples) * 15 + 50
    ).clip(0, 100)
    
    # Create train/test split
    train_size = int(0.7 * n_samples)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Remove targets from test set
    test_df = test_df.drop(columns=['Ca', 'P', 'pH', 'SOC', 'Sand'])
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'sample_train.csv')
    test_path = os.path.join(output_dir, 'sample_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Sample training data saved to: {train_path}")
    print(f"  - Shape: {train_df.shape}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Features: {len(spectral_cols)} spectral + 3 spatial")
    print(f"  - Targets: Ca, P, pH, SOC, Sand")
    
    print(f"\n✓ Sample test data saved to: {test_path}")
    print(f"  - Shape: {test_df.shape}")
    print(f"  - Samples: {len(test_df)}")
    
    print("\nTarget Statistics (Training Data):")
    print(train_df[['Ca', 'P', 'pH', 'SOC', 'Sand']].describe())
    
    print("\n" + "="*70)
    print("Sample data generated successfully!")
    print("="*70)
    print("\nYou can now train a model using:")
    print(f"  python train.py --train_data {train_path}")
    
    return train_df, test_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample soil property data')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--features', type=int, default=3578, help='Number of spectral features')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    generate_sample_data(
        n_samples=args.samples,
        n_features=args.features,
        output_dir=args.output
    )
