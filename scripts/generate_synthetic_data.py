"""
Generate synthetic data for testing and demonstration.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data(n_samples=1000, n_features=3578, n_targets=5, 
                           noise_level=0.1, save_path=None):
    """
    Generate synthetic spectral data for soil property prediction.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of spectral features
    n_targets : int
        Number of target properties
    noise_level : float
        Noise level to add to data
    save_path : str, optional
        Path to save CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with spectral features and target properties
    """
    np.random.seed(42)
    
    # Generate spectral features
    # Simulate realistic spectral patterns
    base_spectrum = np.zeros(n_features)
    
    # Add some absorption peaks
    peak_positions = [500, 1000, 1500, 2000, 2500, 3000]
    for pos in peak_positions:
        if pos < n_features:
            width = 50
            x = np.arange(n_features)
            peak = np.exp(-((x - pos) ** 2) / (2 * width ** 2))
            base_spectrum += peak * np.random.uniform(0.5, 1.5)
    
    # Add baseline
    base_spectrum += np.linspace(0.5, 1.5, n_features)
    
    # Generate samples with variations
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Add random variations
        X[i] = base_spectrum * np.random.uniform(0.8, 1.2) + \
               np.random.randn(n_features) * noise_level
    
    # Generate target properties
    # Ca (Calcium): 0-40,000 ppm
    Ca = np.random.lognormal(8, 1, n_samples)
    Ca = np.clip(Ca, 100, 40000)
    
    # P (Phosphorus): 0-200 ppm
    P = np.random.lognormal(3, 0.8, n_samples)
    P = np.clip(P, 1, 200)
    
    # pH: 4-9
    pH = np.random.normal(6.5, 1.0, n_samples)
    pH = np.clip(pH, 4, 9)
    
    # SOC (Soil Organic Carbon): 0-10%
    SOC = np.random.lognormal(0.5, 0.6, n_samples)
    SOC = np.clip(SOC, 0.1, 10)
    
    # Sand: 0-100%
    Sand = np.random.normal(50, 20, n_samples)
    Sand = np.clip(Sand, 0, 100)
    
    # Create DataFrame
    columns = [f'm{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Add targets
    df['Ca'] = Ca
    df['P'] = P
    df['pH'] = pH
    df['SOC'] = SOC
    df['Sand'] = Sand
    
    # Add sample ID
    df.insert(0, 'PIDN', [f'Sample_{i:04d}' for i in range(n_samples)])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Synthetic data saved to {save_path}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic soil spectral data')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/synthetic_training.csv',
                       help='Output CSV path')
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} synthetic samples...")
    df = generate_synthetic_data(n_samples=args.n_samples, save_path=args.output)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few columns:")
    print(df.iloc[:, :5].head())
    print(f"\nTarget statistics:")
    print(df[['Ca', 'P', 'pH', 'SOC', 'Sand']].describe())
