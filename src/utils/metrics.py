"""
Evaluation metrics for soil property prediction.
Includes RMSE, RPD (Residual Prediction Deviation), and other metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rpd(y_true, y_pred):
    """
    Calculate Residual Prediction Deviation (RPD).
    
    RPD = SD / RMSE
    where SD is the standard deviation of the true values.
    
    RPD interpretation:
    - RPD < 1.4: Poor model
    - 1.4 <= RPD < 2.0: Fair model
    - 2.0 <= RPD < 3.0: Good model
    - RPD >= 3.0: Excellent model
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        RPD value
    """
    sd = np.std(y_true)
    rmse_val = rmse(y_true, y_pred)
    
    return sd / (rmse_val + 1e-10)  # Add small epsilon to avoid division by zero


def mean_columnwise_rmse(y_true, y_pred):
    """
    Calculate Mean Column-wise RMSE (official Kaggle metric).
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values of shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values of shape (n_samples, n_targets)
        
    Returns:
    --------
    float
        Mean of RMSE values across all targets
    """
    if y_true.ndim == 1:
        return rmse(y_true, y_pred)
    
    n_targets = y_true.shape[1]
    rmse_values = []
    
    for i in range(n_targets):
        rmse_values.append(rmse(y_true[:, i], y_pred[:, i]))
    
    return np.mean(rmse_values)


def evaluate_predictions(y_true, y_pred, target_names=None):
    """
    Comprehensive evaluation of predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values of shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values of shape (n_samples, n_targets)
    target_names : list, optional
        Names of target variables
        
    Returns:
    --------
    dict
        Dictionary containing various metrics for each target
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_targets = y_true.shape[1]
    
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]
    
    results = {}
    
    for i, name in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        results[name] = {
            'RMSE': rmse(y_t, y_p),
            'MAE': mean_absolute_error(y_t, y_p),
            'R2': r2_score(y_t, y_p),
            'RPD': rpd(y_t, y_p)
        }
    
    # Calculate mean metrics
    results['Mean'] = {
        'RMSE': np.mean([results[name]['RMSE'] for name in target_names]),
        'MAE': np.mean([results[name]['MAE'] for name in target_names]),
        'R2': np.mean([results[name]['R2'] for name in target_names]),
        'RPD': np.mean([results[name]['RPD'] for name in target_names])
    }
    
    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted table.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from evaluate_predictions
    """
    print("\n" + "="*80)
    print(f"{'Target':<15} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'RPD':<12}")
    print("="*80)
    
    for target, metrics in results.items():
        print(f"{target:<15} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} "
              f"{metrics['R2']:<12.4f} {metrics['RPD']:<12.4f}")
    
    print("="*80)
    
    # Add RPD interpretation for mean
    if 'Mean' in results:
        mean_rpd = results['Mean']['RPD']
        if mean_rpd >= 3.0:
            interpretation = "Excellent"
        elif mean_rpd >= 2.0:
            interpretation = "Good"
        elif mean_rpd >= 1.4:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        print(f"\nOverall Model Performance (Mean RPD = {mean_rpd:.4f}): {interpretation}")
    print()
