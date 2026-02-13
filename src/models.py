"""
Multi-target regression models for soil property prediction.

This module implements various regression models for simultaneously predicting
five soil properties: Ca, P, pH, SOC, and Sand content.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib


class MultiTargetSoilPredictor:
    """Base class for multi-target soil property prediction."""
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'ridge', 'lasso')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.target_names = ['Ca', 'P', 'pH', 'SOC', 'Sand']
        
    def _create_model(self, **kwargs):
        """Create the appropriate model based on model_type."""
        if self.model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            min_samples_split = kwargs.get('min_samples_split', 2)
            n_jobs = kwargs.get('n_jobs', -1)
            random_state = kwargs.get('random_state', 42)
            
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                n_jobs=n_jobs,
                random_state=random_state
            )
        
        elif self.model_type == 'xgboost':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', 6)
            learning_rate = kwargs.get('learning_rate', 0.1)
            n_jobs = kwargs.get('n_jobs', -1)
            random_state = kwargs.get('random_state', 42)
            
            base_model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=n_jobs,
                random_state=random_state
            )
            return MultiOutputRegressor(base_model)
        
        elif self.model_type == 'ridge':
            alpha = kwargs.get('alpha', 1.0)
            return MultiOutputRegressor(Ridge(alpha=alpha))
        
        elif self.model_type == 'lasso':
            alpha = kwargs.get('alpha', 1.0)
            return MultiOutputRegressor(Lasso(alpha=alpha))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets (Ca, P, pH, SOC, Sand)
        """
        print(f"Training {self.model_type} model...")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        self.model.fit(X_train, y_train)
        print("Training completed.")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted target values
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_true: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'overall': {},
            'per_target': {}
        }
        
        # Overall metrics
        metrics['overall']['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['overall']['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['overall']['R2'] = r2_score(y_true, y_pred)
        
        # Per-target metrics
        for i, target_name in enumerate(self.target_names):
            if i < y_true.shape[1]:
                metrics['per_target'][target_name] = {
                    'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                    'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                    'R2': r2_score(y_true[:, i], y_pred[:, i])
                }
        
        return metrics
    
    def print_evaluation(self, metrics):
        """Print evaluation metrics in a readable format."""
        print("\n" + "="*60)
        print(f"Model Evaluation: {self.model_type}")
        print("="*60)
        
        print("\nOverall Metrics:")
        print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
        print(f"  MAE:  {metrics['overall']['MAE']:.4f}")
        print(f"  R2:   {metrics['overall']['R2']:.4f}")
        
        print("\nPer-Target Metrics:")
        for target_name, target_metrics in metrics['per_target'].items():
            print(f"\n  {target_name}:")
            print(f"    RMSE: {target_metrics['RMSE']:.4f}")
            print(f"    MAE:  {target_metrics['MAE']:.4f}")
            print(f"    R2:   {target_metrics['R2']:.4f}")
        
        print("\n" + "="*60)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, models=None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of MultiTargetSoilPredictor instances
        """
        self.models = models if models else []
        self.weights = None
    
    def add_model(self, model):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def train_all(self, X_train, y_train):
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(X_train, y_train)
    
    def predict(self, X, method='average'):
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            method: Ensemble method ('average' or 'weighted')
            
        Returns:
            Ensemble predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted' and self.weights is not None:
            weighted_preds = np.sum(predictions * self.weights[:, np.newaxis, np.newaxis], axis=0)
            return weighted_preds
        else:
            return np.mean(predictions, axis=0)
    
    def optimize_weights(self, X_val, y_val):
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        # Simple implementation: weight by inverse RMSE
        errors = []
        for model in self.models:
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            errors.append(rmse)
        
        errors = np.array(errors)
        self.weights = (1 / errors) / np.sum(1 / errors)
        
        print("Optimized ensemble weights:", self.weights)
