"""
Base model interface for SAGE-Forge models.

Defines the standard API that all SAGE models must implement
for consistency, ensemble compatibility, and professional integration.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from sage_forge.core.config import get_config


@dataclass
class ModelMetrics:
    """Standard metrics for SAGE model performance tracking."""
    
    # Training metrics
    training_iterations: int = 0
    last_training_time: Optional[datetime] = None
    training_duration: float = 0.0
    
    # Prediction metrics  
    predictions_made: int = 0
    last_prediction_time: Optional[datetime] = None
    avg_prediction_time: float = 0.0
    
    # Performance metrics
    mse: float = 0.0
    mae: float = 0.0
    correlation: float = 0.0
    uncertainty_quality: float = 0.0
    
    # Model health
    is_healthy: bool = True
    last_error: Optional[str] = None
    error_count: int = 0


class BaseSAGEModel(ABC):
    """
    Base class for all SAGE models.
    
    Defines the standard API that ensures consistency across the model zoo
    and enables seamless integration with the ensemble framework.
    
    All SAGE models must implement:
    - fit(): Training interface
    - predict(): Prediction interface
    - update(): Online learning interface
    - get_feature_importance(): Feature analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base SAGE model.
        
        Parameters:
        -----------
        config : dict, optional
            Model-specific configuration parameters
        """
        self.config = config or {}
        self.sage_config = get_config()
        
        # Model state
        self.is_trained = False
        self.feature_dim = None
        self.feature_names = []
        
        # Performance tracking
        self.metrics = ModelMetrics()
        
        # Model metadata
        self._creation_time = datetime.now()
        self._model_id = self._generate_model_id()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string."""
        pass
    
    @property
    def model_id(self) -> str:
        """Unique model identifier."""
        return self._model_id
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseSAGEModel':
        """
        Train the model on provided data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        **kwargs : dict
            Additional training parameters
            
        Returns:
        --------
        self : BaseSAGEModel
            Trained model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, return_uncertainty: bool = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on provided data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates
            
        Returns:
        --------
        predictions : np.ndarray or tuple
            Predictions, optionally with uncertainty estimates
        """
        pass
    
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update of the model with new data.
        
        Parameters:
        -----------
        X : np.ndarray
            New feature data
        y : np.ndarray
            New target data
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        importance : dict
            Feature name to importance score mapping
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
        --------
        info : dict
            Model metadata and configuration
        """
        return {
            'model_id': self.model_id,
            'name': self.name,
            'version': self.version,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'feature_names': self.feature_names.copy(),
            'config': self.config.copy(),
            'metrics': self._get_metrics_dict(),
            'creation_time': self._creation_time.isoformat(),
            'class': self.__class__.__name__,
        }
    
    def get_metrics(self) -> ModelMetrics:
        """
        Get model performance metrics.
        
        Returns:
        --------
        metrics : ModelMetrics
            Current performance metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics to initial state."""
        self.metrics = ModelMetrics()
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data format and dimensions.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix to validate
        y : np.ndarray, optional
            Target values to validate
            
        Raises:
        -------
        ValueError
            If input validation fails
        """
        # Check X format
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        # Check feature dimension consistency
        if self.is_trained and self.feature_dim is not None:
            if X.shape[1] != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} features, got {X.shape[1]}")
        
        # Check y format if provided
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be numpy array")
            
            if y.ndim != 1:
                raise ValueError(f"y must be 1D array, got {y.ndim}D")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    def _update_training_metrics(self, duration: float) -> None:
        """Update training-related metrics."""
        self.metrics.training_iterations += 1
        self.metrics.last_training_time = datetime.now()
        self.metrics.training_duration = duration
    
    def _update_prediction_metrics(self, prediction_time: float, n_predictions: int) -> None:
        """Update prediction-related metrics."""
        self.metrics.predictions_made += n_predictions
        self.metrics.last_prediction_time = datetime.now()
        
        # Update average prediction time (exponential moving average)
        alpha = 0.1  # Learning rate for moving average
        if self.metrics.avg_prediction_time == 0:
            self.metrics.avg_prediction_time = prediction_time
        else:
            self.metrics.avg_prediction_time = (
                alpha * prediction_time + (1 - alpha) * self.metrics.avg_prediction_time
            )
    
    def _update_performance_metrics(self, mse: float, mae: float, correlation: float) -> None:
        """Update performance metrics."""
        self.metrics.mse = mse
        self.metrics.mae = mae
        self.metrics.correlation = correlation
        
        # Update model health based on performance
        self.metrics.is_healthy = (
            not np.isnan(mse) and 
            not np.isnan(mae) and 
            not np.isnan(correlation) and
            mse < 100.0  # Reasonable MSE threshold
        )
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and track model errors."""
        self.metrics.error_count += 1
        self.metrics.last_error = str(error)
        self.metrics.is_healthy = False
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'training_iterations': self.metrics.training_iterations,
            'last_training_time': self.metrics.last_training_time.isoformat() if self.metrics.last_training_time else None,
            'training_duration': self.metrics.training_duration,
            'predictions_made': self.metrics.predictions_made,
            'last_prediction_time': self.metrics.last_prediction_time.isoformat() if self.metrics.last_prediction_time else None,
            'avg_prediction_time': self.metrics.avg_prediction_time,
            'mse': self.metrics.mse,
            'mae': self.metrics.mae,
            'correlation': self.metrics.correlation,
            'uncertainty_quality': self.metrics.uncertainty_quality,
            'is_healthy': self.metrics.is_healthy,
            'last_error': self.metrics.last_error,
            'error_count': self.metrics.error_count,
        }
    
    def _generate_model_id(self) -> str:
        """Generate unique model identifier."""
        import hashlib
        
        id_string = f"{self.__class__.__name__}_{self._creation_time.isoformat()}_{id(self)}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def __str__(self) -> str:
        """String representation of model."""
        return f"{self.name}(id={self.model_id}, trained={self.is_trained})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"id='{self.model_id}', "
            f"trained={self.is_trained}, "
            f"feature_dim={self.feature_dim}"
            f")"
        )