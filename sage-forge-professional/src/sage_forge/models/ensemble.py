"""
SAGE Ensemble Framework - Meta-learning with dynamic model weighting.

Provides self-adaptive model combination that:
- Dynamically weights models based on performance
- Handles model uncertainty and confidence
- Provides ensemble-level predictions and uncertainty
- Supports online adaptation and rebalancing
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

from sage_forge.models.base import BaseSAGEModel, ModelMetrics
from sage_forge.core.config import get_config

from rich.console import Console
console = Console()


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble performance and model weights."""
    
    # Model weights and selection
    model_weights: Dict[str, float] = field(default_factory=dict)
    active_models: List[str] = field(default_factory=list)
    
    # Performance tracking
    ensemble_mse: float = 0.0
    ensemble_mae: float = 0.0
    ensemble_correlation: float = 0.0
    
    # Adaptation metrics
    weight_updates: int = 0
    last_rebalance: Optional[datetime] = None
    
    # Model health
    healthy_models: List[str] = field(default_factory=list)
    unhealthy_models: List[str] = field(default_factory=list)


class SAGEEnsemble(BaseSAGEModel):
    """
    SAGE Ensemble Framework for Meta-Learning.
    
    Combines multiple SAGE models with dynamic weighting based on:
    - Individual model performance
    - Prediction uncertainty and confidence
    - Recent performance trends
    - Model health and stability
    
    Features:
    - Adaptive model weighting
    - Uncertainty quantification
    - Online learning capabilities
    - Model health monitoring
    - Performance-based selection
    """
    
    def __init__(self, models: List[Union[str, BaseSAGEModel]], config: Optional[Dict[str, Any]] = None):
        """
        Initialize SAGE ensemble with models.
        
        Parameters:
        -----------
        models : list
            List of model names (str) or model instances (BaseSAGEModel)
        config : dict, optional
            Ensemble configuration parameters
        """
        super().__init__(config)
        
        console.print(f"[yellow]🧠 Initializing SAGE Ensemble with {len(models)} models...[/yellow]")
        
        # Ensemble configuration
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'adaptive')
        self.min_model_weight = self.config.get('min_weight', 0.05)  # 5% minimum
        self.max_model_weight = self.config.get('max_weight', 0.7)   # 70% maximum
        self.uncertainty_weighting = self.config.get('uncertainty_weighting', True)
        self.performance_window = self.config.get('performance_window', 100)
        
        # Initialize models
        self.models = {}
        self._initialize_models(models)
        
        # Ensemble state
        self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        self.model_performance_history = {name: [] for name in self.models.keys()}
        self.prediction_history = []
        
        # Ensemble metrics
        self.ensemble_metrics = EnsembleMetrics()
        self.ensemble_metrics.model_weights = self.model_weights.copy()
        self.ensemble_metrics.active_models = list(self.models.keys())
        
        console.print(f"[bold green]✅ SAGE Ensemble initialized with models: {list(self.models.keys())}[/bold green]")
    
    @property
    def name(self) -> str:
        """Ensemble name."""
        return "sage_ensemble"
    
    @property
    def version(self) -> str:
        """Ensemble version."""
        return "1.0.0"
    
    def _initialize_models(self, models: List[Union[str, BaseSAGEModel]]) -> None:
        """Initialize individual models in the ensemble."""
        for model in models:
            if isinstance(model, str):
                # Create model from string identifier
                model_instance = self._create_model_from_name(model)
                if model_instance is not None:
                    self.models[model] = model_instance
                else:
                    console.print(f"[yellow]⚠️ Skipping unknown model: {model}[/yellow]")
            elif isinstance(model, BaseSAGEModel):
                # Use provided model instance
                self.models[model.name] = model
            else:
                console.print(f"[yellow]⚠️ Skipping invalid model type: {type(model)}[/yellow]")
        
        if not self.models:
            raise ValueError("No valid models provided to ensemble")
    
    def _create_model_from_name(self, model_name: str) -> Optional[BaseSAGEModel]:
        """Create model instance from name string."""
        # This would create actual model instances
        # For now, return None to indicate model not available
        
        model_configs = {
            'alphaforge': {'type': 'feature_engineering', 'features': ['momentum', 'volatility']},
            'catch22': {'type': 'time_series', 'feature_set': 'comprehensive'},
            'tirex': {'type': 'forecasting', 'horizon': 1, 'uncertainty': True},
        }
        
        if model_name.lower() in model_configs:
            # TODO: Import and create actual model instances
            # For now, return None
            console.print(f"[dim yellow]Note: {model_name} model not yet implemented[/dim yellow]")
            return None
        
        return None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SAGEEnsemble':
        """
        Train all models in the ensemble.
        
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
        self : SAGEEnsemble
            Trained ensemble instance
        """
        console.print(f"[blue]🔧 Training SAGE Ensemble on {X.shape[0]} samples...[/blue]")
        
        # Validate input
        self.validate_input(X, y)
        
        # Store feature information
        self.feature_dim = X.shape[1] 
        self.feature_names = [f"feature_{i}" for i in range(self.feature_dim)]
        
        start_time = datetime.now()
        successful_models = []
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                console.print(f"[cyan]  • Training {model_name}...[/cyan]")
                model.fit(X, y, **kwargs)
                successful_models.append(model_name)
                
            except Exception as e:
                console.print(f"[red]  ❌ {model_name} training failed: {e}[/red]")
                # Remove failed model from active ensemble
                self.model_weights.pop(model_name, None)
        
        # Update ensemble state
        if successful_models:
            # Renormalize weights for successful models
            total_weight = sum(self.model_weights.get(name, 0) for name in successful_models)
            if total_weight > 0:
                for name in successful_models:
                    if name in self.model_weights:
                        self.model_weights[name] /= total_weight
            
            self.is_trained = True
            self.ensemble_metrics.active_models = successful_models
            
            training_duration = (datetime.now() - start_time).total_seconds()
            self._update_training_metrics(training_duration)
            
            console.print(f"[green]✅ Ensemble training completed with {len(successful_models)} models[/green]")
        else:
            raise RuntimeError("All models failed to train")
        
        return self
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make ensemble predictions with dynamic weighting.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates
            
        Returns:
        --------
        predictions : np.ndarray or tuple
            Ensemble predictions, optionally with uncertainty estimates
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before making predictions")
        
        self.validate_input(X)
        
        start_time = datetime.now()
        
        # Get predictions from all models
        model_predictions = {}
        model_uncertainties = {}
        
        for model_name, model in self.models.items():
            if model_name in self.ensemble_metrics.active_models:
                try:
                    if hasattr(model, 'predict'):
                        # Get prediction with uncertainty if supported
                        result = model.predict(X, return_uncertainty=True)
                        if isinstance(result, tuple):
                            pred, uncertainty = result
                        else:
                            pred = result
                            uncertainty = np.ones_like(pred) * 0.1  # Default uncertainty
                        
                        model_predictions[model_name] = pred
                        model_uncertainties[model_name] = uncertainty
                    else:
                        # Fallback for models without predict method
                        console.print(f"[yellow]⚠️ {model_name} has no predict method, skipping[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]⚠️ {model_name} prediction failed: {e}[/red]")
                    # Handle model failure in ensemble
                    self._handle_model_failure(model_name)
        
        if not model_predictions:
            raise RuntimeError("No models available for prediction")
        
        # Compute ensemble prediction using dynamic weights
        ensemble_pred, ensemble_uncertainty = self._compute_ensemble_prediction(
            model_predictions, model_uncertainties
        )
        
        # Update metrics
        prediction_time = (datetime.now() - start_time).total_seconds()
        self._update_prediction_metrics(prediction_time, X.shape[0])
        
        # Return appropriate format
        if return_uncertainty or (return_uncertainty is None and self.uncertainty_weighting):
            return ensemble_pred, ensemble_uncertainty
        else:
            return ensemble_pred
    
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update of ensemble with new data.
        
        Parameters:
        -----------
        X : np.ndarray
            New feature data
        y : np.ndarray
            New target data
        """
        if not self.is_trained:
            # Initial training if not trained yet
            self.fit(X, y)
            return
        
        self.validate_input(X, y)
        
        # Update individual models
        for model_name, model in self.models.items():
            if model_name in self.ensemble_metrics.active_models:
                try:
                    model.update(X, y)
                except Exception as e:
                    console.print(f"[red]⚠️ {model_name} update failed: {e}[/red]")
                    self._handle_model_failure(model_name)
        
        # Update ensemble weights based on recent performance
        self._update_model_weights(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get ensemble feature importance scores.
        
        Returns:
        --------
        importance : dict
            Weighted feature importance from all models
        """
        if not self.is_trained:
            return {}
        
        ensemble_importance = {}
        total_weight = 0
        
        # Aggregate importance from all active models
        for model_name, model in self.models.items():
            if model_name in self.ensemble_metrics.active_models:
                try:
                    model_importance = model.get_feature_importance()
                    model_weight = self.model_weights.get(model_name, 0)
                    
                    for feature, importance in model_importance.items():
                        if feature not in ensemble_importance:
                            ensemble_importance[feature] = 0
                        ensemble_importance[feature] += importance * model_weight
                    
                    total_weight += model_weight
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not get importance from {model_name}: {e}[/yellow]")
        
        # Normalize by total weight
        if total_weight > 0:
            for feature in ensemble_importance:
                ensemble_importance[feature] /= total_weight
        
        return ensemble_importance
    
    def _compute_ensemble_prediction(self, 
                                   model_predictions: Dict[str, np.ndarray],
                                   model_uncertainties: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted ensemble prediction and uncertainty."""
        
        if not model_predictions:
            raise ValueError("No model predictions available")
        
        # Get prediction shapes
        n_samples = next(iter(model_predictions.values())).shape[0]
        
        # Initialize ensemble arrays
        ensemble_pred = np.zeros(n_samples)
        ensemble_uncertainty = np.zeros(n_samples)
        total_weight = 0
        
        # Compute weighted predictions
        for model_name, predictions in model_predictions.items():
            model_weight = self.model_weights.get(model_name, 0)
            uncertainties = model_uncertainties.get(model_name, np.ones_like(predictions) * 0.1)
            
            if self.uncertainty_weighting:
                # Weight by inverse uncertainty (more certain predictions get higher weight)
                uncertainty_weight = 1.0 / (uncertainties + 1e-8)
                effective_weight = model_weight * uncertainty_weight
            else:
                effective_weight = model_weight
            
            ensemble_pred += predictions * effective_weight
            ensemble_uncertainty += uncertainties * effective_weight
            total_weight += effective_weight
        
        # Normalize by total weight
        if np.all(total_weight > 0):
            ensemble_pred /= total_weight
            ensemble_uncertainty /= total_weight
        
        return ensemble_pred, ensemble_uncertainty
    
    def _update_model_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update model weights based on recent performance."""
        
        # Get recent predictions from all models
        model_errors = {}
        
        for model_name, model in self.models.items():
            if model_name in self.ensemble_metrics.active_models:
                try:
                    predictions = model.predict(X)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]  # Extract predictions from tuple
                    
                    # Calculate error
                    mse = np.mean((predictions - y) ** 2)
                    model_errors[model_name] = mse
                    
                    # Update performance history
                    self.model_performance_history[model_name].append(mse)
                    
                    # Keep only recent history
                    if len(self.model_performance_history[model_name]) > self.performance_window:
                        self.model_performance_history[model_name] = \
                            self.model_performance_history[model_name][-self.performance_window:]
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not evaluate {model_name}: {e}[/yellow]")
        
        if model_errors:
            # Update weights based on inverse error (lower error = higher weight)
            total_inverse_error = 0
            inverse_errors = {}
            
            for model_name, error in model_errors.items():
                inverse_error = 1.0 / (error + 1e-8)  # Avoid division by zero
                inverse_errors[model_name] = inverse_error
                total_inverse_error += inverse_error
            
            # Update weights with bounds checking
            for model_name in inverse_errors:
                if total_inverse_error > 0:
                    new_weight = inverse_errors[model_name] / total_inverse_error
                    
                    # Apply weight bounds
                    new_weight = max(self.min_model_weight, 
                                   min(self.max_model_weight, new_weight))
                    
                    self.model_weights[model_name] = new_weight
            
            # Renormalize weights to sum to 1
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
            # Update ensemble metrics
            self.ensemble_metrics.model_weights = self.model_weights.copy()
            self.ensemble_metrics.weight_updates += 1
            self.ensemble_metrics.last_rebalance = datetime.now()
    
    def _handle_model_failure(self, model_name: str) -> None:
        """Handle failure of individual model."""
        if model_name in self.ensemble_metrics.active_models:
            self.ensemble_metrics.active_models.remove(model_name)
        
        if model_name in self.ensemble_metrics.healthy_models:
            self.ensemble_metrics.healthy_models.remove(model_name)
        
        if model_name not in self.ensemble_metrics.unhealthy_models:
            self.ensemble_metrics.unhealthy_models.append(model_name)
        
        # Remove weight for failed model
        self.model_weights.pop(model_name, None)
        
        # Renormalize remaining weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""
        base_info = self.get_model_info()
        
        ensemble_info = {
            **base_info,
            'ensemble_metrics': {
                'model_weights': self.ensemble_metrics.model_weights.copy(),
                'active_models': self.ensemble_metrics.active_models.copy(),
                'healthy_models': self.ensemble_metrics.healthy_models.copy(),
                'unhealthy_models': self.ensemble_metrics.unhealthy_models.copy(),
                'weight_updates': self.ensemble_metrics.weight_updates,
                'last_rebalance': self.ensemble_metrics.last_rebalance.isoformat() if self.ensemble_metrics.last_rebalance else None,
                'ensemble_mse': self.ensemble_metrics.ensemble_mse,
                'ensemble_mae': self.ensemble_metrics.ensemble_mae,
                'ensemble_correlation': self.ensemble_metrics.ensemble_correlation,
            },
            'individual_models': {
                name: model.get_model_info() for name, model in self.models.items()
            },
            'config': {
                'rebalance_frequency': self.rebalance_frequency,
                'min_model_weight': self.min_model_weight,
                'max_model_weight': self.max_model_weight,
                'uncertainty_weighting': self.uncertainty_weighting,
                'performance_window': self.performance_window,
            }
        }
        
        return ensemble_info


# Example usage and testing
if __name__ == "__main__":
    console.print("[bold green]🧠 SAGE Ensemble Framework[/bold green]")
    console.print("[dim]Meta-learning with dynamic model weighting[/dim]")
    
    # Basic functionality test
    try:
        # Create ensemble with model names (will show "not implemented" messages)
        ensemble = SAGEEnsemble(['alphaforge', 'catch22', 'tirex'])
        
        # Since no models are actually implemented yet, this will have empty models
        console.print(f"✅ Ensemble created with {len(ensemble.models)} available models")
        console.print("   Note: Individual model implementations coming soon")
        
    except Exception as e:
        console.print(f"❌ Ensemble test failed: {e}")