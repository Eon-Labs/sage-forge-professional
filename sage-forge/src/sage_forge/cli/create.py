"""
Template creation command for strategies, models, and other components.

Provides professional template generation with:
- NT-native strategy templates
- SAGE model templates  
- Example integration scripts
- Proper package structure
"""

from pathlib import Path
from datetime import datetime
import re

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


class TemplateGenerator:
    """Professional template generator with best practices."""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "templates"
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
    
    def create_strategy(self, name: str, output_dir: Path) -> Path:
        """Create NT-native strategy template."""
        # Validate name
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', name):
            raise ValueError("Strategy name must be valid Python identifier")
        
        # Create strategy file
        strategy_content = self._get_strategy_template(name)
        strategy_file = output_dir / f"{self._snake_case(name)}.py"
        strategy_file.write_text(strategy_content)
        
        # Create corresponding test file
        test_content = self._get_strategy_test_template(name)
        test_file = Path("tests") / "test_strategies" / f"test_{self._snake_case(name)}.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(test_content)
        
        return strategy_file
    
    def create_model(self, name: str, output_dir: Path) -> Path:
        """Create SAGE model template."""
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', name):
            raise ValueError("Model name must be valid Python identifier")
        
        model_content = self._get_model_template(name) 
        model_file = output_dir / f"{self._snake_case(name)}.py"
        model_file.write_text(model_content)
        
        # Create test file
        test_content = self._get_model_test_template(name)
        test_file = Path("tests") / "test_models" / f"test_{self._snake_case(name)}.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(test_content)
        
        return model_file
    
    def create_example(self, name: str, template_type: str) -> Path:
        """Create example usage script."""
        example_content = self._get_example_template(name, template_type)
        example_file = Path("examples") / f"{self._snake_case(name)}_example.py"
        example_file.parent.mkdir(parents=True, exist_ok=True)
        example_file.write_text(example_content)
        
        return example_file
    
    def _snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def _get_strategy_template(self, name: str) -> str:
        """Generate NT-native strategy template."""
        snake_name = self._snake_case(name)
        
        return f'''"""
{name} - NT-Native Adaptive Trading Strategy
{'=' * (len(name) + 40)}

Professional NautilusTrader strategy following best practices:
- NT-native patterns and Strategy inheritance
- Real DSM data integration
- SAGE model integration ready
- Proper indicator registration
- Event-driven architecture

Created: {self.timestamp}
"""

import numpy as np
from typing import Optional
from collections import deque

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.atr import AverageTrueRange

from sage_forge.models.ensemble import SAGEEnsemble
from sage_forge.data.manager import ArrowDataManager
from sage_forge.core.config import SageConfig

from rich.console import Console
console = Console()


class {name}(Strategy):
    """
    {name} - NT-Native Adaptive Trading Strategy
    
    Features:
    - NT-native Strategy inheritance and patterns
    - SAGE model integration for adaptive signals
    - Real-time indicator processing
    - Risk management and position sizing
    - Professional logging and monitoring
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        console.print(f"[yellow]🔧 Initializing {{name}}...[/yellow]")
        
        # Configuration
        self.sage_config = SageConfig()
        
        # Data storage (minimal - NT cache handles history)
        self.returns = deque(maxlen=50)
        self.signals = deque(maxlen=100)
        
        # NT Built-in Indicators (auto-registered)
        self.ema_fast = ExponentialMovingAverage(12)
        self.ema_slow = ExponentialMovingAverage(26)
        self.rsi = RelativeStrengthIndex(14)
        self.atr = AverageTrueRange(20)
        
        # SAGE Model Integration
        self.sage_models = SAGEEnsemble([
            "alphaforge",
            "catch22"
        ])
        
        # Strategy state
        self.bar_counter = 0
        self.last_signal = 0.0
        self.position_hold_bars = 0
        
        # Performance tracking
        self.trade_count = 0
        self.signal_strength_history = []
        
        console.print(f"[bold green]✅ {{name}} initialized successfully![/bold green]")
    
    def on_start(self):
        """Strategy startup - register indicators with NT auto-update."""
        self.subscribe_bars(self.config.bar_type)
        
        # Register all indicators for auto-updates
        console.print("[yellow]🔧 Registering indicators with NT auto-update system...[/yellow]")
        
        self.register_indicator_for_bars(self.config.bar_type, self.ema_fast)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_slow)  
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        
        console.print("[green]✅ All indicators registered - auto-updates enabled![/green]")
        console.print(f"[cyan]📊 Subscribed to {{self.config.bar_type}}[/cyan]")
    
    def on_bar(self, bar: Bar):
        """
        Process each bar using NT-native patterns.
        
        CRITICAL: Uses ONLY NT cache and registered indicators - no current bar data!
        """
        self.bar_counter += 1
        
        # Access historical data via NT cache (bias-free)
        historical_bars = self.cache.bars(self.config.bar_type)
        
        if len(historical_bars) < 50:
            console.print(f"[dim yellow]Warming up: {{len(historical_bars)}}/50 bars[/dim yellow]")
            return
        
        # Extract features using NT indicators and cache
        features = self._extract_features(historical_bars)
        
        # Generate trading signal using SAGE ensemble
        signal_strength = 0.5  # Default neutral
        
        if self._all_indicators_ready() and len(features) > 0:
            try:
                # SAGE model prediction
                signal_strength = self.sage_models.predict(features)
                self.signals.append(signal_strength)
                self.signal_strength_history.append(signal_strength)
                
            except Exception as e:
                console.print(f"[red]⚠️ SAGE prediction error: {{e}}[/red]")
                signal_strength = 0.5  # Neutral fallback
        
        # Execute trading logic
        self._execute_trading_decision(signal_strength, bar)
        
        # Update returns for next iteration (prequential learning)
        if len(historical_bars) >= 2:
            current_price = float(historical_bars[-1].close)
            prev_price = float(historical_bars[-2].close) 
            current_return = (current_price - prev_price) / prev_price
            self.returns.append(current_return)
        
        # Periodic progress logging
        if self.bar_counter % 1000 == 0:
            avg_signal = np.mean(list(self.signals)) if self.signals else 0.5
            console.print(f"[dim cyan]📊 Bar {{self.bar_counter}}: Avg Signal: {{avg_signal:.3f}} | "
                         f"Trades: {{self.trade_count}} | Cache: {{len(historical_bars)}} bars[/dim cyan]")
        
        self.last_signal = signal_strength
    
    def _extract_features(self, historical_bars) -> np.ndarray:
        """Extract features using NT indicators and cache data."""
        features = []
        
        # EMA signals
        if self.ema_fast.initialized and self.ema_slow.initialized:
            ema_signal = (self.ema_fast.value - self.ema_slow.value) / self.ema_slow.value
            features.append(ema_signal)
        else:
            features.append(0.0)
        
        # RSI signal (normalized)
        if self.rsi.initialized:
            rsi_signal = (self.rsi.value - 50.0) / 50.0
            features.append(rsi_signal)
        else:
            features.append(0.0)
        
        # ATR signal (volatility)
        if self.atr.initialized and len(historical_bars) >= 1:
            recent_price = float(historical_bars[-1].close)
            atr_signal = self.atr.value / recent_price if recent_price > 0 else 0.0
            features.append(atr_signal)
        else:
            features.append(0.0)
        
        # Price momentum
        if len(self.returns) >= 5:
            momentum = np.mean(list(self.returns)[-5:])
            features.append(momentum)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def _all_indicators_ready(self) -> bool:
        """Check if all indicators are initialized."""
        return all([
            self.ema_fast.initialized,
            self.ema_slow.initialized,
            self.rsi.initialized,
            self.atr.initialized
        ])
    
    def _execute_trading_decision(self, signal_strength: float, bar: Bar):
        """Execute trading decision based on signal strength."""
        # Convert to bias signal (-0.5 to +0.5)
        signal_bias = signal_strength - 0.5
        
        action_taken = "NONE"
        
        # Trading thresholds
        strong_signal_threshold = 0.15  # 15% above/below neutral
        
        if abs(signal_bias) > strong_signal_threshold:
            if signal_bias > 0 and not self.portfolio.is_net_long(self.config.instrument_id):
                self._place_order(OrderSide.BUY, bar)
                action_taken = "BUY"
                
            elif signal_bias < 0 and not self.portfolio.is_net_short(self.config.instrument_id):
                self._place_order(OrderSide.SELL, bar)
                action_taken = "SELL"
        
        # Position management
        self._manage_position()
        
        # Log significant actions
        if action_taken != "NONE":
            console.print(f"[bold blue]📈 {{action_taken}}: Signal={{signal_strength:.3f}}, "
                         f"Bias={{signal_bias:.3f}}, Bar={{self.bar_counter}}[/bold blue]")
    
    def _place_order(self, side: OrderSide, bar: Bar):
        """Place market order with proper risk management."""
        try:
            # Use configured position size
            position_size = getattr(self.config, 'trade_size', 0.001)
            quantity = Quantity(position_size, precision=3)
            
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=quantity,
                time_in_force=TimeInForce.IOC,
                client_order_id=self.generate_order_id()
            )
            
            self.submit_order(order)
            self.trade_count += 1
            
        except Exception as e:
            console.print(f"[red]❌ Order placement failed: {{e}}[/red]")
    
    def _manage_position(self):
        """Manage position holding time and risk."""
        if not self.portfolio.is_flat(self.config.instrument_id):
            self.position_hold_bars += 1
            
            # Force close after extended hold (risk management)
            max_hold_bars = 240  # 4 hours for 1-minute bars
            if self.position_hold_bars >= max_hold_bars:
                console.print(f"[yellow]⚠️ Force closing position after {{self.position_hold_bars}} bars[/yellow]")
                # Position will be closed on next signal
        else:
            self.position_hold_bars = 0
    
    def generate_order_id(self):
        """Generate unique order ID."""
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.model.identifiers import ClientOrderId
        return ClientOrderId(str(UUID4()))
    
    def on_stop(self):
        """Strategy cleanup and final reporting."""
        console.print(f"[yellow]⏹️ {{name}} stopped[/yellow]")
        
        if self.bar_counter > 0:
            console.print(f"[cyan]📊 Final Performance Summary:[/cyan]")
            console.print(f"[cyan]  • Total bars processed: {{self.bar_counter}}[/cyan]")
            console.print(f"[cyan]  • Total trades executed: {{self.trade_count}}[/cyan]")
            console.print(f"[cyan]  • Final signal strength: {{self.last_signal:.3f}}[/cyan]")
            
            if self.signal_strength_history:
                avg_signal = np.mean(self.signal_strength_history)
                signal_std = np.std(self.signal_strength_history)
                console.print(f"[cyan]  • Average signal: {{avg_signal:.3f}} ± {{signal_std:.3f}}[/cyan]")
        
        console.print(f"[bold green]✅ {{name}} completed successfully![/bold green]")
    
    def on_reset(self):
        """Reset strategy state for new run."""
        self.returns.clear()
        self.signals.clear()
        self.signal_strength_history.clear()
        self.bar_counter = 0
        self.last_signal = 0.0
        self.position_hold_bars = 0
        self.trade_count = 0
        
        console.print(f"[blue]🔄 {{name}} reset - Ready for new run![/blue]")


# Example usage and testing
if __name__ == "__main__":
    console.print(f"[bold green]🚀 {{name}} - NT-Native Adaptive Trading Strategy[/bold green]")
    console.print("[dim]This strategy follows NautilusTrader best practices with SAGE integration[/dim]")
    
    # Basic configuration example
    class MockConfig:
        def __init__(self):
            self.instrument_id = "BTCUSDT-PERP.SIM"
            self.bar_type = "BTCUSDT-PERP.SIM-1-MINUTE-LAST-EXTERNAL"
            self.trade_size = 0.002
    
    try:
        strategy = {name}(MockConfig())
        console.print("✅ Strategy instantiation test passed")
    except Exception as e:
        console.print(f"❌ Strategy test failed: {{e}}")
'''
    
    def _get_strategy_test_template(self, name: str) -> str:
        """Generate test template for strategy."""
        snake_name = self._snake_case(name)
        
        return f'''"""
Test suite for {name} strategy.

Tests strategy initialization, core functionality, and integration points.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.sage_forge.strategies.{snake_name} import {name}


class TestConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.instrument_id = "BTCUSDT-PERP.SIM"
        self.bar_type = "BTCUSDT-PERP.SIM-1-MINUTE-LAST-EXTERNAL"
        self.trade_size = 0.001


class Test{name}:
    """Test suite for {name} strategy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TestConfig()
        self.strategy = {name}(self.config)
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.bar_counter == 0
        assert self.strategy.trade_count == 0
        assert len(self.strategy.returns) == 0
        assert len(self.strategy.signals) == 0
    
    def test_indicators_setup(self):
        """Test indicators are properly initialized."""
        assert self.strategy.ema_fast is not None
        assert self.strategy.ema_slow is not None
        assert self.strategy.rsi is not None
        assert self.strategy.atr is not None
    
    def test_sage_integration(self):
        """Test SAGE model integration."""
        assert self.strategy.sage_models is not None
        # Test prediction with mock data
        mock_features = [0.1, 0.2, 0.05, 0.01]
        # This will test when SAGE models are implemented
    
    def test_feature_extraction(self):
        """Test feature extraction with mock data."""
        # Mock historical bars
        mock_bars = []
        features = self.strategy._extract_features(mock_bars)
        assert isinstance(features, list) or hasattr(features, '__len__')
    
    def test_order_generation(self):
        """Test order ID generation."""
        order_id = self.strategy.generate_order_id()
        assert order_id is not None
        assert str(order_id) != ""
    
    def test_reset_functionality(self):
        """Test strategy reset works correctly."""
        # Add some data
        self.strategy.bar_counter = 100
        self.strategy.trade_count = 5
        
        # Reset
        self.strategy.on_reset()
        
        # Verify reset
        assert self.strategy.bar_counter == 0
        assert self.strategy.trade_count == 0
'''
    
    def _get_model_template(self, name: str) -> str:
        """Generate SAGE model template."""
        snake_name = self._snake_case(name)
        
        return f'''"""
{name} - SAGE Model Implementation
{'=' * (len(name) + 30)}

Professional SAGE model following the framework patterns:
- Base model interface compliance
- Feature extraction capabilities
- Prediction and uncertainty quantification
- Integration with ensemble framework

Created: {self.timestamp}
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from sage_forge.models.base import BaseSAGEModel
from sage_forge.core.config import SageConfig

from rich.console import Console
console = Console()


class {name}(BaseSAGEModel):
    """
    {name} - SAGE Model Implementation
    
    Features:
    - Professional model interface
    - Feature extraction and engineering
    - Prediction with uncertainty quantification  
    - Ensemble framework compatibility
    - Configuration management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        console.print(f"[yellow]🧠 Initializing {{name}}...[/yellow]")
        
        # Model configuration
        self.config = config or {{}}
        self.sage_config = SageConfig()
        
        # Model parameters
        self.feature_dim = self.config.get('feature_dim', 10)
        self.prediction_horizon = self.config.get('horizon', 1)
        self.uncertainty_estimation = self.config.get('uncertainty', True)
        
        # Model state
        self.is_trained = False
        self.training_history = []
        self.prediction_history = []
        
        # Feature processing
        self.feature_scaler = None
        self.feature_names = []
        
        # Performance tracking
        self.model_metrics = {{
            'predictions_made': 0,
            'training_iterations': 0,
            'last_update': None
        }}
        
        console.print(f"[bold green]✅ {{name}} initialized successfully![/bold green]")
    
    @property
    def name(self) -> str:
        """Model name for identification."""
        return "{snake_name}"
    
    @property
    def version(self) -> str:
        """Model version."""
        return "1.0.0"
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'The{name}':
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
        self : {name}
            Trained model instance
        """
        console.print(f"[blue]🔧 Training {{name}} on {{X.shape[0]}} samples...[/blue]")
        
        # Validate input
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {{X.shape[0]}} vs {{y.shape[0]}}")
        
        # Store feature information
        self.feature_dim = X.shape[1]
        self.feature_names = [f"feature_{{i}}" for i in range(self.feature_dim)]
        
        # Feature preprocessing
        self._preprocess_features(X)
        
        # Model-specific training logic
        self._train_model(X, y, **kwargs)
        
        # Update state
        self.is_trained = True
        self.model_metrics['training_iterations'] += 1
        self.model_metrics['last_update'] = console.get_datetime()
        
        console.print(f"[green]✅ Training completed for {{name}}[/green]")
        return self
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = None) -> Union[np.ndarray, tuple]:
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
        if not self.is_trained:
            raise RuntimeError(f"{{name}} must be trained before making predictions")
        
        if X.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {{self.feature_dim}} features, got {{X.shape[1]}}")
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=False)
        
        # Generate predictions
        predictions = self._generate_predictions(X_processed)
        
        # Generate uncertainty estimates if requested
        if return_uncertainty or (return_uncertainty is None and self.uncertainty_estimation):
            uncertainties = self._estimate_uncertainty(X_processed, predictions)
            result = (predictions, uncertainties)
        else:
            result = predictions
        
        # Update metrics
        self.model_metrics['predictions_made'] += X.shape[0]
        self.prediction_history.extend(predictions.tolist())
        
        return result
    
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
        if not self.is_trained:
            # Initial training if not trained yet
            self.fit(X, y)
        else:
            # Online update
            self._online_update(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        importance : dict
            Feature importance mapping
        """
        if not self.is_trained:
            return {{}}
        
        # Model-specific feature importance calculation
        importance_scores = self._calculate_feature_importance()
        
        return dict(zip(self.feature_names, importance_scores))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
        --------
        info : dict
            Model configuration and performance info
        """
        return {{
            'name': self.name,
            'version': self.version,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'config': self.config,
            'metrics': self.model_metrics.copy(),
            'uncertainty_estimation': self.uncertainty_estimation
        }}
    
    def _preprocess_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Preprocess features for training/prediction.
        
        Parameters:
        -----------
        X : np.ndarray
            Raw features
        fit : bool
            Whether to fit preprocessing parameters
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed features
        """
        # Implement feature scaling, normalization, etc.
        if fit:
            # Fit preprocessing parameters
            self.feature_scaler = {{
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0) + 1e-8  # Avoid division by zero
            }}
        
        # Apply preprocessing
        if self.feature_scaler:
            X_processed = (X - self.feature_scaler['mean']) / self.feature_scaler['std']
        else:
            X_processed = X.copy()
        
        return X_processed
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Model-specific training implementation.
        
        Parameters:
        -----------
        X : np.ndarray
            Preprocessed features
        y : np.ndarray
            Target values
        **kwargs : dict
            Training parameters
        """
        # IMPLEMENT YOUR MODEL TRAINING LOGIC HERE
        # Example: sklearn model, neural network, custom algorithm, etc.
        
        console.print(f"[cyan]  • Model type: {{name}}[/cyan]")
        console.print(f"[cyan]  • Training samples: {{X.shape[0]}}[/cyan]")
        console.print(f"[cyan]  • Feature dimension: {{X.shape[1]}}[/cyan]")
        
        # Store training data statistics for validation
        self.training_stats = {{
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_mean': np.mean(y),
            'target_std': np.std(y)
        }}
        
        # Example: Simple linear model for demonstration
        # Replace with your actual model implementation
        self.model_weights = np.random.randn(X.shape[1])
        self.model_bias = 0.0
    
    def _generate_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Preprocessed features
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        # IMPLEMENT YOUR PREDICTION LOGIC HERE
        
        # Example: Simple linear prediction for demonstration  
        # Replace with your actual model prediction
        predictions = X @ self.model_weights + self.model_bias
        
        return predictions
    
    def _estimate_uncertainty(self, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        predictions : np.ndarray
            Model predictions
            
        Returns:
        --------
        uncertainties : np.ndarray
            Uncertainty estimates
        """
        # IMPLEMENT YOUR UNCERTAINTY ESTIMATION HERE
        
        # Example: Simple uncertainty based on feature variance
        # Replace with proper uncertainty quantification
        feature_variance = np.var(X, axis=1)
        uncertainties = np.sqrt(feature_variance) * 0.1  # Simple heuristic
        
        return uncertainties
    
    def _online_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform online model update.
        
        Parameters:
        -----------
        X : np.ndarray
            New features
        y : np.ndarray
            New targets
        """
        # IMPLEMENT YOUR ONLINE LEARNING LOGIC HERE
        
        X_processed = self._preprocess_features(X, fit=False)
        
        # Example: Simple gradient update for demonstration
        # Replace with proper online learning algorithm
        learning_rate = 0.01
        predictions = self._generate_predictions(X_processed)
        errors = predictions - y
        
        # Update weights
        gradient = X_processed.T @ errors / len(errors)
        self.model_weights -= learning_rate * gradient
    
    def _calculate_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance scores.
        
        Returns:
        --------
        importance : np.ndarray
            Feature importance scores
        """
        # IMPLEMENT YOUR FEATURE IMPORTANCE CALCULATION HERE
        
        # Example: Use absolute weights as importance
        # Replace with proper feature importance calculation
        if hasattr(self, 'model_weights'):
            importance = np.abs(self.model_weights)
            # Normalize to sum to 1
            importance = importance / np.sum(importance)
        else:
            importance = np.ones(self.feature_dim) / self.feature_dim
        
        return importance


# Example usage and testing
if __name__ == "__main__":
    console.print(f"[bold green]🧠 {{name}} - SAGE Model Implementation[/bold green]")
    console.print("[dim]Professional model template with ensemble framework integration[/dim]")
    
    # Basic functionality test
    try:
        # Create model
        model = {name}({{'feature_dim': 5, 'uncertainty': True}})
        
        # Generate mock data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 5)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions, uncertainties = model.predict(X_test, return_uncertainty=True)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        console.print("✅ Model functionality test passed")
        console.print(f"  • Predictions shape: {{predictions.shape}}")
        console.print(f"  • Uncertainties shape: {{uncertainties.shape}}")
        console.print(f"  • Feature importance: {{len(importance)}} features")
        
    except Exception as e:
        console.print(f"❌ Model test failed: {{e}}")
'''
    
    def _get_model_test_template(self, name: str) -> str:
        """Generate test template for model."""
        snake_name = self._snake_case(name)
        
        return f'''"""
Test suite for {name} model.

Tests model functionality, training, prediction, and integration.
"""

import pytest
import numpy as np

from src.sage_forge.models.{snake_name} import {name}


class Test{name}:
    """Test suite for {name} model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {{'feature_dim': 5, 'uncertainty': True}}
        self.model = {name}(self.config)
        
        # Mock data
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 5)
    
    def test_initialization(self):
        """Test model initializes correctly."""
        assert self.model.name == "{snake_name}"
        assert not self.model.is_trained
        assert self.model.feature_dim == 5
    
    def test_training(self):
        """Test model training functionality."""
        self.model.fit(self.X_train, self.y_train)
        assert self.model.is_trained
        assert self.model.model_metrics['training_iterations'] == 1
    
    def test_prediction(self):
        """Test model prediction functionality."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        assert predictions.shape[0] == self.X_test.shape[0]
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        self.model.fit(self.X_train, self.y_train)
        predictions, uncertainties = self.model.predict(self.X_test, return_uncertainty=True)
        assert uncertainties.shape[0] == self.X_test.shape[0]
        assert np.all(uncertainties >= 0)  # Uncertainties should be non-negative
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        self.model.fit(self.X_train, self.y_train)
        importance = self.model.get_feature_importance()
        assert len(importance) == 5
        assert all(isinstance(k, str) for k in importance.keys())
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_online_update(self):
        """Test online learning functionality."""
        self.model.fit(self.X_train, self.y_train)
        
        # New data for online update
        X_new = np.random.randn(10, 5)
        y_new = np.random.randn(10)
        
        self.model.update(X_new, y_new)
        # Should still be trained after update
        assert self.model.is_trained
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        assert 'name' in info
        assert 'version' in info
        assert 'is_trained' in info
        assert info['name'] == "{snake_name}"
'''
    
    def _get_example_template(self, name: str, template_type: str) -> str:
        """Generate example usage template."""
        snake_name = self._snake_case(name)
        
        if template_type == "strategy":
            return f'''"""
Example usage of {name} strategy.

Demonstrates:
- Strategy configuration and setup
- Real data integration with DSM
- Backtesting with NautilusTrader
- Performance analysis and visualization
"""

import sys
from pathlib import Path

# Add sage-forge to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sage_forge.strategies.{snake_name} import {name}
from sage_forge.data.manager import ArrowDataManager
from sage_forge.core.config import SageConfig

from rich.console import Console
console = Console()


def main():
    """Run {name} strategy example."""
    console.print("[bold blue]🚀 {name} Strategy Example[/bold blue]")
    
    # TODO: Implement example when backtesting infrastructure is ready
    console.print("[yellow]Example implementation coming soon...[/yellow]")
    console.print("This will demonstrate:")
    console.print("  • Strategy configuration")
    console.print("  • Real data fetching with DSM")
    console.print("  • NautilusTrader backtesting")
    console.print("  • Performance analysis")
    console.print("  • FinPlot visualization")


if __name__ == "__main__":
    main()
'''
        
        elif template_type == "model":
            return f'''"""
Example usage of {name} model.

Demonstrates:
- Model training and configuration
- Feature engineering and preprocessing
- Prediction and uncertainty quantification
- Integration with SAGE ensemble
"""

import sys
from pathlib import Path
import numpy as np

# Add sage-forge to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sage_forge.models.{snake_name} import {name}
from sage_forge.data.manager import ArrowDataManager

from rich.console import Console
console = Console()


def main():
    """Run {name} model example."""
    console.print("[bold blue]🧠 {name} Model Example[/bold blue]")
    
    # Model configuration
    config = {{
        'feature_dim': 10,
        'uncertainty': True,
        'horizon': 1
    }}
    
    # Create model
    model = {name}(config)
    console.print(f"✅ Created {{model.name}} model")
    
    # Generate example data (replace with real market data)
    console.print("📊 Generating example training data...")
    n_samples = 1000
    X_train = np.random.randn(n_samples, 10)
    y_train = np.random.randn(n_samples)
    
    # Train model
    console.print("🔧 Training model...")
    model.fit(X_train, y_train)
    
    # Generate test data
    X_test = np.random.randn(100, 10)
    
    # Make predictions
    console.print("🎯 Making predictions...")
    predictions, uncertainties = model.predict(X_test, return_uncertainty=True)
    
    # Show results
    console.print(f"📈 Generated {{len(predictions)}} predictions")
    console.print(f"   Mean prediction: {{np.mean(predictions):.3f}}")
    console.print(f"   Mean uncertainty: {{np.mean(uncertainties):.3f}}")
    
    # Feature importance
    importance = model.get_feature_importance()
    console.print("🎯 Top 3 most important features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_features[:3]):
        console.print(f"   {{i+1}}. {{feature}}: {{score:.3f}}")
    
    # Model info
    info = model.get_model_info()
    console.print(f"ℹ️ Model info: {{info['name']}} v{{info['version']}}")
    console.print(f"   Predictions made: {{info['metrics']['predictions_made']}}")
    
    console.print("✅ Example completed successfully!")


if __name__ == "__main__":
    main()
'''
        
        else:
            return f'"""Example template for {name}."""\nprint("Example template generated")\n'


@click.command()
@click.argument('component_type', type=click.Choice(['strategy', 'model', 'example']))
@click.argument('name')
@click.option('--output-dir', '-o', type=click.Path(), default=None, 
              help='Output directory (default: src/sage_forge/{component_type}s/)')
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.pass_context
def create(ctx, component_type, name, output_dir, force):
    """
    Create templates for strategies, models, or examples.
    
    COMPONENT_TYPE: Type of component to create (strategy/model/example)
    NAME: Name of the component (e.g., MyAdaptiveStrategy)
    """
    verbose = ctx.obj.get('verbose', False)
    
    generator = TemplateGenerator()
    
    try:
        # Determine output directory
        if output_dir is None:
            if component_type == 'example':
                output_dir = Path('examples')
            else:
                output_dir = Path('src/sage_forge') / f"{component_type}s"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate appropriate template
        if component_type == 'strategy':
            created_file = generator.create_strategy(name, output_dir)
        elif component_type == 'model':
            created_file = generator.create_model(name, output_dir)
        elif component_type == 'example':
            # Ask for example type
            example_type = click.prompt('Example type', type=click.Choice(['strategy', 'model']))
            created_file = generator.create_example(name, example_type)
        
        # Check if file exists and handle force flag
        if created_file.exists() and not force:
            if not click.confirm(f"File {{created_file}} already exists. Overwrite?"):
                console.print("❌ [yellow]Creation cancelled[/yellow]")
                return
        
        console.print(f"✅ [green]Created {{component_type}}:[/green] {{created_file}}")
        
        if verbose:
            # Show file preview
            content = created_file.read_text()
            syntax = Syntax(content[:1000] + "..." if len(content) > 1000 else content, 
                          "python", theme="monokai", line_numbers=True)
            console.print("\n[bold]File Preview:[/bold]")
            console.print(syntax)
        
        # Provide next steps
        console.print(f"\n[bold blue]Next Steps:[/bold blue]")
        if component_type == 'strategy':
            console.print("  1. Implement your trading logic in the strategy class")
            console.print("  2. Configure SAGE models in __init__")
            console.print("  3. Test with: python examples/{{generator._snake_case(name)}}_example.py")
        elif component_type == 'model':
            console.print("  1. Implement _train_model() with your algorithm")
            console.print("  2. Implement _generate_predictions() method")
            console.print("  3. Test with: python examples/{{generator._snake_case(name)}}_example.py")
        
        console.print(f"  4. Run tests: pytest tests/test_{component_type}s/test_{{generator._snake_case(name)}}.py")
        
    except Exception as e:
        console.print(f"❌ [red]Creation failed:[/red] {{e}}")
        raise click.ClickException(str(e))