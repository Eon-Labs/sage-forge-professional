#!/usr/bin/env python3
"""
TiRex Model Integration - NX-AI TiRex 35M Parameter Model for SAGE-Forge
Real-time time series forecasting with GPU acceleration for directional trading signals.

Architecture: xLSTM-based transformer with 12 sLSTM blocks (35.3M parameters)
Source: https://github.com/NX-AI/tirex
Paper: https://arxiv.org/abs/2505.23719 (May 2025)
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import logging

from nautilus_trader.model.data import Bar
from rich.console import Console

# Import real TiRex API
console = Console()
try:
    from tirex import load_model, ForecastModel
    TIREX_AVAILABLE = True
    console.print("✅ TiRex library available")
except ImportError as e:
    TIREX_AVAILABLE = False
    ForecastModel = None
    console.print(f"❌ TiRex library not available: {e}")
    console.print("Install with: pip install tirex")

logger = logging.getLogger(__name__)


@dataclass
class TiRexPrediction:
    """TiRex model prediction with confidence metrics."""
    direction: int  # -1: bearish, 0: neutral, 1: bullish
    confidence: float  # Prediction confidence [0, 1]
    raw_forecast: np.ndarray  # Raw TiRex forecast output
    volatility_forecast: float  # Predicted volatility
    processing_time_ms: float  # Inference time
    market_regime: str  # Detected market regime


class TiRexInputProcessor:
    """Preprocesses market data for TiRex model input."""
    
    def __init__(self, sequence_length: int = 128):
        """
        Initialize TiRex input processor.
        
        Args:
            sequence_length: Context window for TiRex model (default 128)
        """
        self.sequence_length = sequence_length
        self.price_buffer = deque(maxlen=sequence_length)
        
        # Market regime detection
        self._min_samples = 50
    
    def add_bar(self, bar: Bar) -> None:
        """Add new bar to the input buffer."""
        # Extract close price (TiRex uses univariate time series)
        close_price = float(bar.close)
        self.price_buffer.append(close_price)
    
    def get_model_input(self) -> Optional[torch.Tensor]:
        """Generate input tensor for TiRex model."""
        if len(self.price_buffer) < self.sequence_length:
            return None
        
        # Convert to tensor format expected by TiRex
        # TiRex expects 1D tensor [sequence_length] for single time series
        price_series = np.array(list(self.price_buffer), dtype=np.float32)
        input_tensor = torch.tensor(price_series)  # 1D tensor: [128]
        
        return input_tensor
    
    def get_market_regime(self) -> str:
        """Detect current market regime from price action."""
        if len(self.price_buffer) < self._min_samples:
            return "insufficient_data"
        
        prices = np.array(list(self.price_buffer))
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility regime
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        if volatility < 0.15:
            vol_regime = "low_vol"
        elif volatility < 0.35:
            vol_regime = "medium_vol"
        else:
            vol_regime = "high_vol"
        
        # Trend regime
        x = np.arange(len(prices))
        trend_slope = np.polyfit(x, prices, 1)[0]
        price_range = prices.max() - prices.min()
        trend_strength = abs(trend_slope) / (price_range / len(prices))
        
        if trend_strength > 0.6:
            trend_regime = "trending"
        elif trend_strength > 0.3:
            trend_regime = "weak_trend" 
        else:
            trend_regime = "ranging"
        
        return f"{vol_regime}_{trend_regime}"


class TiRexModel:
    """
    Real NX-AI TiRex Model Integration for SAGE-Forge.
    
    Uses the official TiRex 35M parameter xLSTM model for zero-shot time series forecasting.
    """
    
    def __init__(self, model_name: str = "NX-AI/TiRex", prediction_length: int = 1):
        """
        Initialize TiRex model.
        
        Args:
            model_name: HuggingFace model identifier
            prediction_length: Forecast horizon (default 1 for next bar)
        """
        if not TIREX_AVAILABLE:
            raise ImportError("TiRex library not available. Install with: pip install tirex")
        
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.model: Optional[ForecastModel] = None
        self.input_processor = TiRexInputProcessor()
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.prediction_history = deque(maxlen=1000)
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the real TiRex model."""
        try:
            console.print(f"🔄 Loading TiRex model: {self.model_name}")
            
            # Load the real TiRex model
            self.model = load_model(self.model_name)
            
            console.print("✅ Real TiRex 35M parameter model loaded successfully")
            console.print("🦖 xLSTM architecture with 12 sLSTM blocks")
            console.print("⚡ Zero-shot forecasting enabled")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TiRex model: {e}")
            console.print(f"❌ TiRex model loading failed: {e}")
            return False
    
    def add_bar(self, bar: Bar) -> None:
        """Add new market data bar for processing."""
        self.input_processor.add_bar(bar)
    
    def predict(self) -> Optional[TiRexPrediction]:
        """Generate real TiRex prediction from current market state."""
        if not self.is_loaded or self.model is None:
            logger.warning("TiRex model not loaded")
            return None
        
        # Get model input
        model_input = self.input_processor.get_model_input()
        if model_input is None:
            return None
        
        # Inference timing
        start_time = time.time()
        
        try:
            # Debug: Log input shape
            logger.debug(f"TiRex input shape: {model_input.shape}, dtype: {model_input.dtype}")
            logger.debug(f"TiRex input sample: {model_input[:5].tolist()}")  # First 5 values
            
            # Real TiRex inference
            # TiRex returns (quantiles, means) tuple
            quantiles, means = self.model.forecast(
                context=model_input, 
                prediction_length=self.prediction_length
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(processing_time)
            
            # Extract forecast data from TiRex output
            # quantiles: [batch, prediction_length, num_quantiles] = [1, 1, 9]
            # means: [batch, prediction_length] = [1, 1]
            mean_forecast = means.squeeze().cpu().numpy()  # Remove batch dimensions
            
            # Calculate uncertainty from quantiles (use std of quantile predictions)
            quantile_values = quantiles.squeeze().cpu().numpy()  # [prediction_length, num_quantiles]
            forecast_std = np.std(quantile_values, axis=-1) if len(quantile_values.shape) > 0 else 0.1
            
            # Convert to directional signal
            # Handle scalar and array cases properly
            if isinstance(mean_forecast, np.ndarray) and mean_forecast.shape == ():
                # Scalar case
                forecast_value = float(mean_forecast)
            elif hasattr(mean_forecast, '__len__') and len(mean_forecast) > 0:
                # Array case
                forecast_value = float(mean_forecast[0])
            else:
                # Fallback
                forecast_value = float(mean_forecast)
                
            current_price = list(self.input_processor.price_buffer)[-1]
            
            # Calculate direction and confidence
            price_change = forecast_value - current_price
            direction = self._interpret_forecast(price_change, current_price)
            
            # Handle forecast_std properly
            if isinstance(forecast_std, np.ndarray) and forecast_std.shape == ():
                std_value = float(forecast_std)
            elif hasattr(forecast_std, '__len__') and len(forecast_std) > 0:
                std_value = float(forecast_std[0])
            else:
                std_value = float(forecast_std)
                
            confidence = self._calculate_confidence(price_change, current_price, std_value)
            volatility_forecast = std_value / current_price  # Normalized volatility
            market_regime = self.input_processor.get_market_regime()
            
            prediction = TiRexPrediction(
                direction=direction,
                confidence=confidence,
                raw_forecast=mean_forecast,
                volatility_forecast=volatility_forecast,
                processing_time_ms=processing_time,
                market_regime=market_regime
            )
            
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"TiRex prediction failed: {e}")
            return None
    
    def _interpret_forecast(self, price_change: float, current_price: float) -> int:
        """Convert price forecast to directional signal."""
        relative_change = price_change / current_price
        
        # Adaptive threshold based on recent volatility
        if len(self.prediction_history) > 10:
            recent_changes = [p.raw_forecast[0] - list(self.input_processor.price_buffer)[-2] 
                             for p in list(self.prediction_history)[-20:] if len(self.input_processor.price_buffer) > 1]
            if recent_changes:
                threshold = np.std(recent_changes) / current_price * 0.5
            else:
                threshold = 0.001  # 0.1% default
        else:
            threshold = 0.001  # 0.1% default
        
        if relative_change > threshold:
            return 1  # Bullish
        elif relative_change < -threshold:
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def _calculate_confidence(self, price_change: float, current_price: float, forecast_std: float) -> float:
        """Calculate prediction confidence based on forecast uncertainty."""
        relative_change = abs(price_change) / current_price
        relative_uncertainty = forecast_std / current_price
        
        # Higher relative change and lower uncertainty = higher confidence
        if relative_uncertainty > 0:
            confidence = relative_change / (relative_change + relative_uncertainty)
        else:
            confidence = 0.5  # Default if no uncertainty info
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_performance_stats(self) -> Dict:
        """Get model performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            "avg_inference_time_ms": np.mean(self.inference_times),
            "max_inference_time_ms": np.max(self.inference_times),
            "min_inference_time_ms": np.min(self.inference_times),
            "total_predictions": len(self.prediction_history),
            "model_name": self.model_name,
            "prediction_length": self.prediction_length,
            "is_real_tirex": True,
            "architecture": "xLSTM (35M parameters)"
        }


# Test function for development
def test_tirex_integration():
    """Test real TiRex model integration."""
    console.print("🧪 Testing Real TiRex Integration")
    
    if not TIREX_AVAILABLE:
        console.print("❌ TiRex library not available")
        return False
    
    try:
        # Initialize model
        tirex = TiRexModel()
        
        if not tirex.is_loaded:
            console.print("❌ TiRex model not loaded")
            return False
        
        # Create synthetic bars for testing
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        from datetime import datetime
        
        # Add bars to build up sequence (need 128+ for TiRex)
        base_price = 50000.0
        for i in range(150):
            # Create realistic price movement
            price_change = np.random.normal(0, 100)  # $100 volatility
            close_price = base_price + price_change + i * 10  # Slight uptrend
            
            bar = Bar(
                bar_type=None,
                open=Price.from_str(f"{close_price - 50}"),
                high=Price.from_str(f"{close_price + 100}"),
                low=Price.from_str(f"{close_price - 100}"),
                close=Price.from_str(f"{close_price}"),
                volume=Quantity.from_int(1000),
                ts_event=dt_to_unix_nanos(datetime.now()),
                ts_init=dt_to_unix_nanos(datetime.now())
            )
            
            tirex.add_bar(bar)
        
        # Test prediction
        prediction = tirex.predict()
        
        if prediction:
            console.print(f"✅ Real TiRex Prediction Generated:")
            console.print(f"   Direction: {prediction.direction}")
            console.print(f"   Confidence: {prediction.confidence:.3f}")
            console.print(f"   Forecast: {prediction.raw_forecast}")
            console.print(f"   Processing Time: {prediction.processing_time_ms:.2f}ms")
            console.print(f"   Market Regime: {prediction.market_regime}")
            console.print(f"   Volatility: {prediction.volatility_forecast:.4f}")
            
            # Performance stats
            stats = tirex.get_performance_stats()
            console.print(f"📊 Performance Stats:")
            for key, value in stats.items():
                console.print(f"   {key}: {value}")
            
            console.print("🎉 Real TiRex 35M parameter model working!")
            return True
        else:
            console.print("❌ TiRex prediction failed")
            return False
            
    except Exception as e:
        console.print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_tirex_integration()