"""
🛡️ Circuit Breaker Shield: Failure Handling Protection

Implements circuit breaker pattern to protect against TiRex cascade failures
with graceful degradation to simple forecasting fallbacks.
"""

import torch
import logging
import time
from typing import Tuple, Optional, Dict, Any
from enum import Enum
from contextlib import contextmanager

from ..exceptions import TiRexServiceUnavailableError, FallbackExhaustionError, ShieldViolation

# Circuit-specific logging
circuit_logger = logging.getLogger('sage_forge.guardian.shields.circuit')


class CircuitState(Enum):
    """Circuit breaker states with intuitive names for LLM agents"""
    CLOSED = "CLOSED"      # ✅ TiRex functioning normally
    OPEN = "OPEN"          # 🚨 TiRex failures detected, using fallbacks
    HALF_OPEN = "HALF_OPEN"  # 🔄 Testing TiRex recovery


class FallbackStrategy(Enum):
    """Fallback strategies when TiRex fails"""
    SIMPLE_MA = "simple_moving_average"     # Moving average of recent values
    LINEAR_TREND = "linear_trend"           # Linear extrapolation
    LAST_VALUE = "last_value"               # Repeat last known value
    ZERO_CHANGE = "zero_change"             # Assume no change


class CircuitShield:
    """
    🛡️ CIRCUIT BREAKER GUARDIAN: Protects against TiRex cascade failures.
    
    Implements intelligent circuit breaker pattern that detects TiRex failures
    and gracefully degrades to simple but reliable fallback forecasting methods.
    
    PROTECTION STRATEGY:
    - Monitor TiRex inference success/failure patterns
    - Open circuit when failure threshold exceeded
    - Provide graceful fallback forecasting during outages
    - Automatically test recovery and close circuit when TiRex recovers
    
    FALLBACK HIERARCHY (from most to least sophisticated):
    1. Simple Moving Average - smoothed recent trend
    2. Linear Trend - extrapolation from recent data
    3. Last Value - repeat last known price (ultimate fallback)
    
    EMPIRICAL BASIS:
    Based on TiRex failure scenarios discovered during security testing where
    model becomes unstable and produces NaN/inf outputs consistently.
    """
    
    def __init__(self, 
                 failure_threshold: int = 3,
                 recovery_timeout: float = 60.0,
                 fallback_strategy: str = "graceful"):
        """
        Initialize Circuit Shield with failure detection and fallback strategies.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery  
            fallback_strategy: "graceful", "strict", "minimal" - fallback approach
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback_strategy = fallback_strategy
        
        # Circuit state management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = time.time()
        
        # Performance tracking
        self.total_inferences = 0
        self.tirex_successes = 0
        self.fallback_uses = 0
        self.circuit_opens = 0
        
        # Fallback configuration
        self._configure_fallback_strategies(fallback_strategy)
        
        circuit_logger.info(
            f"🛡️ CircuitShield initialized - Failure threshold: {failure_threshold}, "
            f"Recovery timeout: {recovery_timeout}s, Strategy: {fallback_strategy}"
        )
    
    def _configure_fallback_strategies(self, strategy: str):
        """Configure fallback behavior based on strategy"""
        if strategy == "graceful":
            # Full fallback hierarchy for maximum resilience
            self.enabled_fallbacks = [
                FallbackStrategy.SIMPLE_MA,
                FallbackStrategy.LINEAR_TREND, 
                FallbackStrategy.LAST_VALUE
            ]
            self.allow_graceful_degradation = True
            
        elif strategy == "strict":
            # Limited fallbacks, prefer failing fast
            self.enabled_fallbacks = [FallbackStrategy.LAST_VALUE]
            self.allow_graceful_degradation = False
            
        elif strategy == "minimal":
            # Minimal protection, mostly monitoring
            self.enabled_fallbacks = [FallbackStrategy.ZERO_CHANGE]
            self.allow_graceful_degradation = False
            
        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")
    
    def protected_inference(self, 
                          context: torch.Tensor,
                          prediction_length: int,
                          **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🛡️ PROTECTED TIREX INFERENCE: Circuit-protected model inference with fallbacks.
        
        Provides the same interface as raw TiRex but with intelligent failure detection
        and graceful fallback strategies when the model becomes unstable.
        
        Args:
            context: Historical data [batch_size, context_length]  
            prediction_length: Number of timesteps to forecast
            **kwargs: Additional TiRex arguments
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (quantiles[B,k,9], mean[B,k])
            
        Raises:
            TiRexServiceUnavailableError: Circuit is open, using fallbacks
            FallbackExhaustionError: All fallback strategies failed
        """
        self.total_inferences += 1
        
        with self._circuit_management():
            if self.state == CircuitState.CLOSED:
                # TiRex is healthy - attempt normal inference
                return self._attempt_tirex_inference(context, prediction_length, **kwargs)
                
            elif self.state == CircuitState.HALF_OPEN:
                # Testing recovery - single TiRex attempt
                return self._test_recovery_inference(context, prediction_length, **kwargs)
                
            elif self.state == CircuitState.OPEN:
                # Circuit open - use fallback strategies
                return self._fallback_inference(context, prediction_length)
    
    @contextmanager
    def _circuit_management(self):
        """Context manager for circuit state management"""
        try:
            yield
        except Exception as error:
            # Any exception during inference counts as failure
            self._record_failure(error)
            raise
    
    def _attempt_tirex_inference(self, 
                               context: torch.Tensor,
                               prediction_length: int,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attempt TiRex inference with failure detection"""
        try:
            # Import TiRex here to avoid circular dependencies
            from repos.tirex import TiRex
            
            model = TiRex()
            quantiles, mean = model.forecast(
                context=context,
                prediction_length=prediction_length,
                **kwargs
            )
            
            # Validate outputs for corruption (empirical vulnerability)
            self._validate_tirex_output(quantiles, mean)
            
            # Success - record and return
            self._record_success()
            return quantiles, mean
            
        except Exception as tirex_error:
            circuit_logger.warning(f"🛡️ TiRex inference failed: {tirex_error}")
            self._record_failure(tirex_error)
            
            # If circuit should open, use fallback
            if self.state == CircuitState.OPEN:
                return self._fallback_inference(context, prediction_length)
            else:
                raise
    
    def _test_recovery_inference(self,
                               context: torch.Tensor, 
                               prediction_length: int,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Test TiRex recovery with single attempt"""
        try:
            # Single recovery test
            quantiles, mean = self._attempt_tirex_inference(context, prediction_length, **kwargs)
            
            # Success - close circuit
            self._close_circuit()
            circuit_logger.info("🛡️ TiRex recovery confirmed - circuit CLOSED")
            
            return quantiles, mean
            
        except Exception as recovery_error:
            # Recovery failed - stay open, use fallback
            circuit_logger.warning(f"🛡️ Recovery test failed: {recovery_error}")
            self._open_circuit()
            return self._fallback_inference(context, prediction_length)
    
    def _fallback_inference(self,
                          context: torch.Tensor,
                          prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute fallback forecasting strategies when TiRex is unavailable.
        
        Tries fallback strategies in order of sophistication until one succeeds.
        """
        self.fallback_uses += 1
        failed_fallbacks = []
        
        circuit_logger.info(f"🛡️ Using fallback strategies - TiRex circuit OPEN")
        
        for fallback in self.enabled_fallbacks:
            try:
                quantiles, mean = self._execute_fallback_strategy(
                    fallback, context, prediction_length
                )
                
                circuit_logger.info(f"🛡️ Fallback success: {fallback.value}")
                return quantiles, mean
                
            except Exception as fallback_error:
                failed_fallbacks.append(fallback.value)
                circuit_logger.warning(f"🛡️ Fallback failed: {fallback.value} - {fallback_error}")
                continue
        
        # All fallbacks failed - critical system failure
        raise FallbackExhaustionError(
            f"All fallback strategies failed: {failed_fallbacks}",
            failed_fallbacks=failed_fallbacks
        )
    
    def _execute_fallback_strategy(self,
                                 strategy: FallbackStrategy,
                                 context: torch.Tensor,
                                 prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute specific fallback forecasting strategy"""
        
        if strategy == FallbackStrategy.SIMPLE_MA:
            return self._simple_moving_average_forecast(context, prediction_length)
            
        elif strategy == FallbackStrategy.LINEAR_TREND:
            return self._linear_trend_forecast(context, prediction_length)
            
        elif strategy == FallbackStrategy.LAST_VALUE:
            return self._last_value_forecast(context, prediction_length)
            
        elif strategy == FallbackStrategy.ZERO_CHANGE:
            return self._zero_change_forecast(context, prediction_length)
            
        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")
    
    def _simple_moving_average_forecast(self, 
                                      context: torch.Tensor,
                                      prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple moving average fallback (most sophisticated)"""
        batch_size = context.shape[0]
        
        # Use last 5 values for moving average (or all if less than 5)
        window_size = min(5, context.shape[-1])
        recent_values = context[..., -window_size:]
        
        # Calculate moving average
        ma_value = recent_values.mean(dim=-1, keepdim=True)  # [B, 1]
        
        # Forecast: repeat moving average
        mean_forecast = ma_value.expand(-1, prediction_length)  # [B, k]
        
        # Generate simple quantiles around mean (±5%, ±10%, ±20%, ±50%)
        quantiles = self._generate_simple_quantiles(mean_forecast)
        
        return quantiles, mean_forecast
    
    def _linear_trend_forecast(self,
                             context: torch.Tensor, 
                             prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear trend extrapolation fallback"""
        batch_size = context.shape[0]
        context_length = context.shape[-1]
        
        if context_length < 2:
            # Need at least 2 points for trend - fall back to last value
            return self._last_value_forecast(context, prediction_length)
        
        # Calculate simple linear trend from first and last values
        first_val = context[..., 0]   # [B]
        last_val = context[..., -1]   # [B]
        
        # Trend per timestep
        trend = (last_val - first_val) / (context_length - 1)  # [B]
        
        # Extrapolate trend
        time_steps = torch.arange(1, prediction_length + 1, dtype=context.dtype, device=context.device)
        future_values = last_val.unsqueeze(-1) + trend.unsqueeze(-1) * time_steps  # [B, k]
        
        # Generate quantiles around trend
        quantiles = self._generate_simple_quantiles(future_values)
        
        return quantiles, future_values
    
    def _last_value_forecast(self,
                           context: torch.Tensor,
                           prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Last value repetition fallback (ultimate fallback)"""
        last_value = context[..., -1].unsqueeze(-1)  # [B, 1]
        mean_forecast = last_value.expand(-1, prediction_length)  # [B, k]
        
        # Generate minimal quantiles around last value
        quantiles = self._generate_simple_quantiles(mean_forecast, spread=0.01)  # 1% spread
        
        return quantiles, mean_forecast
    
    def _zero_change_forecast(self,
                            context: torch.Tensor,
                            prediction_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero change forecast (minimal fallback)"""
        last_value = context[..., -1].unsqueeze(-1)  # [B, 1] 
        mean_forecast = last_value.expand(-1, prediction_length)  # [B, k]
        
        # All quantiles equal to mean (no uncertainty)
        quantiles = mean_forecast.unsqueeze(-1).expand(-1, -1, 9)  # [B, k, 9]
        
        return quantiles, mean_forecast
    
    def _generate_simple_quantiles(self, 
                                 mean_forecast: torch.Tensor,
                                 spread: float = 0.05) -> torch.Tensor:
        """Generate simple quantile structure around mean forecast"""
        batch_size, pred_length = mean_forecast.shape
        
        # TiRex quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # Generate symmetric spread around mean
        quantile_offsets = torch.tensor([
            -4*spread, -3*spread, -2*spread, -1*spread, 0.0,  # 0.1, 0.2, 0.3, 0.4, 0.5
            1*spread, 2*spread, 3*spread, 4*spread            # 0.6, 0.7, 0.8, 0.9
        ], dtype=mean_forecast.dtype, device=mean_forecast.device)
        
        # Apply offsets: [B, k] + [9] -> [B, k, 9]
        quantiles = mean_forecast.unsqueeze(-1) + quantile_offsets * mean_forecast.abs().unsqueeze(-1)
        
        return quantiles
    
    def _validate_tirex_output(self, quantiles: torch.Tensor, mean: torch.Tensor):
        """Validate TiRex outputs for corruption (empirical vulnerability protection)"""
        # Check for NaN corruption (confirmed vulnerability)
        if torch.isnan(quantiles).any() or torch.isnan(mean).any():
            raise ShieldViolation("TiRex output corruption: NaN values detected")
            
        # Check for infinity corruption (confirmed vulnerability)  
        if torch.isinf(quantiles).any() or torch.isinf(mean).any():
            raise ShieldViolation("TiRex output corruption: Inf values detected")
            
        # Check for extreme values (confirmed vulnerability)
        if torch.abs(quantiles).max() > 1e6 or torch.abs(mean).max() > 1e6:
            raise ShieldViolation("TiRex output corruption: Extreme values detected")
    
    def _record_success(self):
        """Record successful TiRex inference"""
        self.tirex_successes += 1
        self.last_success_time = time.time()
        self.failure_count = 0  # Reset failure counter on success
        
        circuit_logger.debug(f"🛡️ TiRex success recorded - failures reset to 0")
    
    def _record_failure(self, error: Exception):
        """Record TiRex failure and update circuit state"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        circuit_logger.warning(
            f"🛡️ TiRex failure #{self.failure_count}: {type(error).__name__}"
        )
        
        # Check if circuit should open
        if (self.failure_count >= self.failure_threshold and 
            self.state == CircuitState.CLOSED):
            self._open_circuit()
    
    def _open_circuit(self):
        """Open circuit due to excessive failures"""
        self.state = CircuitState.OPEN
        self.circuit_opens += 1
        
        circuit_logger.error(
            f"🚨 CIRCUIT OPENED - TiRex failures: {self.failure_count}/"
            f"{self.failure_threshold}, switching to fallbacks"
        )
    
    def _close_circuit(self):
        """Close circuit after successful recovery"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        
        circuit_logger.info("✅ CIRCUIT CLOSED - TiRex operational")
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.state != CircuitState.OPEN:
            return False
            
        if self.last_failure_time is None:
            return False
            
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def attempt_recovery(self):
        """Manually trigger recovery attempt (for testing/admin)"""
        if self.state == CircuitState.OPEN and self._should_attempt_recovery():
            self.state = CircuitState.HALF_OPEN
            circuit_logger.info("🔄 Circuit HALF-OPEN - testing TiRex recovery")
    
    def get_circuit_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker performance statistics"""
        uptime_ratio = self.tirex_successes / max(self.total_inferences, 1)
        fallback_ratio = self.fallback_uses / max(self.total_inferences, 1)
        
        return {
            'shield_type': 'CircuitShield',
            'circuit_state': self.state.value,
            'total_inferences': self.total_inferences,
            'tirex_successes': self.tirex_successes,
            'fallback_uses': self.fallback_uses,
            'circuit_opens': self.circuit_opens,
            'current_failures': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'uptime_ratio': uptime_ratio,
            'fallback_ratio': fallback_ratio,
            'enabled_fallbacks': [f.value for f in self.enabled_fallbacks],
            'last_success': self.last_success_time,
            'last_failure': self.last_failure_time
        }
    
    def __repr__(self) -> str:
        """Intuitive representation for LLM agents"""
        return (f"CircuitShield(state={self.state.value}, "
                f"successes={self.tirex_successes}, "
                f"failures={self.failure_count}/{self.failure_threshold})")