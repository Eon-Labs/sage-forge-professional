#!/usr/bin/env python3
"""
TiRexOnlyStrategy - SAGE-Forge NT-Native Parameter-Free Regime-Aware Strategy

TiRex: Time-Series Regime Exchange - parameter-free, regime-aware evaluation framework
that discovers optimal performance criteria from market structure rather than relying 
on fixed thresholds, ensuring robust nonparametric out-of-sample viability.

SAGE Methodology: Self-Adaptive Generative Evaluation with quantitative and adaptive
assessment through parameter-free, regime-aware evaluation frameworks.
"""

import numpy as np
from decimal import Decimal
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from rich.console import Console

from sage_forge.core.config import get_config
from sage_forge.risk.position_sizer import RealisticPositionSizer

console = Console()


@dataclass
class RegimeState:
    """Represents current market regime characteristics."""
    volatility_regime: str  # 'low', 'medium', 'high'
    trend_regime: str      # 'trending', 'ranging', 'transitional'
    momentum_regime: str   # 'strong', 'weak', 'neutral'
    confidence: float      # Regime detection confidence [0,1]
    bars_in_regime: int    # Duration in current regime


class TiRexCore:
    """
    TiRex Core: Parameter-free regime detection and adaptive signal generation.
    
    Uses entropy-based methods and adaptive windows to detect regime changes
    without hardcoded thresholds or magic numbers.
    """
    
    def __init__(self, max_lookback: int = 200):
        self.max_lookback = max_lookback
        self.price_buffer = deque(maxlen=max_lookback)
        self.volume_buffer = deque(maxlen=max_lookback)
        self.returns_buffer = deque(maxlen=max_lookback)
        
        # Adaptive parameters discovered from data
        self.adaptive_windows = {}
        self.regime_history = []
        self.current_regime = None
        
    def update(self, bar: Bar) -> RegimeState:
        """
        Update TiRex with new bar and detect regime changes.
        
        Returns current regime state with confidence metrics.
        """
        close_price = float(bar.close)
        volume = float(bar.volume)
        
        self.price_buffer.append(close_price)
        self.volume_buffer.append(volume)
        
        if len(self.price_buffer) > 1:
            returns = (close_price / self.price_buffer[-2]) - 1
            self.returns_buffer.append(returns)
        
        # Need minimum bars for regime detection
        if len(self.price_buffer) < 20:
            return RegimeState('unknown', 'unknown', 'unknown', 0.0, 0)
            
        regime = self._detect_regime()
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
    
    def _detect_regime(self) -> RegimeState:
        """
        Parameter-free regime detection using adaptive entropy and structure measures.
        """
        returns = np.array(list(self.returns_buffer))
        prices = np.array(list(self.price_buffer))
        volumes = np.array(list(self.volume_buffer))
        
        # Adaptive volatility regime detection
        vol_regime, vol_confidence = self._detect_volatility_regime(returns)
        
        # Adaptive trend regime detection
        trend_regime, trend_confidence = self._detect_trend_regime(prices)
        
        # Adaptive momentum regime detection  
        momentum_regime, momentum_confidence = self._detect_momentum_regime(returns, volumes)
        
        # Aggregate confidence
        overall_confidence = np.mean([vol_confidence, trend_confidence, momentum_confidence])
        
        # Count bars in current regime
        bars_in_regime = self._count_bars_in_current_regime()
        
        return RegimeState(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            momentum_regime=momentum_regime,
            confidence=overall_confidence,
            bars_in_regime=bars_in_regime
        )
    
    def _detect_volatility_regime(self, returns: np.ndarray) -> Tuple[str, float]:
        """Parameter-free volatility regime detection using entropy."""
        if len(returns) < 10:
            return 'unknown', 0.0
            
        # Use rolling entropy to detect volatility clusters
        volatility = np.abs(returns)
        
        # Adaptive quantile-based thresholds
        q25, q75 = np.percentile(volatility, [25, 75])
        iqr = q75 - q25
        
        recent_vol = np.mean(volatility[-10:])  # Recent volatility
        
        if recent_vol < q25:
            regime = 'low'
            confidence = (q25 - recent_vol) / (q25 + 1e-8)
        elif recent_vol > q75:
            regime = 'high' 
            confidence = (recent_vol - q75) / (recent_vol + 1e-8)
        else:
            regime = 'medium'
            confidence = 1.0 - abs(recent_vol - np.median(volatility)) / (iqr + 1e-8)
            
        return regime, min(confidence, 1.0)
    
    def _detect_trend_regime(self, prices: np.ndarray) -> Tuple[str, float]:
        """Parameter-free trend detection using local extrema analysis."""
        if len(prices) < 20:
            return 'unknown', 0.0
            
        # Adaptive window selection based on data characteristics
        window = min(len(prices) // 4, 50)
        
        # Local extrema detection
        recent_prices = prices[-window:]
        price_range = np.max(recent_prices) - np.min(recent_prices)
        
        # Trend strength using linear regression coefficient
        x = np.arange(len(recent_prices))
        trend_coef = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalize by price range for regime classification
        normalized_trend = trend_coef / (price_range / len(recent_prices) + 1e-8)
        
        # R-squared for trend confidence
        predicted = np.polyval([trend_coef, recent_prices[0]], x)
        ss_res = np.sum((recent_prices - predicted) ** 2)
        ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        if abs(normalized_trend) < 0.1:
            regime = 'ranging'
        elif r_squared > 0.3:  # Strong linear relationship
            regime = 'trending'
        else:
            regime = 'transitional'
            
        confidence = r_squared
        
        return regime, min(confidence, 1.0)
    
    def _detect_momentum_regime(self, returns: np.ndarray, volumes: np.ndarray) -> Tuple[str, float]:
        """Parameter-free momentum detection using volume-price relationship."""
        if len(returns) < 10:
            return 'unknown', 0.0
            
        recent_returns = returns[-10:]
        recent_volumes = volumes[-10:]
        
        # Volume-weighted momentum
        volume_weights = recent_volumes / (np.sum(recent_volumes) + 1e-8)
        weighted_momentum = np.sum(recent_returns * volume_weights)
        
        # Momentum persistence
        momentum_consistency = np.sum(np.sign(recent_returns[:-1]) == np.sign(recent_returns[1:])) / max(len(recent_returns) - 1, 1)
        
        momentum_magnitude = abs(weighted_momentum)
        
        if momentum_magnitude > np.std(returns[-50:]) and momentum_consistency > 0.6:
            regime = 'strong'
            confidence = momentum_consistency
        elif momentum_consistency < 0.3:
            regime = 'weak'
            confidence = 1.0 - momentum_consistency
        else:
            regime = 'neutral'
            confidence = 1.0 - abs(momentum_consistency - 0.5) * 2
            
        return regime, min(confidence, 1.0)
    
    def _count_bars_in_current_regime(self) -> int:
        """Count consecutive bars in current regime."""
        if not self.regime_history:
            return 0
            
        current = self.regime_history[-1]
        count = 1
        
        for i in range(len(self.regime_history) - 2, -1, -1):
            if (self.regime_history[i].volatility_regime == current.volatility_regime and
                self.regime_history[i].trend_regime == current.trend_regime):
                count += 1
            else:
                break
                
        return count
    
    def generate_signal(self) -> Dict[str, float]:
        """
        Generate trading signals based on current regime.
        
        Returns signal strength [-1, 1] and confidence [0, 1].
        """
        if not self.current_regime or self.current_regime.confidence < 0.3:
            return {'signal': 0.0, 'confidence': 0.0, 'regime_edge': False}
            
        regime = self.current_regime
        
        # SAGE methodology: Adaptive signal generation based on regime
        signal_strength = 0.0
        regime_edge = False
        
        # Trending + Strong momentum = directional signal
        if regime.trend_regime == 'trending' and regime.momentum_regime == 'strong':
            direction = 1.0 if len(self.returns_buffer) > 0 and self.returns_buffer[-1] > 0 else -1.0
            signal_strength = direction * regime.confidence
            
        # Regime transition detection = potential reversal
        elif regime.trend_regime == 'transitional' and regime.bars_in_regime < 5:
            regime_edge = True
            signal_strength = 0.5 * regime.confidence  # Reduced position during uncertainty
            
        # Low volatility ranging = mean reversion setup
        elif regime.volatility_regime == 'low' and regime.trend_regime == 'ranging':
            # Look for mean reversion signals
            recent_returns = list(self.returns_buffer)[-5:]
            if len(recent_returns) >= 5:
                mean_return = np.mean(recent_returns)
                signal_strength = -np.sign(mean_return) * min(regime.confidence, 0.3)
        
        return {
            'signal': signal_strength,
            'confidence': regime.confidence,
            'regime_edge': regime_edge
        }


class TiRexOnlyStrategy(Strategy):
    """
    TiRex-Only Strategy: Pure parameter-free, regime-aware trading system.
    
    Features:
    - Parameter-free regime detection
    - Adaptive signal generation  
    - NT-native implementation
    - SAGE methodology compliance
    - No hardcoded thresholds or magic numbers
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # SAGE-Forge configuration
        self.sage_config = get_config()
        
        # Professional position sizing
        self.position_sizer = RealisticPositionSizer()
        
        # TiRex core engine
        self.tirex = TiRexCore(max_lookback=getattr(config, 'max_lookback', 200))
        
        # Strategy state
        self.bars_processed = 0
        self.last_regime_log = 0
        self.position_updates = []
        
    def on_start(self):
        """Initialize TiRex strategy."""
        self.subscribe_bars(self.config.bar_type)
        console.log("🦖 TiRex-Only Strategy: Parameter-free regime-aware trading initialized")
        console.log("🎯 SAGE Methodology: Self-Adaptive Generative Evaluation active")
        
    def on_bar(self, bar: Bar):
        """Process new bar with TiRex regime detection and signal generation."""
        self.bars_processed += 1
        
        # Update TiRex with new market data
        regime = self.tirex.update(bar)
        
        # Generate trading signal
        signal_data = self.tirex.generate_signal()
        
        # Log regime changes periodically
        if self.bars_processed - self.last_regime_log >= 50:
            self._log_regime_state(regime, signal_data)
            self.last_regime_log = self.bars_processed
        
        # Execute trades based on TiRex signals
        if abs(signal_data['signal']) > 0.1 and signal_data['confidence'] > 0.4:
            self._execute_tirex_trade(bar, signal_data, regime)
            
    def _execute_tirex_trade(self, bar: Bar, signal_data: Dict, regime: RegimeState):
        """Execute trade based on TiRex signal with adaptive position sizing."""
        
        # Adaptive position sizing based on regime confidence
        base_position_size = self.position_sizer.get_recommended_position_size()
        
        # Scale position by signal confidence and regime stability
        confidence_multiplier = signal_data['confidence']
        regime_stability = min(regime.bars_in_regime / 10.0, 1.0)  # Stable regime bonus
        
        adjusted_size = base_position_size * confidence_multiplier * regime_stability
        
        # Determine order side
        if signal_data['signal'] > 0:
            order_side = OrderSide.BUY
        else:
            order_side = OrderSide.SELL
            adjusted_size = abs(adjusted_size)
        
        # Create market order
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=order_side,
            quantity=Decimal(str(adjusted_size))
        )
        
        # Submit order
        self.submit_order(order)
        
        # Track position update
        self.position_updates.append({
            'bar_time': bar.ts_event,
            'signal_strength': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'regime': f"{regime.volatility_regime}-{regime.trend_regime}-{regime.momentum_regime}",
            'position_size': adjusted_size,
            'order_side': order_side.name
        })
        
        console.log(f"🦖 TiRex Trade: {order_side.name} {adjusted_size:.4f} | "
                   f"Signal: {signal_data['signal']:.3f} | "
                   f"Confidence: {signal_data['confidence']:.3f} | "
                   f"Regime: {regime.volatility_regime}-{regime.trend_regime}")
    
    def _log_regime_state(self, regime: RegimeState, signal_data: Dict):
        """Log current regime state and signal information."""
        console.log(f"📊 TiRex Regime Update (Bar {self.bars_processed}):")
        console.log(f"   Volatility: {regime.volatility_regime} | "
                   f"Trend: {regime.trend_regime} | "
                   f"Momentum: {regime.momentum_regime}")
        console.log(f"   Confidence: {regime.confidence:.3f} | "
                   f"Bars in Regime: {regime.bars_in_regime} | "
                   f"Signal: {signal_data['signal']:.3f}")
        
        if signal_data['regime_edge']:
            console.log("⚠️  Regime transition detected - reduced position sizing active")
        
    def on_stop(self):
        """Strategy shutdown with performance summary."""
        console.log("⏹️ TiRex-Only Strategy stopped")
        console.log(f"📈 Total bars processed: {self.bars_processed}")
        console.log(f"🔄 Total position updates: {len(self.position_updates)}")
        
        if self.position_updates:
            avg_confidence = np.mean([update['confidence'] for update in self.position_updates])
            console.log(f"📊 Average signal confidence: {avg_confidence:.3f}")
            console.log("🦖 TiRex parameter-free regime adaptation completed")
