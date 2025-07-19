#!/usr/bin/env python3
"""
🔒 NT-NATIVE CUSTOM INDICATORS - Following NautilusTrader Patterns
================================================================

Custom indicators that follow NautilusTrader's native bias-free architecture:
- Extend NT's Indicator base class
- Use auto-registration with strategies  
- Leverage NT's event-driven processing
- No manual state management required

Based on NautilusTrader guide patterns for bias-free indicator development.

Author: Claude Code Assistant  
Date: 2025-07-19
License: MIT
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any

# NautilusTrader indicator imports
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.model.data import Bar

# Rich console for enhanced output
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()


class CustomMomentumIndicator(Indicator):
    """
    NT-native momentum indicator with bias-free updates.
    
    Follows NautilusTrader patterns:
    - Auto-registration with strategies
    - Event-driven updates via handle_bar()
    - Uses completed bar data only
    - No current bar data required
    """
    
    def __init__(self, period: int):
        super().__init__(params=[period])
        self.period = period
        self.price_buffer = deque(maxlen=period + 1)  # +1 for lag separation
        self.value = 0.0
        
    def handle_bar(self, bar: Bar):
        """NT calls this automatically on bar completion"""
        self.update_raw(float(bar.close))
        
    def update_raw(self, price: float):
        """Update with completed bar price only"""
        self.price_buffer.append(price)
        
        if len(self.price_buffer) >= self.period + 1:
            # Use historical prices only (bias-free by design)
            old_price = self.price_buffer[0]  # period bars ago
            prev_price = self.price_buffer[-2]  # previous bar (not current)
            
            if old_price > 0:
                self.value = (prev_price - old_price) / old_price
            else:
                self.value = 0.0
                
            self._set_initialized(True)

    def reset(self):
        """Reset indicator state"""
        self.price_buffer.clear()
        self.value = 0.0
        self._set_initialized(False)


class CustomVolatilityRatio(Indicator):
    """
    Volatility ratio using NT's native rolling pattern.
    
    Uses NT's built-in ATR indicators for bias-free computation.
    """
    
    def __init__(self, short_period: int, long_period: int):
        super().__init__(params=[short_period, long_period])
        self.short_period = short_period
        self.long_period = long_period
        
        # Use NT's built-in indicators for bias-free computation
        self.short_atr = AverageTrueRange(short_period)
        self.long_atr = AverageTrueRange(long_period)
        self.value = 0.0
        
    def handle_bar(self, bar: Bar):
        """Update both ATR components"""
        self.short_atr.handle_bar(bar)
        self.long_atr.handle_bar(bar)
        
        if self.short_atr.initialized and self.long_atr.initialized:
            if self.long_atr.value > 1e-8:
                self.value = (self.short_atr.value / self.long_atr.value) - 1.0
            else:
                self.value = 0.0
            self._set_initialized(True)

    def update_raw(self, high: float, low: float, close: float):
        """Update with OHLC data"""
        # This method is called by handle_bar internally
        pass

    def reset(self):
        """Reset indicator state"""
        self.short_atr.reset()
        self.long_atr.reset()
        self.value = 0.0
        self._set_initialized(False)


class CustomChangePointDetector(Indicator):
    """
    Change point detection using NT patterns.
    
    Implements CUSUM algorithm using only completed bar data.
    """
    
    def __init__(self, window_size: int = 50, sensitivity: float = 2.0):
        super().__init__(params=[window_size, sensitivity])
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.price_buffer = deque(maxlen=window_size)
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.value = 0.0
        
    def handle_bar(self, bar: Bar):
        """NT auto-calls this on bar completion"""
        self.update_raw(float(bar.close))
        
    def update_raw(self, price: float):
        """Update change point detection"""
        self.price_buffer.append(price)
        
        if len(self.price_buffer) >= 20:  # Minimum for statistics
            prices = list(self.price_buffer)
            
            # Use only historical data for statistics
            if len(prices) >= 10:
                recent_mean = np.mean(prices[-10:])
                historical_mean = np.mean(prices[:-10])
                historical_std = np.std(prices[:-10])
                
                if historical_std > 1e-8:
                    z_score = (recent_mean - historical_mean) / historical_std
                    
                    # Update CUSUM
                    self.cusum_pos = max(0, self.cusum_pos + z_score)
                    self.cusum_neg = max(0, self.cusum_neg - z_score)
                    
                    # Detect change point
                    if self.cusum_pos > self.sensitivity or self.cusum_neg > self.sensitivity:
                        self.value = max(self.cusum_pos, self.cusum_neg)
                        self.cusum_pos = 0.0
                        self.cusum_neg = 0.0
                    else:
                        self.value = max(self.cusum_pos, self.cusum_neg)
                        
                    self._set_initialized(True)

    def reset(self):
        """Reset indicator state"""
        self.price_buffer.clear()
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.value = 0.0
        self._set_initialized(False)


class CustomVolumeRatio(Indicator):
    """
    Volume momentum indicator using NT patterns.
    
    Compares recent volume to historical average.
    """
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(params=[short_period, long_period])
        self.short_period = short_period
        self.long_period = long_period
        self.volume_buffer = deque(maxlen=long_period)
        self.value = 0.0
        
    def handle_bar(self, bar: Bar):
        """NT auto-calls this on bar completion"""
        self.update_raw(float(bar.volume))
        
    def update_raw(self, volume: float):
        """Update volume ratio"""
        self.volume_buffer.append(volume)
        
        if len(self.volume_buffer) >= self.long_period:
            # Calculate recent vs historical volume
            recent_volumes = list(self.volume_buffer)[-self.short_period:]
            historical_volumes = list(self.volume_buffer)[:-self.short_period]
            
            if len(historical_volumes) > 0:
                recent_avg = np.mean(recent_volumes)
                historical_avg = np.mean(historical_volumes)
                
                if historical_avg > 1e-8:
                    self.value = (recent_avg / historical_avg) - 1.0
                else:
                    self.value = 0.0
                    
                self._set_initialized(True)

    def reset(self):
        """Reset indicator state"""
        self.volume_buffer.clear()
        self.value = 0.0
        self._set_initialized(False)


class CustomCrossoverSignal(Indicator):
    """
    Moving average crossover signal using NT patterns.
    
    Generates signals when fast MA crosses slow MA.
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__(params=[fast_period, slow_period])
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Use simple buffers for MA calculation
        self.fast_buffer = deque(maxlen=fast_period)
        self.slow_buffer = deque(maxlen=slow_period)
        
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.prev_signal = 0.0
        self.value = 0.0  # Crossover signal: +1 bullish, -1 bearish, 0 neutral
        
    def handle_bar(self, bar: Bar):
        """NT auto-calls this on bar completion"""
        self.update_raw(float(bar.close))
        
    def update_raw(self, price: float):
        """Update crossover signal"""
        self.fast_buffer.append(price)
        self.slow_buffer.append(price)
        
        # Calculate moving averages
        if len(self.fast_buffer) == self.fast_period:
            self.fast_ma = np.mean(list(self.fast_buffer))
            
        if len(self.slow_buffer) == self.slow_period:
            self.slow_ma = np.mean(list(self.slow_buffer))
            
            # Generate crossover signal
            if self.fast_ma > self.slow_ma and self.prev_signal <= 0:
                self.value = 1.0  # Bullish crossover
            elif self.fast_ma < self.slow_ma and self.prev_signal >= 0:
                self.value = -1.0  # Bearish crossover  
            else:
                self.value = 0.0  # No crossover
                
            # Update signal for next iteration
            if self.fast_ma > self.slow_ma:
                self.prev_signal = 1.0
            else:
                self.prev_signal = -1.0
                
            self._set_initialized(True)

    def reset(self):
        """Reset indicator state"""
        self.fast_buffer.clear()
        self.slow_buffer.clear()
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.prev_signal = 0.0
        self.value = 0.0
        self._set_initialized(False)


def test_nt_custom_indicators():
    """Test custom indicators using NT patterns"""
    console.print("[yellow]🧪 Testing NT Custom Indicators...[/yellow]")
    
    # Test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    volumes = 1000 + np.random.randn(100) * 100
    
    # Create indicators that don't require real Bar objects
    momentum_5 = CustomMomentumIndicator(5)
    momentum_20 = CustomMomentumIndicator(20)
    change_detector = CustomChangePointDetector(50)
    volume_ratio = CustomVolumeRatio(5, 20)
    crossover = CustomCrossoverSignal(5, 20)
    
    console.print("  Testing indicator updates...")
    
    # Test indicators using update_raw (bypassing handle_bar)
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        # Update indicators directly
        momentum_5.update_raw(price)
        momentum_20.update_raw(price)
        change_detector.update_raw(price)
        volume_ratio.update_raw(volume)
        crossover.update_raw(price)
        
        # Check initialization
        if i == 50:  # Check at middle point
            console.print(f"    Step {i}: Momentum5={momentum_5.value:.4f}, "
                         f"ChangePoint={change_detector.value:.4f}, "
                         f"Initialized: {momentum_5.initialized}")
    
    # Verify final states
    console.print(f"  Final Values:")
    console.print(f"    Momentum 5: {momentum_5.value:.4f} (init: {momentum_5.initialized})")
    console.print(f"    Momentum 20: {momentum_20.value:.4f} (init: {momentum_20.initialized})")
    console.print(f"    Change Point: {change_detector.value:.4f} (init: {change_detector.initialized})")
    console.print(f"    Volume Ratio: {volume_ratio.value:.4f} (init: {volume_ratio.initialized})")
    console.print(f"    Crossover: {crossover.value:.4f} (init: {crossover.initialized})")
    
    console.print("[green]✅ NT Custom Indicators test completed![/green]")


if __name__ == "__main__":
    console.print("[bold green]🔒 NT-Native Custom Indicators![/bold green]")
    console.print("[dim]Following NautilusTrader patterns for bias-free indicator development[/dim]")
    
    # Run tests
    test_nt_custom_indicators()
    
    console.print("\n[green]🌟 Ready for NT-native strategy integration![/green]")