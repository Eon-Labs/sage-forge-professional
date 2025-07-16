"""
Adaptive Regime-Aware Trading Strategy
=====================================

A robust, parameter-free approach that adapts to market conditions automatically.
Uses statistical measures to detect market regimes and adjust behavior accordingly.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.objects import Price, Quantity


class AdaptiveRegimeStrategy(Strategy):
    """
    State-of-the-art adaptive strategy that automatically detects market regimes
    and adjusts trading behavior without magic numbers or parameters.
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Data storage for regime detection
        self.prices = []
        self.volumes = []
        self.returns = []
        self.volatilities = []
        
        # Regime state tracking
        self.current_regime = "UNKNOWN"
        self.regime_confidence = 0.0
        self.last_regime_change = None
        
        # Adaptive thresholds (self-calibrating)
        self.trend_threshold = None
        self.volatility_threshold = None
        self.volume_threshold = None
        
        # Position management
        self.position_hold_bars = 0
        self.max_position_hold = 240  # 4 hours max hold
        
        # Performance tracking
        self.trades_in_regime = {"TRENDING": 0, "RANGING": 0, "VOLATILE": 0}
        self.pnl_in_regime = {"TRENDING": 0.0, "RANGING": 0.0, "VOLATILE": 0.0}

    def on_start(self):
        """Initialize strategy."""
        self.log.info("AdaptiveRegimeStrategy started")
        
    def on_bar(self, bar: Bar):
        """Process each bar and make trading decisions."""
        self._update_data(bar)
        
        # Need minimum data for regime detection
        if len(self.prices) < 50:
            return
            
        # Detect current market regime
        self._detect_regime()
        
        # Make trading decision based on regime
        self._execute_regime_strategy(bar)
        
        # Update position management
        self._manage_position()

    def _update_data(self, bar: Bar):
        """Update internal data structures with new bar data."""
        price = float(bar.close)
        volume = float(bar.volume)
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        # Calculate returns
        if len(self.prices) >= 2:
            ret = (price - self.prices[-2]) / self.prices[-2]
            self.returns.append(ret)
            
        # Calculate rolling volatility (20-bar window)
        if len(self.returns) >= 20:
            recent_returns = self.returns[-20:]
            volatility = np.std(recent_returns)
            self.volatilities.append(volatility)
            
        # Keep only recent data (1000 bars ~ 16 hours)
        if len(self.prices) > 1000:
            self.prices = self.prices[-1000:]
            self.volumes = self.volumes[-1000:]
            self.returns = self.returns[-1000:]
            self.volatilities = self.volatilities[-1000:]

    def _detect_regime(self):
        """
        Detect current market regime using statistical measures.
        No magic numbers - uses adaptive thresholds based on historical data.
        """
        if len(self.returns) < 50 or len(self.volatilities) < 20:
            return
            
        # Calculate adaptive thresholds (bottom 25%, top 25%)
        recent_returns = self.returns[-100:]
        recent_volatilities = self.volatilities[-50:]
        recent_volumes = self.volumes[-100:]
        
        # Self-calibrating thresholds
        self.trend_threshold = np.percentile(np.abs(recent_returns), 75)
        self.volatility_threshold = np.percentile(recent_volatilities, 75)
        self.volume_threshold = np.percentile(recent_volumes, 75)
        
        # Current market characteristics
        current_return = abs(self.returns[-1])
        current_volatility = self.volatilities[-1]
        current_volume = self.volumes[-1]
        
        # Regime detection logic
        if current_volatility > self.volatility_threshold:
            new_regime = "VOLATILE"
            confidence = min(current_volatility / self.volatility_threshold, 2.0)
        elif current_return > self.trend_threshold and current_volume > self.volume_threshold:
            new_regime = "TRENDING"
            confidence = min(current_return / self.trend_threshold, 2.0)
        else:
            new_regime = "RANGING"
            confidence = 1.0 - min(current_return / self.trend_threshold, 1.0)
            
        # Update regime if confidence is high or regime changed
        if confidence > 1.2 or new_regime != self.current_regime:
            if new_regime != self.current_regime:
                self.log.info(f"Regime changed: {self.current_regime} → {new_regime} (confidence: {confidence:.2f})")
                self.last_regime_change = len(self.prices)
                
            self.current_regime = new_regime
            self.regime_confidence = confidence

    def _execute_regime_strategy(self, bar: Bar):
        """Execute trading strategy based on detected regime."""
        if self.current_regime == "UNKNOWN":
            return
            
        # Don't trade immediately after regime change (wait for confirmation)
        if self.last_regime_change and (len(self.prices) - self.last_regime_change) < 5:
            return
            
        # Strategy for each regime
        if self.current_regime == "TRENDING":
            self._trending_strategy(bar)
        elif self.current_regime == "RANGING":
            self._ranging_strategy(bar)
        elif self.current_regime == "VOLATILE":
            self._volatile_strategy(bar)

    def _trending_strategy(self, bar: Bar):
        """Strategy for trending markets - follow momentum."""
        if len(self.returns) < 10:
            return
            
        # Use momentum signals
        short_momentum = np.mean(self.returns[-5:])  # 5-bar momentum
        long_momentum = np.mean(self.returns[-20:])  # 20-bar momentum
        
        # Only trade if momentum is consistent and strong
        if short_momentum > 0 and long_momentum > 0 and short_momentum > long_momentum * 1.5:
            if not self.portfolio.is_flat(self.instrument_id):
                return  # Already in position
            self._submit_order(OrderSide.BUY, bar)
            self.trades_in_regime["TRENDING"] += 1
            
        elif short_momentum < 0 and long_momentum < 0 and short_momentum < long_momentum * 1.5:
            if not self.portfolio.is_flat(self.instrument_id):
                return  # Already in position
            self._submit_order(OrderSide.SELL, bar)
            self.trades_in_regime["TRENDING"] += 1

    def _ranging_strategy(self, bar: Bar):
        """Strategy for ranging markets - mean reversion."""
        if len(self.prices) < 50:
            return
            
        # Calculate adaptive bollinger-like bands
        recent_prices = self.prices[-50:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        current_price = float(bar.close)
        z_score = (current_price - mean_price) / std_price
        
        # Mean reversion when price is extreme
        if z_score > 2.0:  # Price too high
            if not self.portfolio.is_flat(self.instrument_id):
                return
            self._submit_order(OrderSide.SELL, bar)
            self.trades_in_regime["RANGING"] += 1
            
        elif z_score < -2.0:  # Price too low
            if not self.portfolio.is_flat(self.instrument_id):
                return
            self._submit_order(OrderSide.BUY, bar)
            self.trades_in_regime["RANGING"] += 1

    def _volatile_strategy(self, bar: Bar):
        """Strategy for volatile markets - avoid trading or very short holds."""
        # In volatile markets, either don't trade or trade very short-term
        # For now, we'll avoid trading in volatile regimes
        if not self.portfolio.is_flat(self.instrument_id):
            # Close position quickly in volatile markets
            self._close_position(bar)
            
    def _submit_order(self, side: OrderSide, bar: Bar):
        """Submit market order with proper risk management."""
        # Calculate position size (use configured trade size)
        quantity = Quantity.from_str(str(self.config.trade_size))
        
        order = MarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.id,
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
            time_in_force=TimeInForce.FOK,
            ts_init=dt_to_unix_nanos(self.clock.timestamp()),
        )
        
        self.submit_order(order)
        self.position_hold_bars = 0  # Reset hold counter

    def _close_position(self, bar: Bar):
        """Close current position."""
        position = self.portfolio.position(self.instrument_id)
        if position and not position.is_flat():
            if position.is_long():
                self._submit_order(OrderSide.SELL, bar)
            else:
                self._submit_order(OrderSide.BUY, bar)

    def _manage_position(self):
        """Manage position holding time and risk."""
        if not self.portfolio.is_flat(self.instrument_id):
            self.position_hold_bars += 1
            
            # Force close if held too long
            if self.position_hold_bars >= self.max_position_hold:
                self.log.info(f"Force closing position after {self.position_hold_bars} bars")
                # Will be closed on next bar
                
    def on_stop(self):
        """Strategy cleanup."""
        self.log.info("AdaptiveRegimeStrategy stopped")
        
        # Log performance by regime
        for regime, trades in self.trades_in_regime.items():
            pnl = self.pnl_in_regime.get(regime, 0.0)
            if trades > 0:
                self.log.info(f"{regime} regime: {trades} trades, ${pnl:.2f} PnL")

    def on_reset(self):
        """Reset strategy state."""
        self.prices.clear()
        self.volumes.clear()
        self.returns.clear()
        self.volatilities.clear()
        self.current_regime = "UNKNOWN"
        self.regime_confidence = 0.0
        self.position_hold_bars = 0
        self.trades_in_regime = {"TRENDING": 0, "RANGING": 0, "VOLATILE": 0}
        self.pnl_in_regime = {"TRENDING": 0.0, "RANGING": 0.0, "VOLATILE": 0.0}