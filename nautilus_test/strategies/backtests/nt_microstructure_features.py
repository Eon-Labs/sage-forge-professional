#!/usr/bin/env python3
"""
📊 NT-NATIVE MARKET MICROSTRUCTURE FEATURES 2025
================================================

Advanced market microstructure analysis for high-frequency trading insights.
Follows NautilusTrader patterns for bias-free operation and real-time performance.

Features:
- Order flow imbalance indicators
- Bid-ask spread analysis and modeling
- Volume profile construction
- Market impact estimation
- Liquidity and depth analysis
- Integration with enhanced SOTA strategy

Analysis Methods:
- Order Book Imbalance (OBI) indicators
- Volume-Weighted Average Price (VWAP) deviations
- Tick-by-tick spread dynamics
- Volume at Price (VAP) profiles
- Market impact models (linear and nonlinear)
- Liquidity provision metrics

Author: Claude Code Assistant  
Date: 2025-07-19
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
import warnings
from datetime import datetime, timedelta

# NautilusTrader imports for market data structures
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.indicators.base.indicator import Indicator

# Rich console for enhanced output
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()


@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: float
    size: float
    side: str  # 'bid' or 'ask'


@dataclass
class MarketImpactMeasure:
    """Market impact measurement."""
    trade_size: float
    price_impact: float
    temporary_impact: float
    permanent_impact: float
    timestamp: datetime


class OrderFlowImbalanceIndicator(Indicator):
    """
    📈 Order Flow Imbalance (OFI) Indicator
    
    Measures order flow imbalance using bid-ask volume dynamics.
    Follows NT patterns for bias-free computation.
    """
    
    def __init__(self, window_size: int = 20):
        super().__init__(params=[window_size])
        self.window_size = window_size
        
        # Order flow tracking
        self.bid_volumes = deque(maxlen=window_size)
        self.ask_volumes = deque(maxlen=window_size)
        self.trade_directions = deque(maxlen=window_size)
        
        # Current values
        self.ofi_value = 0.0
        self.volume_imbalance = 0.0
        self.trade_direction_bias = 0.0
        
    def handle_quote_tick(self, quote: QuoteTick):
        """Update with quote tick data."""
        bid_size = float(quote.bid_size)
        ask_size = float(quote.ask_size)
        
        self.bid_volumes.append(bid_size)
        self.ask_volumes.append(ask_size)
        
        self._compute_ofi()
    
    def handle_trade_tick(self, trade: TradeTick):
        """Update with trade tick data."""
        # Classify trade direction (simplified)
        # In practice, would use more sophisticated classification
        direction = 1 if trade.aggressor_side.name == 'BUY' else -1
        self.trade_directions.append(direction)
        
        self._compute_ofi()
    
    def _compute_ofi(self):
        """Compute order flow imbalance metrics."""
        if len(self.bid_volumes) < 5 or len(self.ask_volumes) < 5:
            return
        
        # Volume imbalance
        recent_bid_vol = np.sum(list(self.bid_volumes)[-10:])
        recent_ask_vol = np.sum(list(self.ask_volumes)[-10:])
        
        total_volume = recent_bid_vol + recent_ask_vol
        if total_volume > 0:
            self.volume_imbalance = (recent_bid_vol - recent_ask_vol) / total_volume
        
        # Trade direction bias
        if len(self.trade_directions) >= 5:
            self.trade_direction_bias = np.mean(list(self.trade_directions)[-10:])
        
        # Combined OFI
        self.ofi_value = 0.6 * self.volume_imbalance + 0.4 * self.trade_direction_bias
        
        if not self.initialized:
            self._set_initialized(True)
    
    def reset(self):
        """Reset indicator state."""
        self.bid_volumes.clear()
        self.ask_volumes.clear()
        self.trade_directions.clear()
        self.ofi_value = 0.0
        self.volume_imbalance = 0.0
        self.trade_direction_bias = 0.0
        self._set_initialized(False)


class BidAskSpreadAnalyzer(Indicator):
    """
    📏 Bid-Ask Spread Analysis Indicator
    
    Analyzes spread dynamics and provides liquidity insights.
    """
    
    def __init__(self, window_size: int = 50):
        super().__init__(params=[window_size])
        self.window_size = window_size
        
        # Spread tracking
        self.spreads = deque(maxlen=window_size)
        self.relative_spreads = deque(maxlen=window_size)
        self.mid_prices = deque(maxlen=window_size)
        
        # Current metrics
        self.current_spread = 0.0
        self.avg_spread = 0.0
        self.spread_volatility = 0.0
        self.relative_spread = 0.0
        self.liquidity_score = 0.0
        
    def handle_quote_tick(self, quote: QuoteTick):
        """Update with quote tick data."""
        bid_price = float(quote.bid_price)
        ask_price = float(quote.ask_price)
        
        if bid_price > 0 and ask_price > 0 and ask_price > bid_price:
            spread = ask_price - bid_price
            mid_price = (bid_price + ask_price) / 2.0
            relative_spread = spread / mid_price if mid_price > 0 else 0.0
            
            self.spreads.append(spread)
            self.relative_spreads.append(relative_spread)
            self.mid_prices.append(mid_price)
            
            self._compute_spread_metrics()
    
    def _compute_spread_metrics(self):
        """Compute spread-based metrics."""
        if len(self.spreads) < 10:
            return
        
        spreads_array = np.array(list(self.spreads))
        rel_spreads_array = np.array(list(self.relative_spreads))
        
        # Basic spread metrics
        self.current_spread = spreads_array[-1]
        self.avg_spread = np.mean(spreads_array)
        self.spread_volatility = np.std(spreads_array)
        self.relative_spread = rel_spreads_array[-1]
        
        # Liquidity score (inverse of relative spread)
        avg_rel_spread = np.mean(rel_spreads_array)
        self.liquidity_score = 1.0 / (1.0 + avg_rel_spread * 1000)  # Scale for practical range
        
        if not self.initialized:
            self._set_initialized(True)
    
    def get_spread_regime(self) -> str:
        """Classify current spread regime."""
        if not self.initialized or len(self.relative_spreads) < 20:
            return "UNKNOWN"
        
        current_rel = self.relative_spread
        recent_avg = np.mean(list(self.relative_spreads)[-20:])
        
        if current_rel > recent_avg * 1.5:
            return "WIDE_SPREAD"  # Low liquidity
        elif current_rel < recent_avg * 0.7:
            return "TIGHT_SPREAD"  # High liquidity
        else:
            return "NORMAL_SPREAD"
    
    def reset(self):
        """Reset analyzer state."""
        self.spreads.clear()
        self.relative_spreads.clear()
        self.mid_prices.clear()
        self.current_spread = 0.0
        self.avg_spread = 0.0
        self.spread_volatility = 0.0
        self.relative_spread = 0.0
        self.liquidity_score = 0.0
        self._set_initialized(False)


class VolumeProfileIndicator(Indicator):
    """
    📊 Volume Profile Indicator
    
    Constructs volume at price (VAP) profiles for support/resistance analysis.
    """
    
    def __init__(self, window_size: int = 100, price_levels: int = 20):
        super().__init__(params=[window_size, price_levels])
        self.window_size = window_size
        self.price_levels = price_levels
        
        # Volume profile data
        self.trade_data = deque(maxlen=window_size)
        self.volume_profile = {}  # price_level -> volume
        self.value_area_high = 0.0
        self.value_area_low = 0.0
        self.poc_price = 0.0  # Point of Control (highest volume)
        
        # Profile metrics
        self.volume_imbalance_ratio = 0.0
        self.profile_skewness = 0.0
        
    def handle_trade_tick(self, trade: TradeTick):
        """Update with trade tick data."""
        price = float(trade.price)
        volume = float(trade.size)
        
        self.trade_data.append({
            'price': price,
            'volume': volume,
            'timestamp': trade.ts_init
        })
        
        self._compute_volume_profile()
    
    def handle_bar(self, bar: Bar):
        """Update with bar data."""
        # Use OHLC and volume to estimate volume distribution
        ohlc_prices = [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]
        volume_per_level = float(bar.volume) / 4  # Distribute equally across OHLC
        
        for price in ohlc_prices:
            self.trade_data.append({
                'price': price,
                'volume': volume_per_level,
                'timestamp': bar.ts_init
            })
        
        self._compute_volume_profile()
    
    def _compute_volume_profile(self):
        """Compute volume at price profile."""
        if len(self.trade_data) < 10:
            return
        
        # Get price range
        prices = [t['price'] for t in self.trade_data]
        min_price = min(prices)
        max_price = max(prices)
        
        if max_price <= min_price:
            return
        
        # Create price levels
        price_range = max_price - min_price
        level_size = price_range / self.price_levels
        
        # Initialize profile
        self.volume_profile = {}
        for i in range(self.price_levels):
            level_price = min_price + (i + 0.5) * level_size
            self.volume_profile[level_price] = 0.0
        
        # Aggregate volume by price level
        for trade in self.trade_data:
            price = trade['price']
            volume = trade['volume']
            
            # Find nearest price level
            level_index = int((price - min_price) / level_size)
            level_index = max(0, min(level_index, self.price_levels - 1))
            
            level_price = min_price + (level_index + 0.5) * level_size
            self.volume_profile[level_price] += volume
        
        # Compute profile metrics
        self._compute_profile_metrics()
        
        if not self.initialized:
            self._set_initialized(True)
    
    def _compute_profile_metrics(self):
        """Compute volume profile metrics."""
        if not self.volume_profile:
            return
        
        volumes = list(self.volume_profile.values())
        prices = list(self.volume_profile.keys())
        total_volume = sum(volumes)
        
        if total_volume == 0:
            return
        
        # Point of Control (POC) - price with highest volume
        max_volume_idx = np.argmax(volumes)
        self.poc_price = prices[max_volume_idx]
        
        # Value Area (70% of volume)
        sorted_indices = np.argsort(volumes)[::-1]  # Descending order
        cumulative_volume = 0.0
        value_area_prices = []
        
        for idx in sorted_indices:
            cumulative_volume += volumes[idx]
            value_area_prices.append(prices[idx])
            
            if cumulative_volume >= 0.7 * total_volume:
                break
        
        if value_area_prices:
            self.value_area_high = max(value_area_prices)
            self.value_area_low = min(value_area_prices)
        
        # Volume imbalance (upper vs lower half)
        mid_price = (max(prices) + min(prices)) / 2
        upper_volume = sum(v for p, v in self.volume_profile.items() if p > mid_price)
        lower_volume = sum(v for p, v in self.volume_profile.items() if p <= mid_price)
        
        total_half_volume = upper_volume + lower_volume
        if total_half_volume > 0:
            self.volume_imbalance_ratio = (upper_volume - lower_volume) / total_half_volume
        
        # Profile skewness
        weighted_prices = np.array([p * v for p, v in self.volume_profile.items()])
        mean_price = sum(weighted_prices) / total_volume if total_volume > 0 else 0.0
        
        variance = sum(v * (p - mean_price)**2 for p, v in self.volume_profile.items()) / total_volume
        std_dev = np.sqrt(variance) if variance > 0 else 0.0
        
        if std_dev > 0:
            skewness_num = sum(v * (p - mean_price)**3 for p, v in self.volume_profile.items()) / total_volume
            self.profile_skewness = skewness_num / (std_dev ** 3)
    
    def get_support_resistance_levels(self, num_levels: int = 3) -> Dict[str, List[float]]:
        """Get key support and resistance levels from volume profile."""
        if not self.volume_profile:
            return {"support": [], "resistance": []}
        
        # Sort by volume (descending)
        sorted_levels = sorted(self.volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        # Get top volume levels
        key_levels = [price for price, volume in sorted_levels[:num_levels]]
        
        # Classify as support or resistance based on current price context
        current_price = self.poc_price
        support_levels = [p for p in key_levels if p < current_price]
        resistance_levels = [p for p in key_levels if p >= current_price]
        
        return {
            "support": sorted(support_levels, reverse=True)[:num_levels//2 + 1],
            "resistance": sorted(resistance_levels)[:num_levels//2 + 1]
        }
    
    def reset(self):
        """Reset indicator state."""
        self.trade_data.clear()
        self.volume_profile.clear()
        self.value_area_high = 0.0
        self.value_area_low = 0.0
        self.poc_price = 0.0
        self.volume_imbalance_ratio = 0.0
        self.profile_skewness = 0.0
        self._set_initialized(False)


class MarketImpactEstimator:
    """
    💥 Market Impact Estimator
    
    Estimates market impact of trades using various models.
    """
    
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        
        # Impact measurement data
        self.impact_measurements = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size)
        
        # Model parameters
        self.linear_impact_coeff = 0.0
        self.sqrt_impact_coeff = 0.0
        self.temporary_impact_decay = 0.5
        
        # Current estimates
        self.current_impact = 0.0
        self.avg_impact = 0.0
        self.impact_volatility = 0.0
        
    def update_trade(self, trade_size: float, price_before: float, 
                    price_after: float, price_settled: float, 
                    timestamp: datetime):
        """Update with new trade impact measurement."""
        
        # Calculate impact metrics
        immediate_impact = abs(price_after - price_before)
        temporary_impact = abs(price_after - price_settled)
        permanent_impact = abs(price_settled - price_before)
        
        # Normalize by trade size
        if trade_size > 0:
            normalized_impact = immediate_impact / np.sqrt(trade_size)
        else:
            normalized_impact = 0.0
        
        impact_measure = MarketImpactMeasure(
            trade_size=trade_size,
            price_impact=immediate_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            timestamp=timestamp
        )
        
        self.impact_measurements.append(impact_measure)
        
        # Update model parameters
        self._update_impact_model()
    
    def _update_impact_model(self):
        """Update impact model parameters."""
        if len(self.impact_measurements) < 10:
            return
        
        # Extract data for model fitting
        trade_sizes = []
        price_impacts = []
        
        for measure in self.impact_measurements:
            if measure.trade_size > 0:
                trade_sizes.append(measure.trade_size)
                price_impacts.append(measure.price_impact)
        
        if len(trade_sizes) < 5:
            return
        
        trade_sizes = np.array(trade_sizes)
        price_impacts = np.array(price_impacts)
        
        # Fit linear model: impact = alpha * size
        if np.var(trade_sizes) > 1e-6:
            self.linear_impact_coeff = np.cov(trade_sizes, price_impacts)[0, 1] / np.var(trade_sizes)
        
        # Fit square root model: impact = beta * sqrt(size)
        sqrt_sizes = np.sqrt(trade_sizes)
        if np.var(sqrt_sizes) > 1e-6:
            self.sqrt_impact_coeff = np.cov(sqrt_sizes, price_impacts)[0, 1] / np.var(sqrt_sizes)
        
        # Update current estimates
        recent_impacts = [m.price_impact for m in list(self.impact_measurements)[-20:]]
        self.current_impact = recent_impacts[-1] if recent_impacts else 0.0
        self.avg_impact = np.mean(recent_impacts) if recent_impacts else 0.0
        self.impact_volatility = np.std(recent_impacts) if len(recent_impacts) > 1 else 0.0
    
    def predict_impact(self, trade_size: float, model: str = "sqrt") -> float:
        """Predict market impact for given trade size."""
        if model == "linear":
            return self.linear_impact_coeff * trade_size
        elif model == "sqrt":
            return self.sqrt_impact_coeff * np.sqrt(trade_size)
        else:
            # Combined model
            linear_part = self.linear_impact_coeff * trade_size
            sqrt_part = self.sqrt_impact_coeff * np.sqrt(trade_size)
            return 0.5 * linear_part + 0.5 * sqrt_part
    
    def get_optimal_trade_size(self, max_impact_tolerance: float) -> float:
        """Get optimal trade size given impact tolerance."""
        if self.sqrt_impact_coeff <= 0:
            return float('inf')
        
        # Solve: sqrt_impact_coeff * sqrt(size) = max_impact_tolerance
        optimal_size = (max_impact_tolerance / self.sqrt_impact_coeff) ** 2
        return optimal_size
    
    def get_impact_summary(self) -> Dict[str, Any]:
        """Get market impact analysis summary."""
        return {
            "measurement_count": len(self.impact_measurements),
            "linear_impact_coeff": self.linear_impact_coeff,
            "sqrt_impact_coeff": self.sqrt_impact_coeff,
            "current_impact": self.current_impact,
            "avg_impact": self.avg_impact,
            "impact_volatility": self.impact_volatility,
            "temporary_impact_decay": self.temporary_impact_decay
        }


class MicrostructureFeatureExtractor:
    """
    🔬 Comprehensive Microstructure Feature Extractor
    
    Combines all microstructure indicators for strategy integration.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Initialize all indicators
        self.ofi_indicator = OrderFlowImbalanceIndicator(window_size)
        self.spread_analyzer = BidAskSpreadAnalyzer(window_size)
        self.volume_profile = VolumeProfileIndicator(window_size)
        self.impact_estimator = MarketImpactEstimator(window_size)
        
        # Feature tracking
        self.feature_history = deque(maxlen=window_size)
        self.update_count = 0
        
        console.print(f"[green]🔬 Microstructure Feature Extractor initialized[/green]")
        console.print(f"[cyan]  • Window size: {window_size}[/cyan]")
        console.print(f"[cyan]  • Features: OFI, Spread, Volume Profile, Market Impact[/cyan]")
    
    def update_quote(self, quote: QuoteTick):
        """Update with quote tick."""
        self.ofi_indicator.handle_quote_tick(quote)
        self.spread_analyzer.handle_quote_tick(quote)
        
        self._extract_features()
    
    def update_trade(self, trade: TradeTick):
        """Update with trade tick."""
        self.ofi_indicator.handle_trade_tick(trade)
        self.volume_profile.handle_trade_tick(trade)
        
        self._extract_features()
    
    def update_bar(self, bar: Bar):
        """Update with bar data."""
        self.volume_profile.handle_bar(bar)
        
        self._extract_features()
    
    def _extract_features(self):
        """Extract comprehensive microstructure features."""
        features = {}
        
        # Order Flow Imbalance features
        if self.ofi_indicator.initialized:
            features.update({
                "ofi_value": self.ofi_indicator.ofi_value,
                "volume_imbalance": self.ofi_indicator.volume_imbalance,
                "trade_direction_bias": self.ofi_indicator.trade_direction_bias
            })
        else:
            features.update({
                "ofi_value": 0.0,
                "volume_imbalance": 0.0,
                "trade_direction_bias": 0.0
            })
        
        # Spread features
        if self.spread_analyzer.initialized:
            features.update({
                "relative_spread": self.spread_analyzer.relative_spread,
                "spread_volatility": self.spread_analyzer.spread_volatility,
                "liquidity_score": self.spread_analyzer.liquidity_score,
                "spread_regime": self.spread_analyzer.get_spread_regime()
            })
        else:
            features.update({
                "relative_spread": 0.0,
                "spread_volatility": 0.0,
                "liquidity_score": 0.5,
                "spread_regime": "UNKNOWN"
            })
        
        # Volume Profile features
        if self.volume_profile.initialized:
            features.update({
                "volume_imbalance_ratio": self.volume_profile.volume_imbalance_ratio,
                "profile_skewness": self.volume_profile.profile_skewness,
                "poc_price": self.volume_profile.poc_price,
                "value_area_range": self.volume_profile.value_area_high - self.volume_profile.value_area_low
            })
        else:
            features.update({
                "volume_imbalance_ratio": 0.0,
                "profile_skewness": 0.0,
                "poc_price": 0.0,
                "value_area_range": 0.0
            })
        
        # Market Impact features
        impact_summary = self.impact_estimator.get_impact_summary()
        features.update({
            "linear_impact_coeff": impact_summary["linear_impact_coeff"],
            "sqrt_impact_coeff": impact_summary["sqrt_impact_coeff"],
            "avg_impact": impact_summary["avg_impact"],
            "impact_volatility": impact_summary["impact_volatility"]
        })
        
        # Store features
        features["timestamp"] = datetime.now()
        self.feature_history.append(features)
        self.update_count += 1
    
    def get_feature_vector(self) -> np.ndarray:
        """Get normalized feature vector for ML models."""
        if not self.feature_history:
            return np.zeros(11)  # Default feature count
        
        latest_features = self.feature_history[-1]
        
        # Extract numerical features
        feature_vector = [
            latest_features.get("ofi_value", 0.0),
            latest_features.get("volume_imbalance", 0.0),
            latest_features.get("trade_direction_bias", 0.0),
            latest_features.get("relative_spread", 0.0),
            latest_features.get("spread_volatility", 0.0),
            latest_features.get("liquidity_score", 0.5),
            latest_features.get("volume_imbalance_ratio", 0.0),
            latest_features.get("profile_skewness", 0.0),
            latest_features.get("linear_impact_coeff", 0.0),
            latest_features.get("sqrt_impact_coeff", 0.0),
            latest_features.get("avg_impact", 0.0)
        ]
        
        # Normalize and clip
        feature_vector = np.array(feature_vector)
        feature_vector = np.clip(feature_vector, -10.0, 10.0)
        feature_vector = np.tanh(feature_vector)  # Soft normalization
        
        return feature_vector
    
    def get_trading_signals(self) -> Dict[str, Any]:
        """Get actionable trading signals from microstructure analysis."""
        if not self.feature_history:
            return {"signal_strength": 0.0, "direction": "NEUTRAL", "confidence": 0.0}
        
        latest = self.feature_history[-1]
        
        # Signal components
        signals = []
        
        # Order flow signal
        ofi_signal = latest.get("ofi_value", 0.0)
        if abs(ofi_signal) > 0.3:
            signals.append(("OFI", ofi_signal, 0.8))
        
        # Liquidity signal
        liquidity = latest.get("liquidity_score", 0.5)
        if liquidity > 0.7:
            signals.append(("HIGH_LIQUIDITY", 0.5, 0.6))
        elif liquidity < 0.3:
            signals.append(("LOW_LIQUIDITY", -0.3, 0.7))
        
        # Volume profile signal
        vol_imbalance = latest.get("volume_imbalance_ratio", 0.0)
        if abs(vol_imbalance) > 0.4:
            signals.append(("VOLUME_IMBALANCE", vol_imbalance, 0.7))
        
        # Combine signals
        if not signals:
            return {"signal_strength": 0.0, "direction": "NEUTRAL", "confidence": 0.0}
        
        weighted_signal = sum(signal * confidence for _, signal, confidence in signals)
        total_confidence = sum(confidence for _, _, confidence in signals)
        
        final_signal = weighted_signal / total_confidence if total_confidence > 0 else 0.0
        
        direction = "BUY" if final_signal > 0.1 else "SELL" if final_signal < -0.1 else "NEUTRAL"
        
        return {
            "signal_strength": abs(final_signal),
            "direction": direction,
            "confidence": min(1.0, total_confidence / len(signals)),
            "component_signals": [{"type": t, "value": s, "confidence": c} for t, s, c in signals]
        }
    
    def reset(self):
        """Reset all indicators."""
        self.ofi_indicator.reset()
        self.spread_analyzer.reset()
        self.volume_profile.reset()
        self.feature_history.clear()
        self.update_count = 0


def test_microstructure_features():
    """Test microstructure feature extraction."""
    console.print("[yellow]🧪 Testing Microstructure Features...[/yellow]")
    
    # Initialize extractor
    extractor = MicrostructureFeatureExtractor(window_size=50)
    
    # Simulate market data
    np.random.seed(42)
    base_price = 100.0
    
    console.print("  Simulating market microstructure data...")
    
    for i in range(100):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01)
        base_price += price_change
        
        # Simulate spread
        spread = max(0.01, abs(np.random.normal(0.05, 0.02)))
        bid_price = base_price - spread/2
        ask_price = base_price + spread/2
        
        # Create mock quote tick
        class MockQuote:
            def __init__(self, bid_price, ask_price, bid_size, ask_size):
                self.bid_price = bid_price
                self.ask_price = ask_price
                self.bid_size = bid_size
                self.ask_size = ask_size
        
        # Simulate order book
        bid_size = max(1.0, np.random.exponential(10.0))
        ask_size = max(1.0, np.random.exponential(10.0))
        
        quote = MockQuote(bid_price, ask_price, bid_size, ask_size)
        extractor.update_quote(quote)
        
        # Simulate trade
        if i % 3 == 0:  # Trade every 3 quotes
            trade_price = base_price + np.random.normal(0, spread/4)
            trade_size = max(0.1, np.random.exponential(5.0))
            
            class MockTrade:
                def __init__(self, price, size):
                    self.price = price
                    self.size = size
                    self.ts_init = i
                    self.aggressor_side = type('obj', (object,), {'name': 'BUY' if np.random.random() > 0.5 else 'SELL'})()
            
            trade = MockTrade(trade_price, trade_size)
            extractor.update_trade(trade)
        
        # Log progress
        if i % 25 == 0:
            features = extractor.get_feature_vector()
            signals = extractor.get_trading_signals()
            console.print(f"    Step {i}: Features={len(features)}, Signal={signals['direction']} "
                         f"({signals['signal_strength']:.3f})")
    
    # Final analysis
    final_features = extractor.get_feature_vector()
    final_signals = extractor.get_trading_signals()
    
    console.print(f"  Final Results:")
    console.print(f"    Feature vector size: {len(final_features)}")
    console.print(f"    Feature range: [{np.min(final_features):.3f}, {np.max(final_features):.3f}]")
    console.print(f"    Trading signal: {final_signals['direction']} "
                 f"(strength: {final_signals['signal_strength']:.3f}, "
                 f"confidence: {final_signals['confidence']:.3f})")
    
    if final_signals['component_signals']:
        console.print(f"    Signal components:")
        for comp in final_signals['component_signals']:
            console.print(f"      {comp['type']}: {comp['value']:.3f} (conf: {comp['confidence']:.3f})")
    
    console.print("[green]✅ Microstructure features test completed![/green]")
    
    return extractor


if __name__ == "__main__":
    console.print("[bold green]📊 NT-Native Market Microstructure Features![/bold green]")
    console.print("[dim]Advanced microstructure analysis for high-frequency insights[/dim]")
    
    # Test the microstructure system
    extractor = test_microstructure_features()
    
    console.print("\n[green]🌟 Ready for integration with enhanced SOTA strategy![/green]")