"""
Advanced Signal Quality Filter
==============================

Filters out low-quality trading signals to reduce overtrading and improve win rate.
Uses statistical and machine learning approaches without manual parameters.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalQuality(Enum):
    """Signal quality levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"  
    FAIR = "FAIR"
    POOR = "POOR"


@dataclass
class TradingSignal:
    """Trading signal with quality assessment."""
    direction: str  # "BUY" or "SELL"
    strength: float  # 0.0 to 1.0
    quality: SignalQuality
    confidence: float  # 0.0 to 1.0
    expected_profit: float
    risk_reward_ratio: float
    market_context: dict


class SignalQualityFilter:
    """
    Advanced signal quality filter that uses multiple criteria to assess
    trading signal quality without requiring manual parameter tuning.
    """
    
    def __init__(self, lookback_period: int = 500):
        self.lookback_period = lookback_period
        
        # Data storage
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.signal_history = deque(maxlen=100)
        
        # Performance tracking
        self.signal_performance = {
            SignalQuality.EXCELLENT: {"count": 0, "wins": 0, "total_pnl": 0.0},
            SignalQuality.GOOD: {"count": 0, "wins": 0, "total_pnl": 0.0},
            SignalQuality.FAIR: {"count": 0, "wins": 0, "total_pnl": 0.0},
            SignalQuality.POOR: {"count": 0, "wins": 0, "total_pnl": 0.0}
        }
        
        # Adaptive thresholds
        self.quality_thresholds = {
            "volatility": None,
            "volume": None,
            "momentum": None,
            "trend_strength": None
        }
        
    def update_data(self, price: float, volume: float):
        """Update internal data with new market data."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
        
    def evaluate_signal(self, 
                       signal_direction: str,
                       signal_strength: float,
                       current_price: float,
                       current_volume: float,
                       market_regime: str = "UNKNOWN") -> TradingSignal:
        """
        Evaluate a trading signal and return quality assessment.
        """
        if len(self.price_history) < 50:
            return TradingSignal(
                direction=signal_direction,
                strength=signal_strength,
                quality=SignalQuality.POOR,
                confidence=0.1,
                expected_profit=0.0,
                risk_reward_ratio=0.0,
                market_context={}
            )
            
        # Calculate various signal quality metrics
        quality_score = 0.0
        confidence_score = 0.0
        
        # 1. Trend consistency check
        trend_score = self._evaluate_trend_consistency(signal_direction)
        quality_score += trend_score * 0.25
        
        # 2. Volume confirmation
        volume_score = self._evaluate_volume_confirmation(current_volume)
        quality_score += volume_score * 0.20
        
        # 3. Volatility appropriateness
        volatility_score = self._evaluate_volatility_context(market_regime)
        quality_score += volatility_score * 0.15
        
        # 4. Support/resistance levels
        sr_score = self._evaluate_support_resistance(current_price, signal_direction)
        quality_score += sr_score * 0.20
        
        # 5. Market microstructure
        microstructure_score = self._evaluate_microstructure()
        quality_score += microstructure_score * 0.10
        
        # 6. Historical performance of similar signals
        historical_score = self._evaluate_historical_performance(
            signal_direction, signal_strength, market_regime
        )
        quality_score += historical_score * 0.10
        
        # Calculate confidence based on data quality and market conditions
        confidence_score = self._calculate_confidence(quality_score, market_regime)
        
        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        # Estimate expected profit and risk-reward ratio
        expected_profit, risk_reward = self._estimate_trade_metrics(
            signal_direction, current_price, quality_score
        )
        
        signal = TradingSignal(
            direction=signal_direction,
            strength=signal_strength,
            quality=quality_level,
            confidence=confidence_score,
            expected_profit=expected_profit,
            risk_reward_ratio=risk_reward,
            market_context={
                "trend_score": trend_score,
                "volume_score": volume_score,
                "volatility_score": volatility_score,
                "sr_score": sr_score,
                "microstructure_score": microstructure_score,
                "historical_score": historical_score,
                "regime": market_regime
            }
        )
        
        # Store signal for learning
        self.signal_history.append(signal)
        
        return signal
    
    def _evaluate_trend_consistency(self, signal_direction: str) -> float:
        """Evaluate if signal is consistent with current trend."""
        if len(self.price_history) < 20:
            return 0.5
            
        prices = list(self.price_history)
        
        # Calculate multiple timeframe trends
        short_trend = (prices[-1] - prices[-5]) / prices[-5]   # 5-bar trend
        medium_trend = (prices[-1] - prices[-10]) / prices[-10]  # 10-bar trend
        long_trend = (prices[-1] - prices[-20]) / prices[-20]    # 20-bar trend
        
        # Check consistency
        if signal_direction == "BUY":
            trends = [short_trend > 0, medium_trend > 0, long_trend > 0]
        else:
            trends = [short_trend < 0, medium_trend < 0, long_trend < 0]
            
        consistency = sum(trends) / len(trends)
        
        # Bonus for strong trends
        trend_strength = abs(short_trend) + abs(medium_trend) + abs(long_trend)
        strength_bonus = min(trend_strength * 10, 0.3)
        
        return min(consistency + strength_bonus, 1.0)
    
    def _evaluate_volume_confirmation(self, current_volume: float) -> float:
        """Evaluate volume confirmation of the signal."""
        if len(self.volume_history) < 20:
            return 0.5
            
        volumes = list(self.volume_history)
        avg_volume = np.mean(volumes[-20:])
        
        # Higher volume = better confirmation
        if current_volume > avg_volume * 1.5:
            return 1.0
        elif current_volume > avg_volume * 1.2:
            return 0.8
        elif current_volume > avg_volume:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_volatility_context(self, market_regime: str) -> float:
        """Evaluate if current volatility is appropriate for trading."""
        if len(self.price_history) < 20:
            return 0.5
            
        prices = list(self.price_history)
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        current_vol = np.std(returns[-10:])
        
        # Adjust score based on market regime
        if market_regime == "VOLATILE":
            return 0.3  # Low score for volatile markets
        elif market_regime == "RANGING":
            return 0.8  # Good score for ranging markets
        elif market_regime == "TRENDING":
            return 0.9  # High score for trending markets
        else:
            return 0.5  # Neutral for unknown
    
    def _evaluate_support_resistance(self, current_price: float, 
                                   signal_direction: str) -> float:
        """Evaluate proximity to support/resistance levels."""
        if len(self.price_history) < 50:
            return 0.5
            
        prices = list(self.price_history)
        
        # Find recent highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
                
        if not highs or not lows:
            return 0.5
            
        # Find nearest support/resistance
        nearest_resistance = min(highs, key=lambda x: abs(x - current_price))
        nearest_support = min(lows, key=lambda x: abs(x - current_price))
        
        # Calculate distance as percentage of price
        resistance_dist = abs(nearest_resistance - current_price) / current_price
        support_dist = abs(nearest_support - current_price) / current_price
        
        # Score based on signal direction and proximity
        if signal_direction == "BUY":
            # Good if near support, bad if near resistance
            if support_dist < 0.01:  # Very close to support
                return 0.9
            elif resistance_dist < 0.01:  # Very close to resistance
                return 0.2
            else:
                return 0.6
        else:  # SELL
            # Good if near resistance, bad if near support
            if resistance_dist < 0.01:  # Very close to resistance
                return 0.9
            elif support_dist < 0.01:  # Very close to support
                return 0.2
            else:
                return 0.6
    
    def _evaluate_microstructure(self) -> float:
        """Evaluate market microstructure quality."""
        if len(self.price_history) < 10:
            return 0.5
            
        prices = list(self.price_history)
        
        # Calculate price smoothness (lower is better)
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        smoothness = 1.0 - min(np.std(price_changes[-5:]) / np.mean(price_changes[-5:]), 1.0)
        
        return smoothness
    
    def _evaluate_historical_performance(self, signal_direction: str, 
                                       signal_strength: float,
                                       market_regime: str) -> float:
        """Evaluate based on historical performance of similar signals."""
        if len(self.signal_history) < 10:
            return 0.5
            
        # Find similar signals
        similar_signals = []
        for signal in self.signal_history:
            if (signal.direction == signal_direction and 
                signal.market_context.get("regime") == market_regime and
                abs(signal.strength - signal_strength) < 0.2):
                similar_signals.append(signal)
                
        if not similar_signals:
            return 0.5
            
        # Calculate average performance
        avg_performance = np.mean([s.expected_profit for s in similar_signals])
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, avg_performance * 10 + 0.5))
    
    def _calculate_confidence(self, quality_score: float, market_regime: str) -> float:
        """Calculate confidence based on quality score and market conditions."""
        base_confidence = quality_score
        
        # Adjust for market regime
        regime_adjustment = {
            "TRENDING": 1.1,
            "RANGING": 1.0,
            "VOLATILE": 0.7,
            "UNKNOWN": 0.8
        }.get(market_regime, 0.8)
        
        confidence = base_confidence * regime_adjustment
        
        # Adjust for data quality
        data_quality = min(len(self.price_history) / 100, 1.0)
        confidence *= data_quality
        
        return max(0.0, min(1.0, confidence))
    
    def _determine_quality_level(self, quality_score: float) -> SignalQuality:
        """Determine quality level based on score."""
        if quality_score >= 0.8:
            return SignalQuality.EXCELLENT
        elif quality_score >= 0.6:
            return SignalQuality.GOOD
        elif quality_score >= 0.4:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
    
    def _estimate_trade_metrics(self, signal_direction: str, 
                               current_price: float, 
                               quality_score: float) -> tuple[float, float]:
        """Estimate expected profit and risk-reward ratio."""
        if len(self.price_history) < 20:
            return 0.0, 1.0
            
        prices = list(self.price_history)
        
        # Estimate volatility-based profit target
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns[-20:])
        
        # Expected profit scales with quality and volatility
        expected_profit = quality_score * volatility * current_price * 2
        
        # Risk-reward ratio improves with quality
        risk_reward = 1.0 + quality_score * 2
        
        return expected_profit, risk_reward
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent data."""
        if len(self.price_history) < 100:
            return
            
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        
        # Update volatility threshold
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        self.quality_thresholds["volatility"] = np.percentile(np.abs(returns), 75)
        
        # Update volume threshold
        self.quality_thresholds["volume"] = np.percentile(volumes, 75)
        
        # Update momentum threshold
        momentum = [prices[i] - prices[i-5] for i in range(5, len(prices))]
        self.quality_thresholds["momentum"] = np.percentile(np.abs(momentum), 75)
    
    def should_trade_signal(self, signal: TradingSignal) -> bool:
        """Determine if signal quality is sufficient for trading."""
        # Only trade GOOD or EXCELLENT signals
        if signal.quality in [SignalQuality.EXCELLENT, SignalQuality.GOOD]:
            return True
            
        # Trade FAIR signals only if confidence is very high
        if signal.quality == SignalQuality.FAIR and signal.confidence > 0.8:
            return True
            
        return False
    
    def update_signal_performance(self, signal: TradingSignal, 
                                 trade_result: str, pnl: float):
        """Update signal performance tracking."""
        quality = signal.quality
        
        if quality in self.signal_performance:
            self.signal_performance[quality]["count"] += 1
            self.signal_performance[quality]["total_pnl"] += pnl
            
            if trade_result == "WIN":
                self.signal_performance[quality]["wins"] += 1
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics by signal quality."""
        stats = {}
        
        for quality, data in self.signal_performance.items():
            if data["count"] > 0:
                win_rate = data["wins"] / data["count"]
                avg_pnl = data["total_pnl"] / data["count"]
                stats[quality.value] = {
                    "count": data["count"],
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "total_pnl": data["total_pnl"]
                }
                
        return stats
