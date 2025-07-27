#!/usr/bin/env python3
"""
🚀 NT-NATIVE ENHANCED SOTA STRATEGY 2025
========================================

Enhanced state-of-the-art trading strategy combining:
- NT-native bias-free architecture (Phase 1-8 complete)
- Catch22 canonical time series features (Phase 9.1)
- Online feature selection with auto-parameterization (Phase 9.2)
- FTRL online learning with enhanced features
- Real-time computational efficiency

This strategy builds on the proven bias-free foundation and adds 2025 SOTA ML enhancements
while maintaining NT's native patterns and real-time performance requirements.

Features:
✅ All Phase 1-8 bias prevention guarantees maintained
✅ 22 canonical time series characteristics (Catch22)
✅ Adaptive online feature selection (MI + LASSO + RFE ensemble)
✅ Enhanced FTRL learning with larger feature space
✅ Computational efficiency optimized for live trading
✅ Seamless integration with existing NT infrastructure

Author: Claude Code Assistant  
Date: 2025-07-19
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
import csv
import time
import warnings
from datetime import datetime
from collections import deque, defaultdict

# Import enhanced SOTA components
try:
    # Import Catch22 feature extractor
    from strategies.backtests.nt_sota_feature_engineering import Catch22FeatureExtractor
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False
    warnings.warn("Catch22 feature extractor not available")

try:
    # Import online feature selection
    from strategies.backtests.nt_online_feature_selection import EnsembleFeatureSelector
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    warnings.warn("Online feature selection not available")

# Import NT-native base components
from strategies.backtests.nt_custom_indicators import (
    CustomMomentumIndicator,
    CustomVolatilityRatio,
    CustomChangePointDetector,
    CustomVolumeRatio,
    CustomCrossoverSignal
)
from strategies.backtests.nt_bias_free_config import (
    create_comprehensive_bias_free_config,
    get_bias_free_strategy_config
)

# NautilusTrader imports
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.core.uuid import UUID4

# Built-in NT indicators 
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.atr import AverageTrueRange

# Rich console for enhanced output
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()
    Progress = None


@dataclass
class EnhancedRegimeState:
    """Enhanced regime state with confidence and feature attribution."""
    name: str
    confidence: float
    duration: int
    primary_features: List[str]
    regime_strength: float


class EnhancedFTRLLearner:
    """Enhanced FTRL learner with adaptive feature weighting and regularization."""
    
    def __init__(self, feature_dim: int, alpha: float = 0.1, beta: float = 1.0, 
                 l1_lambda: float = 0.01, l2_lambda: float = 0.01):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # FTRL parameters
        self.weights = np.zeros(feature_dim)
        self.z_weights = np.zeros(feature_dim)
        self.n_weights = np.ones(feature_dim) * 1e-6
        
        # Enhanced tracking
        self.prediction_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.feature_importance = np.zeros(feature_dim)
        self.feature_usage_count = np.zeros(feature_dim)
        
        # Adaptive parameters
        self.adaptive_alpha = alpha
        self.performance_window = deque(maxlen=100)
        
    def predict(self, features: np.ndarray, selected_features: Optional[Set[int]] = None) -> float:
        """Make prediction using current weights, optionally with feature selection."""
        if len(features) != self.feature_dim:
            # Handle dynamic feature dimension
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                padded_features = np.zeros(self.feature_dim)
                padded_features[:len(features)] = features
                features = padded_features
        
        # Apply feature selection if provided
        if selected_features is not None:
            masked_features = np.zeros_like(features)
            for i in selected_features:
                if i < len(features):
                    masked_features[i] = features[i]
            features = masked_features
        
        # Track feature usage
        self.feature_usage_count += (np.abs(features) > 1e-8).astype(float)
        
        # Compute prediction
        logit = np.dot(self.weights, features)
        prediction = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        
        self.prediction_history.append(prediction)
        return prediction
    
    def update(self, features: np.ndarray, target: float, prediction: float,
               selected_features: Optional[Set[int]] = None):
        """Update weights using enhanced FTRL with adaptive regularization."""
        if len(features) != self.feature_dim:
            # Handle dynamic feature dimension
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                padded_features = np.zeros(self.feature_dim)
                padded_features[:len(features)] = features
                features = padded_features
        
        # Apply feature selection if provided
        if selected_features is not None:
            masked_features = np.zeros_like(features)
            for i in selected_features:
                if i < len(features):
                    masked_features[i] = features[i]
            features = masked_features
        
        # Compute loss and gradient
        loss = (prediction - target) ** 2
        gradient = 2 * (prediction - target) * features
        
        # Update FTRL parameters
        self.n_weights += gradient ** 2
        
        # Adaptive learning rate
        sigma = (np.sqrt(self.n_weights) - np.sqrt(self.n_weights - gradient ** 2)) / self.adaptive_alpha
        self.z_weights += gradient - sigma * self.weights
        
        # Enhanced FTRL update with adaptive regularization
        for i in range(self.feature_dim):
            # Adaptive regularization based on feature importance
            adaptive_l1 = self.l1_lambda * (1.0 + 0.5 * self.feature_importance[i])
            
            if abs(self.z_weights[i]) <= adaptive_l1:
                self.weights[i] = 0.0
            else:
                sign_z = np.sign(self.z_weights[i])
                denominator = (self.beta + np.sqrt(self.n_weights[i])) / self.adaptive_alpha + self.l2_lambda
                self.weights[i] = -(self.z_weights[i] - sign_z * adaptive_l1) / denominator
        
        # Update feature importance (exponential moving average)
        current_importance = np.abs(gradient)
        self.feature_importance = 0.95 * self.feature_importance + 0.05 * current_importance
        
        # Track performance for adaptive learning rate
        self.loss_history.append(loss)
        self.performance_window.append(loss)
        
        # Adaptive learning rate adjustment
        if len(self.performance_window) >= 50:
            recent_loss = np.mean(list(self.performance_window)[-25:])
            older_loss = np.mean(list(self.performance_window)[-50:-25])
            
            if recent_loss > older_loss * 1.1:  # Performance degrading
                self.adaptive_alpha *= 0.98  # Decrease learning rate
            elif recent_loss < older_loss * 0.9:  # Performance improving
                self.adaptive_alpha *= 1.02  # Increase learning rate
            
            # Clamp adaptive alpha
            self.adaptive_alpha = np.clip(self.adaptive_alpha, 0.01, 1.0)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get current feature importance scores."""
        return self.feature_importance.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get enhanced performance metrics."""
        if len(self.loss_history) == 0:
            return {"avg_loss": 0.0, "recent_loss": 0.0, "adaptive_alpha": self.adaptive_alpha}
        
        losses = np.array(list(self.loss_history))
        recent_losses = losses[-min(100, len(losses)):]
        
        return {
            "avg_loss": np.mean(losses),
            "recent_loss": np.mean(recent_losses),
            "loss_trend": np.mean(recent_losses[-25:]) - np.mean(recent_losses[-50:-25]) if len(recent_losses) >= 50 else 0.0,
            "adaptive_alpha": self.adaptive_alpha,
            "n_predictions": len(self.prediction_history)
        }


class NTEnhancedSOTAStrategy(Strategy):
    """
    🚀 NT-NATIVE ENHANCED SOTA TRADING STRATEGY 2025
    
    ARCHITECTURAL GUARANTEES (inherited from Phase 1-8):
    ✅ Uses ONLY NT's cache for historical data access
    ✅ Leverages NT's built-in indicators with auto-registration
    ✅ Follows event-driven architecture (Guide Section 3.1)
    ✅ No current bar data usage in feature extraction
    ✅ Trusts NT's stateful, evolving cache design
    ✅ Enables comprehensive bias prevention configuration
    
    ENHANCED SOTA FEATURES (Phase 9+):
    🚀 22 Catch22 canonical time series characteristics
    🎯 Online feature selection with ensemble voting
    🧠 Enhanced FTRL with adaptive regularization
    📊 Real-time feature importance tracking
    ⚡ Computational efficiency optimized for live trading
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        console.print("[yellow]🚀 Initializing NT-Native Enhanced SOTA Strategy...[/yellow]")
        
        # Get strategy configuration
        self.strategy_config = get_bias_free_strategy_config()
        
        # Enhanced feature dimensions
        self.base_features = 10  # Original features from base strategy
        self.catch22_features = 22 if CATCH22_AVAILABLE else 0
        self.total_features = self.base_features + self.catch22_features
        
        console.print(f"[cyan]  • Base features: {self.base_features}[/cyan]")
        console.print(f"[cyan]  • Catch22 features: {self.catch22_features}[/cyan]")
        console.print(f"[cyan]  • Total feature dimension: {self.total_features}[/cyan]")
        
        # Core data storage - minimal buffering (NT cache handles history)
        self.returns = deque(maxlen=50)
        
        # NT Built-in Indicators (auto-registered, bias-free)
        self.ema_short = ExponentialMovingAverage(5)
        self.ema_medium = ExponentialMovingAverage(20)  
        self.ema_long = ExponentialMovingAverage(50)
        self.rsi = RelativeStrengthIndex(14)
        self.atr = AverageTrueRange(20)
        
        # Custom Indicators following NT patterns
        self.momentum_5 = CustomMomentumIndicator(5)
        self.momentum_20 = CustomMomentumIndicator(20)
        self.vol_ratio = CustomVolatilityRatio(5, 20)
        self.change_detector = CustomChangePointDetector(50)
        self.volume_ratio = CustomVolumeRatio(5, 20)
        self.crossover = CustomCrossoverSignal(5, 20)
        
        # Enhanced SOTA Components
        if CATCH22_AVAILABLE:
            self.catch22_extractor = Catch22FeatureExtractor(window_size=100, update_frequency=5)
            console.print("[green]✅ Catch22 feature extractor initialized[/green]")
        else:
            self.catch22_extractor = None
            console.print("[yellow]⚠️ Catch22 features not available[/yellow]")
        
        if FEATURE_SELECTION_AVAILABLE:
            self.feature_selector = EnsembleFeatureSelector(
                max_features=min(15, self.total_features),
                ensemble_method="weighted_voting"
            )
            console.print("[green]✅ Online feature selection initialized[/green]")
        else:
            self.feature_selector = None
            console.print("[yellow]⚠️ Online feature selection not available[/yellow]")
        
        # Enhanced learning setup
        self.enhanced_learner = EnhancedFTRLLearner(self.total_features)
        
        # Enhanced state tracking
        self.current_regime = EnhancedRegimeState("UNKNOWN", 0.0, 0, [], 0.0)
        self.bar_counter = 0
        self.last_signal = 0.0
        self.selected_features = set()
        
        # Previous data for prequential learning (bias-free)
        self.prev_features = None
        self.prev_return = None
        
        # Performance tracking
        self.feature_selection_history = deque(maxlen=1000)
        self.computation_times = deque(maxlen=100)
        
        # Progress tracking for initialization
        self.initialization_progress = None
        self.initialization_complete = False
        
        # Logging setup
        self.setup_enhanced_logging()
        
        console.print("[bold green]🚀 NT-Native Enhanced SOTA Strategy initialized![/bold green]")
    
    def setup_enhanced_logging(self):
        """Setup enhanced logging system with SOTA feature tracking."""
        self.logs_dir = Path("trade_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced trade log
        self.trade_log_file = self.logs_dir / f"nt_enhanced_sota_trades_{timestamp}.csv"
        with open(self.trade_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'bar_number', 'action', 'signal_strength', 'regime',
                'total_features', 'selected_features', 'catch22_available', 
                'feature_importance_top3', 'adaptive_alpha', 'prediction_performance',
                'computation_time_ms', 'cache_bars_count'
            ])
        
        # Feature analysis log
        self.feature_log_file = self.logs_dir / f"nt_sota_features_{timestamp}.csv"
        with open(self.feature_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'bar_number', 'feature_type', 'feature_values',
                'selected_features', 'importance_scores', 'selection_method'
            ])
        
        console.print(f"[cyan]📝 Enhanced SOTA logging: {self.trade_log_file.name}[/cyan]")
        console.print(f"[cyan]📊 Feature analysis logging: {self.feature_log_file.name}[/cyan]")
    
    def on_start(self):
        """Strategy startup with enhanced indicator registration."""
        self.subscribe_bars(self.config.bar_type)
        
        # Register ALL indicators using NT's auto-update system
        console.print("[yellow]🔧 Registering enhanced indicators with NT auto-update system...[/yellow]")
        
        # Built-in NT indicators
        self.register_indicator_for_bars(self.config.bar_type, self.ema_short)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_medium)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_long)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        
        # Custom indicators
        self.register_indicator_for_bars(self.config.bar_type, self.momentum_5)
        self.register_indicator_for_bars(self.config.bar_type, self.momentum_20)
        self.register_indicator_for_bars(self.config.bar_type, self.vol_ratio)
        self.register_indicator_for_bars(self.config.bar_type, self.change_detector)
        self.register_indicator_for_bars(self.config.bar_type, self.volume_ratio)
        self.register_indicator_for_bars(self.config.bar_type, self.crossover)
        
        # Enhanced SOTA indicators
        if self.catch22_extractor:
            self.register_indicator_for_bars(self.config.bar_type, self.catch22_extractor)
            console.print("[green]✅ Catch22 extractor registered[/green]")
        
        console.print("[green]✅ All enhanced indicators registered for auto-updates![/green]")
        console.print(f"[cyan]📊 Subscribed to {self.config.bar_type}[/cyan]")
    
    def on_bar(self, bar: Bar):
        """
        Process each bar using enhanced SOTA features with NT's native bias-free patterns.
        
        CRITICAL: Maintains all Phase 1-8 bias prevention guarantees while adding SOTA features!
        """
        start_time = time.time()
        self.bar_counter += 1
        
        # PHASE 1: Access historical data using NT's cache (unchanged from base strategy)
        historical_bars = self.cache.bars(self.config.bar_type)
        
        if len(historical_bars) < self.strategy_config['min_bars_required']:
            if self.initialization_progress is None:
                self.initialization_progress = Progress(
                    TextColumn("🔧 [bold blue]Warming up indicators & ML models"),
                    BarColumn(complete_style="green"), 
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
                    TextColumn("({task.completed}/{task.total} bars)"),
                    console=console
                )
                self.initialization_progress.start()
                self.init_task = self.initialization_progress.add_task(
                    "Loading historical data for feature extraction", 
                    total=self.strategy_config['min_bars_required']
                )
            
            self.initialization_progress.update(self.init_task, completed=len(historical_bars))
            return
        
        # Complete initialization progress bar once
        if not self.initialization_complete and self.initialization_progress is not None:
            self.initialization_progress.update(self.init_task, completed=self.strategy_config['min_bars_required'])
            self.initialization_progress.stop()
            self.initialization_progress = None  # Clean up reference
            self.initialization_complete = True
            
            # Ensure clean line separation from any previous output
            console.print("\n" + "="*60)
            console.print("[bold green]✅ Enhanced SOTA Strategy initialization complete - Starting trading![/bold green]")
            console.print("="*60 + "\n")
        
        # PHASE 2: Extract ENHANCED features using ONLY NT cache and indicators
        enhanced_features = self._extract_enhanced_sota_features(historical_bars)
        
        # PHASE 3: Apply online feature selection if available
        if self.feature_selector and len(enhanced_features) > 0:
            # Use previous return as target for feature selection
            target = 1.0 if len(self.returns) > 0 and self.returns[-1] > 0 else 0.0
            self.selected_features = self.feature_selector.select_features(enhanced_features, target)
        else:
            # Use all features if selection not available
            self.selected_features = set(range(len(enhanced_features)))
        
        # PHASE 4: Make trading decision using enhanced features
        if self._all_enhanced_indicators_initialized() and len(enhanced_features) > 0:
            signal_strength = self.enhanced_learner.predict(enhanced_features, self.selected_features)
            self._execute_trading_logic(signal_strength, bar)
        else:
            signal_strength = 0.0
        
        # PHASE 5: Update learning with PREVIOUS iteration's outcome (prequential)
        if self.prev_features is not None and len(historical_bars) >= 2:
            # Calculate return from previous bars (bias-free)
            current_price = float(historical_bars[-1].close)
            prev_price = float(historical_bars[-2].close)
            
            current_return = (current_price - prev_price) / prev_price
            self.returns.append(current_return)
            
            # Enhanced prequential learning
            if len(self.returns) >= 2:
                prev_outcome = 1.0 if self.returns[-2] > 0 else 0.0
                prev_prediction = self.enhanced_learner.predict(self.prev_features, self.selected_features)
                self.enhanced_learner.update(self.prev_features, prev_outcome, prev_prediction, self.selected_features)
        
        # PHASE 6: Enhanced regime classification
        self._classify_enhanced_regime(historical_bars, enhanced_features)
        
        # Performance tracking
        computation_time = (time.time() - start_time) * 1000  # ms
        self.computation_times.append(computation_time)
        
        # Log progress with enhanced metrics
        if self.bar_counter % 1000 == 0:
            self._log_enhanced_progress(signal_strength, enhanced_features, computation_time)
        
        # Store current features for next iteration
        self.prev_features = enhanced_features.copy() if len(enhanced_features) > 0 else None
        self.last_signal = signal_strength
    
    def _extract_enhanced_sota_features(self, historical_bars: List[Bar]) -> np.ndarray:
        """
        Extract enhanced SOTA features combining base features + Catch22.
        
        CRITICAL: NO current bar data usage - only completed historical bars and indicators!
        """
        all_features = []
        
        # PART 1: Extract base features (unchanged from base strategy)
        base_features = self._extract_base_features(historical_bars)
        all_features.extend(base_features)
        
        # PART 2: Extract Catch22 features if available
        if self.catch22_extractor and self.catch22_extractor.initialized:
            catch22_features = self.catch22_extractor.get_feature_vector()
            all_features.extend(catch22_features)
        else:
            # Pad with zeros if Catch22 not available
            all_features.extend([0.0] * self.catch22_features)
        
        # Convert to numpy array and apply normalization
        enhanced_features = np.array(all_features)
        
        if self.strategy_config['outlier_clipping']:
            enhanced_features = np.clip(enhanced_features, -10.0, 10.0)
        
        if self.strategy_config['feature_normalization']:
            enhanced_features = np.tanh(enhanced_features)
        
        return enhanced_features
    
    def _extract_base_features(self, historical_bars: List[Bar]) -> List[float]:
        """Extract base features (unchanged from original strategy)."""
        features = []
        
        # Feature 1-2: EMA Signals
        if self.ema_short.initialized and self.ema_medium.initialized:
            ema_signal = (self.ema_short.value - self.ema_medium.value) / self.ema_medium.value
            features.append(ema_signal)
            
            ema_long_signal = (self.ema_medium.value - self.ema_long.value) / self.ema_long.value
            features.append(ema_long_signal)
        else:
            features.extend([0.0, 0.0])
        
        # Feature 3-4: Momentum Signals
        features.append(self.momentum_5.value if self.momentum_5.initialized else 0.0)
        features.append(self.momentum_20.value if self.momentum_20.initialized else 0.0)
        
        # Feature 5: RSI Signal
        if self.rsi.initialized:
            rsi_signal = (self.rsi.value - 50.0) / 50.0
            features.append(rsi_signal)
        else:
            features.append(0.0)
        
        # Feature 6: Volatility Ratio
        features.append(self.vol_ratio.value if self.vol_ratio.initialized else 0.0)
        
        # Feature 7: Change Point Signal
        if self.change_detector.initialized:
            change_signal = self.change_detector.value / 10.0
            features.append(change_signal)
        else:
            features.append(0.0)
        
        # Feature 8: Volume Ratio
        features.append(self.volume_ratio.value if self.volume_ratio.initialized else 0.0)
        
        # Feature 9: Crossover Signal
        features.append(self.crossover.value if self.crossover.initialized else 0.0)
        
        # Feature 10: ATR Signal
        if self.atr.initialized and len(historical_bars) >= 2:
            recent_price = float(historical_bars[-1].close)
            atr_signal = self.atr.value / recent_price if recent_price > 0 else 0.0
            features.append(atr_signal)
        else:
            features.append(0.0)
        
        return features
    
    def _all_enhanced_indicators_initialized(self) -> bool:
        """Check if all enhanced indicators are properly initialized."""
        base_initialized = all([
            self.ema_short.initialized,
            self.ema_medium.initialized,
            self.momentum_5.initialized,
            self.rsi.initialized,
            self.vol_ratio.initialized
        ])
        
        catch22_initialized = (not CATCH22_AVAILABLE or 
                              (self.catch22_extractor and self.catch22_extractor.initialized))
        
        return base_initialized and catch22_initialized
    
    def _classify_enhanced_regime(self, historical_bars: List[Bar], features: np.ndarray):
        """Enhanced regime classification using SOTA features."""
        if len(historical_bars) < 20:
            self.current_regime = EnhancedRegimeState("UNKNOWN", 0.0, 0, [], 0.0)
            return
        
        # Base regime detection using ATR
        regime_name = "UNKNOWN"
        confidence = 0.0
        primary_features = []
        regime_strength = 0.0
        
        if self.atr.initialized:
            current_atr = self.atr.value
            recent_prices = [float(bar.close) for bar in historical_bars[-10:]]
            price_std = np.std(recent_prices)
            avg_price = np.mean(recent_prices)
            
            normalized_vol = price_std / avg_price if avg_price > 0 else 0.0
            
            if normalized_vol > 0.02:
                regime_name = "VOLATILE"
                confidence = min(normalized_vol / 0.02, 1.0)
                primary_features.append("ATR")
            else:
                regime_name = "RANGING"
                confidence = max(0.5, 1.0 - normalized_vol / 0.02)
                primary_features.append("Price_Stability")
            
            regime_strength = normalized_vol
        
        # Enhanced regime features using Catch22 (if available)
        if self.catch22_extractor and self.catch22_extractor.initialized and len(features) > self.base_features:
            catch22_features = features[self.base_features:]
            
            # Use specific Catch22 features for regime enhancement
            if len(catch22_features) >= 22:
                # Use volatility and trend features
                trend_feature = catch22_features[2]  # CO_f1ecac (autocorr)
                complexity_feature = catch22_features[9]  # PD_PeriodicityWang_th0_01
                
                if abs(trend_feature) > 0.5:
                    primary_features.append("Catch22_Autocorr")
                    regime_strength += abs(trend_feature) * 0.3
                
                if abs(complexity_feature) > 0.3:
                    primary_features.append("Catch22_Periodicity")
                    regime_strength += abs(complexity_feature) * 0.2
        
        self.current_regime = EnhancedRegimeState(
            regime_name, confidence, 1, primary_features, regime_strength
        )
    
    def _execute_trading_logic(self, signal_strength: float, bar: Bar):
        """Execute trading logic using enhanced signals (unchanged from base)."""
        signal_bias = signal_strength - 0.5
        action_taken = "NONE"
        
        if abs(signal_bias) > self.strategy_config['signal_threshold']:
            if signal_bias > 0 and not self.portfolio.is_net_long(self.config.instrument_id):
                self._place_order(OrderSide.BUY, bar)
                action_taken = "BUY"
            elif signal_bias < 0 and not self.portfolio.is_net_short(self.config.instrument_id):
                self._place_order(OrderSide.SELL, bar)
                action_taken = "SELL"
        
        self._log_enhanced_trade(bar, action_taken, signal_strength)
    
    def _place_order(self, side: OrderSide, bar: Bar):
        """Place order with configured position size (unchanged from base)."""
        try:
            position_size = self.strategy_config['max_position_size']
            quantity = Quantity(position_size, precision=3)
            
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=quantity,
                time_in_force=TimeInForce.IOC,
                client_order_id=self.generate_order_id()
            )
            
            self.submit_order(order)
            
        except Exception as e:
            console.print(f"[red]❌ Order placement failed: {e}[/red]")
    
    def _log_enhanced_trade(self, bar: Bar, action: str, signal: float):
        """Log enhanced trade decisions with SOTA feature analysis."""
        if self.strategy_config['log_trades']:
            with open(self.trade_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Enhanced metrics
                performance = self.enhanced_learner.get_performance_metrics()
                feature_importance = self.enhanced_learner.get_feature_importance()
                top3_features = np.argsort(feature_importance)[-3:].tolist()
                
                avg_computation_time = np.mean(list(self.computation_times)) if self.computation_times else 0.0
                cache_bars = len(self.cache.bars(self.config.bar_type))
                
                writer.writerow([
                    bar.ts_init, self.bar_counter, action, signal,
                    self.current_regime.name, self.total_features, 
                    len(self.selected_features), CATCH22_AVAILABLE,
                    str(top3_features), performance['adaptive_alpha'],
                    performance['recent_loss'], avg_computation_time, cache_bars
                ])
    
    def _log_enhanced_progress(self, signal: float, features: np.ndarray, computation_time: float):
        """Log enhanced progress with SOTA metrics."""
        performance = self.enhanced_learner.get_performance_metrics()
        feature_importance = self.enhanced_learner.get_feature_importance()
        
        # Ensure output always starts on a fresh line
        console.print(f"\n[dim cyan]🚀 Enhanced Bar {self.bar_counter}: {self.current_regime.name} "
                     f"| Signal: {signal:.3f} | Features: {len(self.selected_features)}/{self.total_features} "
                     f"| α: {performance['adaptive_alpha']:.3f} | Loss: {performance['recent_loss']:.4f} "
                     f"| Catch22: {'✅' if CATCH22_AVAILABLE else '❌'} | Time: {computation_time:.1f}ms[/dim cyan]")
    
    def generate_order_id(self):
        """Generate unique order ID."""
        from nautilus_trader.model.identifiers import ClientOrderId
        return ClientOrderId(str(UUID4()))
    
    def on_stop(self):
        """Strategy cleanup with enhanced reporting."""
        console.print("[yellow]⏹️ NT-Native Enhanced SOTA Strategy stopped[/yellow]")
        
        if self.bar_counter > 0:
            performance = self.enhanced_learner.get_performance_metrics()
            feature_importance = self.enhanced_learner.get_feature_importance()
            
            console.print(f"[cyan]🚀 Final Enhanced Performance:[/cyan]")
            console.print(f"[cyan]  • Total bars processed: {self.bar_counter}[/cyan]")
            console.print(f"[cyan]  • Enhanced feature dimension: {self.total_features}[/cyan]")
            console.print(f"[cyan]  • Catch22 features: {'✅ Active' if CATCH22_AVAILABLE else '❌ Unavailable'}[/cyan]")
            console.print(f"[cyan]  • Feature selection: {'✅ Active' if FEATURE_SELECTION_AVAILABLE else '❌ Unavailable'}[/cyan]")
            console.print(f"[cyan]  • Final adaptive α: {performance['adaptive_alpha']:.4f}[/cyan]")
            console.print(f"[cyan]  • Recent loss: {performance['recent_loss']:.4f}[/cyan]")
            console.print(f"[cyan]  • Avg computation time: {np.mean(list(self.computation_times)):.1f}ms[/cyan]")
            console.print(f"[cyan]  • Final regime: {self.current_regime.name} (features: {self.current_regime.primary_features})[/cyan]")
            
            # Feature importance analysis
            if len(feature_importance) > 0:
                top_features = np.argsort(feature_importance)[-5:][::-1]
                console.print(f"[cyan]  • Top 5 features by importance: {top_features.tolist()}[/cyan]")
        
        console.print("[bold green]🚀 NT-Native Enhanced SOTA strategy completed successfully![/bold green]")
    
    def on_reset(self):
        """Reset enhanced strategy state."""
        # Reset base state
        self.returns.clear()
        self.bar_counter = 0
        self.last_signal = 0.0
        self.prev_features = None
        self.prev_return = None
        self.selected_features.clear()
        
        # Reset enhanced components
        self.computation_times.clear()
        self.feature_selection_history.clear()
        
        if self.feature_selector:
            self.feature_selector.reset()
        
        console.print("[blue]🔄 Enhanced SOTA Strategy reset - Ready for new run![/blue]")


if __name__ == "__main__":
    console.print("[bold green]🚀 NT-NATIVE ENHANCED SOTA TRADING STRATEGY 2025![/bold green]")
    console.print("[dim]Combining NT bias-free patterns with state-of-the-art ML features[/dim]")
    
    # Check component availability
    console.print(f"[cyan]📊 Component Status:[/cyan]")
    console.print(f"[cyan]  • Catch22 Features: {'✅ Available' if CATCH22_AVAILABLE else '❌ Not Available'}[/cyan]")
    console.print(f"[cyan]  • Online Feature Selection: {'✅ Available' if FEATURE_SELECTION_AVAILABLE else '❌ Not Available'}[/cyan]")
    
    console.print("\n[green]🌟 Ready for deployment with enhanced SOTA features![/green]")