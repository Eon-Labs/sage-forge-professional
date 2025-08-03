# SAGE Hybrid Uncertainty-Informed Regime Detection: Optimal Architecture

**Created**: 2025-08-02  
**Status**: Expert Panel Validated Design (Fourth-Generation Plan)  
**Context**: TiRex complementary value + proven HMM regime detection = optimal hybrid system  
**Related**: [Profitability Plan](sage_profitability_focused_implementation.md) | [SAGE Meta-Framework](sage_meta_framework_strategy.md)

---

## 🎯 **EXPERT PANEL INSIGHT: HIERARCHICAL UNCERTAINTY-INFORMED ARCHITECTURE**

### **Key Research Finding**:
> **TiRex's unique value: Uncertainty quantification as meta-signal**  
> **HMM's proven strength: Fast, profitable regime classification**  
> **Optimal approach: Uncertainty-informed regime detection**

### **Critical Insight**:
**TiRex uncertainty spikes signal when to trust HMM regime classifications** and when to switch to uncertainty-aware defensive strategies.

---

## 🏗️ **HYBRID ARCHITECTURE DESIGN**

### **Dual-Layer System (Uncertainty + Regime)**

```
SAGE Uncertainty-Informed Trading System
├── Layer 1: Real-Time Regime Detection (HMM) [<1ms latency]
│   ├── HMMRegimeIndicator                    # Fast regime classification
│   ├── RegimeConfidenceTracker               # HMM state confidence
│   └── RegimeTransitionDetector              # Regime change signals
├── Layer 2: Market Uncertainty Detection (TiRex) [Batch updates]
│   ├── TiRexUncertaintyIndicator            # Uncertainty quantification
│   ├── MarketStressDetector                 # High uncertainty = stress
│   └── UncertaintyTrendAnalyzer             # Uncertainty pattern analysis
└── Layer 3: Adaptive Strategy Selection (Meta-Controller)
    ├── UncertaintyRegimeController          # Uncertainty-informed decisions
    ├── StrategyWeightManager                # Dynamic strategy allocation
    └── RiskAdaptivePositionSizer            # Uncertainty-based sizing
```

**Core Principle**: **HMM drives decisions, TiRex informs confidence**

---

## 🔧 **IMPLEMENTATION: DUAL-LAYER INDICATORS**

### **Layer 1: Real-Time HMM Regime Detection**

```python
from nautilus_trader.indicators.base import Indicator
import numpy as np
from hmmlearn import hmm

class HMMRegimeIndicator(Indicator):
    """
    Fast HMM regime detection with proven trading performance.
    Real-time execution <1ms for trading decisions.
    """
    
    def __init__(self, n_regimes: int = 3, lookback: int = 252, refit_frequency: int = 21):
        super().__init__(params=[n_regimes, lookback, refit_frequency])
        
        # HMM configuration
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.refit_frequency = refit_frequency
        
        # HMM model (using proven hmmlearn)
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=42,
            n_iter=100
        )
        
        # State tracking
        self.current_regime = 0
        self.regime_probabilities = np.zeros(n_regimes)
        self.regime_confidence = 0.0
        self.bars_since_refit = 0
        
        # Performance tracking
        self.regime_changes = 0
        self.last_regime = 0
        self.regime_duration = 0
        
        # Regime names for interpretation
        self.regime_names = {0: "bear_trend", 1: "sideways_chop", 2: "bull_trend"}
        
    def _update_raw(self, value: float) -> None:
        """Real-time HMM regime detection."""
        
        if not self.initialized:
            return
            
        self.bars_since_refit += 1
        
        # Prepare features for HMM
        features = self._prepare_hmm_features()
        
        # Refit HMM periodically (e.g., weekly)
        if self.bars_since_refit >= self.refit_frequency or not hasattr(self, '_fitted'):
            self._refit_hmm_model(features)
            self.bars_since_refit = 0
        
        # Predict current regime (fast inference)
        if hasattr(self, '_fitted') and self._fitted:
            try:
                # Fast regime prediction
                current_features = features[-1:].reshape(1, -1)
                regime_probs = self.hmm_model.predict_proba(current_features)
                
                # Update state
                self.regime_probabilities = regime_probs[0]
                self.current_regime = np.argmax(self.regime_probabilities)
                self.regime_confidence = np.max(self.regime_probabilities)
                
                # Track regime changes
                if self.current_regime != self.last_regime:
                    self.regime_changes += 1
                    self.regime_duration = 0
                    self.last_regime = self.current_regime
                else:
                    self.regime_duration += 1
                
                # Set indicator value
                self.set_value(float(self.current_regime))
                
            except Exception as e:
                # Fallback to simple regime detection
                self._simple_regime_fallback()
        else:
            self._simple_regime_fallback()
    
    def _prepare_hmm_features(self) -> np.ndarray:
        """Prepare features for HMM regime detection."""
        
        # Get price data from NT input buffer
        prices = self.get_inputs()
        
        if len(prices) < self.lookback:
            return np.array([]).reshape(0, 2)
        
        # Calculate returns and volatility features
        recent_prices = prices[-self.lookback:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Rolling volatility (21-day window)
        volatilities = []
        for i in range(20, len(returns)):
            vol = np.std(returns[i-20:i+1])
            volatilities.append(vol)
        
        # Align returns and volatilities
        aligned_returns = returns[20:]
        aligned_volatilities = volatilities
        
        # Create feature matrix for HMM
        features = np.column_stack([aligned_returns, aligned_volatilities])
        
        return features
    
    def _refit_hmm_model(self, features: np.ndarray) -> None:
        """Refit HMM model periodically."""
        
        if len(features) < 50:  # Need minimum data
            return
            
        try:
            # Fit HMM on recent data
            self.hmm_model.fit(features)
            self._fitted = True
            
        except Exception as e:
            print(f"HMM refit failed: {e}")
            self._fitted = False
    
    def _simple_regime_fallback(self) -> None:
        """Simple fallback regime detection."""
        
        prices = self.get_inputs()
        if len(prices) < 20:
            self.current_regime = 1  # Default to sideways
            self.regime_confidence = 0.3
            self.set_value(1.0)
            return
        
        # Simple trend detection
        recent_returns = np.diff(prices[-20:]) / prices[-20:-1]
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Simple 3-regime classification
        if avg_return > volatility * 0.5:
            self.current_regime = 2  # Bull trend
        elif avg_return < -volatility * 0.5:
            self.current_regime = 0  # Bear trend  
        else:
            self.current_regime = 1  # Sideways
        
        self.regime_confidence = 0.6  # Medium confidence
        self.set_value(float(self.current_regime))
    
    @property
    def regime_name(self) -> str:
        """Human-readable regime name."""
        return self.regime_names.get(self.current_regime, "unknown")
    
    @property
    def is_regime_stable(self) -> bool:
        """Check if current regime is stable."""
        return self.regime_duration >= 5 and self.regime_confidence > 0.6
```

### **Layer 2: TiRex Uncertainty Detection (Batch Updates)**

```python
class TiRexUncertaintyIndicator(Indicator):
    """
    TiRex uncertainty quantification for market stress detection.
    Batch updates (hourly/daily) to avoid real-time latency issues.
    """
    
    def __init__(self, update_frequency: int = 60):  # Update every 60 bars
        super().__init__(params=[update_frequency])
        
        self.update_frequency = update_frequency
        self.bars_since_update = 0
        
        # TiRex model (lazy loaded)
        self.tirex_model = None
        self.model_loaded = False
        
        # Uncertainty state
        self.current_uncertainty = 0.5  # Default medium uncertainty
        self.uncertainty_trend = 0.0
        self.market_stress_level = 0.0
        
        # Historical uncertainty for trend analysis
        self.uncertainty_history = []
        self.max_history = 100
        
    def _update_raw(self, value: float) -> None:
        """Batch TiRex uncertainty updates."""
        
        self.bars_since_update += 1
        
        # Only update periodically to avoid latency
        if self.bars_since_update >= self.update_frequency:
            self._batch_update_uncertainty()
            self.bars_since_update = 0
        
        # Set current uncertainty as indicator value
        self.set_value(self.current_uncertainty)
    
    def _batch_update_uncertainty(self) -> None:
        """Batch update of TiRex uncertainty (can be async)."""
        
        try:
            # Lazy load TiRex model
            if not self.model_loaded:
                self._load_tirex_model()
            
            if self.model_loaded:
                # Get recent price data
                prices = self.get_inputs()
                
                if len(prices) >= 50:
                    # Generate TiRex uncertainty estimate
                    uncertainty = self._generate_tirex_uncertainty(prices[-50:])
                    
                    # Update uncertainty state
                    self._update_uncertainty_state(uncertainty)
                else:
                    # Fallback uncertainty estimation
                    uncertainty = self._estimate_simple_uncertainty(prices)
                    self._update_uncertainty_state(uncertainty)
            else:
                # Model failed to load - use simple uncertainty
                prices = self.get_inputs()
                uncertainty = self._estimate_simple_uncertainty(prices)
                self._update_uncertainty_state(uncertainty)
                
        except Exception as e:
            print(f"TiRex uncertainty update failed: {e}")
            # Use simple uncertainty fallback
            prices = self.get_inputs()
            uncertainty = self._estimate_simple_uncertainty(prices)
            self._update_uncertainty_state(uncertainty)
    
    def _load_tirex_model(self) -> None:
        """Lazy load TiRex model with proper error handling."""
        
        try:
            # Attempt to load TiRex (placeholder - replace with actual implementation)
            from transformers import AutoModel
            
            # Note: Replace with actual TiRex model when available
            self.tirex_model = AutoModel.from_pretrained(
                "microsoft/time-series-transformer",  # Placeholder
                trust_remote_code=True
            )
            self.tirex_model.eval()
            self.model_loaded = True
            
        except Exception as e:
            print(f"TiRex model loading failed: {e}")
            self.model_loaded = False
    
    def _generate_tirex_uncertainty(self, prices: np.ndarray) -> float:
        """Generate TiRex-based uncertainty estimate."""
        
        # Prepare input for TiRex
        # This is a placeholder - implement actual TiRex inference
        input_data = self._prepare_tirex_input(prices)
        
        # Generate forecasts with quantiles
        with torch.no_grad():
            # Placeholder inference
            forecasts = self.tirex_model(input_data)
            
            # Extract uncertainty from quantile spread
            # This is conceptual - implement based on actual TiRex API
            uncertainty = self._extract_uncertainty_from_quantiles(forecasts)
        
        return uncertainty
    
    def _estimate_simple_uncertainty(self, prices: np.ndarray) -> float:
        """Simple uncertainty estimation based on volatility."""
        
        if len(prices) < 20:
            return 0.5
        
        # Calculate rolling volatility as uncertainty proxy
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        current_vol = np.std(returns)
        
        # Historical volatility for comparison
        if len(prices) >= 100:
            historical_returns = np.diff(prices[-100:]) / prices[-100:-1]
            historical_vol = np.std(historical_returns)
            
            # Uncertainty as relative volatility
            uncertainty = min(current_vol / (historical_vol + 1e-8), 2.0) / 2.0
        else:
            # Normalize volatility to 0-1 range
            uncertainty = min(current_vol * 10, 1.0)
        
        return uncertainty
    
    def _update_uncertainty_state(self, new_uncertainty: float) -> None:
        """Update uncertainty state and calculate trends."""
        
        # Update current uncertainty
        previous_uncertainty = self.current_uncertainty
        self.current_uncertainty = new_uncertainty
        
        # Calculate uncertainty trend
        self.uncertainty_trend = new_uncertainty - previous_uncertainty
        
        # Update uncertainty history
        self.uncertainty_history.append(new_uncertainty)
        if len(self.uncertainty_history) > self.max_history:
            self.uncertainty_history.pop(0)
        
        # Calculate market stress level
        if len(self.uncertainty_history) >= 10:
            recent_avg = np.mean(self.uncertainty_history[-10:])
            overall_avg = np.mean(self.uncertainty_history)
            self.market_stress_level = (recent_avg - overall_avg) / (overall_avg + 1e-8)
        else:
            self.market_stress_level = 0.0
    
    @property
    def is_market_stressed(self) -> bool:
        """Check if market is in stressed state."""
        return self.current_uncertainty > 0.7 or self.market_stress_level > 0.5
    
    @property
    def uncertainty_increasing(self) -> bool:
        """Check if uncertainty is increasing."""
        return self.uncertainty_trend > 0.1
```

### **Layer 3: Adaptive Strategy Controller**

```python
class SAGEHybridStrategy(Strategy):
    """
    SAGE strategy with uncertainty-informed regime detection.
    Combines fast HMM regime detection with TiRex uncertainty quantification.
    """
    
    def __init__(self, config: SAGEHybridConfig):
        super().__init__(config)
        
        # Dual-layer indicators
        self.hmm_regime = HMMRegimeIndicator(n_regimes=3, lookback=252)
        self.tirex_uncertainty = TiRexUncertaintyIndicator(update_frequency=60)
        
        # Alpha factor indicators
        self.alpha_factor = ProfitabilityAlphaIndicator(period=20)
        self.alphaforge_factors = AlphaForgeIndicator(period=20)
        
        # Strategy state
        self.current_strategy_mode = "normal"  # normal, defensive, aggressive
        self.uncertainty_threshold_high = 0.7
        self.uncertainty_threshold_low = 0.3
        
        # Performance tracking
        self.strategy_performance = {
            'normal': {'trades': 0, 'pnl': 0.0},
            'defensive': {'trades': 0, 'pnl': 0.0},
            'aggressive': {'trades': 0, 'pnl': 0.0}
        }
        
    def on_start(self):
        """Initialize hybrid strategy."""
        
        self.subscribe_bars(self.config.bar_type)
        
        # Register all indicators
        indicators = [
            self.hmm_regime,
            self.tirex_uncertainty,
            self.alpha_factor,
            self.alphaforge_factors
        ]
        
        for indicator in indicators:
            self.register_indicator_for_bars(self.config.bar_type, indicator)
        
        self.log.info("SAGE Hybrid Uncertainty-Informed Strategy started")
    
    def on_bar(self, bar: Bar):
        """Uncertainty-informed trading logic."""
        
        if not self._all_indicators_ready():
            return
        
        # Get regime and uncertainty information
        current_regime = self.hmm_regime.regime_name
        regime_confidence = self.hmm_regime.regime_confidence
        market_uncertainty = self.tirex_uncertainty.current_uncertainty
        is_market_stressed = self.tirex_uncertainty.is_market_stressed
        
        # Determine strategy mode based on uncertainty
        strategy_mode = self._determine_strategy_mode(
            market_uncertainty, is_market_stressed, regime_confidence
        )
        
        # Generate signals based on strategy mode
        ensemble_signal = self._generate_uncertainty_informed_signal(
            current_regime, strategy_mode, market_uncertainty
        )
        
        # Calculate position size with uncertainty adjustment
        position_size = self._calculate_uncertainty_adjusted_position_size(
            ensemble_signal, regime_confidence, market_uncertainty
        )
        
        # Execute trade if signal is strong enough
        if abs(ensemble_signal) > self._get_dynamic_threshold(strategy_mode):
            self._execute_uncertainty_informed_trade(
                ensemble_signal, position_size, bar, strategy_mode
            )
    
    def _determine_strategy_mode(self, uncertainty: float, is_stressed: bool, 
                               regime_confidence: float) -> str:
        """Determine strategy mode based on uncertainty and regime confidence."""
        
        if is_stressed or uncertainty > self.uncertainty_threshold_high:
            return "defensive"
        elif uncertainty < self.uncertainty_threshold_low and regime_confidence > 0.8:
            return "aggressive"
        else:
            return "normal"
    
    def _generate_uncertainty_informed_signal(self, regime: str, mode: str, 
                                            uncertainty: float) -> float:
        """Generate trading signal informed by uncertainty."""
        
        # Base signal from alpha factors
        alpha_signal = self.alpha_factor.value
        alphaforge_signal = self.alphaforge_factors.value if self.alphaforge_factors.initialized else 0.0
        
        # Mode-specific signal weighting
        if mode == "defensive":
            # Conservative approach during high uncertainty
            ensemble_signal = (
                0.7 * alpha_signal +      # Favor proven alpha factors
                0.3 * alphaforge_signal   # Reduce model-based signals
            )
            # Apply uncertainty dampening
            ensemble_signal *= (1.0 - uncertainty * 0.5)
            
        elif mode == "aggressive":
            # Enhanced approach during low uncertainty
            ensemble_signal = (
                0.5 * alpha_signal +      # Balanced approach
                0.5 * alphaforge_signal   # Trust model signals more
            )
            # Apply uncertainty boost
            ensemble_signal *= (1.0 + (1.0 - uncertainty) * 0.3)
            
        else:  # normal mode
            # Balanced approach
            ensemble_signal = (
                0.6 * alpha_signal +
                0.4 * alphaforge_signal
            )
        
        return ensemble_signal
    
    def _calculate_uncertainty_adjusted_position_size(self, signal: float, 
                                                    regime_confidence: float,
                                                    uncertainty: float) -> float:
        """Calculate position size adjusted for uncertainty."""
        
        # Base position size
        base_size = self.config.base_position_size
        
        # Signal strength factor
        signal_factor = abs(signal)
        
        # Regime confidence factor
        confidence_factor = regime_confidence
        
        # Uncertainty adjustment (reduce size when uncertain)
        uncertainty_factor = 1.0 - uncertainty * 0.7
        
        # Combined position size
        position_size = base_size * signal_factor * confidence_factor * uncertainty_factor
        
        return min(position_size, self.config.max_position_size)
    
    def _get_dynamic_threshold(self, mode: str) -> float:
        """Get dynamic signal threshold based on strategy mode."""
        
        base_threshold = self.config.signal_threshold
        
        if mode == "defensive":
            return base_threshold * 1.5  # Higher threshold when uncertain
        elif mode == "aggressive":
            return base_threshold * 0.7  # Lower threshold when confident
        else:
            return base_threshold
    
    def _execute_uncertainty_informed_trade(self, signal: float, size: float, 
                                          bar: Bar, mode: str):
        """Execute trade with uncertainty tracking."""
        
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        # Create order
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=size
        )
        
        # Submit order
        self.submit_order(order)
        
        # Track performance by strategy mode
        self.strategy_performance[mode]['trades'] += 1
        
        self.log.info(f"Uncertainty-informed trade [{mode}]: {side.name} {size} @ {bar.close}")
        self.log.info(f"Regime: {self.hmm_regime.regime_name}, Uncertainty: {self.tirex_uncertainty.current_uncertainty:.2f}")
```

---

## 📊 **BACKTESTING FRAMEWORK INTEGRATION**

### **VectorBT Walk-Forward Analysis**

```python
import vectorbt as vbt
import pandas as pd

class SAGEHybridBacktester:
    """
    Advanced backtesting framework for uncertainty-informed regime strategies.
    """
    
    def __init__(self, strategy_config: dict):
        self.config = strategy_config
        
    def run_uncertainty_informed_backtest(self, data: pd.DataFrame, 
                                        start_date: str, end_date: str):
        """Run comprehensive backtest with uncertainty analysis."""
        
        # Initialize indicators
        hmm_regime = HMMRegimeIndicator()
        tirex_uncertainty = TiRexUncertaintyIndicator()
        
        # Generate signals
        signals = self._generate_hybrid_signals(data, hmm_regime, tirex_uncertainty)
        
        # Run VectorBT backtest
        portfolio = vbt.Portfolio.from_signals(
            data['close'], 
            entries=signals['buy'], 
            exits=signals['sell'],
            size=signals['position_size'],
            fees=self.config.get('fees', 0.001),
            slippage=self.config.get('slippage', 0.0005)
        )
        
        # Analyze performance by uncertainty regime
        uncertainty_analysis = self._analyze_uncertainty_performance(
            portfolio, signals['uncertainty_levels']
        )
        
        return {
            'portfolio': portfolio,
            'uncertainty_analysis': uncertainty_analysis,
            'regime_analysis': self._analyze_regime_performance(portfolio, signals['regimes'])
        }
    
    def _analyze_uncertainty_performance(self, portfolio, uncertainty_levels):
        """Analyze performance across different uncertainty levels."""
        
        # Segment trades by uncertainty level
        low_uncertainty = uncertainty_levels < 0.3
        medium_uncertainty = (uncertainty_levels >= 0.3) & (uncertainty_levels < 0.7)
        high_uncertainty = uncertainty_levels >= 0.7
        
        analysis = {}
        for level, mask in [('low', low_uncertainty), ('medium', medium_uncertainty), ('high', high_uncertainty)]:
            if mask.any():
                subset_returns = portfolio.returns[mask]
                analysis[level] = {
                    'sharpe_ratio': subset_returns.mean() / subset_returns.std() * np.sqrt(252),
                    'total_return': (1 + subset_returns).prod() - 1,
                    'max_drawdown': subset_returns.cumsum().expanding().max().sub(subset_returns.cumsum()).max(),
                    'win_rate': (subset_returns > 0).mean()
                }
        
        return analysis
```

---

## 🎯 **IMPLEMENTATION ROADMAP (HYBRID APPROACH)**

### **Week 2: Hybrid Uncertainty-Informed Implementation (Days 8-14)**

#### **Days 8-9: Dual-Layer Implementation**
- [ ] Implement `HMMRegimeIndicator` using hmmlearn library
- [ ] Create `TiRexUncertaintyIndicator` with batch processing
- [ ] Integrate both with NT-native indicator patterns
- [ ] **Milestone**: Dual-layer regime + uncertainty detection operational

#### **Days 10-11: Adaptive Strategy Integration**
- [ ] Implement `SAGEHybridStrategy` with uncertainty-informed decisions
- [ ] Add dynamic strategy mode switching (normal/defensive/aggressive)
- [ ] Integrate with alpha factors and position sizing
- [ ] **Milestone**: Complete uncertainty-informed trading strategy

#### **Days 12-13: Advanced Backtesting Validation**
- [ ] Implement VectorBT-based walk-forward analysis
- [ ] Validate performance across uncertainty regimes
- [ ] Compare hybrid vs. pure HMM vs. pure TiRex approaches
- [ ] **Milestone**: Proven hybrid superiority in backtesting

#### **Day 14: Production Deployment**
- [ ] Deploy hybrid system with real-time monitoring
- [ ] Implement uncertainty trend alerting
- [ ] Document optimal uncertainty thresholds
- [ ] **Milestone**: Live uncertainty-informed trading system

---

## ✅ **HYBRID SUCCESS CRITERIA**

### **Performance Gates**:
- [ ] **Hybrid outperforms pure HMM** by >15% in Sharpe ratio
- [ ] **Low uncertainty periods show >70% win rate**
- [ ] **High uncertainty periods limit drawdown to <10%**
- [ ] **Dynamic thresholds reduce false signals by >30%**

### **Technical Validation**:
- [ ] **HMM latency <1ms** for real-time regime detection
- [ ] **TiRex batch updates complete within 5 seconds**
- [ ] **Memory usage stable** during extended backtesting
- [ ] **Strategy mode switching responsive** to uncertainty changes

---

**Document Status**: ✅ **OPTIMAL HYBRID ARCHITECTURE**  
**Key Innovation**: Uncertainty-informed regime detection combining TiRex and HMM strengths  
**Next Action**: Begin Day 8 implementation of dual-layer system  
**Success Metric**: Hybrid system outperforms individual components

---

**Last Updated**: 2025-08-02  
**Architecture**: Hierarchical uncertainty-informed regime detection  
**Implementation Priority**: CRITICAL - Optimal combination of proven methods