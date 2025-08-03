# TiRex Uncertainty-Based Regime Detection: NT-Native Implementation Plan

**Created**: 2025-08-02  
**Status**: Ready for Implementation (Week 2 of SAGE Enhanced Phase 0)  
**Context**: Critical integration for SAGE Meta-Framework dynamic model weighting  
**Related**: [SAGE Meta-Framework Strategy](sage_meta_framework_strategy.md) | [Enhanced Phase 0 Progress](enhanced_phase_0_progress.md)

---

## 🎯 Executive Summary

This document details the **critical juncture** implementation where SAGE transforms from individual model validation to true ensemble intelligence. The TiRex uncertainty-based regime detection serves as the **central nervous system** that dynamically adjusts model weights and enables adaptive performance.

**Strategic Importance**: This is the breakthrough component that enables SAGE's self-adaptive capabilities, moving beyond static model combinations to dynamic, regime-aware ensemble intelligence.

### **Integration Challenge**
Integrate a 35M parameter xLSTM model (TiRex) into NT-native backtesting while maintaining streamlined approach and ensuring robust production-ready performance.

### **Success Impact**
Upon completion, SAGE will dynamically adjust model weights based on market regimes, providing the adaptive intelligence that differentiates it from static ensemble approaches.

---

## 🏗️ NT-Native Architecture Design

### **Component Hierarchy (NT Pattern Compliance)**

```
TiRexRegimeActor (Actor)                    # NT-native event processing
├── RegimeDetectionEngine                   # Core TiRex integration  
├── UncertaintyQuantifier                   # Uncertainty extraction
├── RegimeClassifier                        # Market state categorization
└── RegimePublisher                         # NT event distribution

SAGEMetaStrategy (Strategy)                 # NT-native trading strategy
├── RegimeSubscriber                        # Receives regime events
├── ModelWeightManager                      # Dynamic weight allocation
├── EnsembleSignalCombiner                 # Signal aggregation
└── RiskAwarePositionSizer                 # Uncertainty-based sizing

RegimeAwareIndicator (IndicatorBase)        # NT-native technical analysis
├── RegimeStabilityMetrics                 # Regime persistence tracking
├── TransitionProbabilityCalculator        # Change likelihood estimation
└── RegimeAdjustedSignals                  # Context-aware indicators
```

### **NT Integration Points**

#### **Actor Pattern Integration**
```python
class TiRexRegimeActor(Actor):
    """NT-native Actor for TiRex-based regime detection"""
    
    def __init__(self, config: TiRexRegimeConfig):
        super().__init__()
        self.tirex_engine = TiRexRegimeDetectionEngine(config)
        self.regime_cache = {}
        self.uncertainty_buffer = deque(maxlen=100)
        
    async def _on_start(self):
        """NT-native actor initialization"""
        await self.tirex_engine.initialize()
        self._log.info("TiRex regime detection actor started")
        
    async def _on_bar(self, bar: Bar):
        """Process incoming bars with NT-native event handling"""
        try:
            # Generate TiRex prediction
            prediction = await self.tirex_engine.predict(bar)
            
            # Extract regime and uncertainty
            regime_signal = self._extract_regime(prediction)
            uncertainty_score = self._extract_uncertainty(prediction)
            
            # Cache and publish regime event
            self._cache_regime_data(regime_signal, uncertainty_score)
            await self._publish_regime_event(regime_signal, uncertainty_score, bar.ts_event)
            
        except Exception as e:
            self._log.error(f"TiRex processing error: {e}")
            await self._handle_fallback_regime_detection(bar)
```

#### **Strategy Pattern Integration**
```python
class SAGEMetaStrategy(Strategy):
    """NT-native strategy with TiRex regime awareness"""
    
    def __init__(self, config: SAGEMetaStrategyConfig):
        super().__init__(config)
        self.current_regime = None
        self.model_weights = {}
        self.ensemble_uncertainty = 0.0
        
    def on_start(self):
        """NT-native strategy initialization"""
        # Subscribe to regime events from TiRexRegimeActor
        self.msgbus.subscribe(topic="regime_change", handler=self._on_regime_change)
        self.msgbus.subscribe(topic="uncertainty_update", handler=self._on_uncertainty_update)
        
    def _on_regime_change(self, event: RegimeChangeEvent):
        """Handle regime transitions with NT-native event processing"""
        self.current_regime = event.new_regime
        self.ensemble_uncertainty = event.uncertainty_score
        
        # Recalculate model weights based on new regime
        self._update_model_weights(event.regime_confidence)
        
        # Log regime change with NT-native logging
        self.log.info(f"Regime change: {event.old_regime} → {event.new_regime}")
        
    def on_bar(self, bar: Bar):
        """NT-native bar processing with regime awareness"""
        if self.current_regime is None:
            return  # Wait for regime initialization
            
        # Generate ensemble signal using current regime weights
        ensemble_signal = self._generate_ensemble_signal(bar)
        
        # Calculate regime-aware position size
        position_size = self._calculate_regime_position_size(
            ensemble_signal, self.ensemble_uncertainty
        )
        
        # Execute trades using NT-native order management
        if abs(ensemble_signal) > self.config.signal_threshold:
            self._execute_regime_aware_trade(ensemble_signal, position_size)
```

---

## 🔧 TiRex Engine Technical Implementation

### **TiRex Engine Wrapper (NT-Compatible)**

```python
class TiRexRegimeDetectionEngine:
    """Clean TiRex integration with NT-native error handling"""
    
    def __init__(self, config: TiRexConfig):
        self.config = config
        self.model = None
        self.is_initialized = False
        self.prediction_cache = LRUCache(maxsize=1000)
        
    async def initialize(self):
        """Initialize TiRex model with robust error handling"""
        try:
            # HuggingFace API integration
            from transformers import AutoModel, AutoTokenizer
            
            self.model = AutoModel.from_pretrained(
                "tirex-forecasting/tirex-35m-crypto",
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tirex-forecasting/tirex-35m-crypto"
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            # Fallback to synthetic regime detection
            console.print(f"[yellow]⚠️ TiRex initialization failed: {e}[/yellow]")
            console.print("[yellow]🔄 Using fallback regime detection[/yellow]")
            self.is_initialized = False
            return False
    
    async def predict(self, bar: Bar) -> TiRexPrediction:
        """Generate TiRex prediction with caching and error handling"""
        
        # Check cache first
        cache_key = f"{bar.instrument_id}_{bar.ts_event}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        try:
            if self.is_initialized and self.model is not None:
                prediction = await self._generate_tirex_prediction(bar)
            else:
                prediction = await self._generate_fallback_prediction(bar)
                
            # Cache the prediction
            self.prediction_cache[cache_key] = prediction
            return prediction
            
        except Exception as e:
            console.print(f"[yellow]⚠️ TiRex prediction failed: {e}[/yellow]")
            return await self._generate_fallback_prediction(bar)
    
    async def _generate_tirex_prediction(self, bar: Bar) -> TiRexPrediction:
        """Generate real TiRex prediction"""
        
        # Prepare input data for TiRex
        input_sequence = self._prepare_tirex_input(bar)
        
        # Generate prediction with quantiles
        with torch.no_grad():
            outputs = self.model(
                input_sequence,
                return_dict=True,
                output_quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
            )
        
        # Extract predictions and uncertainty
        forecast = outputs.prediction_outputs
        quantiles = outputs.quantile_outputs
        
        # Calculate uncertainty from quantile spread
        uncertainty = self._calculate_uncertainty_from_quantiles(quantiles)
        
        return TiRexPrediction(
            forecast=forecast,
            quantiles=quantiles,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            timestamp=bar.ts_event
        )
    
    async def _generate_fallback_prediction(self, bar: Bar) -> TiRexPrediction:
        """Fallback regime detection using volatility/trend analysis"""
        
        # Simple regime detection based on recent price action
        price_data = self._get_recent_prices(bar, lookback=20)
        
        # Calculate trend and volatility
        returns = np.diff(price_data) / price_data[:-1]
        trend_strength = np.mean(returns)
        volatility = np.std(returns)
        
        # Synthetic uncertainty based on volatility
        uncertainty = min(volatility * 10, 0.9)  # Cap at 90%
        
        # Create synthetic prediction
        return TiRexPrediction(
            forecast=np.array([trend_strength]),
            quantiles=None,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            timestamp=bar.ts_event,
            is_fallback=True
        )
```

### **Regime Classification Logic**

```python
class RegimeClassifier:
    """Classify market regimes from TiRex predictions"""
    
    def __init__(self):
        self.regime_states = [
            'bull_low_vol', 'bull_high_vol', 
            'bear_low_vol', 'bear_high_vol',
            'sideways_low_vol', 'sideways_high_vol'
        ]
        self.transition_threshold = 0.15
        
    def classify_regime(self, prediction: TiRexPrediction, market_data: MarketData) -> RegimeSignal:
        """Classify current market regime using TiRex uncertainty"""
        
        # Extract trend from TiRex forecast
        forecast_trend = np.mean(prediction.forecast)
        uncertainty = prediction.uncertainty
        
        # Calculate market characteristics
        recent_returns = market_data.get_recent_returns(21)
        realized_volatility = np.std(recent_returns)
        
        # Determine base regime
        if forecast_trend > 0.001:  # Bullish
            base_regime = 'bull'
        elif forecast_trend < -0.001:  # Bearish
            base_regime = 'bear'
        else:  # Sideways
            base_regime = 'sideways'
        
        # Determine volatility regime
        vol_threshold = np.median(market_data.get_volatility_history(252))
        if realized_volatility > vol_threshold:
            vol_regime = 'high_vol'
        else:
            vol_regime = 'low_vol'
        
        # Combine regimes
        regime = f"{base_regime}_{vol_regime}"
        
        # Add transition flag based on TiRex uncertainty
        if uncertainty > self.transition_threshold:
            regime += '_transitioning'
        
        return RegimeSignal(
            regime=regime,
            confidence=prediction.confidence,
            uncertainty=uncertainty,
            trend_strength=abs(forecast_trend),
            volatility_level=realized_volatility,
            transition_probability=uncertainty,
            timestamp=prediction.timestamp
        )
```

---

## 🔄 Streamlined Backtesting Integration

### **Integration with Streamlined Specification Approach**

```python
class SAGEStreamlinedBacktestingEngine:
    """Enhanced backtesting engine with TiRex regime detection"""
    
    def __init__(self):
        # Use our streamlined specification manager (P&L essentials only)
        self.spec_manager = BacktestingSpecificationManager()
        
        # Initialize TiRex regime detection
        self.tirex_actor = TiRexRegimeActor(TiRexRegimeConfig())
        
        # Initialize SAGE strategy with regime awareness
        self.sage_strategy = SAGEMetaStrategy(SAGEMetaStrategyConfig())
        
        # Connect regime detection to strategy
        self._setup_regime_integration()
    
    def _setup_regime_integration(self):
        """Connect TiRex regime detection to SAGE strategy"""
        
        # Create message bus for NT-native communication
        self.msgbus = MessageBus()
        
        # Register regime event handlers
        self.msgbus.register_handler(
            topic="regime_change",
            handler=self.sage_strategy._on_regime_change
        )
        
        # Connect actors to message bus
        self.tirex_actor.register_msgbus(self.msgbus)
        self.sage_strategy.register_msgbus(self.msgbus)
    
    async def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run backtest with regime-aware SAGE strategy"""
        
        # Initialize components
        await self.tirex_actor._on_start()
        self.sage_strategy.on_start()
        
        # Load historical data using streamlined approach (essential specs only)
        market_data = self.spec_manager.load_historical_data(start_date, end_date)
        
        # Process bars with regime detection
        for bar in market_data:
            # TiRex processes bar and publishes regime events
            await self.tirex_actor._on_bar(bar)
            
            # Strategy processes bar with regime awareness
            self.sage_strategy.on_bar(bar)
            
        # Generate NT-native reporting
        return self._generate_sage_performance_report()
```

### **Dynamic Model Weight Calculation**

```python
class ModelWeightManager:
    """Manage dynamic model weights based on regime and performance"""
    
    def __init__(self):
        self.performance_window = 63  # 3 months
        self.min_weight = 0.05
        self.regime_adjustments = self._load_regime_preferences()
    
    def calculate_dynamic_weights(self, current_regime: RegimeSignal, 
                                model_performances: Dict[str, float]) -> Dict[str, float]:
        """Calculate model weights using regime and performance data"""
        
        # Base weights from recent performance
        base_weights = self._calculate_performance_weights(model_performances)
        
        # Regime-specific adjustments (from SAGE research)
        regime_multipliers = self._get_regime_multipliers(current_regime.regime)
        
        # Uncertainty-based adjustments
        uncertainty_factor = self._calculate_uncertainty_adjustment(current_regime.uncertainty)
        
        # Combine all factors
        final_weights = {}
        for model_name in ['alphaforge', 'tirex', 'catch22', 'tsfresh']:
            base_weight = base_weights.get(model_name, 0.25)
            regime_mult = regime_multipliers.get(model_name, 1.0)
            uncertainty_adj = uncertainty_factor
            
            adjusted_weight = base_weight * regime_mult * uncertainty_adj
            final_weights[model_name] = max(adjusted_weight, self.min_weight)
        
        # Normalize to sum to 1
        total_weight = sum(final_weights.values())
        normalized_weights = {
            model: weight / total_weight 
            for model, weight in final_weights.items()
        }
        
        return normalized_weights
    
    def _get_regime_multipliers(self, regime: str) -> Dict[str, float]:
        """Get regime-specific model weight multipliers (from SAGE research)"""
        
        regime_preferences = {
            'bull_low_vol': {
                'alphaforge': 1.2,  # Formulaic factors excel in stable trends
                'tirex': 0.9,       # Less uncertainty advantage in stable periods
                'catch22': 1.1,     # Canonical features reliable
                'tsfresh': 1.0      # Neutral
            },
            'bull_high_vol': {
                'alphaforge': 0.8,  # Formulaic may lag in volatility
                'tirex': 1.3,       # Uncertainty modeling valuable
                'catch22': 1.0,     # Neutral
                'tsfresh': 1.1      # Feature diversity helps
            },
            'bear_high_vol': {
                'alphaforge': 0.7,  # Struggle in chaotic periods
                'tirex': 1.4,       # Most valuable in uncertainty
                'catch22': 0.9,     # Some degradation
                'tsfresh': 1.2      # Diversity crucial
            },
            'bear_low_vol': {
                'alphaforge': 1.1,  # Good for systematic downtrends
                'tirex': 1.0,       # Neutral
                'catch22': 1.2,     # Canonical features robust
                'tsfresh': 0.9      # May overfit
            },
            'sideways_low_vol': {
                'alphaforge': 0.9,  # Limited trending signals
                'tirex': 1.1,       # Uncertainty helps in choppy markets
                'catch22': 1.2,     # Canonical features stable
                'tsfresh': 1.0      # Neutral
            },
            'sideways_high_vol': {
                'alphaforge': 0.8,  # Poor in choppy volatile markets
                'tirex': 1.3,       # High uncertainty environment
                'catch22': 1.0,     # Neutral
                'tsfresh': 1.2      # Feature diversity valuable
            }
        }
        
        base_regime = regime.replace('_transitioning', '')
        return regime_preferences.get(base_regime, {})
```

---

## 🛡️ Error Handling & Production Robustness

### **Graceful Degradation Strategy**

```python
class RobustTiRexIntegration:
    """Robust TiRex integration with multiple fallback levels"""
    
    def __init__(self):
        self.fallback_levels = [
            'tirex_full',      # Full TiRex with uncertainty
            'tirex_basic',     # TiRex without uncertainty  
            'synthetic_smart', # Advanced volatility/trend analysis
            'synthetic_simple' # Basic regime detection
        ]
        self.current_level = 'tirex_full'
        self.error_counts = defaultdict(int)
        self.max_errors_per_level = 5
        
    async def get_regime_prediction(self, bar: Bar) -> RegimeSignal:
        """Get regime prediction with automatic fallback"""
        
        for level in self.fallback_levels:
            try:
                if self.error_counts[level] > self.max_errors_per_level:
                    continue  # Skip failed levels
                    
                if level == 'tirex_full':
                    return await self._tirex_full_prediction(bar)
                elif level == 'tirex_basic':
                    return await self._tirex_basic_prediction(bar)
                elif level == 'synthetic_smart':
                    return await self._synthetic_smart_prediction(bar)
                else:  # synthetic_simple
                    return await self._synthetic_simple_prediction(bar)
                    
            except Exception as e:
                self.error_counts[level] += 1
                console.print(f"[yellow]⚠️ {level} failed: {e}, trying next level[/yellow]")
                continue
        
        # Emergency fallback
        return RegimeSignal.create_emergency_fallback()
    
    async def _synthetic_smart_prediction(self, bar: Bar) -> RegimeSignal:
        """Advanced synthetic regime detection using multiple indicators"""
        
        # Get recent price data
        price_data = self._get_recent_price_data(bar, lookback=50)
        
        # Calculate multiple regime indicators
        trend_indicator = self._calculate_trend_strength(price_data)
        volatility_indicator = self._calculate_volatility_regime(price_data)
        momentum_indicator = self._calculate_momentum_regime(price_data)
        
        # Combine indicators for regime classification
        regime = self._classify_regime_from_indicators(
            trend_indicator, volatility_indicator, momentum_indicator
        )
        
        # Estimate uncertainty based on indicator agreement
        uncertainty = self._estimate_uncertainty_from_indicators(
            trend_indicator, volatility_indicator, momentum_indicator
        )
        
        return RegimeSignal(
            regime=regime,
            confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            timestamp=bar.ts_event,
            is_fallback=True,
            fallback_level='synthetic_smart'
        )
```

### **NT-Native Error Integration**

```python
class SAGEErrorHandler:
    """NT-native error handling for SAGE components"""
    
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.error_counts = defaultdict(int)
        self.max_errors = 10
        self.last_known_regime = None
        
    def handle_tirex_error(self, error: Exception, bar: Bar) -> RegimeSignal:
        """Handle TiRex errors with NT-native logging and fallback"""
        
        self.error_counts['tirex'] += 1
        
        # Log error using NT-native logging
        self.strategy.log.error(f"TiRex error: {error}")
        
        # Switch to fallback if too many errors
        if self.error_counts['tirex'] > self.max_errors:
            self.strategy.log.warning("TiRex error threshold exceeded, switching to fallback")
            return self._activate_fallback_regime_detection(bar)
        
        # Return last known good regime with degraded confidence
        if self.last_known_regime:
            degraded_regime = self.last_known_regime.copy()
            degraded_regime.confidence *= 0.5  # Reduce confidence
            degraded_regime.uncertainty = min(degraded_regime.uncertainty * 1.5, 0.9)
            return degraded_regime
        
        # Emergency fallback
        return RegimeSignal.create_emergency_fallback()
    
    def update_last_known_regime(self, regime: RegimeSignal):
        """Update last known good regime for error fallback"""
        if regime.confidence > 0.5:  # Only cache high-confidence regimes
            self.last_known_regime = regime
```

---

## 📊 Performance Optimization for Backtesting

### **Batch Processing Strategy**

```python
class TiRexBatchProcessor:
    """Optimize TiRex for backtesting performance"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.prediction_queue = []
        self.result_cache = LRUCache(maxsize=10000)
        
    async def process_bars_batch(self, bars: List[Bar]) -> List[RegimeSignal]:
        """Process multiple bars in batch for efficiency"""
        
        # Check cache first
        cached_results = []
        uncached_bars = []
        
        for bar in bars:
            cache_key = f"{bar.instrument_id}_{bar.ts_event}"
            if cache_key in self.result_cache:
                cached_results.append(self.result_cache[cache_key])
            else:
                uncached_bars.append(bar)
        
        # Process uncached bars in batch
        if uncached_bars:
            batch_predictions = await self._batch_tirex_prediction(uncached_bars)
            
            # Cache new results
            for bar, prediction in zip(uncached_bars, batch_predictions):
                cache_key = f"{bar.instrument_id}_{bar.ts_event}"
                self.result_cache[cache_key] = prediction
            
            # Combine cached and new results
            all_results = cached_results + batch_predictions
        else:
            all_results = cached_results
            
        return all_results
    
    async def _batch_tirex_prediction(self, bars: List[Bar]) -> List[RegimeSignal]:
        """Generate TiRex predictions in batch"""
        
        # Prepare batch input
        batch_inputs = [self._prepare_tirex_input(bar) for bar in bars]
        batch_tensor = torch.stack(batch_inputs)
        
        # Batch prediction
        with torch.no_grad():
            batch_outputs = self.model(batch_tensor, return_dict=True)
        
        # Convert to regime signals
        regime_signals = []
        for i, bar in enumerate(bars):
            prediction = TiRexPrediction.from_batch_output(batch_outputs, i, bar.ts_event)
            regime_signal = self.regime_classifier.classify_regime(prediction, bar)
            regime_signals.append(regime_signal)
        
        return regime_signals
```

### **Memory Management**

```python
class TiRexMemoryManager:
    """Manage TiRex memory usage during backtesting"""
    
    def __init__(self, max_cache_size: int = 10000):
        self.max_cache_size = max_cache_size
        self.prediction_cache = LRUCache(maxsize=max_cache_size)
        self.regime_history = deque(maxlen=1000)
        self.uncertainty_buffer = deque(maxlen=500)
        self.cleanup_interval = 1000  # Cleanup every 1000 bars
        self.bar_count = 0
        
    def cleanup_memory(self):
        """Periodic memory cleanup during backtesting"""
        
        self.bar_count += 1
        
        if self.bar_count % self.cleanup_interval == 0:
            # Clear old predictions
            if len(self.prediction_cache) > self.max_cache_size * 0.8:
                # Keep only the most recent predictions
                self.prediction_cache.clear()
                
            # Trim buffers
            if len(self.regime_history) > 800:
                self.regime_history = deque(list(self.regime_history)[-500:], maxlen=1000)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory status
            console.print(f"[blue]🧹 Memory cleanup: Cache size {len(self.prediction_cache)}, "
                         f"Regime history {len(self.regime_history)}[/blue]")
```

---

## 🧪 Testing & Validation Framework

### **Integration Testing**

```python
class SAGETiRexIntegrationTests:
    """Comprehensive testing for TiRex-SAGE integration"""
    
    async def test_end_to_end_integration(self):
        """Test complete SAGE workflow with TiRex regime detection"""
        
        # Setup test environment
        test_engine = SAGEStreamlinedBacktestingEngine()
        test_data = self._load_test_btcusdt_data()
        
        # Run integration test
        results = await test_engine.run_backtest(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30)
        )
        
        # Validate results
        assert results.total_trades > 0
        assert results.regime_changes > 5  # Expect regime transitions
        assert results.model_weight_updates > 20  # Dynamic rebalancing
        assert abs(results.sharpe_ratio) > 0.5  # Reasonable performance
        
        console.print(f"[green]✅ End-to-end test passed: {results.total_trades} trades, "
                     f"{results.regime_changes} regime changes[/green]")
        
    async def test_tirex_fallback_behavior(self):
        """Test graceful degradation when TiRex fails"""
        
        # Simulate TiRex failure
        with mock.patch('TiRexRegimeDetectionEngine.predict', side_effect=Exception("API Error")):
            
            strategy = SAGEMetaStrategy(SAGEMetaStrategyConfig())
            error_handler = SAGEErrorHandler(strategy)
            
            # Should fall back to synthetic regime detection
            test_bar = self._create_test_bar()
            regime_signal = error_handler.handle_tirex_error(Exception("Test error"), test_bar)
            
            assert regime_signal is not None
            assert regime_signal.is_fallback == True
            assert regime_signal.confidence < 1.0
            
        console.print("[green]✅ Fallback behavior test passed[/green]")
    
    async def test_dynamic_weight_calculation(self):
        """Test dynamic model weight calculation"""
        
        weight_manager = ModelWeightManager()
        
        # Create test regime signal
        test_regime = RegimeSignal(
            regime='bull_high_vol',
            confidence=0.8,
            uncertainty=0.2,
            timestamp=datetime.now()
        )
        
        # Create test performance data
        test_performances = {
            'alphaforge': 0.6,
            'tirex': 0.8,
            'catch22': 0.7,
            'tsfresh': 0.5
        }
        
        # Calculate weights
        weights = weight_manager.calculate_dynamic_weights(test_regime, test_performances)
        
        # Validate weights
        assert abs(sum(weights.values()) - 1.0) < 0.001  # Sum to 1
        assert all(w >= 0.05 for w in weights.values())  # Minimum weights
        assert weights['tirex'] > weights['alphaforge']  # TiRex higher in high vol
        
        console.print(f"[green]✅ Dynamic weights test passed: {weights}[/green]")
```

### **Performance Benchmarking**

```python
class TiRexPerformanceBenchmark:
    """Benchmark TiRex integration performance"""
    
    async def benchmark_regime_detection_speed(self):
        """Benchmark regime detection latency"""
        
        test_bars = self._generate_test_bars(1000)
        tirex_actor = TiRexRegimeActor(TiRexRegimeConfig())
        await tirex_actor._on_start()
        
        start_time = time.time()
        for bar in test_bars:
            regime_signal = await tirex_actor._on_bar(bar)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / len(test_bars) * 1000  # ms
        
        # Ensure reasonable performance for backtesting
        assert avg_latency < 50  # <50ms per bar for backtesting
        
        console.print(f"[green]✅ Average regime detection latency: {avg_latency:.2f}ms[/green]")
        
        return {
            'avg_latency_ms': avg_latency,
            'total_bars': len(test_bars),
            'total_time_s': end_time - start_time,
            'performance_status': 'PASS' if avg_latency < 50 else 'FAIL'
        }
    
    async def benchmark_memory_usage(self):
        """Benchmark memory usage during extended backtesting"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extended test
        test_bars = self._generate_test_bars(10000)  # Large test
        memory_manager = TiRexMemoryManager()
        
        for bar in test_bars:
            # Simulate regime detection with memory tracking
            memory_manager.cleanup_memory()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Ensure reasonable memory usage
        assert memory_increase < 500  # <500MB increase
        
        console.print(f"[green]✅ Memory usage test: {memory_increase:.1f}MB increase[/green]")
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'memory_status': 'PASS' if memory_increase < 500 else 'FAIL'
        }
```

---

## 🎯 Implementation Roadmap & Milestones

### **Week 2 Implementation Sequence (Days 8-14)**

#### **Day 8-9: Core TiRex Integration**
- [ ] Implement `TiRexRegimeDetectionEngine` with HuggingFace integration
- [ ] Create fallback regime detection logic
- [ ] Build NT-native `TiRexRegimeActor` wrapper
- [ ] **Milestone**: TiRex engine operational with fallback capability

#### **Day 10-11: Regime Classification System**
- [ ] Implement `RegimeClassifier` with uncertainty-based logic
- [ ] Create regime transition detection algorithms
- [ ] Build regime event publishing system
- [ ] **Milestone**: Regime classification producing consistent signals

#### **Day 12-13: SAGE Strategy Integration**
- [ ] Integrate regime awareness into `SAGEMetaStrategy`
- [ ] Implement `ModelWeightManager` for dynamic weight calculation
- [ ] Connect with streamlined backtesting approach
- [ ] **Milestone**: Dynamic model weights responding to regime changes

#### **Day 14: Testing & Validation**
- [ ] End-to-end integration testing
- [ ] Performance benchmarking (<50ms latency target)
- [ ] Error handling validation
- [ ] **Milestone**: Production-ready TiRex regime detection system

### **Success Criteria & Gates**

#### **Week 2 Completion Gates**
- ✅ TiRex regime detection functional with <50ms latency per bar
- ✅ Dynamic model weights respond appropriately to regime changes
- ✅ Graceful fallback when TiRex fails (no system interruption)
- ✅ NT-native patterns maintained throughout
- ✅ Integration with streamlined backtesting successful
- ✅ SAGE ensemble shows improved Sharpe ratio vs individual models

#### **Performance Targets**
- **Latency**: <50ms per bar for regime detection
- **Memory**: <500MB increase during extended backtesting
- **Accuracy**: >70% regime detection accuracy on historical data
- **Reliability**: <1% system failures with fallback active

#### **Quality Gates**
- All integration tests pass
- Performance benchmarks meet targets
- Error handling covers all failure scenarios
- Documentation complete and reviewed

---

## 📋 Risk Mitigation & Contingency Plans

### **Technical Risks**

1. **TiRex Model Availability Risk**
   - **Mitigation**: Multiple fallback levels implemented
   - **Contingency**: Synthetic regime detection as backup

2. **Performance Degradation Risk**
   - **Mitigation**: Batch processing and caching strategies
   - **Contingency**: Simplified regime detection for real-time use

3. **Memory Usage Risk**
   - **Mitigation**: Aggressive memory management and cleanup
   - **Contingency**: Reduced cache sizes and buffer limits

### **Integration Risks**

1. **NT Pattern Compliance Risk**
   - **Mitigation**: Strict adherence to Actor/Strategy patterns
   - **Contingency**: Custom wrapper layer if needed

2. **Streamlined Backtesting Compatibility Risk**
   - **Mitigation**: Use existing specification manager
   - **Contingency**: Adapt regime detection to work with any data source

---

## 📚 Related Documentation & Cross-References

### **Direct Dependencies**
- [SAGE Meta-Framework Strategy](sage_meta_framework_strategy.md) - Parent strategy document
- [Enhanced Phase 0 Progress](enhanced_phase_0_progress.md) - Implementation tracking
- [Pending Research Topics](../research/pending_research_topics.md) - Progress tracking

### **Technical Integration**
- `sage-forge/src/sage_forge/market/backtesting_specs.py` - Streamlined specification manager
- `nautilus_test/sage/models/tirex_wrapper.py` - Existing TiRex wrapper
- `validate_btcusdt_models.py` - Model validation framework

### **Research Foundation**
- [NT Implementation Priority Matrix](../research/nt_implementation_priority_matrix_2025.md) - Implementation priorities
- [Algorithm Taxonomy](../research/adaptive_algorithm_taxonomy_2024_2025.md) - SOTA model categorization
- [Expert Analysis](../research/cfup_afpoe_expert_analysis_2025.md) - Multi-model validation insights

---

## 📝 Implementation Notes & Considerations

### **Critical Design Decisions**

1. **Actor vs Strategy for Regime Detection**
   - **Decision**: Use Actor pattern for regime detection
   - **Rationale**: Separates regime logic from trading logic, enables reuse

2. **Fallback Strategy Levels**
   - **Decision**: 4-level fallback system (TiRex full → basic → synthetic smart → simple)
   - **Rationale**: Graceful degradation without system failure

3. **Integration with Streamlined Backtesting**
   - **Decision**: Maintain P&L-essential focus, regime detection as additional layer
   - **Rationale**: Preserves streamlined approach benefits

### **Future Evolution Paths**

1. **Multi-Asset Regime Detection**
   - Extend TiRex to handle multiple instruments
   - Cross-asset regime correlation analysis

2. **Advanced Uncertainty Modeling**
   - Incorporate additional uncertainty sources
   - Ensemble uncertainty from multiple regime detectors

3. **Real-Time Optimization**
   - Streaming regime detection for live trading
   - GPU acceleration for TiRex inference

---

**Document Status**: ✅ **READY FOR IMPLEMENTATION**  
**Next Action**: Begin Day 8 implementation - Core TiRex Integration  
**Success Metric**: TiRex regime detection operational by Day 14  
**Critical Path**: This implementation enables Week 3 SAGE meta-combination engine

---

**Last Updated**: 2025-08-02  
**Version**: 1.0  
**Review Status**: Pending technical review and adversarial analysis