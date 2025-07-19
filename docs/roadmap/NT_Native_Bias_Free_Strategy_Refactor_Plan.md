# 🏗️ NT-Native Bias-Free Strategy Refactor Plan

**Target Directory**: `/Users/terryli/eon/nt/nautilus_test/strategies/backtests`  
**Based on**: NautilusTrader Nutshell Guide principles  
**Objective**: Replace manual bias-prevention with NT's native patterns  

## 📋 **EXECUTIVE SUMMARY**

### **Current Problem Analysis**
- ❌ Manual `PureLag1RollingStats` implementations fight NT's architecture
- ❌ Feature extraction requires `current_price`, `current_volume` (impossible in live trading)
- ❌ Complex temporal separation logic reinvents what NT already provides
- ❌ Custom rolling windows ignore NT's bias-free cache system
- ❌ Multiple failed attempts at manual bias prevention

### **NT Guide Solution**
- ✅ Use NT's cache system for inherent bias-free historical access
- ✅ Leverage built-in indicators with auto-registration
- ✅ Follow event-driven architecture instead of vectorized processing
- ✅ Trust NT's stateful, evolving cache design
- ✅ Enable comprehensive bias prevention configuration

---

## 🎯 **PHASE 1: CLEANUP & PREPARATION**

### **Files to Delete**
- [ ] `pure_lag1_rolling_windows.py` - Replace with NT cache patterns
- [ ] `truly_lagged_rolling_windows.py` - Redundant manual implementation
- [ ] `mathematically_guaranteed_bias_free_strategy_2025.py` - Architecturally flawed
- [ ] `enhanced_sota_strategy_2025.py` - Contains bias violations
- [ ] `corrected_bias_free_strategy_2025.py` - Still biased despite name
- [ ] `final_bias_free_strategy_2025.py` - Multiple bias issues remain

### **Files to Create**
- [ ] `nt_native_bias_free_strategy_2025.py` - Main strategy using NT patterns
- [ ] `nt_custom_indicators.py` - Custom indicators following NT conventions
- [ ] `nt_bias_free_config.py` - Configuration with all bias prevention enabled

### **Files to Update**
- [ ] `sota_strategy_span_1.py` - Point to new NT-native strategy
- [ ] Update import statements throughout codebase

---

## 🏗️ **PHASE 2: NT-NATIVE ARCHITECTURE IMPLEMENTATION**

### **Core Strategy Class (`nt_native_bias_free_strategy_2025.py`)**

#### **Constructor Setup**
- [ ] Import NT built-in indicators (EMA, RSI, ATR)
- [ ] Initialize custom indicators following NT patterns
- [ ] Keep FTRL online learner (bias-free by design)
- [ ] Remove all manual rolling window implementations
- [ ] Remove current bar data parameters

#### **Indicator Registration (`on_start`)**
- [ ] `self.register_indicator_for_bars()` for all indicators
- [ ] Verify auto-update system activation
- [ ] Test indicator initialization sequence
- [ ] Validate bias-free update timing

#### **Event-Driven Processing (`on_bar`)**
```python
def on_bar(self, bar: Bar):
    # ✅ CORRECT: Use cache for historical access only
    historical_bars = self.cache.bars(self.config.bar_type)
    
    # ❌ NEVER: Use current bar data
    # current_price = float(bar.close)  # This is look-ahead bias!
    
    if len(historical_bars) < 50:
        return
        
    # Extract features using NT indicators and cache only
    features = self._extract_nt_native_features()
    
    # Make prediction and execute trades
    if self._all_indicators_initialized():
        signal = self.signal_learner.predict(features)
        self._execute_trading_logic(signal, bar)
    
    # Update learning with previous outcome
    self._update_learning()
```

**Checklist:**
- [ ] Implement cache-only historical access
- [ ] Remove all `current_price`, `current_volume` usage
- [ ] Use NT indicator `.value` properties only
- [ ] Implement prequential learning updates
- [ ] Add comprehensive logging

---

## 🛠️ **PHASE 3: CUSTOM INDICATORS (`nt_custom_indicators.py`)**

### **Momentum Indicator**
```python
class CustomMomentumIndicator(Indicator):
    def __init__(self, period: int):
        super().__init__(params=[period])
        self.period = period
        self.price_buffer = deque(maxlen=period + 1)
        self.value = 0.0
    
    def handle_bar(self, bar: Bar):
        """NT auto-calls this on bar completion"""
        self.update_raw(float(bar.close))
    
    def update_raw(self, price: float):
        self.price_buffer.append(price)
        if len(self.price_buffer) >= self.period + 1:
            # Use historical prices only (excluding current)
            old_price = self.price_buffer[0]
            prev_price = self.price_buffer[-2]  # Previous bar, not current
            self.value = (prev_price - old_price) / old_price
            self._set_initialized(True)
```

**Checklist:**
- [ ] `CustomMomentumIndicator(5)` for short-term momentum
- [ ] `CustomMomentumIndicator(20)` for medium-term momentum
- [ ] `CustomVolatilityRatio(5, 20)` using built-in ATR indicators
- [ ] `CustomChangePointDetector(50)` with CUSUM algorithm
- [ ] Test auto-registration with strategy
- [ ] Verify bias-free update timing

### **Volatility Ratio Indicator**
- [ ] Use NT's built-in `AverageTrueRange` indicators internally
- [ ] Compute ratio in `handle_bar()` method
- [ ] Ensure proper initialization sequence

### **Change Point Detector**
- [ ] Implement CUSUM algorithm using price buffer
- [ ] Use historical statistics only for z-score computation
- [ ] Reset CUSUM on change point detection
- [ ] Return normalized signal strength

---

## ⚙️ **PHASE 4: BIAS PREVENTION CONFIGURATION (`nt_bias_free_config.py`)**

### **Data Engine Configuration**
```python
data_config = DataEngineConfig(
    validate_data_sequence=True,           # Reject out-of-sequence data
    time_bars_timestamp_on_close=True,     # Proper bar timestamping
    time_bars_build_with_no_updates=True,  # Build bars without updates
    time_bars_skip_first_non_full_bar=True, # Skip incomplete first bar
    time_bars_build_delay=15,              # 15µs delay for completeness
    buffer_deltas=True,                    # Buffer order book deltas
)
```

**Checklist:**
- [ ] Enable all data validation flags
- [ ] Configure proper bar timestamping
- [ ] Set appropriate build delays
- [ ] Test sequence validation

### **Backtest Engine Configuration**
```python
engine_config = BacktestEngineConfig(
    latency_model=LatencyModel(
        base_latency_nanos=1_000_000,      # 1ms base latency
        insert_latency_nanos=2_000_000,    # 2ms order submission
        update_latency_nanos=1_500_000,    # 1.5ms modifications
        cancel_latency_nanos=1_000_000,    # 1ms cancellations
    ),
    validate_data_sequence=True,
)
```

**Checklist:**
- [ ] Configure realistic latency model
- [ ] Enable sequence validation
- [ ] Test order execution delays
- [ ] Verify no instant execution

### **Venue Configuration**
```python
venue_config = {
    'oms_type': OmsType.NETTING,
    'account_type': AccountType.CASH,
    'bar_adaptive_high_low_ordering': True,  # Realistic OHLC sequencing
    'fill_model': FillModel(
        prob_fill_on_limit=0.8,            # 80% fill probability
        prob_slippage=0.3,                 # 30% slippage chance
    ),
}
```

**Checklist:**
- [ ] Enable adaptive OHLC ordering
- [ ] Configure realistic fill models
- [ ] Set appropriate slippage probabilities
- [ ] Test execution realism

---

## 📊 **PHASE 5: FEATURE EXTRACTION REFACTOR**

### **Cache-Based Feature Extraction**
```python
def _extract_nt_native_features(self) -> np.ndarray:
    """Extract features using ONLY NT's cache and indicators"""
    
    # Get historical bars from cache (bias-free by design)
    bars = self.cache.bars(self.config.bar_type)
    features = []
    
    # Feature 1-2: Moving Average Signals
    if self.ema_short.initialized and self.ema_medium.initialized:
        ma_signal = (self.ema_short.value - self.ema_medium.value) / self.ema_medium.value
        features.append(ma_signal)
    else:
        features.append(0.0)
    
    # Feature 3-4: Momentum Signals
    if self.momentum_5.initialized:
        features.append(self.momentum_5.value)
    else:
        features.append(0.0)
    
    # Continue for all features...
    return np.array(features)
```

**Checklist:**
- [ ] Remove all manual rolling window computations
- [ ] Use NT indicator `.value` properties exclusively
- [ ] Access historical data via `self.cache.bars()` only
- [ ] Handle indicator initialization states
- [ ] Clip extreme values for stability
- [ ] Test feature consistency

### **Features to Implement**
- [ ] **EMA Signals**: Short/medium/long EMA relationships
- [ ] **Momentum Signals**: 5-period and 20-period momentum
- [ ] **RSI Signal**: Normalized RSI around 50
- [ ] **Volatility Ratio**: Short/long volatility relationship
- [ ] **Volume Momentum**: Recent vs historical volume ratios
- [ ] **Change Point Signal**: CUSUM-based regime detection
- [ ] **ATR Signal**: Normalized average true range

---

## 🧪 **PHASE 6: TESTING & VALIDATION**

### **NT-Native Bias Tests**
```python
def test_nt_native_bias_prevention():
    """Test strategy using NT's native bias prevention mechanisms"""
    
    # Test 1: Verify cache-only access
    def test_cache_access_only():
        # Verify no current bar data usage in feature extraction
        # Architecture prevents this by design
        pass
    
    # Test 2: Verify indicator consistency
    def test_indicator_temporal_consistency():
        # Process same historical sequence multiple times
        # Verify deterministic indicator outputs
        pass
    
    # Test 3: Verify chronological processing
    def test_chronological_processing():
        # NT's BacktestDataIterator guarantees this
        # Trust NT's implementation
        pass
```

**Testing Checklist:**
- [ ] Test cache-only access patterns
- [ ] Verify indicator auto-registration
- [ ] Test feature extraction consistency
- [ ] Validate chronological processing
- [ ] Test with realistic latency
- [ ] Verify no instant execution
- [ ] Test data sequence validation
- [ ] Validate OHLC ordering

### **Performance Validation**
- [ ] Compare with previous biased implementations
- [ ] Measure strategy performance metrics
- [ ] Validate online learning convergence
- [ ] Test with different market conditions
- [ ] Verify live trading compatibility

---

## 🚀 **PHASE 7: INTEGRATION & DEPLOYMENT**

### **Update Main Runner (`sota_strategy_span_1.py`)**
```python
# Replace biased strategy imports
from nt_native_bias_free_strategy_2025 import NTNativeBiasFreeStrategy
from nt_bias_free_config import create_nt_bias_free_config

# Use new strategy
def run_nt_native_strategy():
    config = create_nt_bias_free_config()
    strategy = NTNativeBiasFreeStrategy(config)
    # Run backtest...
```

**Integration Checklist:**
- [ ] Update import statements
- [ ] Configure bias prevention settings
- [ ] Test with existing data
- [ ] Validate performance metrics
- [ ] Update documentation
- [ ] Run comprehensive backtests

### **Documentation Updates**
- [ ] Update strategy documentation
- [ ] Document NT pattern usage
- [ ] Create bias prevention guide
- [ ] Update performance benchmarks
- [ ] Document configuration options

---

## 📈 **PHASE 8: MONITORING & OPTIMIZATION**

### **Performance Monitoring**
- [ ] Track strategy performance metrics
- [ ] Monitor indicator convergence
- [ ] Validate learning algorithm performance
- [ ] Check execution latency simulation
- [ ] Monitor bias prevention effectiveness

### **Optimization Opportunities**
- [ ] Fine-tune indicator parameters
- [ ] Optimize feature selection
- [ ] Adjust learning rates
- [ ] Configure latency models
- [ ] Tune fill probability models

---

## ✅ **SUCCESS CRITERIA**

### **Bias Elimination**
- [ ] ✅ Zero current bar data usage in feature extraction
- [ ] ✅ All features use NT cache or indicator values only
- [ ] ✅ Proper chronological processing verified
- [ ] ✅ Realistic execution latency simulation
- [ ] ✅ No instant order execution

### **Performance Validation**
- [ ] ✅ Strategy performance improves vs biased versions
- [ ] ✅ Online learning convergence validated
- [ ] ✅ Live trading compatibility confirmed
- [ ] ✅ Code maintainability improved
- [ ] ✅ NT best practices followed

### **Architecture Quality**
- [ ] ✅ Simplified codebase vs manual implementations
- [ ] ✅ Leverages NT's optimized components
- [ ] ✅ Follows established NT conventions
- [ ] ✅ Comprehensive bias prevention enabled
- [ ] ✅ Production-ready configuration

---

## 🔄 **ROLLBACK PLAN**

If refactoring introduces issues:

1. **Immediate Rollback**: Keep current files as `.backup` during refactor
2. **Partial Implementation**: Test each phase independently
3. **Performance Comparison**: Benchmark against previous implementations
4. **Gradual Migration**: Migrate features one by one
5. **Validation Gates**: Don't proceed without passing bias tests

---

## 📞 **SUPPORT RESOURCES**

- **NT Documentation**: `/docs/concepts/` for backtesting and indicators
- **NT Examples**: `examples/backtest/` for strategy templates
- **NT Community**: Discord for architecture questions
- **Guide Reference**: `NautilusTrader_Nutshell_Guide.md` sections 3.1, 8.2

---

**Last Updated**: 2025-07-19  
**Status**: Planning Phase  
**Next Action**: Begin Phase 1 cleanup and preparation