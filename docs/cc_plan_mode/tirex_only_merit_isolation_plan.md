# TiRex-Only Merit Isolation Plan: Scientific Standalone Validation

**Created**: 2025-08-03  
**Status**: Week 2 Enhanced Phase 0 Implementation  
**Context**: Isolate TiRex's trading merit before ensemble complexity per SAGE principle  
**Related**: [Enhanced Phase 0 Progress](enhanced_phase_0_progress.md) | [SAGE Meta-Framework](sage_meta_framework_strategy.md)

---

## 🎯 **STRATEGIC RATIONALE: SCIENTIFIC VARIABLE ISOLATION**

### **Core Research Questions**:
1. **Signal Quality**: Can TiRex forecasts alone generate profitable trades vs buy-and-hold?
2. **Uncertainty Edge**: Do TiRex confidence estimates improve Sharpe ratios and drawdown control?
3. **Optimal Horizon**: What forecast length (1h, 4h, 24h) works best for crypto trading?
4. **Market Regime Performance**: Where does TiRex excel vs. fail across market conditions?
5. **Transaction Cost Break-Even**: At what accuracy does TiRex overcome trading fees and slippage?

### **SAGE Principle Application**:
> **"Let TiRex discover its own evaluation criteria from market structure rather than inherit our biases about what matters"**

---

## 🏗️ **NT-NATIVE TIREX-ONLY ARCHITECTURE**

### **Pure TiRex Strategy Implementation**

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
import torch
from transformers import AutoModel

class TiRexOnlyStrategy(Strategy):
    """
    Pure TiRex forecasting strategy for merit isolation.
    NT-native implementation using existing infrastructure.
    """
    
    def __init__(self, config: TiRexOnlyConfig):
        super().__init__(config)
        
        # TiRex model configuration
        self.forecast_horizons = config.forecast_horizons  # [1, 4, 24] hours
        self.current_horizon = config.forecast_horizons[0]  # Start with 1h
        self.confidence_threshold = config.confidence_threshold  # 0.6
        
        # TiRex model (lazy loaded)
        self.tirex_model = None
        self.model_loaded = False
        
        # Performance tracking by horizon
        self.horizon_performance = {
            horizon: {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0}
            for horizon in self.forecast_horizons
        }
        
        # Market data buffer for forecasting
        self.price_buffer = []
        self.max_buffer_size = 200  # Context for TiRex
        
        # Signal tracking
        self.last_forecast = None
        self.last_uncertainty = 0.5
        self.bars_since_forecast = 0
        
    def on_start(self):
        """Initialize TiRex-only strategy."""
        
        self.subscribe_bars(self.config.bar_type)
        self.log.info("TiRex-Only Merit Isolation Strategy started")
        
        # Lazy load TiRex model
        self._initialize_tirex_model()
        
    def on_bar(self, bar: Bar):
        """Pure TiRex trading logic - no ensemble complexity."""
        
        # Update price buffer
        self._update_price_buffer(bar.close.as_double())
        
        # Skip if insufficient data or model not loaded
        if not self._ready_for_forecasting():
            return
            
        # Generate TiRex forecast and uncertainty
        forecast_signal, uncertainty = self._generate_tirex_forecast()
        
        # Apply uncertainty-based position sizing
        position_size = self._calculate_uncertainty_position_size(
            forecast_signal, uncertainty
        )
        
        # Execute trade if signal meets confidence threshold
        if abs(forecast_signal) > self._get_dynamic_threshold(uncertainty):
            self._execute_tirex_trade(forecast_signal, position_size, bar)
            
    def _initialize_tirex_model(self):
        """Initialize TiRex model with proper error handling."""
        
        try:
            # Load TiRex model
            self.tirex_model = AutoModel.from_pretrained(
                "NX-AI/TiRex",
                trust_remote_code=True
            )
            self.tirex_model.eval()
            self.model_loaded = True
            self.log.info("TiRex model loaded successfully")
            
        except Exception as e:
            self.log.error(f"TiRex model loading failed: {e}")
            self.model_loaded = False
            
    def _update_price_buffer(self, price: float):
        """Maintain price buffer for TiRex context."""
        
        self.price_buffer.append(price)
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer.pop(0)
            
    def _ready_for_forecasting(self) -> bool:
        """Check if ready for TiRex forecasting."""
        
        return (
            self.model_loaded and 
            len(self.price_buffer) >= 50 and
            self.bars_since_forecast >= self.current_horizon
        )
        
    def _generate_tirex_forecast(self) -> tuple[float, float]:
        """Generate TiRex forecast and uncertainty estimate."""
        
        try:
            # Prepare input data for TiRex
            input_data = self._prepare_tirex_input()
            
            # Generate forecast with uncertainty quantification
            with torch.no_grad():
                # TiRex inference for specified horizon
                forecast_result = self.tirex_model.forecast(
                    context=input_data,
                    prediction_length=self.current_horizon
                )
                
                # Extract signal and uncertainty
                signal = self._extract_trading_signal(forecast_result)
                uncertainty = self._extract_uncertainty(forecast_result)
                
            self.last_forecast = signal
            self.last_uncertainty = uncertainty
            self.bars_since_forecast = 0
            
            return signal, uncertainty
            
        except Exception as e:
            self.log.error(f"TiRex forecasting failed: {e}")
            # Fallback to neutral signal
            return 0.0, 1.0  # High uncertainty when model fails
            
    def _prepare_tirex_input(self) -> torch.Tensor:
        """Prepare price data for TiRex input."""
        
        # Use recent price history as context
        context_prices = self.price_buffer[-100:]  # Last 100 bars
        
        # Convert to returns for TiRex
        returns = []
        for i in range(1, len(context_prices)):
            ret = (context_prices[i] - context_prices[i-1]) / context_prices[i-1]
            returns.append(ret)
            
        # Convert to tensor format expected by TiRex
        input_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)
        
        return input_tensor
        
    def _extract_trading_signal(self, forecast_result) -> float:
        """Extract trading signal from TiRex forecast."""
        
        # Extract median forecast and compare to current price
        forecast_prices = forecast_result.prediction.median
        current_price = self.price_buffer[-1]
        
        # Calculate expected return
        expected_return = (forecast_prices[-1] - current_price) / current_price
        
        # Convert to signal (-1 to 1)
        signal = np.tanh(expected_return * 10)  # Scale and bound
        
        return float(signal)
        
    def _extract_uncertainty(self, forecast_result) -> float:
        """Extract uncertainty from TiRex quantile predictions."""
        
        # Calculate uncertainty from quantile spread
        q10 = forecast_result.prediction.quantiles[0.1]
        q90 = forecast_result.prediction.quantiles[0.9]
        median = forecast_result.prediction.median
        
        # Uncertainty as relative quantile spread
        uncertainty = (q90[-1] - q10[-1]) / (median[-1] + 1e-8)
        uncertainty = min(max(uncertainty, 0.0), 1.0)  # Bound 0-1
        
        return float(uncertainty)
        
    def _calculate_uncertainty_position_size(self, signal: float, uncertainty: float) -> float:
        """Calculate position size based on TiRex uncertainty."""
        
        # Base position size
        base_size = self.config.base_position_size
        
        # Signal strength factor
        signal_factor = abs(signal)
        
        # Uncertainty adjustment (reduce size when uncertain)
        uncertainty_factor = 1.0 - uncertainty * 0.8
        
        # Combined position size
        position_size = base_size * signal_factor * uncertainty_factor
        
        return min(position_size, self.config.max_position_size)
        
    def _get_dynamic_threshold(self, uncertainty: float) -> float:
        """Dynamic signal threshold based on uncertainty."""
        
        base_threshold = self.config.signal_threshold
        
        # Higher threshold when uncertain
        uncertainty_adjustment = uncertainty * 0.5
        dynamic_threshold = base_threshold + uncertainty_adjustment
        
        return dynamic_threshold
        
    def _execute_tirex_trade(self, signal: float, size: float, bar: Bar):
        """Execute trade based on TiRex signal."""
        
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        # Create market order
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=size
        )
        
        # Submit order
        self.submit_order(order)
        
        # Track performance by horizon
        self.horizon_performance[self.current_horizon]['trades'] += 1
        
        self.log.info(f"TiRex-only trade [{self.current_horizon}h]: {side.name} {size} @ {bar.close}")
        self.log.info(f"Signal: {signal:.3f}, Uncertainty: {self.last_uncertainty:.3f}")
        
    def on_stop(self):
        """Log TiRex-only performance summary."""
        
        self.log.info("=== TiRex-Only Merit Isolation Results ===")
        for horizon, perf in self.horizon_performance.items():
            self.log.info(f"{horizon}h horizon: {perf['trades']} trades, PnL: {perf['pnl']:.2f}")
```

---

## 📊 **MULTI-HORIZON TESTING FRAMEWORK**

### **Systematic Horizon Comparison**

```python
class TiRexHorizonBacktester:
    """
    NT-native backtesting for TiRex horizon optimization.
    """
    
    def __init__(self, horizons=[1, 4, 24]):
        self.horizons = horizons
        self.results = {}
        
    def run_horizon_comparison(self, symbol="BTCUSDT", start_date="2024-01-01"):
        """Compare TiRex performance across forecast horizons."""
        
        for horizon in self.horizons:
            # Configure TiRex strategy for this horizon
            config = TiRexOnlyConfig(
                forecast_horizons=[horizon],
                confidence_threshold=0.6,
                base_position_size=0.002,  # Conservative sizing
                signal_threshold=0.3
            )
            
            # Run NT native backtest
            engine = self._create_backtest_engine()
            strategy = TiRexOnlyStrategy(config)
            
            # Execute backtest
            results = engine.run(
                strategies=[strategy],
                symbol=symbol,
                start=start_date
            )
            
            # Store results
            self.results[f"{horizon}h"] = {
                'total_return': results.total_return(),
                'sharpe_ratio': results.sharpe_ratio(),
                'max_drawdown': results.max_drawdown(),
                'win_rate': results.win_rate(),
                'total_trades': results.total_trades()
            }
            
        return self.results
        
    def _create_backtest_engine(self):
        """Create NT backtesting engine."""
        
        # Use existing NT infrastructure
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.config import BacktestEngineConfig
        
        config = BacktestEngineConfig(
            # Leverage existing data pipeline
            data_engine_config=self._get_dsm_data_config(),
            risk_engine_config=self._get_risk_config(),
            exec_engine_config=self._get_execution_config()
        )
        
        return BacktestEngine(config)
```

---

## 🔬 **BENCHMARK COMPARISON FRAMEWORK**

### **TiRex vs Simple Strategies**

```python
class TiRexBenchmarkAnalysis:
    """
    Compare TiRex-only against simple benchmarks.
    """
    
    def __init__(self):
        self.benchmark_strategies = [
            'buy_and_hold',
            'momentum_20',
            'mean_reversion_20',
            'random_walk'
        ]
        
    def run_benchmark_comparison(self):
        """Run TiRex vs benchmarks on same data."""
        
        results = {}
        
        # TiRex-only strategy
        tirex_results = self._run_tirex_strategy()
        results['tirex_only'] = tirex_results
        
        # Simple benchmarks
        for benchmark in self.benchmark_strategies:
            bench_results = self._run_benchmark_strategy(benchmark)
            results[benchmark] = bench_results
            
        # Comparative analysis
        comparison = self._analyze_relative_performance(results)
        
        return results, comparison
        
    def _analyze_relative_performance(self, results):
        """Analyze TiRex vs benchmark performance."""
        
        tirex_sharpe = results['tirex_only']['sharpe_ratio']
        
        comparison = {}
        for benchmark, result in results.items():
            if benchmark != 'tirex_only':
                sharpe_diff = tirex_sharpe - result['sharpe_ratio']
                comparison[benchmark] = {
                    'sharpe_improvement': sharpe_diff,
                    'return_difference': results['tirex_only']['total_return'] - result['total_return'],
                    'drawdown_improvement': result['max_drawdown'] - results['tirex_only']['max_drawdown']
                }
                
        return comparison
```

---

## 🎯 **IMPLEMENTATION ROADMAP (DAYS 8-14)**

### **Day 8-9: TiRex-Only Framework**
- [ ] Implement `TiRexOnlyStrategy` with NT-native patterns
- [ ] Multi-horizon testing framework (1h, 4h, 24h)
- [ ] Uncertainty-based position sizing implementation
- [ ] **Milestone**: Pure TiRex backtesting operational

### **Day 10-11: Merit Analysis**
- [ ] Run TiRex vs benchmark comparison on BTCUSDT data
- [ ] Identify optimal forecast horizon through empirical testing
- [ ] Analyze TiRex uncertainty edge in position sizing
- [ ] **Milestone**: Quantified TiRex trading merit established

### **Day 12-13: Validation Framework**
- [ ] NT native walk-forward analysis with TiRex-only signals
- [ ] Transaction cost break-even analysis (fees, slippage)
- [ ] Market regime performance segmentation analysis
- [ ] **Milestone**: TiRex viability confirmed across market conditions

### **Day 14: Documentation & Baseline**
- [ ] Document TiRex standalone merit and optimal configurations
- [ ] Establish performance baseline for future ensemble comparisons
- [ ] Create TiRex trading merit assessment report
- [ ] **Milestone**: Complete TiRex-only validation ready for ensemble consideration

---

## ✅ **SUCCESS CRITERIA**

### **Merit Validation Gates**:
- [ ] **Profitability**: TiRex-only strategy beats buy-and-hold baseline
- [ ] **Optimal Horizon**: Clear best-performing forecast window identified
- [ ] **Uncertainty Edge**: TiRex confidence estimates improve risk-adjusted returns
- [ ] **Transaction Viability**: TiRex signals overcome real trading costs
- [ ] **Market Robustness**: Positive performance across different market regimes

### **Technical Validation**:
- [ ] **NT Compliance**: Full integration with existing NT infrastructure
- [ ] **DSM Integration**: Seamless data pipeline utilization
- [ ] **Resource Efficiency**: TiRex inference completes within acceptable timeframes
- [ ] **Error Handling**: Graceful degradation when TiRex model fails

---

**Document Status**: ✅ **SCIENTIFIC BASELINE STRATEGY**  
**Key Innovation**: Pure TiRex merit isolation using NT-native backtesting  
**Next Action**: Begin Day 8 implementation of TiRex-only framework  
**Success Metric**: Proven TiRex standalone trading value before ensemble complexity

---

**Last Updated**: 2025-08-03  
**Architecture**: NT-native TiRex-only validation  
**Implementation Priority**: CRITICAL - Scientific variable isolation for SAGE framework