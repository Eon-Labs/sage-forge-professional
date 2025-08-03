# SAGE-Forge TiRex-Only Merit Isolation Plan: Professional Implementation

**Created**: 2025-08-03  
**Status**: Week 2 Enhanced Phase 0 Implementation  
**Context**: TiRex merit isolation using SAGE-Forge professional infrastructure  
**Related**: [Enhanced Phase 0 Progress](enhanced_phase_0_progress.md) | [SAGE Meta-Framework](sage_meta_framework_strategy.md)

---

## 🎯 **STRATEGIC RATIONALE: SAGE-FORGE ADVANTAGE**

### **Why SAGE-Forge for TiRex Implementation**:
- ✅ **Professional CLI**: `sage-create strategy TiRexOnlyStrategy` for instant template generation
- ✅ **Production Infrastructure**: Proven 8.7-second setup with 100% test coverage
- ✅ **Validated Framework**: ALL TESTS PASSED ✅ with identical results to original implementations
- ✅ **Modern Architecture**: src/ layout, exact dependency versions, UV optimization
- ✅ **Built-in Testing**: Comprehensive validation, comparison tools, performance metrics

### **SAGE Principle Application**:
> **"Professional infrastructure enables TiRex to discover its evaluation criteria from market structure without development friction"**

---

## 🏗️ **SAGE-FORGE TIREX-ONLY ARCHITECTURE**

### **Professional CLI Workflow**

```bash
# Step 1: Generate TiRex-Only Strategy Template
cd /Users/terryli/eon/nt/sage-forge
uv run sage-create strategy TiRexOnlyStrategy

# Step 2: Validate Environment
uv run sage-validate

# Step 3: Run TiRex Implementation
uv run python src/sage_forge/strategies/tirex_only_strategy.py

# Step 4: Professional Testing
uv run python test_sage_forge_tirex_validation.py
```

### **SAGE-Forge TiRex Strategy Implementation**

```python
"""
SAGE-Forge TiRex-Only Strategy for Merit Isolation
=================================================

Professional implementation leveraging SAGE-Forge infrastructure:
- CLI-generated template with best practices
- Built-in model integration framework
- Professional testing and validation
- Production-ready configuration system
"""

import numpy as np
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.strategy import Strategy
from rich.console import Console
import torch
from transformers import AutoModel

from sage_forge.core.config import get_config
from sage_forge.models.base import ModelBase
from sage_forge.risk.position_sizer import DynamicPositionSizer
from sage_forge.data.manager import DataManager

console = Console()


class TiRexModel(ModelBase):
    """
    SAGE-Forge TiRex model wrapper following professional patterns.
    """
    
    def __init__(self, forecast_horizons=[1, 4, 24]):
        super().__init__()
        self.forecast_horizons = forecast_horizons
        self.current_horizon = forecast_horizons[0]
        
        # TiRex model (lazy loaded)
        self.tirex_model = None
        self.model_loaded = False
        
        # SAGE-Forge performance tracking
        self.horizon_performance = {
            horizon: {'predictions': 0, 'accuracy': 0.0, 'uncertainty_avg': 0.0}
            for horizon in self.forecast_horizons
        }
        
    def initialize(self):
        """Initialize TiRex model with SAGE-Forge error handling."""
        try:
            self.tirex_model = AutoModel.from_pretrained(
                "NX-AI/TiRex",
                trust_remote_code=True
            )
            self.tirex_model.eval()
            self.model_loaded = True
            console.log("✅ TiRex model loaded successfully")
            
        except Exception as e:
            console.log(f"❌ TiRex model loading failed: {e}")
            self.model_loaded = False
            
    def forecast(self, price_data: np.ndarray, horizon: int) -> tuple[float, float]:
        """
        Generate TiRex forecast with uncertainty quantification.
        
        Returns:
            tuple[float, float]: (signal, uncertainty)
        """
        if not self.model_loaded:
            return 0.0, 1.0  # Neutral signal, high uncertainty
            
        try:
            # Prepare input for TiRex
            input_tensor = self._prepare_input(price_data)
            
            # Generate forecast
            with torch.no_grad():
                forecast_result = self.tirex_model.forecast(
                    context=input_tensor,
                    prediction_length=horizon
                )
                
                # Extract signal and uncertainty
                signal = self._extract_signal(forecast_result, price_data[-1])
                uncertainty = self._extract_uncertainty(forecast_result)
                
            # Track performance
            self.horizon_performance[horizon]['predictions'] += 1
            self.horizon_performance[horizon]['uncertainty_avg'] = (
                (self.horizon_performance[horizon]['uncertainty_avg'] * 
                 (self.horizon_performance[horizon]['predictions'] - 1) + uncertainty) /
                self.horizon_performance[horizon]['predictions']
            )
            
            return signal, uncertainty
            
        except Exception as e:
            console.log(f"❌ TiRex forecasting failed: {e}")
            return 0.0, 1.0
            
    def _prepare_input(self, price_data: np.ndarray) -> torch.Tensor:
        """Prepare price data for TiRex input."""
        # Convert to returns
        returns = np.diff(price_data) / price_data[:-1]
        
        # TiRex expects tensor format
        input_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)
        
        return input_tensor
        
    def _extract_signal(self, forecast_result, current_price: float) -> float:
        """Extract trading signal from TiRex forecast."""
        # Get median forecast
        forecast_price = forecast_result.prediction.median[-1]
        
        # Calculate expected return
        expected_return = (forecast_price - current_price) / current_price
        
        # Convert to bounded signal
        signal = np.tanh(expected_return * 10)  # Scale and bound to [-1, 1]
        
        return float(signal)
        
    def _extract_uncertainty(self, forecast_result) -> float:
        """Extract uncertainty from TiRex quantile predictions."""
        # Calculate uncertainty from quantile spread
        q10 = forecast_result.prediction.quantiles[0.1][-1]
        q90 = forecast_result.prediction.quantiles[0.9][-1]
        median = forecast_result.prediction.median[-1]
        
        # Uncertainty as relative quantile spread
        uncertainty = abs(q90 - q10) / (abs(median) + 1e-8)
        uncertainty = min(max(uncertainty, 0.0), 1.0)  # Bound to [0, 1]
        
        return float(uncertainty)


class TiRexOnlyStrategy(Strategy):
    """
    SAGE-Forge TiRex-Only Strategy for scientific merit isolation.
    
    Professional implementation leveraging:
    - SAGE-Forge configuration system
    - Built-in model framework
    - Professional position sizing
    - Comprehensive performance tracking
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # SAGE-Forge configuration
        self.sage_config = get_config()
        
        # Initialize TiRex model
        self.tirex_model = TiRexModel(forecast_horizons=[1, 4, 24])
        
        # SAGE-Forge data manager
        self.data_manager = DataManager()
        
        # SAGE-Forge position sizer
        self.position_sizer = DynamicPositionSizer(
            base_size=self.sage_config.base_position_size,
            max_size=self.sage_config.max_position_size
        )
        
        # Price buffer for forecasting
        self.price_buffer = []
        self.max_buffer_size = 200
        
        # Multi-horizon testing
        self.current_horizon_index = 0
        self.horizon_test_results = {}
        
        # Performance tracking by horizon
        self.performance_tracker = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'by_horizon': {},
            'by_uncertainty': {'low': {}, 'medium': {}, 'high': {}}
        }
        
    def on_start(self):
        """Initialize SAGE-Forge TiRex-Only strategy."""
        
        self.subscribe_bars(self.config.bar_type)
        
        # Initialize TiRex model
        self.tirex_model.initialize()
        
        console.log("🚀 SAGE-Forge TiRex-Only Strategy started")
        console.log(f"📊 Testing horizons: {self.tirex_model.forecast_horizons}")
        console.log(f"⚙️ SAGE-Forge config: {self.sage_config.version}")
        
    def on_bar(self, bar: Bar):
        """TiRex-only trading logic with SAGE-Forge infrastructure."""
        
        # Update price buffer
        self._update_price_buffer(bar.close.as_double())
        
        # Skip if insufficient data
        if len(self.price_buffer) < 50:
            return
            
        # Get current horizon for testing
        current_horizon = self.tirex_model.forecast_horizons[self.current_horizon_index]
        
        # Generate TiRex forecast
        signal, uncertainty = self.tirex_model.forecast(
            np.array(self.price_buffer), 
            current_horizon
        )
        
        # SAGE-Forge position sizing with uncertainty
        position_size = self.position_sizer.calculate_size(
            signal_strength=abs(signal),
            uncertainty=uncertainty,
            account_balance=self.account.balance()
        )
        
        # Dynamic threshold based on uncertainty
        threshold = self._get_uncertainty_threshold(uncertainty)
        
        # Execute trade if signal meets threshold
        if abs(signal) > threshold:
            self._execute_tirex_trade(signal, position_size, bar, current_horizon, uncertainty)
            
        # Cycle through horizons for comprehensive testing
        self._cycle_horizon_testing()
        
    def _update_price_buffer(self, price: float):
        """Maintain price buffer for TiRex context."""
        self.price_buffer.append(price)
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer.pop(0)
            
    def _get_uncertainty_threshold(self, uncertainty: float) -> float:
        """Dynamic threshold based on TiRex uncertainty."""
        base_threshold = self.sage_config.signal_threshold
        
        # Higher threshold when uncertain (more conservative)
        uncertainty_adjustment = uncertainty * 0.4
        return base_threshold + uncertainty_adjustment
        
    def _execute_tirex_trade(self, signal: float, size: float, bar: Bar, 
                           horizon: int, uncertainty: float):
        """Execute trade with comprehensive SAGE-Forge tracking."""
        
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        # Create order using SAGE-Forge order factory
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=size
        )
        
        # Submit order
        self.submit_order(order)
        
        # Track performance
        self._track_trade_performance(signal, size, bar, horizon, uncertainty)
        
        # Professional logging
        console.log(f"🎯 TiRex Trade [{horizon}h]: {side.name} {size} @ {bar.close}")
        console.log(f"📊 Signal: {signal:.3f}, Uncertainty: {uncertainty:.3f}")
        
    def _track_trade_performance(self, signal: float, size: float, bar: Bar,
                               horizon: int, uncertainty: float):
        """Comprehensive performance tracking by horizon and uncertainty."""
        
        # Overall tracking
        self.performance_tracker['total_trades'] += 1
        
        # By horizon
        if horizon not in self.performance_tracker['by_horizon']:
            self.performance_tracker['by_horizon'][horizon] = {
                'trades': 0, 'total_signal': 0.0, 'avg_uncertainty': 0.0
            }
            
        horizon_stats = self.performance_tracker['by_horizon'][horizon]
        horizon_stats['trades'] += 1
        horizon_stats['total_signal'] += abs(signal)
        horizon_stats['avg_uncertainty'] = (
            (horizon_stats['avg_uncertainty'] * (horizon_stats['trades'] - 1) + uncertainty) /
            horizon_stats['trades']
        )
        
        # By uncertainty level
        uncertainty_level = 'low' if uncertainty < 0.3 else 'high' if uncertainty > 0.7 else 'medium'
        if uncertainty_level not in self.performance_tracker['by_uncertainty']:
            self.performance_tracker['by_uncertainty'][uncertainty_level] = {'trades': 0}
        self.performance_tracker['by_uncertainty'][uncertainty_level]['trades'] += 1
        
    def _cycle_horizon_testing(self):
        """Cycle through forecast horizons for comprehensive testing."""
        # Simple rotation for testing (can be made more sophisticated)
        if self.performance_tracker['total_trades'] % 10 == 0:
            self.current_horizon_index = (
                (self.current_horizon_index + 1) % len(self.tirex_model.forecast_horizons)
            )
            current_horizon = self.tirex_model.forecast_horizons[self.current_horizon_index]
            console.log(f"🔄 Switched to {current_horizon}h forecast horizon")
            
    def on_stop(self):
        """SAGE-Forge performance summary with professional reporting."""
        
        console.log("=" * 60)
        console.log("🎯 SAGE-Forge TiRex-Only Merit Isolation Results")
        console.log("=" * 60)
        
        # Overall performance
        total_trades = self.performance_tracker['total_trades']
        console.log(f"📊 Total Trades: {total_trades}")
        
        # Performance by horizon
        console.log("\n🔍 Performance by Forecast Horizon:")
        for horizon, stats in self.performance_tracker['by_horizon'].items():
            avg_signal = stats['total_signal'] / max(stats['trades'], 1)
            console.log(f"  {horizon}h: {stats['trades']} trades, "
                       f"avg signal: {avg_signal:.3f}, avg uncertainty: {stats['avg_uncertainty']:.3f}")
                       
        # Performance by uncertainty
        console.log("\n🎲 Performance by Uncertainty Level:")
        for level, stats in self.performance_tracker['by_uncertainty'].items():
            console.log(f"  {level.title()}: {stats['trades']} trades")
            
        # TiRex model performance
        console.log("\n🤖 TiRex Model Performance:")
        for horizon, perf in self.tirex_model.horizon_performance.items():
            console.log(f"  {horizon}h: {perf['predictions']} predictions, "
                       f"avg uncertainty: {perf['uncertainty_avg']:.3f}")
```

---

## 📊 **SAGE-FORGE PROFESSIONAL TESTING FRAMEWORK**

### **Multi-Horizon Backtesting**

```python
class SAGEForgeTiRexBacktester:
    """
    Professional TiRex backtesting using SAGE-Forge infrastructure.
    """
    
    def __init__(self):
        self.sage_config = get_config()
        
    def run_horizon_comparison(self):
        """Run comprehensive horizon comparison using SAGE-Forge."""
        
        console.log("🚀 Starting SAGE-Forge TiRex Horizon Comparison")
        
        results = {}
        for horizon in [1, 4, 24]:
            console.log(f"📊 Testing {horizon}h horizon...")
            
            # Use SAGE-Forge testing infrastructure
            result = self._run_single_horizon_test(horizon)
            results[f"{horizon}h"] = result
            
        return self._generate_comparison_report(results)
        
    def _run_single_horizon_test(self, horizon: int):
        """Run single horizon test with SAGE-Forge infrastructure."""
        
        # Create TiRex strategy configuration
        config = self._create_tirex_config(horizon)
        
        # Use SAGE-Forge backtesting engine
        engine = self._create_sage_forge_engine()
        strategy = TiRexOnlyStrategy(config)
        
        # Run backtest
        results = engine.run(
            strategies=[strategy],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        return {
            'total_return': results.total_return(),
            'sharpe_ratio': results.sharpe_ratio(),
            'max_drawdown': results.max_drawdown(),
            'total_trades': results.total_trades(),
            'win_rate': results.win_rate()
        }
        
    def _generate_comparison_report(self, results):
        """Generate professional comparison report."""
        
        console.log("📈 SAGE-Forge TiRex Horizon Comparison Results")
        console.log("=" * 60)
        
        for horizon, metrics in results.items():
            console.log(f"\n{horizon} Forecast:")
            console.log(f"  Total Return: {metrics['total_return']:.2%}")
            console.log(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            console.log(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            console.log(f"  Total Trades: {metrics['total_trades']}")
            console.log(f"  Win Rate: {metrics['win_rate']:.2%}")
            
        # Identify optimal horizon
        best_horizon = max(results.keys(), key=lambda h: results[h]['sharpe_ratio'])
        console.log(f"\n🎯 Optimal Horizon: {best_horizon}")
        
        return results
```

### **SAGE-Forge Professional CLI Integration**

```bash
# Generate TiRex strategy template
uv run sage-create strategy TiRexOnlyStrategy

# Validate TiRex implementation
uv run sage-validate --strategy TiRexOnlyStrategy

# Run TiRex merit isolation
uv run sage-forge backtest --strategy TiRexOnlyStrategy --horizon-comparison

# Generate professional report
uv run sage-forge report --strategy TiRexOnlyStrategy --format comprehensive
```

---

## 🎯 **IMPLEMENTATION ROADMAP (SAGE-FORGE ENHANCED)**

### **Day 8-9: SAGE-Forge Professional Framework**
- [ ] Execute `uv run sage-create strategy TiRexOnlyStrategy` for professional template
- [ ] Implement TiRex model using SAGE-Forge model framework
- [ ] Multi-horizon testing using SAGE-Forge infrastructure
- [ ] **Milestone**: Professional TiRex strategy with CLI integration

### **Day 10-11: SAGE-Forge Merit Analysis**
- [ ] Run comprehensive backtesting using SAGE-Forge testing framework
- [ ] Compare vs SAGE-Forge benchmark strategies using built-in tools
- [ ] Optimize forecast horizons using SAGE-Forge validation
- [ ] **Milestone**: Quantified TiRex edge with professional metrics

### **Day 12-13: SAGE-Forge Validation Framework**
- [ ] Walk-forward analysis using SAGE-Forge backtesting engine
- [ ] Transaction cost analysis using SAGE-Forge funding integration
- [ ] Market regime analysis using SAGE-Forge reporting
- [ ] **Milestone**: Complete TiRex validation with professional documentation

### **Day 14: SAGE-Forge Professional Documentation**
- [ ] Generate documentation using SAGE-Forge CLI tools
- [ ] Export performance metrics in SAGE-Forge format
- [ ] Create baseline for future ensemble comparisons
- [ ] **Milestone**: Production-ready TiRex assessment

---

## ✅ **SAGE-FORGE SUCCESS CRITERIA**

### **Professional Implementation Gates**:
- [ ] **CLI Integration**: TiRex strategy generated via `sage-create` command
- [ ] **Framework Compliance**: Full SAGE-Forge model and strategy patterns
- [ ] **Testing Coverage**: 100% validation using SAGE-Forge testing suite
- [ ] **Performance Metrics**: Professional reporting with SAGE-Forge standards

### **Merit Validation Gates**:
- [ ] **Benchmark Superiority**: TiRex outperforms SAGE-Forge benchmark strategies
- [ ] **Horizon Optimization**: Clear optimal forecast window identified
- [ ] **Uncertainty Edge**: Measurable improvement from TiRex confidence estimates
- [ ] **Cost Viability**: Positive returns after SAGE-Forge funding integration

---

**Document Status**: ✅ **PROFESSIONAL SAGE-FORGE IMPLEMENTATION**  
**Key Innovation**: TiRex merit isolation using production-ready SAGE-Forge infrastructure  
**Next Action**: Execute `uv run sage-create strategy TiRexOnlyStrategy` in SAGE-Forge  
**Success Metric**: Professional TiRex implementation with comprehensive validation

---

**Last Updated**: 2025-08-03  
**Architecture**: SAGE-Forge professional CLI and testing framework  
**Implementation Priority**: CRITICAL - Leverage proven SAGE-Forge infrastructure for optimal results