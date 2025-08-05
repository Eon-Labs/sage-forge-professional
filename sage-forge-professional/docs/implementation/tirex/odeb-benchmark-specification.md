# ODEB: Omniscient Directional Efficiency Benchmark Specification

**Version**: 1.0  
**Date**: August 5, 2025  
**Context**: Phase 3A Evolution following Phase 2 Completion  
**Dependencies**: TiRex adversarial audit remediation, walk-forward optimization framework  
**Status**: Implementation specification for session continuity

---

## Executive Summary

The **Omniscient Directional Efficiency Benchmark (ODEB)** provides a systematic methodology for evaluating TiRex strategy performance against a theoretical perfect-information baseline. This evolution from Phase 2 completion addresses the critical gap between signal generation quality and portfolio management effectiveness through risk-adjusted directional capture analysis.

### Core Innovation

ODEB measures how effectively TiRex strategies capture directional moves compared to a theoretical "crystal ball" strategy that knows the optimal direction for the entire trading period, using identical position sizing and duration constraints.

---

## Mathematical Framework

### 1. Time-Weighted Average Exposure (TWAE)

**Purpose**: Calculate oracle position size matching TiRex capital deployment pattern

**Formula**:
```
Oracle_Position_Size = Σ(Position_Size_i × Duration_i) / Total_Active_Duration
```

**Implementation**:
```python
def calculate_twae(self, positions: List[Position]) -> float:
    """Calculate time-weighted average exposure across all positions."""
    total_exposure_time = 0.0
    total_duration = 0.0
    
    for position in positions:
        duration = (position.close_time - position.open_time).total_seconds() / 86400  # days
        exposure = abs(position.size_usd) * duration
        total_exposure_time += exposure
        total_duration += duration
    
    return total_exposure_time / total_duration if total_duration > 0 else 0.0
```

### 2. Duration-Scaled Quantile Market Noise Floor (DSQMNF)

**Purpose**: Establish minimum adverse excursion threshold to handle near-zero drawdown cases

**Formula**:
```
Min_Adverse_Excursion = Historical_Noise_Floor × sqrt(Position_Duration_Days / Median_Historical_Duration)
```

**Implementation**:
```python
def calculate_noise_floor(self, market_data: pd.DataFrame, position_duration_days: int, lookback_days: int = 252) -> float:
    """Calculate duration-scaled quantile market noise floor."""
    # Calculate historical adverse excursions for perfect positions
    adverse_excursions = []
    
    for window_start in range(len(market_data) - position_duration_days):
        window_end = window_start + position_duration_days
        window_data = market_data.iloc[window_start:window_end]
        
        # Simulate perfect directional position
        direction = np.sign(window_data.iloc[-1]['close'] - window_data.iloc[0]['close'])
        perfect_pnl_series = direction * (window_data['close'] - window_data.iloc[0]['close'])
        
        # Calculate maximum adverse excursion
        running_min = perfect_pnl_series.expanding().min()
        max_adverse = abs(running_min.min())
        adverse_excursions.append(max_adverse)
    
    # 15th percentile represents typical market noise
    historical_noise_floor = np.percentile(adverse_excursions, 15)
    
    # Duration scaling (square root of time for Brownian motion)
    median_duration = np.median([pos.duration_days for pos in self.historical_positions])
    duration_scalar = np.sqrt(position_duration_days / median_duration)
    
    return historical_noise_floor * duration_scalar
```

### 3. Directional Efficiency Ratio

**Purpose**: Risk-adjusted performance comparison between TiRex and oracle strategies

**Formula**:
```
Efficiency_Ratio = Final_PnL / max(Max_Drawdown, Noise_Floor)
Directional_Capture = TiRex_Efficiency_Ratio / Oracle_Efficiency_Ratio
```

---

## Implementation Blueprint

### Core Classes

#### 1. OmniscientDirectionalEfficiencyBenchmark

**Location**: `src/sage_forge/reporting/performance.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

@dataclass
class OdebResult:
    """Results from ODEB analysis."""
    tirex_efficiency_ratio: float
    oracle_efficiency_ratio: float
    directional_capture_pct: float
    oracle_direction: int  # 1 for LONG, -1 for SHORT
    oracle_position_size: float
    noise_floor_applied: float
    tirex_final_pnl: float
    oracle_final_pnl: float
    tirex_max_drawdown: float
    oracle_max_drawdown: float
    
@dataclass 
class Position:
    """Position data structure for ODEB analysis."""
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    size_usd: float
    pnl: float
    direction: int  # 1 for LONG, -1 for SHORT
    
    @property
    def duration_days(self) -> float:
        return (self.close_time - self.open_time).total_seconds() / 86400

class OmniscientDirectionalEfficiencyBenchmark:
    """
    Omniscient Directional Efficiency Benchmark (ODEB) Framework
    
    Measures TiRex strategy effectiveness against theoretical perfect-information baseline
    using time-weighted position sizing and duration-scaled noise floor methodology.
    """
    
    def __init__(self, historical_positions: List[Position]):
        self.historical_positions = historical_positions
        
    def calculate_twae(self, positions: List[Position]) -> float:
        """Calculate time-weighted average exposure."""
        # Implementation as shown above
        
    def calculate_noise_floor(self, market_data: pd.DataFrame, position_duration_days: int) -> float:
        """Calculate duration-scaled quantile market noise floor."""
        # Implementation as shown above
        
    def simulate_oracle_strategy(self, 
                                direction: int, 
                                position_size: float, 
                                market_data: pd.DataFrame,
                                start_time: pd.Timestamp,
                                end_time: pd.Timestamp) -> Tuple[float, float]:
        """Simulate oracle strategy performance."""
        entry_price = market_data.loc[start_time, 'close']
        exit_price = market_data.loc[end_time, 'close']
        
        # Calculate oracle P&L
        oracle_pnl = direction * position_size * (exit_price - entry_price) / entry_price
        
        # Calculate oracle maximum drawdown
        period_data = market_data.loc[start_time:end_time]
        oracle_pnl_series = direction * position_size * (period_data['close'] - entry_price) / entry_price
        running_max = oracle_pnl_series.expanding().max()
        drawdown_series = oracle_pnl_series - running_max
        oracle_max_drawdown = abs(drawdown_series.min())
        
        return oracle_pnl, oracle_max_drawdown
        
    def calculate_odeb_ratio(self, 
                           tirex_positions: List[Position],
                           market_data: pd.DataFrame) -> OdebResult:
        """Calculate complete ODEB analysis."""
        
        # Determine oracle direction (sign of net price movement)
        start_time = min(pos.open_time for pos in tirex_positions)
        end_time = max(pos.close_time for pos in tirex_positions)
        
        start_price = market_data.loc[start_time, 'close']
        end_price = market_data.loc[end_time, 'close']
        oracle_direction = 1 if end_price > start_price else -1
        
        # Calculate time-weighted average exposure
        oracle_position_size = self.calculate_twae(tirex_positions)
        
        # Simulate oracle performance
        oracle_pnl, oracle_max_drawdown = self.simulate_oracle_strategy(
            oracle_direction, oracle_position_size, market_data, start_time, end_time
        )
        
        # Calculate TiRex performance metrics
        tirex_final_pnl = sum(pos.pnl for pos in tirex_positions)
        tirex_max_drawdown = self._calculate_tirex_max_drawdown(tirex_positions, market_data)
        
        # Apply noise floor
        position_duration = (end_time - start_time).days
        noise_floor = self.calculate_noise_floor(market_data, position_duration)
        
        # Calculate efficiency ratios
        tirex_efficiency = tirex_final_pnl / max(tirex_max_drawdown, noise_floor)
        oracle_efficiency = oracle_pnl / max(oracle_max_drawdown, noise_floor)
        
        directional_capture = (tirex_efficiency / oracle_efficiency * 100) if oracle_efficiency != 0 else 0
        
        return OdebResult(
            tirex_efficiency_ratio=tirex_efficiency,
            oracle_efficiency_ratio=oracle_efficiency,
            directional_capture_pct=directional_capture,
            oracle_direction=oracle_direction,
            oracle_position_size=oracle_position_size,
            noise_floor_applied=noise_floor,
            tirex_final_pnl=tirex_final_pnl,
            oracle_final_pnl=oracle_pnl,
            tirex_max_drawdown=tirex_max_drawdown,
            oracle_max_drawdown=oracle_max_drawdown
        )
    
    def _calculate_tirex_max_drawdown(self, positions: List[Position], market_data: pd.DataFrame) -> float:
        """Calculate maximum drawdown for TiRex strategy."""
        # Reconstruct P&L series from positions
        pnl_series = pd.Series(dtype=float)
        
        for position in positions:
            position_data = market_data.loc[position.open_time:position.close_time]
            position_pnl = position.direction * position.size_usd * (
                position_data['close'] - position_data.iloc[0]['close']
            ) / position_data.iloc[0]['close']
            pnl_series = pd.concat([pnl_series, position_pnl])
        
        pnl_series = pnl_series.sort_index()
        running_max = pnl_series.expanding().max()
        drawdown_series = pnl_series - running_max
        
        return abs(drawdown_series.min())
```

### Integration Points

#### 1. Walk-Forward Optimization Integration

**Location**: `src/sage_forge/optimization/tirex_parameter_optimizer.py`

```python
# Add to TiRexParameterOptimizer class
def evaluate_with_odeb(self, optimization_result: OptimizationResult) -> Dict[str, float]:
    """Evaluate optimization results using ODEB methodology."""
    
    odeb_benchmark = OmniscientDirectionalEfficiencyBenchmark(self.historical_positions)
    
    # Calculate ODEB for each walk-forward window
    window_results = []
    for window in self.walk_forward_windows:
        window_positions = self._get_window_positions(optimization_result, window)
        window_market_data = self._get_window_market_data(window)
        
        odeb_result = odeb_benchmark.calculate_odeb_ratio(window_positions, window_market_data)
        window_results.append(odeb_result)
    
    # Aggregate results
    avg_directional_capture = np.mean([r.directional_capture_pct for r in window_results])
    avg_efficiency_ratio = np.mean([r.tirex_efficiency_ratio for r in window_results])
    
    return {
        'odeb_directional_capture': avg_directional_capture,
        'odeb_efficiency_ratio': avg_efficiency_ratio,
        'odeb_consistency': np.std([r.directional_capture_pct for r in window_results])
    }
```

#### 2. Backtesting Engine Integration  

**Location**: `src/sage_forge/backtesting/tirex_backtest_engine.py`

```python
# Add ODEB analysis to backtest results
def run_backtest_with_odeb(self, config: BacktestConfig) -> Dict[str, Any]:
    """Run backtest with ODEB analysis."""
    
    # Standard backtest execution
    backtest_result = self.run_standard_backtest(config)
    
    # Extract positions from backtest result
    positions = self._extract_positions_from_result(backtest_result)
    
    # Run ODEB analysis
    odeb_benchmark = OmniscientDirectionalEfficiencyBenchmark(self.historical_positions)
    odeb_result = odeb_benchmark.calculate_odeb_ratio(positions, self.market_data)
    
    # Combine results
    return {
        'backtest_result': backtest_result,
        'odeb_analysis': odeb_result,
        'combined_metrics': self._combine_metrics(backtest_result, odeb_result)
    }
```

---

## Validation Framework

### Test Cases

#### 1. Perfect Directional Strategy Test
```python
def test_perfect_directional_strategy():
    """Test ODEB with perfect directional strategy (should achieve 100% capture)."""
    # Create synthetic perfect positions that match oracle direction
    # Expected: directional_capture_pct ≈ 100%
```

#### 2. Random Strategy Test  
```python
def test_random_strategy():
    """Test ODEB with random directional strategy."""
    # Create random positions
    # Expected: directional_capture_pct ≈ 50% over many trials
```

#### 3. Noise Floor Validation Test
```python
def test_noise_floor_calculation():
    """Validate noise floor calculation methodology."""
    # Test with different market conditions and position durations
    # Expected: noise floor scales appropriately with duration and volatility
```

#### 4. Edge Cases
```python
def test_zero_drawdown_case():
    """Test ODEB when strategy has zero drawdown."""
    # Expected: noise floor applied correctly
    
def test_single_position():
    """Test ODEB with single position."""
    # Expected: TWAE equals position size
```

### Performance Benchmarks

**Expected Ranges for Different Strategy Types**:
- **Trend Following**: 60-80% directional capture
- **Mean Reversion**: 40-60% directional capture  
- **Random/Noise**: 45-55% directional capture
- **Perfect Oracle**: 100% directional capture

---

## Usage Examples

### Basic ODEB Analysis
```python
from sage_forge.reporting.performance import OmniscientDirectionalEfficiencyBenchmark

# Initialize ODEB with historical data
odeb = OmniscientDirectionalEfficiencyBenchmark(historical_positions)

# Analyze TiRex strategy performance
tirex_positions = load_tirex_positions("2024-01-01", "2024-03-31")
market_data = load_market_data("2024-01-01", "2024-03-31")

result = odeb.calculate_odeb_ratio(tirex_positions, market_data)

print(f"Directional Capture: {result.directional_capture_pct:.1f}%")
print(f"TiRex Efficiency: {result.tirex_efficiency_ratio:.3f}")
print(f"Oracle Efficiency: {result.oracle_efficiency_ratio:.3f}")
```

### Walk-Forward ODEB Analysis
```python
from sage_forge.optimization.tirex_parameter_optimizer import TiRexParameterOptimizer

optimizer = TiRexParameterOptimizer(config)
optimization_result = optimizer.optimize_parameters()

# Evaluate with ODEB
odeb_metrics = optimizer.evaluate_with_odeb(optimization_result)
print(f"Average Directional Capture: {odeb_metrics['odeb_directional_capture']:.1f}%")
```

---

## Implementation Checklist

### Phase 3A.1: Core ODEB Framework (Week 1)
- [ ] Create `OmniscientDirectionalEfficiencyBenchmark` class
- [ ] Implement TWAE calculation method
- [ ] Implement DSQMNF noise floor methodology
- [ ] Create `OdebResult` data structure
- [ ] Add basic validation tests

### Phase 3A.2: Integration Layer (Week 2)
- [ ] Integrate ODEB into `TiRexParameterOptimizer`
- [ ] Add ODEB support to backtesting engine
- [ ] Create position extraction utilities
- [ ] Implement market data alignment functions

### Phase 3A.3: Validation & Testing (Week 3)
- [ ] Comprehensive test suite for all edge cases
- [ ] Performance benchmark validation
- [ ] Statistical significance testing
- [ ] Documentation validation examples

### Phase 3A.4: Production Integration (Week 4)
- [ ] Real-time ODEB calculation support
- [ ] Performance monitoring dashboard
- [ ] Automated reporting integration
- [ ] Final validation and deployment

---

## Future Session Recovery

### Quick Start Guide
1. **Check Implementation Status**: Review this specification for current progress
2. **Core Files**: Start with `src/sage_forge/reporting/performance.py` for main ODEB class
3. **Integration Points**: Reference `src/sage_forge/optimization/tirex_parameter_optimizer.py` for walk-forward integration
4. **Test Files**: Look for `test_odeb_*.py` files for validation status
5. **Examples**: Check `demos/` directory for ODEB usage examples

### Key Reference Files
- **Mathematical Foundation**: This specification document
- **Existing Patterns**: `validate_phase2_completion.py` for validation framework
- **TiRex Integration**: `test_authentic_tirex_signals.py` for TiRex usage patterns
- **Performance Reporting**: Current `performance.py` for integration patterns

---

## SR&ED Evidence Chain

### Technical Advancement
ODEB represents systematic advancement in quantitative finance benchmarking through:
1. **Time-weighted exposure matching** between actual and theoretical strategies
2. **Duration-scaled noise floor methodology** for robust risk adjustment
3. **Directional capture efficiency** as novel performance metric

### Innovation Context
Evolution from Phase 2 TiRex integration addressing:
- Gap between signal quality and portfolio performance
- Need for risk-adjusted strategy evaluation
- Requirement for theoretical performance bounds

### Research Contribution
ODEB methodology provides:
- Parameter-free benchmarking framework
- Statistical significance for strategy evaluation  
- Integration with walk-forward validation systems

---

**Document Status**: Complete specification for implementation continuity  
**Next Steps**: Begin implementation with `OmniscientDirectionalEfficiencyBenchmark` class  
**Dependencies**: Phase 2 completion, existing TiRex infrastructure, walk-forward optimization framework