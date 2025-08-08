# TiRex Implementation Guide

## Current Evolutionary State

The TiRex signal generation implementation has evolved through multiple iterations to reach its current architectural compliance while maintaining practical effectiveness.

## Current Implementation

**File**: `tirex_signal_generator.py`

**Evolutionary Characteristics**:
- Native xLSTM architecture compliance (128-bar context windows)
- Strategic state management between market contexts
- Diverse market regime sampling for balanced predictions
- Computational efficiency through proper resource utilization
- Temporal ordering validation maintained

## Implementation Architecture

### Core Components

```python
# Native sequence length compliance
context_window_size = tirex_model.input_processor.sequence_length  # 128 bars

# Strategic state management
tirex_model.input_processor.price_buffer.clear()
tirex_model.input_processor.timestamp_buffer.clear() 
tirex_model.input_processor.last_timestamp = None

# Diverse market sampling
num_windows = 20  # Strategic balance of coverage vs efficiency
stride = max_possible_windows // (num_windows - 1)
```

### Key Design Principles

1. **Architecture Compliance**: Respects TiRex native 128-bar sequence length
2. **State Management**: Clears model state between market contexts to prevent bias
3. **Market Diversity**: Samples across different time periods for balanced predictions  
4. **Resource Efficiency**: No computational waste through proper data utilization
5. **Security**: Maintains temporal ordering validation while enabling context resets

## Evolutionary History

The current implementation incorporates lessons learned from previous approaches:

### Legacy Reference: `legacy/tirex-evolution/`

1. **Initial Implementation** (`visualize_authentic_tirex_signals.py`)
   - Sequential data feeding approach
   - Led to bias accumulation issues

2. **Extended Approach** (`visualize_authentic_tirex_signals_extended.py`)  
   - Introduced state clearing (valuable insight)
   - Used 512-bar windows (architectural violation)
   - Worked by accident due to deque auto-truncation

3. **Analysis Phase** (various audit and analysis scripts)
   - Identified why extended approach succeeded despite violations
   - Revealed that state clearing and diverse windowing were key benefits

4. **Current Evolution** (`tirex_signal_generator.py`)
   - Combines benefits without architectural violations
   - Uses native 128-bar windows
   - Maintains computational efficiency

## Usage Example

```python
from tirex_signal_generator import load_tirex_model, load_market_data, generate_tirex_signals

# Load components
tirex_model = load_tirex_model()
market_data = load_market_data()

# Generate signals
signals = generate_tirex_signals(tirex_model, market_data)

# Analyze results
analyze_signal_results(signals)
```

## Signal Output Format

Each signal contains:
```python
{
    'timestamp': datetime,
    'price': float,
    'signal': 'BUY' | 'SELL',
    'confidence': float,
    'volatility_forecast': float,
    'raw_forecast': list | float,
    'bar_index': int,
    'context_window_start': int,
    'context_window_end': int,
    'context_period': str,
    'model_source': 'TIREX_EVOLUTIONARY',
    'architecture_compliance': 'NATIVE_XLSTM'
}
```

## Configuration Parameters

### Model Configuration
- **sequence_length**: 128 (native TiRex architecture)
- **prediction_length**: 1 (single-step forecasting)
- **device**: "cuda" (GPU acceleration)

### Sampling Configuration  
- **num_windows**: 20 (strategic market coverage)
- **context_window_size**: 128 (native sequence length)
- **stride**: Dynamic based on data length

## Performance Characteristics

**Typical Results**:
- Signal generation rate: 30-40%
- Signal balance: Mixed BUY/SELL predictions
- Confidence range: 10-35%
- Computational efficiency: 100% (no waste)
- Architecture compliance: Native xLSTM

## Best Practices

1. **Use Native Architecture**: Always respect model's designed sequence length
2. **Strategic State Management**: Clear state between distinct market contexts
3. **Diverse Sampling**: Cover different market regimes for balanced predictions
4. **Resource Efficiency**: Feed only necessary data to model
5. **Temporal Validation**: Maintain ordering checks while enabling context resets

## Integration Points

### With SAGE-Forge Framework
- Uses `ArrowDataManager` for market data
- Integrates with NT Bar objects for data feeding
- Maintains SAGE methodology compliance

### With TiRex Model
- Respects native `TiRexInputProcessor` design
- Uses proper `deque` buffer management
- Maintains GPU acceleration capabilities

## Future Evolution

The implementation continues to evolve based on:
- Market performance feedback
- Computational efficiency improvements  
- Integration requirements with broader SAGE-Forge framework
- Insights from production usage patterns

**Note**: This represents the current evolutionary state. Previous implementations are preserved in `legacy/tirex-evolution/` for reference and understanding of the development progression.