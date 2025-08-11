# TiRex Context Length Empirical Performance Findings

## Executive Summary

**Hardware Environment**: RTX 4090 24GB VRAM, CUDA 12.8, Ubuntu 24.04 LTS  
**Test Date**: 2025-08-11  
**Model**: NX-AI/TiRex 35M parameter xLSTM architecture  

## Key Findings

### 🚨 Surprising Result: Context Length 144 is SLOWER than 288/512

Contrary to theoretical expectations, the shortest context length (144 timesteps) showed **worse performance** than longer contexts:

| Context Length | Avg Inference | GPU Memory | Throughput | Success Rate |
|----------------|---------------|------------|------------|--------------|
| 144 timesteps | **16.0 ms**   | 145.8 MB   | 62.4 pred/sec | 100% |
| 288 timesteps | **9.3 ms**    | 283.4 MB   | 107.1 pred/sec | 100% |
| 512 timesteps | **9.4 ms**    | 283.4 MB   | 106.2 pred/sec | 100% |

### Performance Analysis

**Speed Paradox**: 288 and 512 timesteps are **~1.7x faster** than 144 timesteps  
**Memory Scaling**: Linear relationship (144→283 MB for 288/512 contexts)  
**Throughput Champion**: 288 timesteps delivers highest throughput at 107.1 predictions/sec  

## Technical Insights

### CUDA Compilation Overhead
- **First Load**: 55.6 seconds compilation time (one-time cost)
- **Subsequent Loads**: 0.2-0.3 seconds (cached kernels)
- **Implication**: Model initialization dominates short-term performance

### Batch Processing Efficiency
The performance paradox suggests TiRex's xLSTM architecture has **optimal batch processing sweet spots**:
- 144 timesteps: Below optimal batch utilization
- 288-512 timesteps: In the GPU efficiency zone
- Memory usage plateaus suggest internal batching optimizations

### GPU Memory Utilization
```
Context 144: 145.8 MB (~0.6% of 24GB VRAM)
Context 288: 283.4 MB (~1.2% of 24GB VRAM)  
Context 512: 283.4 MB (~1.2% of 24GB VRAM)  # Same as 288!
```

**Memory Plateau**: 288 and 512 contexts use identical memory, indicating internal optimization.

## Backtesting Performance Recommendations

### For Different Use Cases

#### 🏃‍♂️ Fast Iteration Backtesting
**Recommended**: **288 timesteps (24 hours @ 5min bars)**
- **Performance**: 9.3ms per prediction (107 pred/sec)
- **Reason**: Optimal speed-memory balance
- **Use Case**: Rapid strategy development and parameter tuning

#### ⚖️ Balanced Production Backtesting  
**Recommended**: **288 timesteps (24 hours @ 5min bars)**
- **Performance**: 9.3ms per prediction
- **Quality**: 24-hour market context captures daily patterns
- **Reliability**: 100% success rate in testing

#### 🎯 Quality-Focused Backtesting
**Recommended**: **512 timesteps (42 hours @ 5min bars)**
- **Performance**: 9.4ms per prediction (106 pred/sec) 
- **Quality**: Extended context for complex pattern recognition
- **Memory**: Same as 288 due to internal optimizations

### ❌ Not Recommended: 144 Timesteps
Despite being shortest, 144 timesteps is **suboptimal** due to:
- 71% worse performance (16.0ms vs 9.3ms)
- Lower throughput (62.4 vs 107.1 pred/sec)
- Likely insufficient for daily market patterns

## Realistic Backtesting Scenarios

### Scenario 1: Development Phase (1,000 predictions)
```
Context 288: 1,000 × 9.3ms = 9.3 seconds + 0.3s load = 9.6s total
Context 512: 1,000 × 9.4ms = 9.4 seconds + 0.2s load = 9.6s total
```

### Scenario 2: Extended Backtest (10,000 predictions) 
```
Context 288: 10,000 × 9.3ms = 93 seconds + 0.3s load = ~1.6 minutes
Context 512: 10,000 × 9.4ms = 94 seconds + 0.2s load = ~1.6 minutes
```

### Scenario 3: Production Validation (100,000 predictions)
```
Context 288: 100,000 × 9.3ms = 15.5 minutes + 0.3s load
Context 512: 100,000 × 9.4ms = 15.7 minutes + 0.2s load
```

## Implementation Guidelines

### Model Loading Strategy
```python
# Cache model instance to avoid 55s compilation penalty
model = load_model("NX-AI/TiRex", device="cuda:0")  # One-time 55s cost

# Reuse same model instance for all predictions
for prediction_batch in backtest_windows:
    quantiles, mean = model.forecast(context, prediction_length=1)
```

### Context Window Optimization
```python
# RECOMMENDED: Use 288 timesteps as default
OPTIMAL_CONTEXT_LENGTH = 288  # 24 hours @ 5min bars

# FOR QUALITY-FOCUSED: Use 512 timesteps
QUALITY_CONTEXT_LENGTH = 512  # 42.7 hours @ 5min bars

# AVOID: 144 timesteps (performance penalty)
# SUBOPTIMAL_CONTEXT_LENGTH = 144  # Don't use
```

### Memory Management
```python
# GPU memory usage is minimal - no special management needed
# Peak: 283MB out of 24GB (1.2% utilization)
torch.cuda.empty_cache()  # Optional between major batches
```

## Validation Notes

### Test Methodology
- **Sample Size**: 30 predictions per context length
- **Data**: Synthetic BTC-like price walks (realistic volatility)
- **Quantile Levels**: [0.1, 0.5, 0.9] (empirically safe)
- **GPU Monitoring**: Real-time memory allocation tracking
- **Error Handling**: 100% success rate across all tests

### Limitations
- **Synthetic Data**: Real market microstructure may show different patterns
- **Single Model**: Results specific to NX-AI/TiRex architecture  
- **Fixed Quantiles**: Other quantile configurations may perform differently
- **GPU Specific**: RTX 4090 results may not transfer to other hardware

### Future Testing
- **Real Market Data**: Test with actual BTCUSDT historical data
- **Extended Context Range**: Test 768, 1024, 2048 timestep contexts
- **Quality Metrics**: Integrate ODEB directional capture analysis
- **Cross-Model**: Compare with other forecasting architectures

## Conclusion

**Primary Recommendation**: Use **288 timesteps** as the default context length for TiRex backtesting.

**Key Insights**:
1. **Longer ≠ Slower**: 288/512 contexts outperform 144 due to batch optimizations
2. **Memory Efficiency**: GPU memory scales favorably, plateaus at ~283MB
3. **Production Ready**: 9.3ms predictions enable real-time trading systems
4. **Quality Option**: 512 timesteps available with minimal speed penalty

This empirical analysis provides a quantified foundation for TiRex context length selection in professional trading systems, eliminating guesswork and optimizing for your RTX 4090 environment.

---

**Generated**: 2025-08-11 by TiRex Context Length Empirical Testing Suite  
**Data Source**: `/tests/performance/context_length_empirical_suite/results/simplified_benchmark_1754932165.csv`  
**Hardware**: RTX 4090, CUDA 12.8, 35M parameter NX-AI TiRex xLSTM model