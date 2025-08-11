# TiRex Context Length Performance Testing Suite

## Overview

Empirical testing framework for TiRex context length performance optimization on RTX 4090 GPU environment. This suite focuses on realistic backtesting performance analysis to help determine optimal context lengths for different trading scenarios.

## Hardware Environment

- **GPU**: RTX 4090 24GB VRAM
- **CUDA**: 12.6
- **Platform**: Ubuntu 24.04 LTS
- **Performance Baseline**: 45-130ms per prediction, 1,549 timesteps/sec (context length dependent)

## Test Architecture

### Proof of Concept Test (`proof_of_concept_context_benchmark.py`)

**Primary Test Matrix:**
- 144 timesteps (12h @ 5min bars) - Fast baseline
- 288 timesteps (24h @ 5min bars) - Common usage  
- 512 timesteps (42h @ 5min bars) - Quality optimized

**Metrics Tracked:**
1. **Inference Speed** - Milliseconds per prediction
2. **GPU Memory Usage** - Peak VRAM consumption in MB
3. **Throughput** - Predictions per second
4. **Forecast Quality** - Reliability and directional capture
5. **Success Rate** - Successful predictions vs errors/blocks

**Integration Features:**
- ✅ **Guardian System Protection** - All TiRex calls through `TiRexGuardian.safe_forecast()`
- ✅ **Real Market Data** - DSM integration for authentic BTCUSDT 5min data
- ✅ **Rich Console Output** - Professional progress tracking and result tables
- ✅ **Vulnerability Avoidance** - Uses proven safe quantile levels [0.1, 0.5, 0.9]
- ✅ **GPU Resource Tracking** - Real-time VRAM monitoring and cache management
- ✅ **Export Capabilities** - CSV and Excel output for analysis

## Usage

### Quick Start
```bash
# Navigate to test directory
cd /home/tca/eon/nt/tests/performance/context_length_empirical_suite

# Run proof of concept benchmark (recommended first test)
python proof_of_concept_context_benchmark.py
```

### Prerequisites
```bash
# Ensure TiRex is installed
pip install tirex

# Verify CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Check SAGE-Forge Guardian system
python -c "from sage_forge.guardian.core import TiRexGuardian; print('Guardian Available')"
```

### Expected Output

The benchmark will display:

1. **Real-time Progress** - Progress bars for each context length test
2. **Performance Comparison Table** - Side-by-side metrics comparison
3. **Benchmark Analysis** - Recommendations for different use cases
4. **Export Confirmation** - CSV/Excel file locations

**Sample Output:**
```
📊 Context Length Performance Comparison
┌──────────────┬──────────────┬─────────────┬────────────┬─────────────┬──────────────┐
│ Context Length │ Inference Time │ GPU Memory │ Throughput │ Success Rate │ Quality Score │
├──────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────────┤
│ 144          │ 67.2 ms      │ 1,234.5 MB  │ 14.9 pred/sec │ 96.7%       │ 87.3         │
│ 288          │ 89.4 ms      │ 1,456.2 MB  │ 11.2 pred/sec │ 98.1%       │ 91.2         │
│ 512          │ 124.6 ms     │ 1,789.8 MB  │ 8.0 pred/sec  │ 97.3%       │ 94.5         │
└──────────────┴──────────────┴─────────────┴────────────┴─────────────┴──────────────┘

🏆 Performance Analysis:
⚡ Fastest: 144 timesteps (67.2ms)
🎯 Highest Quality: 512 timesteps (score: 94.5)
⚖️ Optimal Balance: 288 timesteps
```

## Results

### Result Files

All benchmark results are exported to `results/`:
- **CSV Format**: `proof_of_concept_benchmark_[timestamp].csv` 
- **Excel Format**: `proof_of_concept_benchmark_[timestamp].xlsx`

### Result Analysis

**Performance Patterns Expected:**
- **Memory Usage**: Linear scaling with context length (~3-5 MB per 100 timesteps)
- **Inference Speed**: Sub-linear scaling (batch processing efficiency)
- **Quality Score**: Generally improves with longer context (diminishing returns)
- **Throughput**: Inverse relationship with inference time

**Backtesting Recommendations:**
- **Fast Iteration**: Use shortest context that maintains >95% success rate
- **Final Validation**: Use highest quality context for production validation
- **Production Balance**: Optimal speed/quality tradeoff for live trading

## Implementation Notes

### Hard-Learnt Pattern Integration

The test framework incorporates lessons learned from previous TiRex integration:

1. **Guardian System Mandatory** - All TiRex calls protected against vulnerabilities
2. **Safe Quantile Levels** - Uses [0.1, 0.5, 0.9] to avoid extrapolation issues
3. **Proper Error Handling** - Robust fallbacks for NaN inputs and edge cases
4. **GPU Cache Management** - Explicit memory clearing between tests
5. **Temporal Validation** - Ensures proper data ordering and context boundaries

### Integration Points

**SAGE-Forge Components Used:**
- `sage_forge.guardian.core.TiRexGuardian` - Vulnerability protection
- `sage_forge.reporting.performance` - ODEB framework (future integration)

**External Dependencies:**
- `tirex` - Official NX-AI TiRex model (35M parameters)
- `core.sync.data_source_manager.ArrowDataManager` - Real market data
- `torch` - GPU acceleration and memory tracking
- `rich` - Professional console output

### Performance Optimization

**GPU Resource Management:**
- Automatic CUDA cache clearing between tests
- Peak memory tracking during inference
- Synchronization points for accurate timing

**Data Pipeline Optimization:**
- Efficient univariate series extraction from OHLCV data
- Proper tensor format conversion for TiRex input requirements
- Batch processing where applicable

## Future Enhancements

### Phase 2: Extended Context Length Matrix
- **Fast Range**: [96, 144, 192, 240, 288] 
- **Quality Range**: [384, 512, 768, 1024]
- **Extreme Range**: [1536, 2048] (memory permitting)

### Phase 3: ODEB Integration
- Full directional capture efficiency analysis
- Market regime-specific performance assessment
- Time-weighted position sizing validation

### Advanced Features
- Multi-GPU scaling analysis
- Quantization impact on speed/quality
- Real-time streaming performance
- Cross-model comparison framework

## Troubleshooting

### Common Issues

**TiRex Loading Failures:**
```
❌ Failed to load TiRex model: ...
```
- Check CUDA installation: `nvcc --version`
- Verify PyTorch CUDA compatibility: `torch.cuda.is_available()`
- Ensure sufficient VRAM (>2GB required)

**Guardian System Unavailable:**
```
⚠️ Guardian system not available: ...
```
- Verify SAGE-Forge installation in Python path
- Check import paths in test script
- Run in fallback mode (direct TiRex calls)

**Data Source Issues:**
```
⚠️ DSM not available, will use synthetic data: ...
```
- Verify Data Source Manager installation
- Check network connectivity for real data
- Synthetic data fallback will still provide valid performance metrics

### Performance Debugging

**Slow Inference Times:**
- Check GPU utilization: `nvidia-smi`
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Verify model is on GPU: `model.device`

**Memory Issues:**
- Reduce batch size or context length
- Check available VRAM: `torch.cuda.memory_allocated()`
- Clear Python variables: `del model; gc.collect()`

## Contact and Support

This testing framework was developed as part of the SAGE-Forge professional trading system. For issues or enhancements, refer to the main project documentation or create detailed issue reports with:

1. Full error messages and stack traces
2. GPU hardware specifications
3. CUDA/PyTorch versions
4. Test parameters used
5. Expected vs actual behavior

---

**Status**: Production ready for RTX 4090 environment  
**Last Updated**: 2025-08-11  
**Version**: 1.0.0 - Proof of Concept