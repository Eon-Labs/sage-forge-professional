# Examples

This directory contains example scripts organized by context and purpose:

## Directory Structure

- **backtest/**: Historical data with simulated venues
- **sandbox/**: Real-time data with simulated venues  
- **live/**: Real-time data with live venues (paper or real trading)
- **funding examples**: Crypto perpetual futures funding rate integration

## Running Examples

Use the provided Makefile:
```bash
make run-example
```

Or run directly with uv:
```bash
uv run python examples/native_funding_complete.py
```

## Example Categories

### 🏭 Production-Ready Examples
**For actual trading and backtesting implementations**

#### Funding Rate Integration
- **native_funding_complete.py** ⭐ **RECOMMENDED FOR PRODUCTION**
  - 100% Native NautilusTrader patterns with FundingActor
  - Event-driven architecture via MessageBus
  - Zero direct portfolio manipulation
  - Mathematical validation: 6/6 tests pass
  - Full BacktestEngine integration with realistic data
  - Comprehensive native pattern demonstration

### 📚 Educational & Validation Examples
**For learning and mathematical verification**

#### Funding Rate Education
- **funding_integration_complete.py** 📖 **EDUCATIONAL**
  - Mathematical foundation and formula verification
  - Temporally accurate 8-hour funding cycle demonstration
  - Sign convention and payment direction validation
  - Synthetic data generation for learning
  - Step-by-step funding calculation breakdown
  - **Note**: Use native_funding_complete.py for actual implementations

### 🧪 Sandbox & Development
**For testing and hybrid implementations**

#### Advanced Integrations
- **sandbox/enhanced_dsm_hybrid_integration.py** 🔬 **EXPERIMENTAL**
  - Hybrid DSM + Direct API data integration
  - Real-time Binance specifications
  - Embedded finplot integration (development only)
  - Testing ground for new features
  - **Status**: ⚠️ Development use (see decoupled version for production)
- **sandbox/live_plotter_decoupled.py** 🚀 **PRODUCTION-READY CHARTING**
  - Decoupled finplot dashboard via Redis MessageBus
  - External process prevents event loop blocking
  - Recommended production pattern for chart visualization
  - **Status**: ✅ Production-ready

### 📊 Data Integration Examples

#### Traditional NautilusTrader Demos
- **sandbox/dsm_integration_demo.py**: Real market data via DSM
  - Binance data via Failover Control Protocol (FCP)
  - Modern Polars/Arrow data pipeline
  - Interactive charting with trade visualization
- **sandbox/synthetic_data_example.py**: Educational synthetic data
  - Basic NautilusTrader concepts demonstration
  - Learning-focused implementation

## Quick Start Guide

### For Production Funding Integration:
```bash
# Run the native production example
uv run python examples/native_funding_complete.py
```

### For Learning Funding Mathematics:
```bash
# Run the educational validation example
uv run python examples/funding_integration_complete.py
```

### For Testing New Features:
```bash
# Run the experimental sandbox example
uv run python examples/sandbox/enhanced_dsm_hybrid_integration.py
```

## Integration Hierarchy

```
Funding Rate Integration
├── 🏭 Production (USE THIS)
│   └── native_funding_complete.py          # Native NautilusTrader patterns
├── 📚 Educational 
│   └── funding_integration_complete.py     # Mathematical validation
└── 🧪 Experimental
    └── sandbox/enhanced_dsm_hybrid_integration.py  # Advanced features
```

## Choosing the Right Example

| Purpose | Example | Native Patterns | Production Ready |
|---------|---------|-----------------|------------------|
| **Live Trading** | native_funding_complete.py | ✅ 100% | ✅ Yes |
| **Backtesting** | native_funding_complete.py | ✅ 100% | ✅ Yes |
| **Learning Math** | funding_integration_complete.py | ❌ No | ❌ Educational |
| **Research** | sandbox/enhanced_dsm_hybrid_integration.py | ⚠️ TBD | 🧪 Experimental |

**⚠️ Important**: Always use `native_funding_complete.py` for production implementations. Other examples are for education and research only.