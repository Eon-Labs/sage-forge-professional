# 🔥 SAGE-Forge

**Self-Adaptive Generative Evaluation Framework for NautilusTrader**

A professional, bulletproof infrastructure for developing adaptive trading strategies with zero trial-and-error setup.

## ✨ Features

- 🎯 **SAGE Meta-Framework**: Self-adaptive model ensemble with dynamic weighting
- 🚀 **NT-Native Integration**: Proper NautilusTrader Strategy patterns and BacktestEngine
- 📊 **Real Data Pipeline**: DSM integration with 100% real market data (no synthetic)
- 📈 **Professional Visualization**: FinPlot actors with enhanced charting
- 💰 **Production Funding**: Real funding rate integration with cost tracking
- ⚡ **Zero Setup Time**: 30-second bulletproof environment with UV
- 🏗️ **Modern Architecture**: src/ layout, professional CLI, reproducible builds

## 🚀 Quick Start

### One-Command Setup
```bash
# Clone and setup (30 seconds total)
git clone <repo-url> sage-forge
cd sage-forge
uv sync                    # Install all dependencies
uv run sage-setup         # Validate environment
```

### Create Your First Strategy
```bash
# Generate strategy template
uv run sage-create strategy MyAdaptiveStrategy

# Run the strategy
uv run python examples/my_adaptive_strategy.py
```

### Professional CLI
```bash
sage-forge --help         # Main CLI
sage-setup                # Environment setup & validation  
sage-create strategy Name  # Create strategy template
sage-validate              # Quick environment check
```

## 📦 Installation

### Requirements
- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager

### From Source
```bash
git clone <repo-url>
cd sage-forge
uv sync                    # Install with exact dependency versions
uv run sage-setup         # Validate installation
```

### As Package Dependency
```bash
uv add sage-forge
```

## 🏗️ Architecture

### Package Structure
```
sage-forge/
├── src/sage_forge/           # Main package
│   ├── cli/                  # Professional CLI commands
│   ├── core/                 # Core infrastructure
│   ├── models/               # SAGE model implementations
│   ├── strategies/           # NautilusTrader strategies
│   ├── data/                 # DSM data integration
│   ├── visualization/        # FinPlot components
│   ├── funding/              # Funding rate system
│   └── utils/                # Shared utilities
├── examples/                 # Usage examples
├── tests/                    # Comprehensive test suite
└── docs/                     # Documentation
```

### Model Zoo
- **AlphaForge**: Feature engineering with domain expertise
- **Catch22**: 22 canonical time series characteristics  
- **TiRex**: Zero-shot forecasting with uncertainty quantification
- **SAGE Ensemble**: Meta-framework for dynamic model weighting

### Data Integration
- **Real Market Data**: 100% real data via Data Source Manager (DSM)
- **Binance Integration**: Real API specifications and funding rates
- **Quality Validation**: Comprehensive data quality checks
- **Caching System**: Smart caching for performance

## 📊 Usage Examples

### Basic Strategy
```python
from sage_forge.strategies import AdaptiveRegimeStrategy
from sage_forge.data import ArrowDataManager
from sage_forge.models import SAGEEnsemble

# Create data manager
data_manager = ArrowDataManager()

# Fetch real market data
data = data_manager.fetch_real_market_data(
    symbol="BTCUSDT",
    timeframe="1m", 
    limit=2000
)

# Create SAGE ensemble
sage = SAGEEnsemble([
    "alphaforge",
    "catch22", 
    "tirex"
])

# Run adaptive strategy
strategy = AdaptiveRegimeStrategy(sage)
# ... backtest execution
```

### Advanced SAGE Ensemble
```python
from sage_forge.models import SAGEEnsemble, AlphaForge, Catch22, TiRex

# Create individual models
models = [
    AlphaForge(config={"features": ["momentum", "volatility"]}),
    Catch22(config={"feature_set": "comprehensive"}),
    TiRex(config={"horizon": 24, "uncertainty": True})
]

# Meta-framework with dynamic weighting
sage = SAGEEnsemble(
    models=models,
    weighting_method="adaptive",
    rebalance_frequency="daily"
)

# Self-adaptive evaluation
performance = sage.evaluate_and_adapt(market_data)
```

## 🧪 Development

### Setup Development Environment
```bash
# Install with development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Code formatting
uv run black src tests examples
uv run ruff check src tests examples

# Type checking
uv run mypy src
```

### Creating New Models
```bash
# Generate model template
uv run sage-create model MyCustomModel

# Implement in src/sage_forge/models/my_custom_model.py
# Add tests in tests/test_models/test_my_custom_model.py
```

## 📈 Performance

- **Setup Time**: 30 seconds (vs 15+ minutes traditional)
- **Data Quality**: 100% real market data validation
- **Memory Efficient**: Smart caching and lazy loading
- **Fast Execution**: UV package management + optimized data pipeline
- **Zero Debugging**: Bulletproof dependency management

## 🔧 Configuration

### Environment Variables
```bash
export SAGE_FORGE_DATA_DIR="/path/to/data"
export SAGE_FORGE_CACHE_DIR="/path/to/cache"  
export SAGE_FORGE_LOG_LEVEL="INFO"
```

### DSM Integration
Configure Data Source Manager in `~/.sage-forge/config.yaml`:
```yaml
dsm:
  provider: "binance"
  market_type: "futures_usdt"
  cache_enabled: true
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`uv sync --group dev`)
4. Make changes and add tests
5. Run quality checks (`uv run pytest && uv run ruff check`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) - Professional trading platform
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager
- [FinPlot](https://github.com/highfestiva/finplot) - Financial visualization
- Contributors and the quantitative finance community

---

**Built with ❤️ for the quantitative trading community**