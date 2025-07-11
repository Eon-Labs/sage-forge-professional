# Complete Workspace Documentation
**NautilusTrader Development Environment**

## 🏗️ Workspace Structure Overview

```
/workspaces/nt/
├── 📁 nautilus_test/           # Your Development Environment
├── 📁 nt_reference/            # Official NautilusTrader Source Code
├── 📄 CLAUDE.md               # Project Instructions & Memory
├── 📄 README.md               # Basic workspace introduction
├── 📄 WORKSPACE_OVERVIEW.md   # This comprehensive documentation
└── 🔧 run.sh                  # Workspace startup script
```

---

## 📦 Binary Builds & Installation Details

### Where NautilusTrader is Installed
**Location**: `/home/vscode/.local/lib/python3.12/site-packages/nautilus_trader/`
- **Installation Method**: Standard Python package via `uv` package manager
- **Version**: 1.219.0 (production release)
- **Total Size**: 246MB installed
- **Binary Extensions**: 140 compiled `.so` files (Cython/PyO3 extensions)

### Binary Architecture
- **Platform**: `aarch64-linux-gnu` (ARM64 Linux)
- **Python Version**: CPython 3.12
- **Core Engine**: Largest binary is `engine.cpython-312-aarch64-linux-gnu.so` (65MB)
- **Performance**: Rust-powered core with Python bindings for maximum speed

### Key Compiled Components
```
engine.so           - 65MB  (Backtesting engine core)
matching_engine.so  - 8MB   (Order matching system)  
data_client.so      - 293KB (Data handling)
exchange.so         - 351KB (Exchange simulation)
+ 136 other .so files       (Various trading components)
```

---

## 🎯 Your Development Environment: `/workspaces/nt/nautilus_test/`

### Project Structure
```
nautilus_test/
├── 🔧 Makefile                    # Development commands (make help, test, format, etc.)
├── ⚙️ pyproject.toml              # Project configuration & dependencies
├── 🔒 uv.lock                     # Dependency lock file
├── 📖 README.md                   # Project documentation
│
├── 📁 src/nautilus_test/          # Main Python package
│   ├── __init__.py
│   ├── 📁 strategies/             # Your trading strategies
│   ├── 📁 adapters/               # Custom exchange adapters  
│   └── 📁 utils/                  # Utility functions
│
├── 📁 tests/                      # Test files
│   ├── __init__.py
│   └── test_basic.py              # Basic functionality tests
│
├── 📁 examples/                   # Example scripts
│   ├── README.md
│   ├── 📁 backtest/               # Historical testing examples
│   ├── 📁 live/                   # Live trading examples  
│   └── 📁 sandbox/                # Safe testing environment
│       └── basic_test.py          # Working basic example
│
├── 📁 learning_notes/             # Your learning documentation
│   ├── README.md                  # Navigation guide
│   ├── 01_project_overview.md     # What is NautilusTrader
│   ├── 02_testing_and_commands.md # Testing results & commands
│   ├── 03_strategies_and_adapters.md # Available tools
│   └── 04_next_steps_and_learning_path.md # Learning roadmap
│
├── 📁 docs/                       # Future documentation
└── 📁 scripts/                    # Utility scripts
```

### Development Tools & Commands
```bash
make help           # Show all available commands
make install        # Install dependencies with uv
make test          # Run pytest tests
make format        # Format code with black/ruff  
make lint          # Lint code with ruff
make typecheck     # Type check with mypy
make run-example   # Run basic sandbox example
make clean         # Clean build artifacts
```

### Dependencies (32 packages total)
```
Core:
├── nautilus-trader v1.219.0    # Main trading platform
├── numpy v2.3.1                # Numerical computing
├── pandas v2.3.1               # Data analysis
└── pyarrow v20.0.0             # Columnar data

Development:
├── black v25.1.0               # Code formatting
├── ruff v0.12.2                # Fast linting
├── mypy v1.16.1                # Type checking
└── pytest v8.4.1              # Testing framework
```

---

## 📚 Reference Repository: `/workspaces/nt/nt_reference/`

### Complete Source Code Structure
```
nt_reference/
├── 📄 README.md                   # Project overview
├── 📄 LICENSE                     # LGPL v3.0 license
├── 📄 CONTRIBUTING.md             # How to contribute
├── ⚙️ pyproject.toml              # Python package config
├── ⚙️ Cargo.toml                  # Rust workspace config
├── 🔒 uv.lock                     # Python dependencies
├── 🔒 Cargo.lock                  # Rust dependencies
│
├── 🦀 crates/                     # Rust core components (50+ crates)
│   ├── core/                      # Fundamental types & utilities
│   ├── model/                     # Trading domain models
│   ├── backtest/                  # Backtesting engine
│   ├── live/                      # Live trading infrastructure
│   ├── data/                      # Data management
│   ├── execution/                 # Order execution
│   ├── network/                   # HTTP/WebSocket components
│   └── adapters/                  # Exchange-specific integrations
│
├── 🐍 nautilus_trader/            # Python package (mirrors your installation)
│   ├── adapters/                  # 10+ exchange adapters
│   │   ├── binance/               # Binance integration
│   │   ├── bybit/                 # Bybit integration
│   │   ├── interactive_brokers/   # Interactive Brokers
│   │   ├── databento/             # Market data provider
│   │   └── [8 more adapters]/
│   │
│   ├── examples/                  # Reference implementations
│   │   └── strategies/            # 15+ example strategies
│   │
│   ├── backtest/                  # Backtesting modules
│   ├── live/                      # Live trading modules
│   ├── model/                     # Trading models
│   ├── indicators/                # Technical indicators
│   └── [15 more modules]/
│
├── 📖 docs/                       # Complete documentation
│   ├── api_reference/             # API documentation
│   ├── concepts/                  # Trading concepts
│   ├── developer_guide/           # Development guides
│   ├── getting_started/           # Tutorials
│   ├── integrations/              # Exchange guides
│   └── tutorials/                 # Jupyter notebooks
│
├── 🧪 examples/                   # Working examples
│   ├── backtest/                  # 25+ backtest examples
│   │   ├── crypto_ema_cross_ethusdt_trade_ticks.py
│   │   ├── fx_ema_cross_audusd_bars.py
│   │   ├── databento_ema_cross_long_only_aapl_bars.py
│   │   └── [22 more examples]/
│   │
│   ├── live/                      # Live trading examples
│   └── sandbox/                   # Safe testing examples
│
├── 🧪 tests/                      # Comprehensive test suite
│   ├── unit_tests/                # Unit tests
│   ├── integration_tests/         # Integration tests
│   ├── performance_tests/         # Performance benchmarks
│   └── acceptance_tests/          # End-to-end tests
│
└── 🛠️ scripts/                   # Build & utility scripts
```

### Key Reference Files
- **Strategy Examples**: `/examples/backtest/*.py` (25+ working strategies)
- **Adapter Code**: `/nautilus_trader/adapters/*/` (10+ exchange integrations)
- **Documentation**: `/docs/` (Complete API and usage docs)
- **Tests**: `/tests/` (Learn from comprehensive test examples)

---

## 🔧 Environment Details

### System Information
```
Platform:      Linux (Docker container)
Architecture:  aarch64 (ARM64)
OS:           Debian GNU/Linux 12
Python:       3.12.11
Rust:         Latest stable (via rustup)
Package Mgr:  uv (NautilusTrader recommended)
```

### DevContainer Setup
```
Base Image:    mcr.microsoft.com/devcontainers/base:debian
Extensions:    Python, Rust Analyzer, Ruff
Tools:         uv, rustup, git, make
Auto-format:   Enabled (black, ruff)
```

### Python Virtual Environment
```
Location:      Managed by uv
Site-packages: /home/vscode/.local/lib/python3.12/site-packages/
Binaries:      /home/vscode/.local/bin/
Cache:         /home/vscode/.cache/uv/
```

---

## 📋 Configuration Files

### Key Config Files
```
/workspaces/nt/CLAUDE.md           # Project instructions & standards
/workspaces/nt/nautilus_test/pyproject.toml    # Your project config
/workspaces/nt/nautilus_test/Makefile          # Development commands
/workspaces/nt/nt_reference/pyproject.toml     # Reference config
```

### Development Standards
- **Line Length**: 100 characters
- **Python Version**: 3.11+ required
- **Code Style**: black formatter
- **Linting**: ruff
- **Type Checking**: mypy with strict settings
- **Testing**: pytest

---

## 🎯 Workflow Recommendations

### Daily Development Flow
1. **Navigate**: `cd /workspaces/nt/nautilus_test/`
2. **Check**: `make lint && make typecheck`
3. **Test**: `make test`
4. **Develop**: Work in `src/nautilus_test/strategies/`
5. **Reference**: Study examples in `/workspaces/nt/nt_reference/examples/`

### Learning Path
1. **Start**: Review `/workspaces/nt/nautilus_test/learning_notes/`
2. **Practice**: Modify `/workspaces/nt/nautilus_test/examples/sandbox/basic_test.py`
3. **Study**: Explore `/workspaces/nt/nt_reference/examples/backtest/`
4. **Build**: Create strategies in `/workspaces/nt/nautilus_test/src/nautilus_test/strategies/`

---

## 🚀 Getting Started Commands

```bash
# Navigate to your development environment
cd /workspaces/nt/nautilus_test/

# Install dependencies
make install

# Run basic tests
make test

# Check code quality
make format && make lint && make typecheck

# Run example
make run-example

# Explore learning materials
ls learning_notes/
```

---

**Created**: 2025-07-11  
**Environment**: NautilusTrader Development Workspace  
**Purpose**: Complete workspace documentation and reference