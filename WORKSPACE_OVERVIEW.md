# Complete Workspace Documentation
**NautilusTrader Development Environment - Production Ready**

## 🏗️ Workspace Structure Overview

```
/workspaces/nt/                     # 🏠 Workspace Root
├── 📁 .vscode/                     # 🔧 VS Code workspace configuration
│   ├── settings.json               # Python interpreter & extension settings
│   ├── tasks.json                  # Development task automation
│   ├── extensions.json             # Recommended extensions
│   └── keybindings.json           # Custom keyboard shortcuts
├── 📁 .claude/                     # 🤖 Claude Code configuration
│   ├── settings.json               # Bypass permissions for streamlined workflow
│   └── README.md                   # Configuration documentation
├── 📄 pyrightconfig.json           # 🐍 Python type checking configuration
├── 📄 SETUP.md                     # 🚀 Comprehensive setup guide
├── 📄 CLAUDE.md                    # 📋 Project instructions & memory
├── 📄 README.md                    # 🏠 Basic workspace introduction
├── 📄 WORKSPACE_OVERVIEW.md        # 📖 This comprehensive documentation
├── 📁 session_logs/                # 📝 Development session tracking
│   ├── LATEST.md -> current        # Auto-discovery symlink
│   ├── INDEX.md                    # Session registry
│   └── 2025/07/                    # Organized by date
├── 📁 nautilus_test/               # 🎯 Your Development Environment
├── 📁 nt_reference/                # 📚 Official NautilusTrader Source
└── 🔧 run.sh                       # 🚀 Workspace startup script
```

---

## 📦 Binary Builds & Installation Details

### Where NautilusTrader is Installed
**Location**: `/workspaces/nt/nautilus_test/.venv/lib/python3.12/site-packages/nautilus_trader/`
- **Installation Method**: UV virtual environment (isolated & reproducible)
- **Version**: 1.219.0+ (latest available)
- **Total Size**: ~300MB installed (including dependencies)
- **Binary Extensions**: 140+ compiled `.so` files (Cython/PyO3 extensions)

### Binary Architecture  
- **Platform**: `aarch64-linux-gnu` (ARM64 Linux)
- **Python Version**: CPython 3.12.11
- **Core Engine**: Rust-powered with Python bindings for maximum performance
- **Dependencies**: pandas 2.0+, Rich 14.0+, requests 2.32+

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
nautilus_test/                     # 🎯 Production-Ready Development Environment
├── 📁 .venv/                      # 🐍 UV-managed virtual environment
├── 🔧 Makefile                    # Development commands (make help, test, format, etc.)
├── ⚙️ pyproject.toml              # Project configuration & dependencies (Python 3.12)
├── 🔒 uv.lock                     # Dependency lock file (all versions pinned)
├── 📖 README.md                   # Project documentation
├── 📍 .python-version             # Python version pinning (3.12.11)
│
├── 📁 src/nautilus_test/          # Main Python package
│   ├── __init__.py
│   ├── 📁 strategies/             # Your trading strategies
│   ├── 📁 adapters/               # Custom exchange adapters  
│   └── 📁 utils/                  # Utility functions
│
├── 📁 tests/                      # Comprehensive test suite
│   ├── __init__.py
│   ├── test_basic.py              # Basic functionality tests
│   └── test_bars_functionality.py # OHLC bars testing (pytest)
│
├── 📁 examples/                   # Example scripts & demonstrations
│   ├── README.md
│   ├── 📁 backtest/               # Historical testing examples
│   ├── 📁 live/                   # Live trading examples  
│   └── 📁 sandbox/                # 🧪 Safe testing environment
│       └── simple_bars_test.py    # Comprehensive OHLC bars demo (Rich UI)
│
├── 📁 docs/                       # Documentation & learning materials
│   ├── README_BAR_TESTS.md        # OHLC bars testing guide
│   └── 📁 learning_notes/         # Learning documentation
│       ├── README.md              # Navigation guide
│       ├── 01_project_overview.md # What is NautilusTrader
│       ├── 02_testing_and_commands.md # Testing results & commands
│       ├── 03_strategies_and_adapters.md # Available tools
│       └── 04_next_steps_and_learning_path.md # Learning roadmap
│
└── 📁 scripts/                    # Utility & setup scripts
    ├── uv_python.sh               # UV Python wrapper
    └── setup_dev_env.sh           # 🚀 Automated environment setup
```

### Development Tools & Commands
```bash
# Quick setup (run once)
./scripts/setup_dev_env.sh       # 🚀 Automated environment setup

# Daily development workflow
make help                        # Show all available commands
make install                     # Install dependencies with uv
make test                        # Run pytest tests
make format                      # Format code with black/ruff  
make lint                        # Lint code with ruff
make typecheck                   # Type check with mypy
make dev-workflow               # Complete development check
make clean                      # Clean build artifacts

# Direct script execution
uv run python examples/sandbox/simple_bars_test.py  # Rich OHLC bars demo
uv run pytest tests/                                # Run test suite
```

### Key Dependencies
```
Core Trading:
├── nautilus-trader >=1.219.0   # Main trading platform
├── pandas >=2.0.0              # Data manipulation
├── numpy                       # Numerical computing (via nautilus)
└── pyarrow                     # Columnar data (via nautilus)

Enhanced Development:
├── rich >=14.0.0               # Beautiful terminal output
├── requests >=2.32.4           # HTTP operations
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

### Development Standards & Configuration
- **Line Length**: 100 characters (black, ruff, pyproject.toml)
- **Python Version**: 3.12+ required (pinned to 3.12.11)
- **Virtual Environment**: UV-managed `.venv/` (isolated dependencies)
- **Code Style**: black formatter + ruff linting
- **Type Checking**: mypy + Pyright (VS Code integration)
- **Testing**: pytest with coverage
- **IDE**: VS Code with robust Python path resolution

---

## 🚀 Quick Start Guide

### First Time Setup (Run Once)
```bash
# Navigate to development environment
cd /workspaces/nt/nautilus_test/

# Run automated setup
./scripts/setup_dev_env.sh

# Reload VS Code window
# Ctrl+Shift+P → "Developer: Reload Window"
```

### Daily Development Flow
```bash
cd /workspaces/nt/nautilus_test/

# Quick development check
make dev-workflow

# Run OHLC bars demo
uv run python examples/sandbox/simple_bars_test.py

# Develop strategies
# Edit: src/nautilus_test/strategies/
# Test: uv run pytest tests/
```

### Learning Path
1. **📚 Study**: Read `docs/learning_notes/` and `docs/README_BAR_TESTS.md`
2. **🧪 Experiment**: Run and modify `examples/sandbox/simple_bars_test.py`
3. **📖 Reference**: Explore `/workspaces/nt/nt_reference/examples/backtest/`
4. **🔨 Build**: Create strategies in `src/nautilus_test/strategies/`

### VS Code Integration
- **Python Interpreter**: Auto-configured to `.venv/bin/python`
- **Import Resolution**: No more missing import errors
- **Testing**: Integrated pytest discovery
- **Linting**: Real-time ruff + mypy checking

---

## ✅ Current Achievements

### 🏗️ **Infrastructure Complete**
- ✅ Robust VS Code configuration with zero import errors
- ✅ UV virtual environment with pinned dependencies
- ✅ Automated setup script for consistent environment
- ✅ Comprehensive OHLC bars testing framework

### 🧪 **Testing Framework Ready**
- ✅ `simple_bars_test.py` - Rich UI OHLC bars demonstration
- ✅ `test_bars_functionality.py` - Professional pytest suite
- ✅ Synthetic + real FXCM data support
- ✅ Multiple strategies (EMA Cross + Bracket orders)

### 📖 **Documentation Complete**
- ✅ Session logging system with auto-discovery
- ✅ Learning notes organized and accessible
- ✅ Setup guides and troubleshooting documentation
- ✅ Git workflow with conventional commits

---

## 🔧 Advanced Features

### Session Management
- **Auto-discovery**: `session_logs/LATEST.md` → current session
- **Organization**: Date-structured (`2025/07/YYYY-MM-DD-NNN.md`)
- **Templates**: Standardized session documentation

### Claude Code Integration
- **Permissions**: `bypassPermissions` for maximum workflow freedom
- **Task agents**: Full tool access without confirmations
- **Memory**: Project context persisted in CLAUDE.md

### Development Automation
- **setup_dev_env.sh**: Validates entire environment
- **Makefile**: Standardized development commands
- **pyrightconfig.json**: Workspace-level type checking

---

**Created**: 2025-07-11  
**Updated**: 2025-07-11 (Production Ready)  
**Environment**: NautilusTrader Development Workspace  
**Purpose**: Complete workspace documentation and reference  
**Repository**: https://github.com/terrylica/nautilus-trader-workspace  
**Status**: ✅ Production ready with robust OHLC bars testing framework