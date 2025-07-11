# NautilusTrader Learning Notes - Project Overview

## What is NautilusTrader?

NautilusTrader is a high-performance algorithmic trading platform built in Rust with Python bindings. It provides a comprehensive framework for developing, testing, and deploying trading strategies.

### Key Features
- **High Performance**: Built in Rust for speed and reliability
- **Python Interface**: Easy to use Python API for strategy development
- **Backtesting**: Comprehensive historical testing capabilities
- **Live Trading**: Real-time trading execution
- **Multi-Asset**: Support for forex, crypto, stocks, futures, options
- **Risk Management**: Built-in position sizing and risk controls
- **Market Data**: Real-time and historical data processing

## Project Structure

Our workspace contains two main directories:

### `/workspaces/nt/nautilus_test/`
- **Purpose**: Our development/learning environment
- **Structure**:
  ```
  nautilus_test/
  ├── src/nautilus_test/          # Main package
  │   ├── strategies/             # Trading strategies
  │   ├── adapters/               # Trading adapters
  │   └── utils/                  # Utility functions
  ├── tests/                      # Test files
  ├── examples/                   # Example scripts
  ├── learning_notes/             # This documentation
  ├── Makefile                    # Development commands
  └── pyproject.toml             # Project configuration
  ```

### `/workspaces/nt/nt_reference/`
- **Purpose**: Complete NautilusTrader source code for reference
- **Contains**: Full implementation, examples, documentation

## Development Environment

### Package Management
- **Tool**: `uv` (NautilusTrader's recommended package manager)
- **Python**: 3.11+ required
- **Dependencies**: NautilusTrader >= 1.219.0

### Development Tools
- **Formatter**: black (100-character line length)
- **Linter**: ruff
- **Type Checker**: mypy
- **Testing**: pytest

### Code Standards
- Line length: 100 characters
- Type hints required for all functions
- Follow NautilusTrader's coding conventions

## Current Environment Status

✅ **Working Components**:
- uv package management
- All development tools (black, ruff, mypy, pytest)
- Basic functionality tests passing
- Make commands working
- Code formatting and linting passing

## What We've Learned So Far

1. **Make Tool**: Build automation that reads Makefile to execute project tasks
2. **Project Setup**: Proper Python project structure with setuptools
3. **Development Workflow**: Format → Lint → Test → Type Check cycle
4. **NautilusTrader Basics**: Core concepts and capabilities

Date: 2025-07-11
Environment: Linux (Docker container with devcontainer setup)