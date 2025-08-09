# NautilusTrader Test Environment

A testing and learning environment for algorithmic trading strategies using the NautilusTrader platform, organized following NautilusTrader's best practices.

## Project Structure

```
nautilus_test/
├── src/nautilus_test/          # Main package
│   ├── strategies/             # Trading strategies
│   ├── adapters/              # Trading adapters
│   └── utils/                 # Utility functions
├── tests/                     # Test files
├── examples/                  # Example scripts
│   ├── backtest/             # Historical data examples
│   ├── sandbox/              # Real-time simulation examples
│   └── live/                 # Live trading examples
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
└── Makefile                  # Development commands
```

## Setup

Install dependencies using uv (recommended):

```bash
make install
# or
uv sync --all-extras
```

## Development

Common development tasks:

```bash
make help           # Show available commands
make install        # Install dependencies
make test          # Run tests
make format        # Format code
make lint          # Lint code
make typecheck     # Type check
make run-example   # Run basic example
```

## Examples

Run the basic functionality test:

```bash
make run-example
```

Or explore examples in the `examples/` directory organized by context.

## Requirements

- Python 3.11+
- uv package manager (recommended)
- NautilusTrader platform
