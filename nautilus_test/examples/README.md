# Examples

This directory contains example scripts organized by context:

## Directory Structure

- **backtest/**: Historical data with simulated venues
- **sandbox/**: Real-time data with simulated venues  
- **live/**: Real-time data with live venues (paper or real trading)

## Running Examples

Use the provided Makefile:
```bash
make run-example
```

Or run directly with uv:
```bash
uv run python examples/sandbox/basic_test.py
```

## Example Scripts

### Sandbox
- `basic_test.py`: Basic NautilusTrader functionality test