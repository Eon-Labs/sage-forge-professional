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
- **dsm_integration_demo.py**: Complete real market data integration using Data Source Manager
  - Real Binance data via Failover Control Protocol (FCP)
  - Modern Polars/Arrow data pipeline
  - Interactive charting with trade visualization
  - Production-ready backtesting workflow
- **synthetic_data_example.py**: Educational example with synthetic data
  - Basic NautilusTrader concepts demonstration
  - Synthetic data generation for learning purposes
  - For production use, refer to dsm_integration_demo.py