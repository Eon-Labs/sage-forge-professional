# NautilusTrader Learning Notes - Testing & Commands

## Make Commands Reference

### Available Commands (from `make help`)

```bash
make help                # Show available commands
make install             # Install all dependencies with uv
make test               # Run tests
make format             # Format code with black and ruff
make lint               # Lint code with ruff
make typecheck          # Run type checking with mypy
make run-example        # Run basic sandbox example
make clean              # Clean build artifacts and cache
```

## Testing Results

### Basic Functionality Test

**Command**: `make test` **Result**: ✅ PASSED

```
============================= test session starts ==============================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0
rootdir: /workspaces/nt/nautilus_test
configfile: pyproject.toml
plugins: cov-6.2.1
collected 1 item

tests/test_basic.py .                                                    [100%]

============================== 1 passed in 0.01s
```

### Code Quality Tests

#### Formatting

**Command**: `make format` **Result**: ✅ PASSED

```
Formatting code...
uv run black src/ tests/ examples/
uv run ruff check --fix src/ tests/ examples/
All checks passed!
All done! ✨ 🍰 ✨
7 files left unchanged.
```

#### Linting

**Command**: `make lint` **Result**: ✅ PASSED

```
Linting code...
uv run ruff check src/ tests/ examples/
All checks passed!
```

#### Type Checking

**Command**: `make typecheck` **Result**: ✅ PASSED

```
Type checking...
uv run mypy src/
Success: no issues found in 4 source files
```

### Basic Example Test

**Command**: `make run-example` **Result**: ✅ PASSED

```
Running basic sandbox example...
uv run python examples/sandbox/basic_test.py
Testing NautilusTrader functionality...

1. Creating test instrument:
   Instrument: EUR/USD.SIM
   Symbol: EUR/USD
   Venue: SIM

2. Creating price and quantity objects:
   Price: 1.08250
   Quantity: 100000

3. Creating quote tick:
   Quote: EUR/USD.SIM,1.08240,1.08250,1000000,1000000,1752220105638048000

✅ NautilusTrader basic functionality test completed successfully!
```

### Reference Examples Testing

#### Attempted Tests

1. **Indicator Example**: `/nt_reference/examples/backtest/example_07_using_indicators/`
   - **Status**: ❌ Failed (module import issues)
   - **Error**: `ModuleNotFoundError: No module named 'examples'`
   - **Reason**: Examples need to be run from specific directory context

2. **Synthetic Data Example**: `/nt_reference/examples/backtest/synthetic_data_pnl_test.py`
   - **Status**: ❌ Failed (attribute error)
   - **Error**: `AttributeError: 'MinimalStrategy' object has no attribute 'portfolio_realized_pnl_values'`
   - **Observation**: Shows NautilusTrader engine starting up (good system info displayed)

## What the Tests Tell Us

### Environment Health

- ✅ Python 3.12.11 working correctly
- ✅ All development tools properly installed
- ✅ Code meets quality standards
- ✅ Basic NautilusTrader functionality working

### Core NautilusTrader Features Tested

1. **Instrument Creation**: EUR/USD forex pair
2. **Price Objects**: Decimal precision handling
3. **Market Data**: Quote tick generation
4. **Venue Simulation**: SIM (simulated) venue

### Areas Needing Further Testing

1. **Backtesting Engine**: Need working backtest example
2. **Strategy Development**: Create custom strategies
3. **Live Market Data**: Test with real data feeds
4. **Multiple Asset Classes**: Beyond forex
5. **Risk Management**: Position sizing and controls

## Testing Best Practices Learned

1. **Always run tests in sequence**: format → lint → typecheck → test
2. **Use make commands**: Standardized, reliable execution
3. **Check environment first**: Ensure all tools working before development
4. **Start simple**: Basic functionality before complex features

Date: 2025-07-11 System: Linux with Python 3.12.11, pytest-8.4.1
