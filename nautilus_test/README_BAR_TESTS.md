# NautilusTrader Bar Test Files

## Current Structure

### Primary Files
- **`simple_bars_test.py`** - **Main file** with all features combined
- **`tests/test_bars_functionality.py`** - Comprehensive pytest test suite

### Backup Files (for reference)
- `backup_ohlc_bars_test.py` - Original basic OHLC test
- `backup_improved_bars_test.py` - Added error handling version
- `backup_rich_bars_test.py` - Rich library focused version

## Features Consolidated into `simple_bars_test.py`

### From `ohlc_bars_test.py`:
✅ EMACrossBracket strategy support (configurable)
✅ Real CSV data loading capability (FXCM data)
✅ QuoteTickDataWrangler integration
✅ Multiple instruments (EUR/USD, GBP/USD)
✅ Higher starting capital options

### From `improved_bars_test.py`:
✅ Enhanced error handling with try/catch
✅ Currency and percentage formatting utilities
✅ Commission extraction and tracking
✅ Safer data parsing (string/float handling)

### From `rich_bars_test.py`:
✅ Progress indicators with spinners
✅ Rich table formatting and panels
✅ Beautiful terminal output

### Original Features:
✅ Synthetic OHLC bar generation
✅ EMA Cross strategy execution
✅ Performance summary reporting
✅ Trade history display

## Configuration Options

The main file now supports:
- `USE_REAL_DATA` - Toggle between synthetic and real FXCM data
- `USE_BRACKET_STRATEGY` - Choose between EMA Cross and EMA Cross Bracket
- `STARTING_CAPITAL` - Configurable starting balance

## Testing

Run the comprehensive test suite:
```bash
uv run pytest tests/test_bars_functionality.py -v
```

Tests cover:
- Bar creation and validation
- Backtest engine setup
- Strategy execution
- Performance metrics
- Different trade sizes
- Bar type consistency

## Usage

```bash
# Run the main enhanced test
uv run python simple_bars_test.py

# Run pytest tests
uv run pytest tests/ -v
```

This consolidation eliminates confusion while preserving all functionality in a single, well-organized file.