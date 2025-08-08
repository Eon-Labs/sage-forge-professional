# Misleading Demos Archive

**Purpose**: Scripts that claim to use TiRex but actually use different strategies, causing user confusion.

## Archived Files

### `ultimate_complete_demo.py`
- **Misleading Claim**: Presented as "Ultimate Complete Trading System" with TiRex features
- **Actual Implementation**: Uses `EMACross` strategy from NautilusTrader examples
- **Problem**: Generated zero trades despite claiming "214-order demo" 
- **User Impact**: Confused users expecting TiRex signals but getting EMA crossover strategy
- **Archive Date**: August 2025
- **Archive Reason**: Fixed documentation to point to actual working TiRex demos

## Correct TiRex Demos

For actual working TiRex demonstrations, see:
- `demos/tirex_backtest_demo.py` - TiRex demo with sample visualization
- `demos/adaptive_tirex_backtest_demo.py` - Real TiRex optimization (Sharpe 1.46, 20.9% return, 57.1% win rate)
- `demos/tirex_demo.py` - Simple TiRex signal generation test

## Notes

This archive prevents user confusion by removing demos that don't match their descriptions. The archived file used sophisticated Binance API integration and professional risk management, but with the wrong underlying strategy.