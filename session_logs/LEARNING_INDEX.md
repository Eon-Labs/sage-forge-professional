# Learning Index

Aggregated knowledge and discoveries from all sessions.

## NautilusTrader Knowledge Base

### Backtesting Framework
**Source**: [2025-07-11-001](./2025/07/2025-07-11-001.md)
**Status**: Research Complete

**Key Capabilities:**
- **API Levels**: Low-level `BacktestEngine` + High-level `BacktestNode`
- **Data Granularities**: L3/L2 orderbook → Ticks → OHLC bars (descending detail)
- **Markets**: Crypto (Binance, Bybit), Forex, Equities, Derivatives, Sports betting
- **Advanced Execution**: Realistic fill models, slippage simulation, partial fills, latency modeling

**OHLC Bar Features:**
- Smart sequencing: Open→High→Low→Close or adaptive High/Low ordering
- Volume distribution across OHLC points  
- ~75-85% accuracy with adaptive mode

**Key Finding**: Professional-grade backtesting rivaling commercial platforms

### Development Environment
**Source**: [2025-07-11-001](./2025/07/2025-07-11-001.md)

**Package Management**: uv (official recommendation)
**Code Standards**: 100-character line length (black/ruff)
**Testing**: pytest, comprehensive test suite

## Session Management System
**Source**: [2025-07-11-001](./2025/07/2025-07-11-001.md)
**Status**: Implemented & Enhanced

**Evolution:**
1. **v1**: Single SESSION_LOG.md (manual management)
2. **v2**: Daily session files (manual import updates)  
3. **v3**: Auto-discovery with LATEST.md symlink system

**Current Architecture:**
- **CLAUDE.md**: Principles only
- **LATEST.md**: Auto-discovery symlink
- **INDEX.md**: Session registry
- **Organized storage**: YYYY/MM/filename structure

## Topics to Explore

### Immediate Focus
- **OHLC Bars Handling**: Start with basic EMA cross strategy
- **Target Example**: `crypto_ema_cross_ethusdt_bars.py`
- **Learning Goals**: Bar execution logic, data loading, basic backtesting

### Future Research
- L2 orderbook backtesting with realistic fills
- Custom strategy development with risk management
- Multi-timeframe strategies
- Performance analysis and reporting

---

*This index aggregates knowledge across all sessions for quick reference and learning continuity.*