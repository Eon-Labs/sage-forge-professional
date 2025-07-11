# NautilusTrader Learning Notes - Strategies & Adapters

## Available Trading Strategies

### Built-in Strategy Examples
Located in: `/nt_reference/nautilus_trader/examples/strategies/`

1. **blank.py** - Template strategy
2. **ema_cross.py** - Exponential Moving Average crossover
3. **ema_cross_bracket.py** - EMA cross with bracket orders
4. **ema_cross_bracket_algo.py** - EMA cross with algorithmic execution
5. **ema_cross_hedge_mode.py** - EMA cross for hedge mode trading
6. **ema_cross_long_only.py** - EMA cross for long-only strategies
7. **ema_cross_stop_entry.py** - EMA cross with stop entry orders
8. **ema_cross_trailing_stop.py** - EMA cross with trailing stops
9. **ema_cross_twap.py** - EMA cross with TWAP execution
10. **market_maker.py** - Market making strategy
11. **orderbook_imbalance.py** - Order flow imbalance strategy
12. **orderbook_imbalance_rust.py** - Rust implementation of order flow
13. **signal_strategy.py** - Event-driven signal strategy
14. **subscribe.py** - Market data subscription example
15. **volatility_market_maker.py** - Dynamic spread market maker

### Strategy Categories

#### Trend Following
- **EMA Cross Variants**: Multiple implementations for different use cases
- **Features**: Moving average crossovers, trend detection, momentum
- **Complexity**: Beginner to intermediate

#### Market Making
- **Market Maker**: Provide liquidity with bid/ask quotes
- **Volatility Market Maker**: Adjust spreads based on volatility
- **Features**: Spread management, inventory control, risk limits

#### Mean Reversion
- **Orderbook Imbalance**: Trade based on order flow analysis
- **Features**: Microstructure analysis, short-term reversals

#### Signal-Based
- **Signal Strategy**: React to external signals/events
- **Features**: Event-driven trading, flexible signal sources

## Available Market Adapters

### Cryptocurrency Exchanges
Located in: `/nt_reference/nautilus_trader/adapters/`

#### Binance (`binance/`)
- **Markets**: Spot, Futures
- **Features**: 
  - REST API and WebSocket support
  - Multiple order types
  - Real-time market data
  - Historical data access
- **Files**: Common utilities, HTTP client, WebSocket client

#### Bybit (`bybit/`)
- **Markets**: Derivatives, Spot
- **Features**:
  - Position management
  - Leverage trading
  - Advanced order types
- **Structure**: Comprehensive endpoint coverage

#### OKX (`okx/`)
- **Markets**: Spot, Futures, Options
- **Features**: 
  - Multi-asset trading
  - Position tiers
  - Advanced trading features

#### Coinbase Pro/Advanced (`coinbase_intx/`)
- **Markets**: Institutional trading
- **Features**: Professional trading interface

#### dYdX (`dydx/`)
- **Markets**: Decentralized derivatives
- **Features**: 
  - DeFi integration
  - Perpetual futures
  - GRPC and HTTP APIs

### Traditional Markets

#### Interactive Brokers (`interactive_brokers/`)
- **Markets**: Stocks, Options, Futures, Forex
- **Features**:
  - Global market access
  - Professional trading tools
  - Historical data
  - Complex order types

#### Databento (`databento/`)
- **Markets**: US equities, options, futures
- **Features**:
  - High-quality market data
  - Historical data access
  - Real-time feeds

### Alternative Markets

#### Betfair (`betfair/`)
- **Markets**: Sports betting exchange
- **Features**: 
  - Betting exchange mechanics
  - Live odds
  - Order book trading

#### Polymarket (`polymarket/`)
- **Markets**: Prediction markets
- **Features**: Event-based trading

### Data Providers

#### Tardis (`tardis/`)
- **Purpose**: Historical cryptocurrency data
- **Features**: High-quality historical market data

## Example Strategy Patterns

### Basic Structure (from reference examples)
```python
class ExampleStrategy(Strategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Initialize indicators, parameters
        
    def on_start(self):
        # Strategy startup logic
        
    def on_data(self, data):
        # Process incoming market data
        
    def on_event(self, event):
        # Handle trading events
```

### Common Components
1. **Indicators**: Technical analysis tools (EMA, RSI, etc.)
2. **Risk Management**: Position sizing, stop losses
3. **Order Management**: Entry/exit logic
4. **Data Handlers**: Market data processing

## Integration Capabilities

### Supported Order Types
- Market orders
- Limit orders
- Stop orders
- Bracket orders
- Trailing stops
- Time-in-force options

### Risk Management Features
- Position sizing
- Maximum position limits
- Stop losses
- Portfolio-level risk controls

### Data Types Supported
- Trade ticks
- Quote ticks (bid/ask)
- Bars/Candles (multiple timeframes)
- Order book data
- Alternative data feeds

Date: 2025-07-11
Note: This represents available capabilities in the reference implementation