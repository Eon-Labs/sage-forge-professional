# NautilusTrader Nutshell Guide

A concise guide for users of NautilusTrader (NT), focusing on backtesting, bias prevention, custom indicators, and rolling windows. This draws from the NT repository (`repos/nautilus_trader/`) and best practices.

## 1. Getting Started

- **Installation**: Use `pip install nautilus-trader` or build from source in `repos/nautilus_trader/`.
- **Key Components**:
  - **Strategies**: Subclass `Strategy` in `nautilus_trader/trading/strategy.py`.
  - **Backtesting**: Use `BacktestEngine` (low-level) or `BacktestNode` (high-level) in `nautilus_trader/backtest/`.
  - **Live Trading**: Configure via adapters (e.g., Binance in `nautilus_trader/adapters/binance/`).
- **Documentation**: Check `docs/` for concepts like [backtesting](../../repos/nautilus_trader/docs/concepts/backtesting.md) and [indicators](../../repos/nautilus_trader/docs/api_reference/indicators.md).
- **Examples**: See `examples/backtest/` for strategy templates.

## 2. Backtesting Basics

- **Data Sequencing**: NT processes data chronologically via timestamps (`ts_event`, `ts_init`) to prevent look-ahead bias.
- **Venue Config**: Set `book_type` (L1/L2/L3) and `bar_execution=True` for bar data. Use `bar_adaptive_high_low_ordering=True` for realistic High/Low sequencing.
- **Fill Model**: Simulate slippage and queue positions with `FillModel` (e.g., `prob_slippage=0.5` for 50% chance of 1-tick slip).
- **Account Types**: Choose `CASH`, `MARGIN`, or `BETTING` when adding venues.
- **High-Level API Example**:

  ```python
  from nautilus_trader.backtest.node import BacktestNode
  from nautilus_trader.backtest.config import BacktestRunConfig

  config = BacktestRunConfig(...)  # Define venues, data, strategies
  node = BacktestNode(configs=[config])
  results = node.run()
  ```

- **Tip**: For large datasets, use high-level API with Parquet catalogs for streaming.

## 3. Preventing Look-Ahead Bias: Core Mechanisms

### 3.1 **Architectural Design Philosophy for Bias Prevention**

NautilusTrader's defense against look-ahead bias is not just a set of features, but a core part of its architecture. Understanding this philosophy helps in using the platform correctly.

- **Event-Driven, Not Vectorized**: Unlike backtesters that process entire data series (e.g., pandas DataFrames) at once, NT is event-driven. It processes one event (a tick, a bar closing, an order update) at a time, moving the simulation clock forward sequentially. This design inherently prevents looking at future data because, from the strategy's perspective, the "future" literally does not exist yet.
- **Encapsulation and Data Hiding**: A `Strategy` operates in a sandboxed environment. It cannot directly access the full historical dataset or the `BacktestEngine`'s data iterator. All historical data must be explicitly requested through the `Cache` or `request_*` methods, which have built-in safeguards (like time-bounding) to prevent access to data beyond the current simulation time.
- **Stateful, Evolving Cache**: The `Cache` is not a static database of the entire backtest period. It is a stateful component that is populated _as the simulation progresses_. When a strategy queries the cache at a given timestamp, it is only seeing a snapshot of the world as it was known at that exact moment.
- **Deterministic Message Passing**: Components communicate via a deterministic `MessageBus`. The order of messages follows the strict chronological flow of the simulation, ensuring that a strategy's reaction to an event (e.g., placing an order) is processed before the next market event is delivered.

### 3.2 **Chronological Data Processing**

- **BacktestDataIterator**: Uses strict chronological ordering by `ts_init` timestamps via heap-based merging.
- **Single vs Multi-Stream**: Optimized for single streams, k-way merge for multiple data streams.
- **Stream Priority**: Higher priority streams processed first when timestamps are identical.

### 3.3 **Data Validation & Sequence Checking**

- **validate_data_sequence**: Enable in `DataEngineConfig` to automatically reject out-of-sequence data.
  ```python
  config = DataEngineConfig(validate_data_sequence=True)
  ```
- **Sequence Validation**: Warns and skips bars/ticks with `ts_event < last_ts_event` or `ts_init < last_ts_init`.
- **Future Data Detection**: Catalog queries automatically prevent data from future timestamps.

### 3.4 **Dual Timestamp System**

- **ts_event**: When event actually occurred (exchange time).
- **ts_init**: When NT received/created the object (system time).
- **Latency Analysis**: Calculate `ts_init - ts_event` for realistic system delays.
- **Environment Behavior**:
  - **Backtesting**: Data ordered by `ts_init` with stable sort.
  - **Live**: Real-time processing with latency detection.

### 3.5 **Bar Timestamp Convention**

- **Critical Rule**: Bar timestamps must represent **closing time**, not opening time.
- **Adapter Support**: Use `bars_timestamp_on_close=True` for automatic adjustment.
- **Manual Adjustment**: Shift timestamps by bar duration (e.g., add 1 minute for 1-MINUTE bars).
- **ts_init_delta**: Control via `BarDataWrangler` parameter.

### 3.6 **Cache Access Patterns**

- **Historical Access Only**: Cache methods like `bars()`, `quotes()`, `trades()` return only past data.
- **Index-Based Access**: `cache.bar(bar_type, index=1)` gets previous bar safely.
- **Range Queries**: `bars_range()`, `quotes_range()` with time bounds prevent future access.
- **Reverse Indexing**: Index 0 = most recent, Index 1 = previous, etc.

### 3.7 **Execution Sequencing**

- **Data-First Processing**: Market data processed before strategy callbacks in backtest loop.
- **Message Bus Ordering**: Commands and events flow through MessageBus in correct sequence.
- **Execution Pipeline**: `Strategy` → `OrderEmulator` → `ExecAlgorithm` → `RiskEngine` → `ExecutionEngine`.

### 3.8 **Latency Models for Realism**

- **LatencyModel**: Simulates realistic order processing delays.

  ```python
  from nautilus_trader.backtest.models import LatencyModel

  latency = LatencyModel(
      base_latency_nanos=1_000_000,      # 1ms base
      insert_latency_nanos=2_000_000,    # 2ms order submission
      update_latency_nanos=1_500_000,    # 1.5ms modifications
      cancel_latency_nanos=1_000_000,    # 1ms cancellations
  )
  ```

- **Inflight Queue**: Commands processed with realistic delays via priority queue.
- **Prevents Instant Execution**: Orders can't execute immediately, mimicking real venues.

### 3.9 **Data Integrity Checks**

- **Timestamp Validation**: `check_ascending_timestamps()` ensures chronological order.
- **Bar Validation**: OHLC relationships checked (`high >= low`, `high >= close`, etc.).
- **Revision Handling**: `is_revision` flag for bar corrections without breaking sequence.

### 3.10 **Request Time Bounds**

- **Historical Requests**: Automatically cap `end` time to current timestamp.
- **Future Prevention**: Queries beyond `ts_now` truncated with warnings.
- **Range Validation**: `start <= end` enforced with detailed error messages.

### 3.11 **Time-Bar Aggregation Safeguards**

- **skip_first_non_full_bar**: Prevents use of an _incomplete_ first bar (which would embed future information) when aggregation starts mid-interval.
  ```python
  config = DataEngineConfig(time_bars_skip_first_non_full_bar=True)
  ```
- **time_bars_build_delay**: Micro-delay (µs) before building composite bars so _all_ underlying updates arrive; avoids early bar finalization that could leak future info.
  ```python
  config = DataEngineConfig(time_bars_build_delay=15)  # 15 µs
  ```
- **time_bars_build_with_no_updates**: Ensures bar builder still emits bars when no trades occur (maintains consistent timeline without guessing future trades).

### 3.12 **Order-Book Delta Buffering**

- **buffer_deltas** (DataEngineConfig): Buffers incoming `OrderBookDelta` messages until the venue flag `F_LAST` is received, guaranteeing a _complete_ order-book snapshot before it's forwarded to strategies/indicators.
  ```python
  config = DataEngineConfig(buffer_deltas=True)
  ```
- **Why it matters**: Prevents partial book states that could inadvertently expose future liquidity (or hide present liquidity) within the same nanosecond.

### 3.13 **Book-Type Consistency Checks**

- NT validates that provided data depth matches venue `book_type`.
  - **Example**: Supplying L1 bar/quote data while venue is configured as `L2_MBP` triggers an `InvalidConfiguration` error.
  - **Action**: Always align `book_type` with data granularity (set to `L1_MBP` for bars/quotes; `L2_MBP` or `L3_MBO` when supplying depth data).

These additional configuration knobs complement earlier mechanisms, giving you **granular control** over bar aggregation timing, order-book completeness, and venue/data alignment—further eliminating sources of forward-looking bias.

## 4. Built-in Indicators

- **Categories**: Moving averages (SMA, EMA), momentum (RSI, Bollinger Bands), volatility (ATR), etc. (See `crates/indicators/` or `nautilus_trader/indicators/`).
- **Usage**:

  ```python
  from nautilus_trader.indicators.average.ema import ExponentialMovingAverage

  self.ema = ExponentialMovingAverage(10)
  self.register_indicator_for_bars(BAR_TYPE, self.ema)  # Auto-updates
  ```

- **Rolling Window Pattern**: Indicators use deques or `ArrayDeque` for fixed windows, updating via `update_raw`.

## 5. Custom Indicators

- **Python Subclassing**: Extend `Indicator` from `nautilus_trader/indicators/base/indicator.pyx`.

  - Implement `update_raw`, `handle_bar`, etc.
  - Example:

    ```python
    from nautilus_trader.indicators.base.indicator import Indicator
    from collections import deque

    class CustomIndicator(Indicator):
        def __init__(self, period: int):
            super().__init__(params=[period])
            self.window = deque(maxlen=period)
            self.value = 0.0

        def update_raw(self, value: float):
            self.window.append(value)
            if len(self.window) == self.period:
                self.value = sum(self.window) / self.period
                self._set_initialized(True)
    ```

- **Rust Implementation**: Implement `Indicator` trait in `crates/indicators/src/indicator.rs`. Expose via PyO3.
- **Integration**: Register in strategy's `on_start`.

## 6. Integrating Third-Party Libraries (e.g., Change Point Detection)

- **Pattern**: Use in `update_raw` of custom indicators. Combine with rolling windows for online processing.
- **Example with `ruptures`** (install via pip):

  ```python
  import ruptures as rpt
  from collections import deque
  from nautilus_trader.indicators.base.indicator import Indicator

  class CPDIndicator(Indicator):
      def __init__(self, window_size: int):
          super().__init__(params=[window_size])
          self.window = deque(maxlen=window_size)
          self.change_points = []

      def update_raw(self, value: float):
          self.window.append(value)
          if len(self.window) == self.window_size:
              signal = list(self.window)
              algo = rpt.Pelt(model="rbf").fit(signal)
              self.change_points = algo.predict(pen=3)
              self._set_initialized(True)
  ```

- **Tips**: For online CPD, use `changefinder`. Ensure thread-safety for live trading.

## 7. Rolling Windows Best Practices

- **Native Structures**: `deque(maxlen=period)` (Python) or `ArrayDeque` (Rust) for O(1) updates.
- **Bias-Free**: Update only on new data; check `initialized` before use.
- **Multi-Timeframe**: Use multiple indicators with different periods.
- **Efficiency**: Avoid full recomputes; leverage NT's sequential engine.

## 8. Advanced Bias Prevention Patterns

### 8.1 **Configuration-Based Protection**

```python
# Enable comprehensive data validation
config = DataEngineConfig(
    validate_data_sequence=True,           # Reject out-of-sequence data
    time_bars_timestamp_on_close=True,     # Proper bar timestamping
    time_bars_build_with_no_updates=True,  # Build bars even without updates
)

# Backtest engine with latency simulation
engine_config = BacktestEngineConfig(
    latency_model=LatencyModel(base_latency_nanos=1_000_000),
    validate_data_sequence=True,
)
```

### 8.2 **Safe Historical Data Access**

```python
# In strategies, always use cache methods for historical access
class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # Safe: Gets only historical bars
        last_10_bars = self.cache.bars(self.bar_type)[:10]

        # Safe: Index-based access to previous bars
        prev_bar = self.cache.bar(self.bar_type, index=1)

        # Safe: Time-bounded historical requests
        historical_bars = self.request_bars(
            bar_type=self.bar_type,
            start=datetime.utcnow() - timedelta(days=1),
            end=datetime.utcnow(),  # Capped to current time
        )
```

### 8.3 **Multi-Stream Data Synchronization**

```python
# BacktestDataIterator automatically handles multiple data streams
# with proper chronological ordering and priority handling
iterator = BacktestDataIterator()
iterator.append_data(bars_stream_1, append_data=False)    # Higher priority
iterator.append_data(bars_stream_2, append_data=True)     # Lower priority

# Data yielded in strict ts_init order with priority tiebreaking
```

### 8.4 **Realistic Execution Simulation**

```python
# Configure venue with realistic constraints
engine.add_venue(
    venue=VENUE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    starting_balances=[Money(10_000, USD)],
    latency_model=LatencyModel(
        insert_latency_nanos=2_000_000,  # 2ms order submission delay
        cancel_latency_nanos=1_000_000,  # 1ms cancellation delay
    ),
    fill_model=FillModel(
        prob_fill_on_limit=0.8,          # 80% fill probability
        prob_slippage=0.3,               # 30% slippage chance
    ),
    bar_adaptive_high_low_ordering=True, # Realistic OHLC sequencing
)
```

## 9. Native Long/Short Target and Stop Loss Management

NautilusTrader provides comprehensive native support for position management through sophisticated order types, bracket orders, and trailing stops. This section covers all available mechanisms for setting targets and stop losses in both backtesting and live trading environments.

### 9.1 **Order Types for Position Management**

NautilusTrader supports a comprehensive set of order types that enable sophisticated risk management:

#### **Core Order Types**

- **`MARKET`**: Immediate execution at best available price
- **`LIMIT`**: Execute at specific price or better
- **`STOP_MARKET`**: Conditional market order triggered at stop price
- **`STOP_LIMIT`**: Conditional limit order triggered at stop price
- **`MARKET_IF_TOUCHED`**: Market order triggered when price touches trigger level
- **`LIMIT_IF_TOUCHED`**: Limit order triggered when price touches trigger level

#### **Advanced Order Types**

- **`TRAILING_STOP_MARKET`**: Dynamic stop that trails favorable price movement
- **`TRAILING_STOP_LIMIT`**: Trailing stop with limit price protection
- **`MARKET_TO_LIMIT`**: Market order that converts to limit if partially filled

### 9.2 **Position Sides and Order Sides**

Understanding the relationship between position sides and order sides is crucial:

```python
from nautilus_trader.model.enums import OrderSide, PositionSide

# Position sides
PositionSide.LONG    # Long position (expecting price to rise)
PositionSide.SHORT   # Short position (expecting price to fall)
PositionSide.FLAT    # No position

# Order sides for position management
OrderSide.BUY        # Creates LONG position or closes SHORT position
OrderSide.SELL       # Creates SHORT position or closes LONG position

# Position closing logic
def get_closing_side(position_side: PositionSide) -> OrderSide:
    if position_side == PositionSide.LONG:
        return OrderSide.SELL    # Sell to close long
    elif position_side == PositionSide.SHORT:
        return OrderSide.BUY     # Buy to close short
    else:
        return OrderSide.NO_ORDER_SIDE
```

### 9.3 **Bracket Orders: Complete Position Management**

Bracket orders are the most powerful tool for comprehensive position management, allowing you to set entry, take-profit, and stop-loss orders simultaneously.

#### **Basic Bracket Order Example**

```python
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce

# Create a bracket order for a long position
bracket_order_list = self.order_factory.bracket(
    instrument_id=self.instrument_id,
    order_side=OrderSide.BUY,                    # Long position
    quantity=self.instrument.make_qty(100),      # Position size

    # Entry order (default: MARKET)
    entry_order_type=OrderType.MARKET,

    # Take-profit order (target)
    tp_price=self.instrument.make_price(last_price + (20 * tick_size)),  # 20 ticks profit
    tp_order_type=OrderType.LIMIT,                # Default: LIMIT

    # Stop-loss order
    sl_trigger_price=self.instrument.make_price(last_price - (10 * tick_size)),  # 10 ticks risk
    sl_order_type=OrderType.STOP_MARKET,          # Default: STOP_MARKET

    time_in_force=TimeInForce.GTC,
)

# Submit the bracket order
self.submit_order_list(bracket_order_list)
```

#### **Advanced Bracket Order Configuration**

```python
from nautilus_trader.model.enums import ContingencyType, TriggerType
from decimal import Decimal

# Advanced bracket with conditional entry
bracket_order_list = self.order_factory.bracket(
    instrument_id=self.instrument_id,
    order_side=OrderSide.SELL,                   # Short position
    quantity=self.instrument.make_qty(100),

    # Conditional entry order
    entry_order_type=OrderType.LIMIT_IF_TOUCHED,
    entry_price=self.instrument.make_price(entry_limit_price),
    entry_trigger_price=self.instrument.make_price(entry_trigger_price),

    # Take-profit with trigger
    tp_order_type=OrderType.LIMIT_IF_TOUCHED,
    tp_price=self.instrument.make_price(target_price),
    tp_trigger_price=self.instrument.make_price(target_trigger),
    tp_trigger_type=TriggerType.BID_ASK,

    # Trailing stop-loss
    sl_order_type=OrderType.TRAILING_STOP_MARKET,
    sl_trailing_offset=Decimal("0.50"),          # 50 cent trailing offset
    sl_trailing_offset_type=TrailingOffsetType.PRICE,
    sl_trigger_type=TriggerType.LAST_PRICE,

    # Contingency relationship
    contingency_type=ContingencyType.OUO,        # One-Updates-Other

    # Emulation for unsupported venues
    emulation_trigger=TriggerType.BID_ASK,

    time_in_force=TimeInForce.GTC,
)
```

### 9.4 **Trailing Stops: Dynamic Risk Management**

Trailing stops automatically adjust stop prices as the market moves favorably, locking in profits while maintaining downside protection.

#### **Trailing Stop Market Orders**

```python
from nautilus_trader.model.enums import TrailingOffsetType
from decimal import Decimal

# Trailing stop for long position (sell to close)
trailing_stop = self.order_factory.trailing_stop_market(
    instrument_id=self.instrument_id,
    order_side=OrderSide.SELL,                   # Close long position
    quantity=position.quantity,                   # Full position size

    # Trailing configuration
    trailing_offset=Decimal("100"),               # Offset amount
    trailing_offset_type=TrailingOffsetType.BASIS_POINTS,  # 1% trailing

    # Activation price (when trailing starts)
    activation_price=self.instrument.make_price(current_price + profit_buffer),

    # Trigger type for monitoring
    trigger_type=TriggerType.LAST_PRICE,

    # Risk management
    reduce_only=True,                            # Only reduce position
    time_in_force=TimeInForce.GTC,
)

self.submit_order(trailing_stop)
```

#### **Trailing Stop Limit Orders**

```python
# More sophisticated trailing stop with limit protection
trailing_stop_limit = self.order_factory.trailing_stop_limit(
    instrument_id=self.instrument_id,
    order_side=OrderSide.BUY,                    # Close short position
    quantity=position.quantity,

    # Initial limit price
    price=self.instrument.make_price(initial_limit_price),

    # Trailing offsets
    trailing_offset=Decimal("0.25"),             # 25 cent trailing for trigger
    limit_offset=Decimal("0.10"),               # 10 cent offset for limit from trigger
    trailing_offset_type=TrailingOffsetType.PRICE,

    # Activation and trigger
    activation_price=self.instrument.make_price(activation_level),
    trigger_type=TriggerType.BID_ASK,

    reduce_only=True,
    time_in_force=TimeInForce.GTC,
)
```

#### **Trailing Offset Types**

```python
# Different ways to specify trailing offsets
TrailingOffsetType.PRICE          # Fixed price offset (e.g., $0.50)
TrailingOffsetType.BASIS_POINTS   # Percentage offset (e.g., 100bp = 1%)
TrailingOffsetType.TICKS          # Number of ticks (e.g., 5 ticks)
TrailingOffsetType.PRICE_TIER     # Venue-specific price tier

# Examples for each type
trailing_price = Decimal("0.50")      # $0.50 trailing
trailing_bp = Decimal("100")          # 1% trailing (100 basis points)
trailing_ticks = Decimal("5")         # 5 tick trailing
```

### 9.5 **Position-Aware Order Management**

Strategies can intelligently manage orders based on current position state:

```python
class PositionAwareStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)

        if position is None or position.side == PositionSide.FLAT:
            self.enter_position(bar)
        elif position.side == PositionSide.LONG:
            self.manage_long_position(position, bar)
        elif position.side == PositionSide.SHORT:
            self.manage_short_position(position, bar)

    def manage_long_position(self, position, bar):
        """Manage existing long position with dynamic stops/targets."""
        unrealized_pnl = position.unrealized_pnl(bar.close)

        # Dynamic stop loss based on ATR
        atr_stop = bar.close - (2.0 * self.atr.value)
        current_stop = self.get_current_stop_price()

        # Trail stop higher if profitable
        if unrealized_pnl.as_double() > 0 and atr_stop > current_stop:
            self.update_stop_loss(atr_stop)

        # Take partial profits at targets
        if unrealized_pnl.as_double() > self.profit_target:
            self.take_partial_profits(position.quantity * 0.5)

    def update_stop_loss(self, new_stop_price):
        """Update existing stop loss order."""
        # Find current stop loss order
        stop_orders = [order for order in self.cache.orders_open()
                      if order.order_type == OrderType.STOP_MARKET
                      and order.order_side == OrderSide.SELL]

        if stop_orders:
            # Modify existing stop
            stop_order = stop_orders[0]
            self.modify_order(
                order=stop_order,
                trigger_price=self.instrument.make_price(new_stop_price)
            )
        else:
            # Create new stop loss
            stop_order = self.order_factory.stop_market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.cache.position_for_instrument(self.instrument_id).quantity,
                trigger_price=self.instrument.make_price(new_stop_price),
                reduce_only=True,
            )
            self.submit_order(stop_order)
```

### 9.6 **Venue-Specific Capabilities**

Different venues support different order types and features:

#### **Binance Futures**

```python
# Binance supports comprehensive order types
- MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
- MARKET_IF_TOUCHED, LIMIT_IF_TOUCHED
- TRAILING_STOP_MARKET (with activation_price)
- reduce_only instruction supported
- Hedging mode with position IDs

# Binance trailing stops require activation_price
trailing_stop = self.order_factory.trailing_stop_market(
    instrument_id=instrument_id,
    order_side=OrderSide.SELL,
    quantity=quantity,
    activation_price=activation_price,        # Required for Binance
    trailing_offset=Decimal("100"),           # Basis points only
    trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
)
```

#### **Interactive Brokers**

```python
# IB supports most order types with OCA (One-Cancels-All) groups
- Full bracket order support
- Configurable OCA behavior (pro-rate vs full cancel)
- Advanced trigger types (BID_ASK, LAST_PRICE, etc.)
- Comprehensive trailing stop support

# IB bracket orders can be configured for partial fills
bracket_order_list = self.order_factory.bracket(
    instrument_id=instrument_id,
    order_side=OrderSide.BUY,
    quantity=quantity,
    tp_price=target_price,
    sl_trigger_price=stop_price,
    contingency_type=ContingencyType.OUO,     # Partial fill handling
)
```

#### **dYdX v4**

```python
# dYdX supports basic order types with on-chain conditions
- MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
- Oracle-based trigger prices
- Size-exact bracket conditions
- No partial fill complications (all-or-nothing)
```

### 9.7 **Risk Management Best Practices**

#### **Position Sizing and Risk**

```python
class RiskManagedStrategy(Strategy):
    def calculate_position_size(self, entry_price, stop_price, risk_per_trade):
        """Calculate position size based on risk management rules."""
        risk_per_share = abs(entry_price - stop_price)
        max_shares = risk_per_trade / risk_per_share

        # Account for minimum tick size
        tick_size = self.instrument.price_increment
        adjusted_shares = int(max_shares / tick_size) * tick_size

        return self.instrument.make_qty(adjusted_shares)

    def create_risk_managed_bracket(self, signal_strength):
        """Create bracket order with dynamic risk/reward."""
        current_price = self.last_quote.bid_price
        atr_value = self.atr.value

        # Dynamic stops based on volatility and signal strength
        stop_multiplier = 2.0 / signal_strength  # Tighter stops for stronger signals
        target_multiplier = 3.0 * signal_strength  # Bigger targets for stronger signals

        stop_price = current_price - (stop_multiplier * atr_value)
        target_price = current_price + (target_multiplier * atr_value)

        # Calculate position size for fixed risk
        position_size = self.calculate_position_size(
            entry_price=current_price,
            stop_price=stop_price,
            risk_per_trade=self.account_balance * 0.02  # 2% risk per trade
        )

        return self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=position_size,
            tp_price=self.instrument.make_price(target_price),
            sl_trigger_price=self.instrument.make_price(stop_price),
        )
```

#### **Multi-Level Exit Strategy**

```python
def create_scaled_exit_strategy(self, position_size, entry_price):
    """Create multiple exit orders for scaled position management."""
    orders = []

    # First target: 50% at 1:1 risk/reward
    target_1 = entry_price + (entry_price - self.stop_price)
    orders.append(self.order_factory.limit(
        instrument_id=self.instrument_id,
        order_side=OrderSide.SELL,
        quantity=position_size * 0.5,
        price=self.instrument.make_price(target_1),
        reduce_only=True,
    ))

    # Second target: 30% at 2:1 risk/reward
    target_2 = entry_price + 2 * (entry_price - self.stop_price)
    orders.append(self.order_factory.limit(
        instrument_id=self.instrument_id,
        order_side=OrderSide.SELL,
        quantity=position_size * 0.3,
        price=self.instrument.make_price(target_2),
        reduce_only=True,
    ))

    # Trailing stop for remaining 20%
    orders.append(self.order_factory.trailing_stop_market(
        instrument_id=self.instrument_id,
        order_side=OrderSide.SELL,
        quantity=position_size * 0.2,
        trailing_offset=Decimal("50"),  # 50bp trailing
        trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
        reduce_only=True,
    ))

    return orders
```

### 9.8 **Backtesting Considerations**

#### **Realistic Execution Simulation**

```python
# Configure venues for realistic bracket order simulation
engine.add_venue(
    venue=VENUE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,           # Enable short selling
    starting_balances=[Money(100_000, USD)],

    # Realistic latency for order processing
    latency_model=LatencyModel(
        base_latency_nanos=1_000_000,          # 1ms base latency
        insert_latency_nanos=2_000_000,        # 2ms order submission
        update_latency_nanos=1_500_000,        # 1.5ms order modifications
        cancel_latency_nanos=1_000_000,        # 1ms order cancellations
    ),

    # Realistic fill simulation
    fill_model=FillModel(
        prob_fill_on_limit=0.8,                # 80% fill probability on limit orders
        prob_slippage=0.3,                     # 30% chance of slippage on market orders
    ),

    # Realistic OHLC sequencing for intrabar fills
    bar_adaptive_high_low_ordering=True,
)
```

#### **Stop Loss and Take Profit Validation**

```python
def validate_bracket_execution(self, bracket_results):
    """Validate that bracket orders executed properly in backtest."""
    entry_order, stop_order, target_order = bracket_results.orders

    # Verify only one exit order filled (OUO behavior)
    exit_fills = [order for order in [stop_order, target_order] if order.is_closed()]
    assert len(exit_fills) == 1, "Multiple exit orders filled - OUO not working"

    # Verify realistic fill prices
    if entry_order.is_filled():
        fill_price = entry_order.last_event.last_px
        market_price = self.cache.quote_tick(self.instrument_id).bid_price
        slippage = abs(fill_price - market_price)
        assert slippage <= self.max_expected_slippage

    # Verify stop loss triggered at correct price
    if stop_order.is_filled():
        trigger_price = stop_order.trigger_price
        fill_price = stop_order.last_event.last_px
        # Stop should fill at or worse than trigger price
        if entry_order.order_side == OrderSide.BUY:  # Long position
            assert fill_price <= trigger_price, "Stop filled better than trigger"
```

### 9.9 **Live Trading Considerations**

#### **Order Management in Live Trading**

```python
class LivePositionManager:
    def __init__(self, strategy):
        self.strategy = strategy
        self.active_stops = {}
        self.active_targets = {}

    def on_position_opened(self, event):
        """Set up stop loss and take profit when position opens."""
        position = event.position

        # Calculate stop and target prices
        entry_price = position.avg_px_open
        atr_value = self.strategy.atr.value

        stop_price = entry_price - (2.0 * atr_value) if position.side == PositionSide.LONG else entry_price + (2.0 * atr_value)
        target_price = entry_price + (3.0 * atr_value) if position.side == PositionSide.LONG else entry_price - (3.0 * atr_value)

        # Create stop loss
        stop_order = self.strategy.order_factory.stop_market(
            instrument_id=position.instrument_id,
            order_side=position.closing_order_side(),
            quantity=position.quantity,
            trigger_price=self.strategy.instrument.make_price(stop_price),
            reduce_only=True,
        )

        # Create take profit
        target_order = self.strategy.order_factory.limit(
            instrument_id=position.instrument_id,
            order_side=position.closing_order_side(),
            quantity=position.quantity,
            price=self.strategy.instrument.make_price(target_price),
            reduce_only=True,
        )

        # Submit orders
        self.strategy.submit_order(stop_order)
        self.strategy.submit_order(target_order)

        # Track orders
        self.active_stops[position.id] = stop_order
        self.active_targets[position.id] = target_order

    def on_position_closed(self, event):
        """Clean up orders when position closes."""
        position_id = event.position.id

        # Cancel remaining orders
        if position_id in self.active_stops:
            remaining_stop = self.active_stops[position_id]
            if not remaining_stop.is_closed():
                self.strategy.cancel_order(remaining_stop)
            del self.active_stops[position_id]

        if position_id in self.active_targets:
            remaining_target = self.active_targets[position_id]
            if not remaining_target.is_closed():
                self.strategy.cancel_order(remaining_target)
            del self.active_targets[position_id]
```

This comprehensive coverage of NautilusTrader's native long/short target and stop loss capabilities provides traders with all the tools needed for sophisticated position management in both backtesting and live trading environments.

## 10. Common Pitfalls & Tips

- **Bias**: Always verify timestamps and use cache methods.
- **Performance**: Profile heavy computations; use Rust for speed.
- **Testing**: Run small backtests; check logs for sequence warnings.
- **Data Quality**: Enable `validate_data_sequence` to catch timestamp issues.
- **Latency Realism**: Use `LatencyModel` for production-ready backtests.
- **Cache Usage**: Prefer cache methods over direct data manipulation.
- **Extensions**: Contribute to NT's open-source repo.
- **Resources**: `docs/concepts/`, `examples/`, Discord community.

For more, explore the repo or ask in the NT Discord!
