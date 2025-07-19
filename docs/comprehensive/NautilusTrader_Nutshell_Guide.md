# NautilusTrader Nutshell Guide

A concise guide for users of NautilusTrader (NT), focusing on backtesting, bias prevention, custom indicators, and rolling windows. This draws from the NT repository (`repos/nautilus_trader/`) and best practices.

## 1. Getting Started

- **Installation**: Use `pip install nautilus-trader` or build from source in `repos/nautilus_trader/`.
- **Key Components**:
  - **Strategies**: Subclass `Strategy` in `nautilus_trader/trading/strategy.py`.
  - **Backtesting**: Use `BacktestEngine` (low-level) or `BacktestNode` (high-level) in `nautilus_trader/backtest/`.
  - **Live Trading**: Configure via adapters (e.g., Binance in `nautilus_trader/adapters/binance/`).
- **Documentation**: Check `docs/` for concepts like [backtesting](concepts/backtesting.md) and [indicators](concepts/indicators.md).
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

- **buffer_deltas** (DataEngineConfig): Buffers incoming `OrderBookDelta` messages until the venue flag `F_LAST` is received, guaranteeing a _complete_ order-book snapshot before it’s forwarded to strategies/indicators.
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

## 9. Common Pitfalls & Tips

- **Bias**: Always verify timestamps and use cache methods.
- **Performance**: Profile heavy computations; use Rust for speed.
- **Testing**: Run small backtests; check logs for sequence warnings.
- **Data Quality**: Enable `validate_data_sequence` to catch timestamp issues.
- **Latency Realism**: Use `LatencyModel` for production-ready backtests.
- **Cache Usage**: Prefer cache methods over direct data manipulation.
- **Extensions**: Contribute to NT's open-source repo.
- **Resources**: `docs/concepts/`, `examples/`, Discord community.

For more, explore the repo or ask in the NT Discord!
