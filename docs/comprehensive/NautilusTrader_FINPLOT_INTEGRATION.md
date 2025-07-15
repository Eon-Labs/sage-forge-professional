# NautilusTrader finplot Integration Guide

> **A cookbook-style integration guide that adheres to NautilusTrader’s native idioms (MessageBus + Actors + StreamingConfig) while incorporating finplot for both post-run and live charting.**

Below is a **cookbook-style** integration guide that sticks to NautilusTrader’s native idioms (MessageBus + Actors + StreamingConfig) while wiring in **finplot** for both _post-run_ and _live_ charting. After each section, you’ll see the exact docs/discussions to read next, so you never lose the breadcrumb trail.

---

## 1 What the Docs Already Give You

| Concept                                             | Where it’s Documented                                                                                                   | Why it Matters                                                                   |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **MessageBus** & deterministic event order          | [Concepts → Message Bus](https://nautilustrader.io/docs/latest/concepts/message_bus/)                                   | Your chart actor just _subscribes_, exactly like a strategy.                     |
| **Actor** lifecycle & `on_data`                     | [Concepts → Actors](https://nautilustrader.io/docs/latest/concepts/actors/)                                             | Put your finplot calls here; no threads required.                                |
| **StreamingConfig** → writes **Feather** files      | [API Reference → Config](https://nautilustrader.io/docs/latest/api_reference/config/#streamingconfig) (see `streaming`) | Easiest offline-chart path: read those Feather frames with pandas, then finplot. |
| GitHub Q&A: _“Plotting entries, exits, indicators”_ | [Discussion #1187](https://github.com/nautechsystems/nautilus_trader/discussions/1187), answer by @limx0                | Confirms Feather-streaming is the blessed route today.                           |

---

## 2 Two Integration Modes that Stay 100% “Native”

| Mode                                                         | When to Use                                                 | Key Nautilus Keywords                                              | Finplot Recipe                                                                      |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **A. Post-run visual audit**                                 | Back-test finished, you want rich OHLC + signals inspection | `StreamingConfig`, `convert_stream_to_data`, `pandas.read_feather` | Load Feather → DataFrame → `finplot.plot(df)` (5-liners).                           |
| **B. Decoupled Live Dashboard (Recommended for Production)** | You need charts updating while the strategy runs            | `Actor`, `subscribe_data`, `publish_signal`                        | Build a `FinplotActor` that pushes rows into a finplot `qplot_ohlc` via PyQt timer. |

---

## 3 Mode A – Five-line Offline Plot

```python
# read_stream.py
import pandas as pd, finplot as fplt
df = pd.read_feather('backtest_stream/BinanceBar.feather')  # OHLCV per bar
plot = fplt.create_plot('Audit', init_zoom_periods=200)
fplt.candlestick_ochl(df[['open','close','high','low']], ax=plot)
fplt.show()
```

_Feather is columnar; load is < 1 s for millions of rows._ See finplot’s examples for styling tweaks ([GitHub](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py)).

---

## 4 Mode B – Decoupled Live Dashboard (Recommended for Production)

```python
# In Your Strategy/Actor (Publish Data Natively):
from nautilus_trader.trading.strategy import Strategy

class MyStrategy(Strategy):
    # ... (your existing init and on_start)

    def on_bar(self, bar):
        # Your logic...
        # Publish bar data as a signal (native to MessageBus)
        self.publish_signal("live_bar", bar)  # Or custom dict with indicators
```

```python
# live_plotter.py (run this in a separate terminal/process)
import redis
import finplot as fplt
import pyqtgraph as pg
from nautilus_trader.config import MessageBusConfig  # For config consistency

# Connect to Redis (match your Nautilus MessageBus config)
r = redis.Redis(host='localhost', port=6379, db=0)  # Adjust if needed
pubsub = r.pubsub()
pubsub.subscribe("signals:live_bar")  # Matches your publish_signal topic

# Finplot setup
ax = fplt.create_plot('Live Dashboard', maximize=False)
ohlc = []  # Rolling buffer

# Timer for refresh (independent of Nautilus)
timer = pg.QtCore.QTimer()
timer.timeout.connect(lambda: refresh_plot())
timer.start(100)  # 100ms refresh

def refresh_plot():
    if not ohlc: return
    ts, o, c, h, l = zip(*ohlc)
    fplt.candlestick_ochl((o, c, h, l), dates=ts, ax=ax, clear=True)
    ohlc.clear()

# Listen for messages
for message in pubsub.listen():
    if message['type'] == 'message':
        data = message['data']  # Deserialize if needed (e.g., JSON)
        # Assume data is a dict like {'ts': ts_ns, 'open': o, ...}
        ohlc.append((data['ts'] / 1e9, data['open'], data['close'], data['high'], data['low']))

# Run finplot show in this process
fplt.show()
```

_How to Run_: Start your NautilusTrader system first, then run `python live_plotter.py` in another terminal. Data streams via Redis without touching the trading thread.

_Config Tip_: In your Nautilus config, set `MessageBusConfig(backend='redis')` to enable Redis persistence ([Config docs](https://nautilustrader.io/docs/latest/api_reference/config/)).

### Mode B Variant: Embedded (For Non-Production Use)

**Warning**: This embeds plotting in the Actor, sharing the main thread. It risks blocking the event loop in high-frequency scenarios—use only for low-frequency debugging. For production, prefer the decoupled approach above.

```python
from nautilus_trader.trading import Actor
import finplot as fplt, pyqtgraph as pg

class FinplotActor(Actor):
    def on_start(self):
        self._ax = fplt.create_plot('Live', maximize=False)
        self._ohlc = []                          # rolling buffer

        # timer that refreshes the canvas every 100 ms
        timer = pg.QtCore.QTimer();  timer.timeout.connect(self._refresh)
        timer.start(100)

    def on_data(self, data):
        if isinstance(data, Bar):   # Use isinstance for type safety
            self._ohlc.append(
                (data.ts_event/1e9, data.open, data.close, data.high, data.low)
            )

    def _refresh(self):
        if not self._ohlc: return
        ts, o, c, h, l = zip(*self._ohlc)
        fplt.candlestick_ochl((o,c,h,l), dates=ts, ax=self._ax, clear=True)
        self._ohlc.clear()
```

---

## 5 Frequently-Missed Native Keywords (Keep Them in Your Code / Config)

- `StreamingConfig` – turns on Feather writers.
- `FinplotActor` – inherits `Actor`, uses `on_data`.
- `publish_signal(name, value)` – push custom indicators; FinplotActor can subscribe to them too.
- `Portfolio.credit()` – if you also plot funding cash-flows, call this inside your FundingActor before chart refresh so equity matches the candles.

---

## 6 Pitfalls & Fixes

| Pitfall                                   | Symptom                       | Fix                                                                                                                                                                            |
| ----------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Running finplot in a _second_ thread      | Charts freeze                 | Keep everything in the Nautilus thread; use Qt timer.                                                                                                                          |
| Forgetting nanosecond→datetime conversion | X-axis jumbled                | Divide `ts_event` by 1e9 before sending to finplot.                                                                                                                            |
| Feather missing custom indicator columns  | Only price shows up           | Add `stream_indicators=True` (coming patch) or write them via `publish_signal`; pending issue #1356 ([GitHub](https://github.com/nautechsystems/nautilus_trader/issues/1356)). |
| Embedded GUI blocking event loop          | Strategy delays/misses events | Decouple plotting to an external process via MessageBus/Redis.                                                                                                                 |

---

## 7 Where to Dive Deeper Next

- **Architecture & MessageBus** – [/concepts/architecture/](https://nautilustrader.io/docs/latest/concepts/architecture/) and [/concepts/message_bus/](https://nautilustrader.io/docs/latest/concepts/message_bus/)
- **StreamingConfig struct** – [/api_reference/config/#streamingconfig](https://nautilustrader.io/docs/latest/api_reference/config/#streamingconfig)
- **Finplot real-time tricks** – GitHub discussions #164 & example `complicated.py` in the repo ([GitHub](https://github.com/highfestiva/finplot/discussions/164), [GitHub](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py))
- **External Processes & Redis** – [Message Bus docs](https://nautilustrader.io/docs/latest/concepts/message_bus/) for streaming to decoupled tools; see user examples like chrwoizi/naytrading-trader.

Follow the two modes above and you stay entirely inside NautilusTrader’s official extension points **while getting finplot’s slick, 60 FPS charts**—no forks, no hidden globals, just clean Actor code and Feather files.

[1]: https://nautilustrader.io/docs/latest/concepts/message_bus/?utm_source=chatgpt.com "Message Bus | NautilusTrader Documentation"
[2]: https://nautilustrader.io/docs/latest/concepts/architecture/ "Architecture | NautilusTrader Documentation"
[3]: https://nautilustrader.io/docs/latest/api_reference/config/?utm_source=chatgpt.com "Config | NautilusTrader Documentation"
[4]: https://github.com/nautechsystems/nautilus_trader/discussions/1187?utm_source=chatgpt.com "Plotting entries, exits, indicators, etc - GitHub"
[5]: https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py?utm_source=chatgpt.com "finplot/finplot/examples/complicated.py at master - GitHub"
[6]: https://github.com/highfestiva/finplot/discussions/164?utm_source=chatgpt.com "Realtime / live plotting help · highfestiva finplot · Discussion #164"
[7]: https://github.com/nautechsystems/nautilus_trader/issues/1356?utm_source=chatgpt.com "Add indicator values to streamed feather files #1356 - GitHub"
