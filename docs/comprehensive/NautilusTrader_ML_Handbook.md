# NautilusTrader + Machine-Learning

**A reproducible handbook for end-to-end ML-driven trading, ready for LLM-agent automation**

---

## 0 · TL;DR quickstart

```bash
# ① Clone + install the framework (already cloned into ./nt_reference)
cd nt_reference
uv pip install -e .        # fast, locked dependency graph introduced in v1.220.0

# ② Self-test the build
uv run pytest
```

---

## 1 · Mental model

| Pillar                  | What it is                                                                      | Why ML cares                                                           |
| ----------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Actor**               | Lightweight concurrent component running on a single asyncio/uvloop event loop  | Drop-in async inference tasks without extra threads                    |
| **Strategy**            | `class MyStrategy(Strategy)` – inherits all Actor capability + order management | Hook ML predictions in `on_bar`, `on_tick`, etc. ([NautilusTrader][1]) |
| **Trader / Controller** | Orchestrates actors; registers, (de)starts, disposes components                 | Clean lifecycle to preload models only once ([NautilusTrader][2])      |
| **Cache + MessageBus**  | Zero-copy data fan-out                                                          | Share features between ML and risk actors                              |
| **BacktestEngine**      | Deterministic replay on historical stream                                       | Offline training & walk-forward analysis                               |

The design is “AI-first” – research, backtest, and live share the _same_ Python code path. ([GitHub][3])

---

## 2 · Cookbook patterns for ML integration

### 2.1 Offline training loop

```python
from nautilus_trader.backtest.engine import BacktestEngine
from mllib.dataset import build_feature_matrix   # your helper

engine = BacktestEngine()
engine.configure("cfg/backtest_eth.yml")
engine.run()                          # → caching of Bars/Ticks

df = engine.cache.bars().to_pandas()  # zero-copy view
X, y = build_feature_matrix(df)
model = trainer.fit(X, y)             # scikit-learn / PyTorch / XGBoost – your choice
model.save("models/eth_bars_lstm.pt")
```

_Use engine cache to avoid redundant I/O; keep feature generation deterministic._

### 2.2 Real-time inference (in-process)

```python
import torch
from nautilus_trader.trading import Strategy

class MLSignal(Strategy):
    def on_start(self):
        self.model = torch.jit.load("models/eth_bars_lstm.pt").eval()
        self.window = []

    def on_bar(self, bar):
        self.window.append([bar.open, bar.high, bar.low, bar.close, bar.volume])
        if len(self.window) < 60:
            return

        x = torch.tensor([self.window[-60:]], dtype=torch.float32)
        long_prob = self.model(x).item()

        if long_prob > 0.60 and not self.position_long():
            self.buy(bar.instrument_id, qty=1)
        elif long_prob < 0.40 and self.position_long():
            self.sell(bar.instrument_id, qty=1)
```

_Single GPU/CPU batch inference keeps latency sub-millisecond on uvloop._

### 2.3 Real-time inference (micro-service)

| Component                   | Why                                           | How                                |
| --------------------------- | --------------------------------------------- | ---------------------------------- |
| **FastAPI + TorchServe**    | hot-reload models, multi-model serving        | `/predict` endpoint returning prob |
| **Actor → AsyncHTTPClient** | non-blocking call inside `on_bar`             | `await self.client.post(...)`      |
| **CircuitBreaker Actor**    | fall-back to heuristic if service unavailable | `on_fault()` hook in Strategy      |

### 2.4 Feature pipelining patterns

1. **Rolling N-bars tensor** – feed LSTM/CNN.
2. **Cross-asset cache join** – pair trading (see Databento pair-trading example). ([Medium][4])
3. **Synthetic instruments** – create spread instrument via `add_synthetic`, then treat as vanilla instrument. ([NautilusTrader][2])
4. **Latency budget** – keep feature engineering pure-Python, push heavy transforms to torch/numba compiled functions.

---

## 3 · Backtest → walk-forward evaluation

```python
from mlutils.metrics import sharpe, max_dd

bt = BacktestEngine()
bt.configure("cfg/ml_walk.yml")
bt.run()

perf = bt.results().to_pandas()
print("Sharpe:", sharpe(perf.pnl))
print("Max DD :", max_dd(perf.equity_curve))
```

Backtest re-uses _identical_ Strategy class – zero code drift between training and live. ([NautilusTrader][1])

---

## 4 · Deployment recipes

| Scenario                | Cmd / infra                                                | Notes                                |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------ |
| **Docker one-shot**     | `docker run ghcr.io/nautechsystems/nautilus_trader:latest` | GPU pass-through: `--gpus all`       |
| **Kubernetes rolling**  | Helm chart publishes Strategy image + ConfigMap            | version pin your models              |
| **On-prem low-latency** | `uv run python live.py --uvloop`                           | Place strategy and model on RAM-disk |

---

## 5 · API cheat-sheet (most ML-relevant types)

| Symbol                                     | Description                               |
| ------------------------------------------ | ----------------------------------------- |
| `Strategy.on_bar/on_tick/on_data/on_event` | call-backs where you inject ML            |
| `BacktestEngine`, `TradingNode`            | offline vs live harness                   |
| `CacheFacade.bars()/ticks()`               | zero-copy feature store                   |
| `Order`, `Position`, `PortfolioFacade`     | consumption of ML decisions               |
| `Clock`                                    | deterministic timestamps for model inputs |

Full signature list lives in the _trading_ API section. ([NautilusTrader][2])

---

## 6 · Best-practice checklist

| ✓   | Recommendation                                                          | Source/why                          |
| --- | ----------------------------------------------------------------------- | ----------------------------------- |
| ☐   | **Load ML model in `on_start` only** – avoid per-bar disk I/O           | strategy docs ([NautilusTrader][1]) |
| ☐   | **Use TorchScript or ONNX** – removes Python GIL bottleneck             |                                     |
| ☐   | **Pin uvloop** (`uvloop.install()`) for tighter latency                 | overview ([NautilusTrader][5])      |
| ☐   | **Backfill cache before live-switch** – identical feature windows       |                                     |
| ☐   | **Version everything**: model hash ↔ strategy hash ↔ config             |                                     |
| ☐   | **Health-check Actor** pings inference service; fail-safe to `on_fault` |                                     |

---

## 7 · Resources & further reading

- Official docs index / Python API ([NautilusTrader][6])
- Strategy concept guide ([NautilusTrader][1])
- Latest 1.219.0 beta release notes (Python 3.13 support) ([GitHub][7])
- Medium “plug-in ML with PyTorch/TensorFlow” overview ([Medium][8])
- Pair-trading walkthrough (feature engineering template) ([Medium][4])

Happy building — your next step is to drop a trained model in `models/`, subclass `Strategy`, and let NautilusTrader do the heavy lifting.

[1]: https://nautilustrader.io/docs/latest/concepts/strategies/ "Strategies | NautilusTrader Documentation"
[2]: https://nautilustrader.io/docs/latest/api_reference/trading/ "Trading | NautilusTrader Documentation"
[3]: https://github.com/nautechsystems/nautilus_trader "nautechsystems/nautilus_trader: A high-performance ... - GitHub"
[4]: https://medium.com/@truongb.duy/how-to-fetch-databento-data-in-nautilus-trader-a8287022e2fa "Pair Trading with Databento and NautilusTrader - Medium"
[5]: https://nautilustrader.io/docs/latest/concepts/overview/ "Overview | NautilusTrader Documentation"
[6]: https://nautilustrader.io/docs/latest/api_reference/ "Python API | NautilusTrader Documentation"
[7]: https://github.com/nautechsystems/nautilus_trader/releases "Releases · nautechsystems/nautilus_trader - GitHub"
[8]: https://medium.com/@gwrx2005/top-10-ai-powered-crypto-trading-repositories-on-github-0041862546b6 "Top 10 AI-Powered Crypto Trading Repositories on GitHub - Medium"
