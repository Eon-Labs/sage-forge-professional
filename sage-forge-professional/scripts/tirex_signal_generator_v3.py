#!/usr/bin/env python3
"""
TiRex Signal Generator v3.0 - Native Outputs (Bias-Free Walk-Forward)

Purpose
- Perform non-look-ahead walk-forward (step size = 1) on real DSM historical OHLCV data
- Output ALL TiRex native outputs (quantiles and mean) for EACH bar after warm-up
- Persist results to CSV under results/ for auditing and downstream analytics

Key Properties
- Bias-free: strictly uses only past N bars to predict the next bar
- Step size: 1 bar (100% data coverage after warm-up)
- Context window: configurable (default 128, native for TiRex)
- Outputs per bar: timestamp, OHLCV, mean, q10..q90, std, direction, confidence, timings

Usage
  uv run python scripts/tirex_signal_generator_v3.py --context-size 128 --symbol BTCUSDT --timeframe 1h --no-viz

Output
  results/tirex_walkforward_v3_<symbol>_<timeframe>_<YYYYMMDDHHMMSS>.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
import os
import platform
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------- Configuration ----------
DEFAULT_CONTEXT = 128  # TiRex native sequence length
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BarOutput:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    median: float
    mean: float
    q10: float
    q20: float
    q30: float
    q40: float
    q50: float
    q60: float
    q70: float
    q80: float
    q90: float
    forecast_std: float
    direction: int  # -1, 0, 1
    confidence: float
    processing_time_ms: float
    context_start_index: int
    context_end_index: int
    bar_index: int
    # New metadata for better visual reference
    symbol: str
    timeframe: str
    prediction_available: bool
    context_ready: bool


def load_market_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load DSM real historical OHLCV data and return pandas DataFrame with columns:
    ['timestamp','open','high','low','close','volume'].
    """
    console.print("📊 Loading market data via DSM …")
    try:
        from sage_forge.data.manager import ArrowDataManager

        dm = ArrowDataManager()
        # Align with v2 script behaviour (fixed test window)
        from datetime import datetime
        start_time = datetime(2024, 10, 1, 0, 0, 0)
        end_time = datetime(2024, 10, 17, 0, 0, 0)

        console.print(f"🌐 Fetching {symbol} {timeframe} data: {start_time} → {end_time}")
        df = dm.fetch_real_market_data(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe,
        )
        if df is None or df.height == 0:
            console.print("❌ DSM returned no data")
            return None
        # Convert polars → pandas, ensure timestamp exists
        pdf = df.to_pandas()
        if "timestamp" not in pdf.columns:
            # The DSM helper indicates 'close_time' included; derive timestamps
            if "close_time" in pdf.columns:
                pdf["timestamp"] = pd.to_datetime(pdf["close_time"])  # derive
            else:
                raise ValueError("No timestamp or close_time column found.")
        pdf = pdf.sort_values("timestamp").reset_index(drop=True)
        # Keep canonical columns only
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in pdf.columns]
        pdf = pdf[keep]
        return pdf
    except Exception as e:
        console.print(f"❌ Failed to load data: {e}")
        return None


def compute_direction_and_confidence(current_price: float, central_forecast: float, forecast_std: float) -> tuple[int, float]:
    """
    Direction: sign of price change (central - current)
    Confidence: relative_change / (relative_change + relative_uncertainty)
    where relative_uncertainty = std / current_price
    """
    price_change = float(central_forecast) - float(current_price)
    # Direction with a small dead-zone could be considered; keep straight sign for transparency
    if price_change > 0:
        direction = 1
    elif price_change < 0:
        direction = -1
    else:
        direction = 0

    current_price = max(current_price, 1e-9)
    relative_change = abs(price_change) / current_price
    relative_unc = float(forecast_std) / current_price if current_price > 0 else 0.0
    conf = relative_change / (relative_change + relative_unc) if (relative_change + relative_unc) > 0 else 0.0
    return direction, float(np.clip(conf, 0.0, 1.0))


def _collect_cuda_arch_list() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "(CUDA not available)"
        num = torch.cuda.device_count()
        caps = set()
        for i in range(num):
            maj, minr = torch.cuda.get_device_capability(i)
            caps.add(f"{maj}.{minr}")
        # Sort high to low for readability
        ordered = sorted(caps, key=lambda s: tuple(map(int, s.split('.'))), reverse=True)
        return " ".join(ordered)
    except Exception as exc:
        return f"(error collecting arch list: {exc})"


def _print_debug_preflight(symbol: str, timeframe: str, context_size: int) -> None:
    console.rule("Debug: Environment & Device Diagnostics")
    py_ver = platform.python_version()
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        dev_count = torch.cuda.device_count() if cuda_available else 0
        device_lines = []
        if cuda_available:
            for i in range(dev_count):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory // (1024**2)
                device_lines.append(f"GPU[{i}]: {name} cc={cap[0]}.{cap[1]} mem={total_mem}MB")
        else:
            device_lines.append("No CUDA device detected")
    except Exception as e:
        torch_ver = "(torch not importable)"
        cuda_available = False
        dev_count = 0
        device_lines = [f"Torch import error: {e}"]

    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>")
    arch_reco = _collect_cuda_arch_list()

    table = Table(title="Preflight")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Python", py_ver)
    table.add_row("Torch", torch_ver)
    table.add_row("CUDA available", str(cuda_available))
    table.add_row("CUDA devices", str(dev_count))
    table.add_row("TORCH_CUDA_ARCH_LIST", arch_env)
    table.add_row("Recommended TORCH_CUDA_ARCH_LIST", arch_reco)
    table.add_row("Symbol", symbol)
    table.add_row("Timeframe", timeframe)
    table.add_row("Context size", str(context_size))
    console.print(table)

    if device_lines:
        dev_table = Table(title="CUDA Devices")
        dev_table.add_column("Info", style="magenta")
        for line in device_lines:
            dev_table.add_row(line)
        console.print(dev_table)


def _apply_torch_amp_shim() -> None:
    """
    Conform to Torch deprecation by mapping deprecated
    torch.cuda.amp.custom_fwd/custom_bwd to torch.amp equivalents with device_type='cuda'.
    This prevents FutureWarnings from third-party libs using the old decorators.
    """
    try:
        import torch  # type: ignore
        # Only patch if torch.amp exists
        if hasattr(torch, "amp"):
            # Fetch new decorators
            new_custom_fwd = getattr(torch.amp, "custom_fwd", None)
            new_custom_bwd = getattr(torch.amp, "custom_bwd", None)
            if new_custom_fwd is not None and new_custom_bwd is not None:
                # Build wrappers that fix device_type
                def _shim_fwd(*args, **kwargs):
                    return new_custom_fwd(device_type="cuda", *args, **kwargs)

                def _shim_bwd(*args, **kwargs):
                    return new_custom_bwd(device_type="cuda", *args, **kwargs)

                # Install onto torch.cuda.amp namespace
                if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                    torch.cuda.amp.custom_fwd = _shim_fwd  # type: ignore[attr-defined]
                    torch.cuda.amp.custom_bwd = _shim_bwd  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; non-fatal if patching fails
        pass


def _ensure_torch_cuda_arch_env() -> str:
    """Set TORCH_CUDA_ARCH_LIST if unset to current GPU cc (e.g., '8.9').
    Returns the effective value after ensuring.
    """
    val = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if val and val.strip():
        return val
    reco = _collect_cuda_arch_list()
    # Pick first capability (highest) if multiple
    first = reco.split()[0] if isinstance(reco, str) and reco.strip() else ""
    if first and first[0].isdigit():
        os.environ["TORCH_CUDA_ARCH_LIST"] = first
        return first
    return val or ""


def run_walk_forward(symbol: str, timeframe: str, context_size: int, debug: bool = False) -> Path:
    """
    Perform non-look-ahead bias walk-forward (step size = 1) and output TiRex native outputs
    (quantiles + mean) for each bar after warm-up.

    Returns: path to the CSV results file.
    """
    # 1) Load data
    market = load_market_data(symbol, timeframe)
    if market is None or len(market) == 0:
        raise RuntimeError("No market data available.")

    total_bars = len(market)
    if total_bars <= context_size:
        raise RuntimeError("Insufficient data for the requested context size.")

    console.print(Panel.fit(
        f"Walk-Forward v3\nSymbol={symbol}  TF={timeframe}\nBars={total_bars}  Context={context_size}  Step=1",
        title="Non-Look-Ahead Walk-Forward",
        border_style="green",
    ))

    if debug:
        # Dataset summary
        console.rule("Debug: Dataset Summary")
        t_min = pd.to_datetime(market["timestamp"].min())
        t_max = pd.to_datetime(market["timestamp"].max())
        table = Table(title="Market Window")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Rows", str(total_bars))
        table.add_row("Start", str(t_min))
        table.add_row("End", str(t_max))
        table.add_row("Context size", str(context_size))
        table.add_row("Warm-up rows", str(context_size))
        table.add_row("Prediction rows", str(total_bars - context_size))
        console.print(table)

        # Glimpse first/last rows
        head_str = market.head(2).to_string(index=False)
        tail_str = market.tail(2).to_string(index=False)
        console.print(Panel.fit(head_str, title="Head(2)"))
        console.print(Panel.fit(tail_str, title="Tail(2)"))

    # 2) Load TiRex native model
    _apply_torch_amp_shim()
    console.print("🦖 Loading TiRex native model …")
    from tirex import load_model  # native API
    model = load_model("NX-AI/TiRex")
    console.print("ℹ️ TiRex returns 'means' as an alias of q50 (median). Ignoring returned 'means' and using q50 as 'median' for decisions and outputs.")

    outputs: List[BarOutput] = []

    # 3) Walk-forward loop (step = 1) — now emit ALL bars (including warm-up)
    closes = market["close"].astype(float).to_numpy()
    volumes = market["volume"].astype(float).to_numpy() if "volume" in market.columns else np.zeros_like(closes)

    import torch

    for current_idx in range(0, total_bars):
        current_bar = market.iloc[current_idx]

        # Warm-up rows (no prediction)
        if current_idx < context_size:
            context_start = 0
            context_end = current_idx - 1
            outputs.append(
                BarOutput(
                    timestamp=pd.to_datetime(current_bar["timestamp"]),
                    open=float(current_bar.get("open", np.nan)),
                    high=float(current_bar.get("high", np.nan)),
                    low=float(current_bar.get("low", np.nan)),
                    close=float(current_bar.get("close", np.nan)),
                    volume=float(current_bar.get("volume", np.nan)),
                    median=float("nan"),
                    mean=float("nan"),
                    q10=float("nan"), q20=float("nan"), q30=float("nan"), q40=float("nan"),
                    q50=float("nan"), q60=float("nan"), q70=float("nan"), q80=float("nan"), q90=float("nan"),
                    forecast_std=float("nan"),
                    direction=0,
                    confidence=float("nan"),
                    processing_time_ms=0.0,
                    context_start_index=context_start,
                    context_end_index=context_end,
                    bar_index=current_idx,
                    symbol=symbol,
                    timeframe=timeframe,
                    prediction_available=False,
                    context_ready=False,
                )
            )
            continue

        # Prediction rows (context available)
        context_start = current_idx - context_size
        context_end = current_idx - 1
        context_prices = closes[context_start:current_idx]

        ctx_tensor = torch.tensor(context_prices, dtype=torch.float32)

        start = time.time()
        quantiles, _means = model.forecast(context=ctx_tensor, prediction_length=1)
        elapsed_ms = (time.time() - start) * 1000.0

        q = quantiles.squeeze().cpu().numpy()  # [9,]
        if np.ndim(q) == 0:
            q = np.array([float(q)])
        if q.size == 9:
            q10, q20, q30, q40, q50, q60, q70, q80, q90 = q.tolist()
        else:
            q_list = q.tolist()
            q_list += [float("nan")] * (9 - len(q_list))
            q10, q20, q30, q40, q50, q60, q70, q80, q90 = q_list[:9]

        forecast_std = float(np.std(q)) if len(q) > 0 else 0.0
        median_val = float(q50)
        direction, confidence = compute_direction_and_confidence(
            current_price=float(current_bar["close"]),
            central_forecast=median_val,
            forecast_std=forecast_std,
        )

        outputs.append(
            BarOutput(
                timestamp=pd.to_datetime(current_bar["timestamp"]),
                open=float(current_bar.get("open", np.nan)),
                high=float(current_bar.get("high", np.nan)),
                low=float(current_bar.get("low", np.nan)),
                close=float(current_bar.get("close", np.nan)),
                volume=float(current_bar.get("volume", np.nan)),
                median=median_val,
                mean=float("nan"),
                q10=float(q10), q20=float(q20), q30=float(q30), q40=float(q40),
                q50=float(q50), q60=float(q60), q70=float(q70), q80=float(q80), q90=float(q90),
                forecast_std=forecast_std,
                direction=direction,
                confidence=confidence,
                processing_time_ms=elapsed_ms,
                context_start_index=context_start,
                context_end_index=context_end,
                bar_index=current_idx,
                symbol=symbol,
                timeframe=timeframe,
                prediction_available=True,
                context_ready=True,
            )
        )

    # 4) Persist results
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    out_path = RESULTS_DIR / f"tirex_walkforward_v3_{symbol}_{timeframe}_{ts}.csv"
    df_out = pd.DataFrame([o.__dict__ for o in outputs])
    # Column ordering for readability
    preferred = [
        "timestamp","symbol","timeframe","bar_index",
        "context_ready","prediction_available","context_start_index","context_end_index",
        "open","high","low","close","volume",
        "median","mean","q10","q20","q30","q40","q50","q60","q70","q80","q90",
        "forecast_std","direction","confidence","processing_time_ms",
    ]
    cols = [c for c in preferred if c in df_out.columns] + [c for c in df_out.columns if c not in preferred]
    df_out = df_out[cols]
    df_out.to_csv(out_path, index=False)

    # Summary
    table = Table(title="TiRex v3 Native Outputs - Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Bars processed", str(total_bars))
    table.add_row("Warm-up (context)", str(context_size))
    table.add_row("Rows output", str(len(df_out)))
    table.add_row("Output file", str(out_path))
    console.print(table)

    if debug:
        console.rule("Debug: Output Glimpse")
        try:
            preview = df_out.head(3).to_string(index=False)
            console.print(Panel.fit(preview, title="CSV Head(3)"))
            tail_prev = df_out.tail(3).to_string(index=False)
            console.print(Panel.fit(tail_prev, title="CSV Tail(3)"))
        except Exception as e:
            console.print(f"(debug) failed to preview CSV dataframe: {e}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="TiRex Signal Generator v3.0 - Native Outputs (Bias-Free Walk-Forward)",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol, e.g., BTCUSDT")
    parser.add_argument("--timeframe", default="1h", help="Timeframe, e.g., 1h")
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT, help="Context window size (bars)")
    parser.add_argument("--no-viz", action="store_true", help="No visualization (default)")
    parser.add_argument("--debug", action="store_true", help="Print debug diagnostics")
    args = parser.parse_args()

    console.print(Panel.fit(
        "🦖 TiRex Signal Generator v3.0\n"
        "Native Outputs • Bias-Free Walk-Forward • Per-Bar Export",
        title="Production v3",
        border_style="blue",
    ))

    # Ensure targeted CUDA arch builds to avoid broad-arch warnings
    effective_arch = _ensure_torch_cuda_arch_env()
    if args.debug:
        console.print(f"ENV TORCH_CUDA_ARCH_LIST={effective_arch or '<unset>'}")

    if args.debug:
        _print_debug_preflight(symbol=args.symbol, timeframe=args.timeframe, context_size=args.context_size)

    out_file = run_walk_forward(
        symbol=args.symbol,
        timeframe=args.timeframe,
        context_size=args.context_size,
        debug=args.debug,
    )
    console.print(f"✅ Done. Results saved to: {out_file}")


if __name__ == "__main__":
    main()
