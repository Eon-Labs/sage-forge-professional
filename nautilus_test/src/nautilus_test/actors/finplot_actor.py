"""
Finplot actor following NautilusTrader patterns.

Provides real-time chart visualization during backtests.
"""

from typing import Any

import finplot as fplt
import pandas as pd
import pyqtgraph as pg
from nautilus_trader.common.actor import Actor
from rich.console import Console

console = Console()


class FinplotActor(Actor):
    """
    Native NautilusTrader Actor for experimental finplot chart integration.

    ⚠️ EXPERIMENTAL USE ONLY - Updated finplot integration guidelines recommend
    decoupled external processes for production. This embedded approach is kept
    for experimental/development purposes only.

    For production: Use publish_signal() to Redis + external live_plotter.py
    For development: This embedded FinplotActor (may block event loop)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._ax = None
        self._ax2 = None
        self._ohlc_buffer: list[dict[str, Any]] = []
        self._volume_buffer: list[dict[str, Any]] = []
        self._funding_events: list[dict[str, Any]] = []
        self._timer = None
        self._backtest_mode = True  # Default to backtest mode

        # Skip chart styling in backtest mode to prevent window creation
        # Theme will be set up by post-backtest visualization
        if not self._backtest_mode:
            self._setup_chart_theme()

        console.print("[green]✅ Native FinplotActor initialized[/green]")

    def _setup_chart_theme(self) -> None:
        """Setup enhanced dark theme for real data visualization."""
        fplt.foreground = "#f0f6fc"
        fplt.background = "#0d1117"

        pg.setConfigOptions(
            foreground=fplt.foreground,
            background=fplt.background,
            antialias=True,
        )

        fplt.odd_plot_background = fplt.background
        fplt.candle_bull_color = "#26d0ce"
        fplt.candle_bear_color = "#f85149"
        fplt.candle_bull_body_color = "#238636"
        fplt.candle_bear_body_color = "#da3633"
        fplt.volume_bull_color = "#26d0ce40"
        fplt.volume_bear_color = "#f8514940"
        fplt.cross_hair_color = "#58a6ff"

    def on_start(self) -> None:
        """
        Called when the actor starts.

        ⚠️ EXPERIMENTAL: In backtest mode, this creates charts but doesn't show them
        to avoid conflicts with post-backtest visualization.
        For live trading, this would display real-time charts.
        """
        # In backtest mode, skip creating the plot window to avoid duplicate windows
        # For live trading, uncomment the following lines:
        # self._ax, self._ax2 = fplt.create_plot('Live NautilusTrader Data', rows=2, maximize=False)
        # self._timer = pg.QtCore.QTimer()
        # self._timer.timeout.connect(self._refresh_chart)
        # self._timer.start(100)  # 100ms refresh rate for smooth updates

        # NTPA: Use NT-native logging
        self.log.info("FinplotActor started (backtest mode - chart creation skipped)")
        console.print(
            "[blue]🚀 FinplotActor started - backtest mode "
            "(post-backtest chart will be shown)[/blue]"
        )

    def on_stop(self) -> None:
        """Called when the actor stops."""
        if self._timer:
            self._timer.stop()
        self.log.info("FinplotActor stopped")
        console.print("[yellow]⏹️ FinplotActor stopped[/yellow]")

    def on_reset(self) -> None:
        """Called when the actor resets."""
        self._ohlc_buffer.clear()
        self._volume_buffer.clear()
        self._funding_events.clear()
        self.log.info("FinplotActor reset")
        console.print("[blue]🔄 FinplotActor reset[/blue]")

    def on_data(self, data) -> None:
        """
        Handle incoming data using native patterns.

        This method receives all data types through MessageBus.
        Following NautilusTrader_FINPLOT_INTEGRATION.md guidelines.
        """
        # Handle Bar data (OHLCV)
        if hasattr(data, "open") and hasattr(data, "close"):  # Bar-like data
            # Convert nanosecond timestamp to datetime (native pattern)
            timestamp = data.ts_event / 1e9

            self._ohlc_buffer.append({
                "timestamp": timestamp,
                "open": float(data.open),
                "close": float(data.close),
                "high": float(data.high),
                "low": float(data.low),
            })
            
            # Diagnostic logging for first few bars
            if len(self._ohlc_buffer) <= 3:
                finplot_timestamp = pd.Timestamp(timestamp, unit="s")
                console.print(f"[bold red]🔍 DIAGNOSTIC 3: Finplot buffer #{len(self._ohlc_buffer)} timestamp: {finplot_timestamp}[/bold red]")
                
            # Diagnostic logging for progress
            if len(self._ohlc_buffer) in [100, 500, 1000, 1500, 2000]:
                bar_timestamp = pd.Timestamp(timestamp, unit="s")
                console.print(f"[bold blue]🔍 DIAGNOSTIC 4: Bar #{len(self._ohlc_buffer)} processed at {bar_timestamp}[/bold blue]")

            if hasattr(data, "volume"):
                self._volume_buffer.append({
                    "timestamp": timestamp,
                    "open": float(data.open),
                    "close": float(data.close),
                    "volume": float(data.volume),
                })

        # Handle Funding events (if available)
        try:
            from nautilus_test.funding.data import FundingPaymentEvent
            if isinstance(data, FundingPaymentEvent):
                timestamp = data.ts_event / 1e9
                self._funding_events.append({
                    "timestamp": timestamp,
                    "amount": float(data.payment_amount),
                    "is_payment": data.is_payment,
                })

                console.print(
                    f"[cyan]📊 Chart: Funding {'payment' if data.is_payment else 'receipt'} "
                    f"${float(data.payment_amount):.2f}[/cyan]",
                )
        except ImportError:
            pass  # Funding system not available

    def _refresh_chart(self) -> None:
        """
        Refresh chart with buffered data.

        Called by Qt timer every 100ms to update charts smoothly.
        Following finplot maintainer's recommended timer-based pattern.
        """
        # Skip if axes not created (backtest mode)
        if self._ax is None or self._ax2 is None:
            return

        # Update OHLC chart
        if self._ohlc_buffer:
            df_ohlc = pd.DataFrame(self._ohlc_buffer)

            # Clear and replot (efficient for real-time updates)
            if self._ax:
                self._ax.clear()
            fplt.candlestick_ochl(
                df_ohlc[["open", "close", "high", "low"]],
                ax=self._ax,
            )

            # Clear buffer after plotting
            self._ohlc_buffer.clear()

        # Update volume chart
        if self._volume_buffer:
            df_vol = pd.DataFrame(self._volume_buffer)

            if self._ax2:
                self._ax2.clear()
            fplt.volume_ocv(
                df_vol[["open", "close", "volume"]],
                ax=self._ax2,
            )

            # Clear buffer after plotting
            self._volume_buffer.clear()

        # Add funding event markers if any
        if self._funding_events:
            for event in self._funding_events:
                color = "#f85149" if event["is_payment"] else "#26d0ce"
                # Add funding marker to chart
                fplt.plot(
                    [event["timestamp"]], [0],
                    ax=self._ax2,
                    style="o",
                    color=color,
                    width=6,
                    legend=f"Funding: ${event['amount']:.2f}",
                )

            self._funding_events.clear()
