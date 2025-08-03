#!/usr/bin/env python3
"""
Native FinplotActor - MessageBus-Integrated Visualization
=========================================================

Native NautilusTrader Actor for experimental finplot chart integration.
100% identical to the original script's FinplotActor.
"""

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
        self._ohlc_buffer = []
        self._volume_buffer = []
        self._funding_events = []
        self._timer = None
        self._backtest_mode = True  # Default to backtest mode
        
        # Skip chart styling in backtest mode to prevent window creation
        # Theme will be set up by post-backtest visualization
        if not self._backtest_mode:
            self._setup_chart_theme()
        
        console.print("[green]✅ Native FinplotActor initialized[/green]")

    def _setup_chart_theme(self):
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
            
            if hasattr(data, "volume"):
                self._volume_buffer.append({
                    "timestamp": timestamp,
                    "open": float(data.open),
                    "close": float(data.close),
                    "volume": float(data.volume),
                })
            
            # Log every 100th bar to avoid spam
            if len(self._ohlc_buffer) % 100 == 0:
                self.log.info(f"FinplotActor: Buffered {len(self._ohlc_buffer)} bars")
        
        # Handle funding events (if available)
        elif hasattr(data, "funding_rate"):
            self._funding_events.append({
                "timestamp": data.ts_event / 1e9,
                "funding_rate": float(data.funding_rate),
                "next_funding_time": getattr(data, "next_funding_time", None),
            })
            
            self.log.info(f"FinplotActor: Recorded funding event {data.funding_rate}")

    def _refresh_chart(self):
        """
        Refresh the chart with buffered data (for live mode only).
        
        ⚠️ EXPERIMENTAL: This would be called by timer in live mode.
        In backtest mode, this is not used to avoid performance issues.
        """
        if not self._ax or not self._ohlc_buffer:
            return
        
        try:
            # Convert buffer to DataFrame
            df_ohlc = pd.DataFrame(self._ohlc_buffer)
            df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], unit="s")
            df_ohlc.set_index("timestamp", inplace=True)
            
            # Clear existing plots
            self._ax.clear()
            self._ax2.clear()
            
            # Plot OHLC
            fplt.candlestick_ochl(df_ohlc[["open", "close", "high", "low"]], ax=self._ax)
            
            # Plot volume if available
            if self._volume_buffer:
                df_vol = pd.DataFrame(self._volume_buffer)
                df_vol["timestamp"] = pd.to_datetime(df_vol["timestamp"], unit="s")
                df_vol.set_index("timestamp", inplace=True)
                fplt.volume_ocv(df_vol[["open", "close", "volume"]], ax=self._ax2)
            
            # Add funding rate markers if available
            if self._funding_events:
                for event in self._funding_events[-10:]:  # Show last 10 events
                    timestamp = pd.to_datetime(event["timestamp"], unit="s")
                    if timestamp in df_ohlc.index:
                        price = df_ohlc.loc[timestamp, "close"]
                        color = "#00ff00" if event["funding_rate"] > 0 else "#ff0000"
                        fplt.plot([timestamp], [price], ax=self._ax, color=color, style="o", width=8)
            
        except Exception as e:
            self.log.error(f"FinplotActor chart refresh error: {e}")

    def get_visualization_stats(self) -> dict:
        """Get statistics about buffered visualization data."""
        return {
            "bars_processed": len(self._ohlc_buffer),
            "volume_points": len(self._volume_buffer),
            "funding_events": len(self._funding_events),
            "backtest_mode": self._backtest_mode,
        }

    def get_ohlc_data(self) -> list:
        """Get buffered OHLC data for external visualization."""
        return self._ohlc_buffer.copy()

    def get_volume_data(self) -> list:
        """Get buffered volume data for external visualization."""
        return self._volume_buffer.copy()

    def get_funding_events(self) -> list:
        """Get buffered funding events for external visualization."""
        return self._funding_events.copy()

    def set_backtest_mode(self, enabled: bool):
        """Set backtest mode (prevents live chart creation)."""
        self._backtest_mode = enabled
        if enabled:
            console.print("[blue]📊 FinplotActor: Backtest mode enabled (no live charts)[/blue]")
        else:
            console.print("[green]📊 FinplotActor: Live mode enabled (real-time charts)[/green]")
            self._setup_chart_theme()

    def create_post_backtest_chart(self, title: str = "Post-Backtest Analysis"):
        """Create a post-backtest chart using buffered data."""
        console.print(f"[cyan]📊 Creating post-backtest chart: {title}...[/cyan]")
        
        if not self._ohlc_buffer:
            console.print("[yellow]⚠️ No OHLC data available for chart[/yellow]")
            return None, None
        
        try:
            # Setup theme
            self._setup_chart_theme()
            
            # Convert data to DataFrame
            df_ohlc = pd.DataFrame(self._ohlc_buffer)
            df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], unit="s")
            df_ohlc.set_index("timestamp", inplace=True)
            
            # Create plot
            ax, ax2 = fplt.create_plot(title, rows=2, maximize=True)
            
            # Plot OHLC
            fplt.candlestick_ochl(df_ohlc[["open", "close", "high", "low"]], ax=ax)
            
            # Plot volume
            if self._volume_buffer:
                df_vol = pd.DataFrame(self._volume_buffer)
                df_vol["timestamp"] = pd.to_datetime(df_vol["timestamp"], unit="s")
                df_vol.set_index("timestamp", inplace=True)
                fplt.volume_ocv(df_vol[["open", "close", "volume"]], ax=ax2)
            
            # Add funding events as markers
            if self._funding_events:
                for event in self._funding_events:
                    timestamp = pd.to_datetime(event["timestamp"], unit="s")
                    if timestamp in df_ohlc.index:
                        price = df_ohlc.loc[timestamp, "close"]
                        color = "#00ff00" if event["funding_rate"] > 0 else "#ff0000"
                        fplt.plot([timestamp], [price], ax=ax, color=color, style="o", width=8)
            
            console.print(f"[green]✅ Post-backtest chart created with {len(df_ohlc)} bars[/green]")
            return ax, ax2
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create post-backtest chart: {e}[/red]")
            return None, None