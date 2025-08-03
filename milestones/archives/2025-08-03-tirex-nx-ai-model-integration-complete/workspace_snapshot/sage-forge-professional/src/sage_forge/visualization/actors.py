"""
SAGE-Forge enhanced visualization actors for NautilusTrader integration.

Provides professional charting and visualization capabilities with real-time
data processing, SAGE-Forge configuration integration, and enhanced analytics.
"""

from typing import Any

import finplot as fplt
import pandas as pd
import pyqtgraph as pg
from nautilus_trader.common.actor import Actor
from rich.console import Console

from sage_forge.core.config import get_config
from sage_forge.funding.data import FundingPaymentEvent

console = Console()


class EnhancedFinPlotActor(Actor):
    """
    SAGE-Forge enhanced FinPlot actor for professional visualization.
    
    Features:
    - Real-time chart visualization during backtests and live trading
    - SAGE-Forge configuration system integration
    - Enhanced funding event visualization
    - Professional chart themes and styling
    - Optimized performance with buffered updates
    - NautilusTrader-native MessageBus integration
    
    Note: For production environments, consider using decoupled external
    processes. This embedded approach is optimized for development and
    experimental use cases.
    """

    def __init__(self, config=None):
        super().__init__(config)
        
        # SAGE-Forge configuration integration
        self.sage_config = get_config()
        vis_config = self.sage_config.get_visualization_config()
        
        self._ax = None
        self._ax2 = None
        self._ohlc_buffer: list[dict[str, Any]] = []
        self._volume_buffer: list[dict[str, Any]] = []
        self._funding_events: list[dict[str, Any]] = []
        self._signal_events: list[dict[str, Any]] = []  # SAGE-Forge strategy signals
        self._timer = None
        
        # SAGE-Forge configuration
        self._backtest_mode = vis_config.get("backtest_mode", True)
        self._refresh_rate = vis_config.get("refresh_rate_ms", 100)
        self._max_buffer_size = vis_config.get("max_buffer_size", 10000)
        self._show_funding_events = vis_config.get("show_funding_events", True)
        self._show_strategy_signals = vis_config.get("show_strategy_signals", True)
        
        # Performance tracking
        self._bars_processed = 0
        self._funding_events_processed = 0
        self._chart_updates = 0

        # Setup chart theme based on configuration
        if not self._backtest_mode:
            self._setup_sage_chart_theme()

        console.print("[green]✅ SAGE-Forge EnhancedFinPlotActor initialized[/green]")
        console.print(f"[blue]🔧 Config: backtest_mode={self._backtest_mode}, refresh_rate={self._refresh_rate}ms[/blue]")

    def _setup_sage_chart_theme(self) -> None:
        """Setup SAGE-Forge enhanced chart theme for professional visualization."""
        # SAGE-Forge premium dark theme
        theme_config = self.sage_config.get_visualization_config().get("theme", {})
        
        fplt.foreground = theme_config.get("foreground", "#f0f6fc")
        fplt.background = theme_config.get("background", "#0d1117")

        pg.setConfigOptions(
            foreground=fplt.foreground,
            background=fplt.background,
            antialias=True,
        )

        # SAGE-Forge enhanced color scheme
        fplt.odd_plot_background = fplt.background
        fplt.candle_bull_color = theme_config.get("candle_bull", "#26d0ce")
        fplt.candle_bear_color = theme_config.get("candle_bear", "#f85149")
        fplt.candle_bull_body_color = theme_config.get("candle_bull_body", "#238636")
        fplt.candle_bear_body_color = theme_config.get("candle_bear_body", "#da3633")
        fplt.volume_bull_color = theme_config.get("volume_bull", "#26d0ce40")
        fplt.volume_bear_color = theme_config.get("volume_bear", "#f8514940")
        fplt.cross_hair_color = theme_config.get("crosshair", "#58a6ff")
        
        console.print("[cyan]🎨 SAGE-Forge premium chart theme configured[/cyan]")

    def on_start(self) -> None:
        """Start SAGE-Forge enhanced visualization with configuration."""
        if not self._backtest_mode:
            # Create live chart windows for real-time trading
            title = f"SAGE-Forge Live Trading - {self.sage_config.version}"
            self._ax, self._ax2 = fplt.create_plot(title, rows=2, maximize=False)
            
            # Setup real-time refresh timer
            self._timer = pg.QtCore.QTimer()
            self._timer.timeout.connect(self._refresh_chart)
            self._timer.start(self._refresh_rate)
            
            console.print(f"[green]📈 Live charts created with {self._refresh_rate}ms refresh[/green]")
        else:
            console.print("[blue]📊 Backtest mode - charts will be shown post-backtest[/blue]")

        self.log.info(f"SAGE-Forge EnhancedFinPlotActor started (backtest_mode={self._backtest_mode})")
        console.print("[blue]🚀 SAGE-Forge visualization ready for market data[/blue]")

    def on_stop(self) -> None:
        """Stop with SAGE-Forge performance reporting."""
        if self._timer:
            self._timer.stop()
            
        self.log.info("SAGE-Forge EnhancedFinPlotActor stopped")
        
        # SAGE-Forge performance summary
        console.print("[yellow]⏹️ SAGE-Forge visualization stopped[/yellow]")
        console.print(f"[cyan]📊 Performance: {self._bars_processed} bars, {self._funding_events_processed} funding events[/cyan]")
        console.print(f"[cyan]📈 Chart updates: {self._chart_updates}[/cyan]")

    def on_reset(self) -> None:
        """Reset with SAGE-Forge state cleanup."""
        self._ohlc_buffer.clear()
        self._volume_buffer.clear()
        self._funding_events.clear()
        self._signal_events.clear()
        
        # Reset performance counters
        self._bars_processed = 0
        self._funding_events_processed = 0
        self._chart_updates = 0
        
        self.log.info("SAGE-Forge EnhancedFinPlotActor reset")
        console.print("[blue]🔄 SAGE-Forge visualization reset with analytics cleared[/blue]")

    def on_data(self, data) -> None:
        """Handle incoming data with SAGE-Forge enhanced processing."""
        try:
            # Handle Bar data (OHLCV) with SAGE-Forge optimization
            if hasattr(data, "open") and hasattr(data, "close"):  # Bar-like data
                self._process_bar_data(data)
                
            # Handle SAGE-Forge funding events
            if isinstance(data, FundingPaymentEvent) and self._show_funding_events:
                self._process_funding_event(data)
                
            # Handle legacy funding events for backward compatibility
            self._try_handle_legacy_funding(data)
            
        except Exception as e:
            self.log.error(f"SAGE-Forge visualization error processing data: {e}")
            console.print(f"[red]❌ Visualization error: {e}[/red]")

    def _process_bar_data(self, data) -> None:
        """Process bar data with SAGE-Forge optimizations."""
        # Convert nanosecond timestamp to datetime (native pattern)
        timestamp = data.ts_event / 1e9
        self._bars_processed += 1

        # SAGE-Forge buffer management
        bar_data = {
            "timestamp": timestamp,
            "open": float(data.open),
            "close": float(data.close),
            "high": float(data.high),
            "low": float(data.low),
        }
        
        self._ohlc_buffer.append(bar_data)
        
        # SAGE-Forge diagnostic logging for early data
        if self._bars_processed <= 3:
            bar_timestamp = pd.Timestamp(timestamp, unit="s")
            console.print(f"[bold green]🔍 SAGE-Forge bar #{self._bars_processed}: {bar_timestamp}[/bold green]")
            
        # Progress logging at key milestones
        if self._bars_processed in [100, 500, 1000, 1500, 2000]:
            bar_timestamp = pd.Timestamp(timestamp, unit="s")
            console.print(f"[bold blue]📊 SAGE-Forge progress: {self._bars_processed} bars @ {bar_timestamp}[/bold blue]")

        # Add volume data if available
        if hasattr(data, "volume"):
            volume_data = {
                "timestamp": timestamp,
                "open": float(data.open),
                "close": float(data.close),
                "volume": float(data.volume),
            }
            self._volume_buffer.append(volume_data)

        # SAGE-Forge buffer size management
        if len(self._ohlc_buffer) > self._max_buffer_size:
            self._ohlc_buffer = self._ohlc_buffer[-self._max_buffer_size:]
        if len(self._volume_buffer) > self._max_buffer_size:
            self._volume_buffer = self._volume_buffer[-self._max_buffer_size:]

    def _process_funding_event(self, event: FundingPaymentEvent) -> None:
        """Process SAGE-Forge funding events for visualization."""
        timestamp = event.ts_event / 1e9
        self._funding_events_processed += 1
        
        funding_data = {
            "timestamp": timestamp,
            "amount": float(event.payment_amount),
            "is_payment": event.is_payment,
            "funding_rate": event.funding_rate,
            "position_size": float(event.position_size),
        }
        
        self._funding_events.append(funding_data)
        
        direction = "payment" if event.is_payment else "receipt"
        console.print(
            f"[cyan]📊 SAGE-Forge funding {direction}: ${float(event.payment_amount):.4f} "
            f"@ {event.funding_rate*100:.4f}%[/cyan]"
        )

    def _try_handle_legacy_funding(self, data) -> None:
        """Handle legacy funding events for backward compatibility."""
        try:
            from nautilus_test.funding.data import FundingPaymentEvent as LegacyFundingPaymentEvent
            
            if isinstance(data, LegacyFundingPaymentEvent) and self._show_funding_events:
                timestamp = data.ts_event / 1e9
                self._funding_events_processed += 1
                
                funding_data = {
                    "timestamp": timestamp,
                    "amount": float(data.payment_amount),
                    "is_payment": data.is_payment,
                    "funding_rate": getattr(data, "funding_rate", 0.0),
                    "position_size": getattr(data, "position_size", 0.0),
                }
                
                self._funding_events.append(funding_data)
                
                direction = "payment" if data.is_payment else "receipt"
                console.print(f"[cyan]📊 Legacy funding {direction}: ${float(data.payment_amount):.4f}[/cyan]")
                
        except ImportError:
            pass  # Legacy funding system not available

    def _refresh_chart(self) -> None:
        """Refresh chart with SAGE-Forge optimized rendering."""
        # Skip if axes not created (backtest mode)
        if self._ax is None or self._ax2 is None:
            return

        chart_updated = False

        # Update OHLC chart with SAGE-Forge optimizations
        if self._ohlc_buffer:
            df_ohlc = pd.DataFrame(self._ohlc_buffer)
            df_ohlc.set_index("timestamp", inplace=True)

            # Clear and replot (efficient for real-time updates)
            if self._ax:
                self._ax.clear()
            fplt.candlestick_ochl(
                df_ohlc[["open", "close", "high", "low"]],
                ax=self._ax,
            )
            
            chart_updated = True
            self._ohlc_buffer.clear()

        # Update volume chart
        if self._volume_buffer:
            df_vol = pd.DataFrame(self._volume_buffer)
            df_vol.set_index("timestamp", inplace=True)

            if self._ax2:
                self._ax2.clear()
            fplt.volume_ocv(
                df_vol[["open", "close", "volume"]],
                ax=self._ax2,
            )
            
            chart_updated = True
            self._volume_buffer.clear()

        # Add SAGE-Forge funding event markers
        if self._funding_events and self._show_funding_events:
            for event in self._funding_events:
                # Color coding for funding events
                color = "#f85149" if event["is_payment"] else "#26d0ce"
                marker_size = min(abs(event["amount"]) * 2, 10)  # Size based on amount
                
                # Add funding marker to volume chart
                fplt.plot(
                    [event["timestamp"]], [0],
                    ax=self._ax2,
                    style="o",
                    color=color,
                    width=marker_size,
                    legend=f"Funding: ${event['amount']:.4f} @ {event['funding_rate']*100:.3f}%",
                )
                
            chart_updated = True
            self._funding_events.clear()

        # Track chart updates for performance monitoring
        if chart_updated:
            self._chart_updates += 1

    # SAGE-Forge analytics and reporting methods
    
    def get_visualization_stats(self) -> dict:
        """Get comprehensive SAGE-Forge visualization statistics."""
        return {
            "sage_forge_version": self.sage_config.version,
            "bars_processed": self._bars_processed,
            "funding_events_processed": self._funding_events_processed,
            "chart_updates": self._chart_updates,
            "buffer_sizes": {
                "ohlc_buffer": len(self._ohlc_buffer),
                "volume_buffer": len(self._volume_buffer),
                "funding_events": len(self._funding_events),
            },
            "configuration": {
                "backtest_mode": self._backtest_mode,
                "refresh_rate_ms": self._refresh_rate,
                "max_buffer_size": self._max_buffer_size,
                "show_funding_events": self._show_funding_events,
                "show_strategy_signals": self._show_strategy_signals,
            },
            "performance": {
                "avg_bars_per_update": self._bars_processed / max(self._chart_updates, 1),
                "funding_event_rate": self._funding_events_processed / max(self._bars_processed, 1),
            }
        }

    def export_chart_data(self) -> dict:
        """Export current chart data for external analysis."""
        return {
            "ohlc_data": self._ohlc_buffer.copy(),
            "volume_data": self._volume_buffer.copy(),
            "funding_events": self._funding_events.copy(),
            "signal_events": self._signal_events.copy(),
            "stats": self.get_visualization_stats(),
        }

    def add_strategy_signal(self, signal_type: str, price: float, timestamp: int, metadata: dict = None) -> None:
        """Add strategy signal markers to charts."""
        if not self._show_strategy_signals:
            return
            
        signal_data = {
            "timestamp": timestamp / 1e9,  # Convert nanoseconds to seconds
            "signal_type": signal_type,  # "BUY", "SELL", "ENTRY", "EXIT"
            "price": price,
            "metadata": metadata or {},
        }
        
        self._signal_events.append(signal_data)
        console.print(f"[green]📊 Strategy signal: {signal_type} @ ${price:.4f}[/green]")