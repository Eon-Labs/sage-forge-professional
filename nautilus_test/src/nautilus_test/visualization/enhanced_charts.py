"""Enhanced visualization functions for NautilusTrader backtesting."""

import finplot as fplt
import pandas as pd
import pyqtgraph as pg
from nautilus_trader.model.data import Bar
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def prepare_bars_dataframe(bars: list[Bar]) -> pd.DataFrame:
    """Convert NautilusTrader Bar objects to DataFrame for visualization."""
    data = []
    for bar in bars:
        timestamp = pd.Timestamp(bar.ts_event, unit="ns")
        data.append({
            "time": timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        })

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df


def create_enhanced_candlestick_chart(df: pd.DataFrame, title: str = "Enhanced OHLC Chart with Real Specs"):
    """Create candlestick chart with enhanced dark theme for real data."""
    import pyqtgraph as pg

    # Enhanced dark theme for real data visualization
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

    # Create figure with enhanced styling
    ax, ax2 = fplt.create_plot(title, rows=2)

    # Plot with real data
    fplt.candlestick_ochl(df[["open", "close", "high", "low"]], ax=ax)
    fplt.volume_ocv(df[["open", "close", "volume"]], ax=ax2)

    return ax, ax2


def add_enhanced_indicators(df: pd.DataFrame, ax, fast_period: int = 10, slow_period: int = 21):
    """Add enhanced indicators with real specification validation."""
    # Calculate indicators
    df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

    # Plot with enhanced colors
    fplt.plot(df["ema_fast"], ax=ax, color="#58a6ff", width=2, legend=f"EMA {fast_period}")
    fplt.plot(df["ema_slow"], ax=ax, color="#ff7b72", width=2, legend=f"EMA {slow_period}")
    fplt.plot(df["bb_upper"], ax=ax, color="#7c3aed", width=1, style="--", legend="BB Upper")
    fplt.plot(df["bb_lower"], ax=ax, color="#7c3aed", width=1, style="--", legend="BB Lower")

    return df


def add_realistic_trade_markers(df: pd.DataFrame, fills_report: pd.DataFrame, ax):
    """Add trade markers positioned with realistic position sizes."""
    if fills_report.empty:
        return

    buy_times, buy_prices = [], []
    sell_times, sell_prices = [], []

    for _, fill in fills_report.iterrows():
        timestamp_val = fill["ts_init"]
        if isinstance(timestamp_val, pd.Series):
            timestamp_val = timestamp_val.iloc[0] if not timestamp_val.empty else None
        if timestamp_val is not None:
            try:
                # Safely convert to timestamp
                if hasattr(timestamp_val, "timestamp") and hasattr(timestamp_val, "floor"):
                    timestamp = timestamp_val
                else:
                    timestamp = pd.Timestamp(timestamp_val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                continue  # Skip invalid timestamps

            try:
                # Ensure we have a proper Timestamp object
                if not isinstance(timestamp, pd.Timestamp):
                    timestamp = pd.Timestamp(timestamp)  # type: ignore
                trade_time = timestamp.floor("min")

                if trade_time in df.index:
                    bar_row = df.loc[trade_time]
                else:
                    nearest_idx = df.index.get_indexer([trade_time], method="nearest")[0]
                    bar_row = df.iloc[nearest_idx]

                bar_high = float(bar_row["high"])
                bar_low = float(bar_row["low"])

                if fill["side"] == "BUY":
                    buy_times.append(timestamp)
                    buy_prices.append(bar_low - (bar_high - bar_low) * 0.05)
                else:
                    sell_times.append(timestamp)
                    sell_prices.append(bar_high + (bar_high - bar_low) * 0.05)

            except (IndexError, KeyError, TypeError):
                price = float(fill["avg_px"])
                price_offset = price * 0.001

                if fill["side"] == "BUY":
                    buy_times.append(timestamp)
                    buy_prices.append(price - price_offset)
                else:
                    sell_times.append(timestamp)
                    sell_prices.append(price + price_offset)

    # Enhanced trade markers
    if buy_times:
        buy_df = pd.DataFrame({"price": buy_prices}, index=pd.Index(buy_times))
        fplt.plot(buy_df, ax=ax, style="^", color="#26d0ce", width=4, legend="Buy (Realistic Size)")

    if sell_times:
        sell_df = pd.DataFrame({"price": sell_prices}, index=pd.Index(sell_times))
        fplt.plot(sell_df, ax=ax, style="v", color="#f85149", width=4, legend="Sell (Realistic Size)")


def display_enhanced_chart(
    bars: list[Bar],
    fills_report: pd.DataFrame,
    instrument_id: str,
    specs: dict,
    position_calc: dict,
    fast_ema: int = 10,
    slow_ema: int = 21,
):
    """Display ultimate chart with real specs + realistic positions + rich visualization."""
    # Convert bars to DataFrame
    df = prepare_bars_dataframe(bars)

    # Create enhanced chart
    chart_title = f"{instrument_id} - Real Binance Specs + Realistic Positions + Rich Visualization"
    ax, _ = create_enhanced_candlestick_chart(df, chart_title)  # ax2 used internally for volume

    # Add indicators
    add_enhanced_indicators(df, ax, fast_ema, slow_ema)

    # Add realistic trade markers
    add_realistic_trade_markers(df, fills_report, ax)

    # Add specification info to chart
    info_text = (
        f"Real Specs: {specs['tick_size']} tick, {specs['step_size']} step | "
        f"Realistic Position: {position_calc['position_size_btc']:.3f} BTC (${position_calc['notional_value']:.0f})"
    )
    console.print(f"[cyan]📊 Chart Info: {info_text}[/cyan]")

    # Show enhanced visualization
    fplt.show()

    return df


def create_post_backtest_chart(bars, fills_report, specs, position_calc):
    """Create post-backtest chart using existing enhanced visualization."""
    return display_enhanced_chart(
        bars, fills_report, "BTC/USDT Enhanced System",
        specs, position_calc, fast_ema=10, slow_ema=21,
    )


def display_ultimate_performance_summary(
    account_report, fills_report, starting_balance, specs, position_calc, funding_summary=None, adjusted_final_balance=None,
):
    """Display ultimate performance summary combining all enhancements."""
    table = Table(
        title="🏆 Ultimate Performance Summary (Real Specs + Realistic Positions + Rich Visualization)",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Category", style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # Specifications section
    table.add_row("📊 Real Specifications", "Price Precision", str(specs["price_precision"]))
    table.add_row("", "Size Precision", str(specs["quantity_precision"]))
    table.add_row("", "Tick Size", specs["tick_size"])
    table.add_row("", "Step Size", specs["step_size"])
    table.add_row("", "Min Notional", f"${specs['min_notional']}")
    table.add_row("", "", "")  # Separator

    # Position sizing section
    table.add_row("💰 Realistic Positions", "Position Size", f"{position_calc['position_size_btc']:.3f} BTC")
    table.add_row("", "Trade Value", f"${position_calc['notional_value']:.2f}")
    table.add_row("", "Account Risk", f"{position_calc['risk_percentage']:.1f}%")
    table.add_row("", "vs Dangerous 1 BTC", f"{119000/position_calc['notional_value']:.0f}x safer")
    table.add_row("", "", "")  # Separator

    # Performance section
    if not account_report.empty:
        try:
            original_final_balance = float(account_report.iloc[-1]["total"])
            original_pnl = original_final_balance - starting_balance
            original_pnl_pct = (original_pnl / starting_balance) * 100
            original_pnl_color = "green" if original_pnl >= 0 else "red"

            table.add_row("📈 Trading Performance", "Starting Balance", f"${starting_balance:,.2f}")
            table.add_row("", "Original Final Balance", f"[{original_pnl_color}]${original_final_balance:,.2f}[/{original_pnl_color}]")
            table.add_row("", "Original P&L", f"[{original_pnl_color}]{original_pnl:+,.2f} ({original_pnl_pct:+.2f}%)[/{original_pnl_color}]")

            # Add funding-adjusted P&L if available
            if adjusted_final_balance is not None:
                adjusted_pnl = adjusted_final_balance - starting_balance
                adjusted_pnl_pct = (adjusted_pnl / starting_balance) * 100
                adjusted_pnl_color = "green" if adjusted_pnl >= 0 else "red"
                funding_cost = original_final_balance - adjusted_final_balance

                table.add_row("", "Funding Costs", f"[red]${funding_cost:+.2f}[/red]")
                table.add_row("", "Adjusted Final Balance", f"[{adjusted_pnl_color}]${adjusted_final_balance:,.2f}[/{adjusted_pnl_color}]")
                table.add_row("", "Funding-Adjusted P&L", f"[{adjusted_pnl_color}]{adjusted_pnl:+,.2f} ({adjusted_pnl_pct:+.2f}%)[/{adjusted_pnl_color}]")

            table.add_row("", "Total Trades", str(len(fills_report)))

        except Exception as e:
            table.add_row("📈 Trading Performance", "Error", str(e))

    # Funding costs section (if available)
    if funding_summary and funding_summary.get("total_events", 0) > 0:
        table.add_row("", "", "")  # Separator

        # Use production funding data
        total_funding_cost = funding_summary.get("total_funding_cost", 0)
        impact_pct = funding_summary.get("account_impact_pct", 0)

        funding_impact = total_funding_cost * -1  # Negative if cost
        impact_color = "red" if funding_impact < 0 else "green"
        table.add_row("💸 Production Funding", "Total Events", str(funding_summary["total_events"]))
        table.add_row("", "Net Funding Impact (negative = cost)", f"[{impact_color}]${funding_impact:+.2f}[/{impact_color}]")
        table.add_row("", "Account Impact", f"{impact_pct:.3f}% of capital")
        table.add_row("", "Data Source", funding_summary.get("data_source", "Unknown"))
        table.add_row("", "Temporal Accuracy", funding_summary.get("temporal_accuracy", "Unknown"))
        table.add_row("", "Math Integrity", funding_summary.get("mathematical_integrity", "Unknown"))

    console.print(table)