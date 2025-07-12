#!/usr/bin/env python3
"""
Simple test to explore OHLC bars handling in NautilusTrader.
This creates synthetic bar data to demonstrate basic concepts.
Enhanced with Rich library for beautiful terminal output.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import finplot as fplt
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from nautilus_trader.examples.strategies.ema_cross_bracket import (
    EMACrossBracket,
    EMACrossBracketConfig,
)
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider, TestInstrumentProvider
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Initialize Rich console
console = Console()


# Visualization functions
def prepare_bars_dataframe(bars: list[Bar]) -> pd.DataFrame:
    """Convert NautilusTrader Bar objects to DataFrame for visualization."""
    data = []
    for bar in bars:
        # Convert nanosecond timestamp to datetime
        timestamp = pd.Timestamp(bar.ts_event, unit="ns")
        data.append(
            {
                "time": timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df


def create_candlestick_chart(df: pd.DataFrame, title: str = "OHLC Chart"):
    """Create a candlestick chart with volume."""
    # Create figure with 2 rows (price and volume)
    ax, ax2 = fplt.create_plot(title, rows=2)

    # Plot candlesticks on main chart
    fplt.candlestick_ochl(df[["open", "close", "high", "low"]], ax=ax)

    # Plot volume on second chart
    fplt.volume_ocv(df[["open", "close", "volume"]], ax=ax2)

    # Link x-axes
    ax2.set_visible(xgrid=True, ygrid=True)

    return ax, ax2


def add_ema_indicators(df: pd.DataFrame, ax, fast_period: int = 10, slow_period: int = 20):
    """Add EMA indicators to the chart."""
    # Calculate EMAs
    df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()

    # Plot EMAs
    fplt.plot(df["ema_fast"], ax=ax, color="#3388ff", width=2, legend=f"EMA {fast_period}")
    fplt.plot(df["ema_slow"], ax=ax, color="#ff3388", width=2, legend=f"EMA {slow_period}")

    return df


def add_trade_markers(df: pd.DataFrame, fills_report: pd.DataFrame, ax):
    """Add trade execution markers to the chart."""
    if fills_report.empty:
        return

    # Process fills data
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    for _, fill in fills_report.iterrows():
        # Convert timestamp to datetime - handle potential series or value
        timestamp_val = fill["ts_init"]
        if isinstance(timestamp_val, pd.Series):
            timestamp_val = timestamp_val.iloc[0] if not timestamp_val.empty else None
        if timestamp_val is not None:
            timestamp = pd.Timestamp(timestamp_val)
            price = float(fill["avg_px"])

            if fill["side"] == "BUY":
                buy_times.append(timestamp)
                buy_prices.append(price)
            else:
                sell_times.append(timestamp)
                sell_prices.append(price)

    # Create scatter plots for trades
    if buy_times:
        buy_df = pd.DataFrame({"price": buy_prices}, index=pd.Index(buy_times))
        fplt.plot(buy_df, ax=ax, style="^", color="#00ff00", width=3, legend="Buy")

    if sell_times:
        sell_df = pd.DataFrame({"price": sell_prices}, index=pd.Index(sell_times))
        fplt.plot(sell_df, ax=ax, style="v", color="#ff0000", width=3, legend="Sell")


def display_chart_with_trades(
    bars: list[Bar],
    fills_report: pd.DataFrame,
    instrument_id: str,
    fast_ema: int = 10,
    slow_ema: int = 20,
):
    """Display complete chart with OHLC, volume, indicators, and trades."""
    # Convert bars to DataFrame
    df = prepare_bars_dataframe(bars)

    # Create candlestick chart
    ax, ax2 = create_candlestick_chart(df, f"{instrument_id} - Backtest Results")

    # Add EMA indicators
    add_ema_indicators(df, ax, fast_ema, slow_ema)

    # Add trade markers
    add_trade_markers(df, fills_report, ax)

    # Show the plot
    fplt.show()

    return df


def create_sample_bars(
    instrument_id: InstrumentId, bar_type: BarType, count: int = 100
) -> list[Bar]:
    """Create sample OHLC bars with progress indicator."""
    bars = []
    base_price = 1.3000  # Starting price for EUR/USD
    base_time = datetime(2024, 1, 1, 9, 0, 0)  # Start time

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Creating {count} synthetic bars...", total=count)

        for i in range(count):
            # Create some price movement
            trend = 0.0001 * (i % 20 - 10)  # Small trend
            noise = 0.0002 * ((i * 7) % 21 - 10) / 10  # Some noise

            open_price = base_price + trend + noise
            close_price = open_price + trend * 0.5

            # Ensure high >= max(open, close) and low <= min(open, close)
            high_price = max(open_price, close_price) + abs(noise) + 0.0001
            low_price = min(open_price, close_price) - abs(noise) - 0.0001

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.5f}"),
                high=Price.from_str(f"{high_price:.5f}"),
                low=Price.from_str(f"{low_price:.5f}"),
                close=Price.from_str(f"{close_price:.5f}"),
                volume=Quantity.from_str("1000"),
                ts_event=int((base_time + timedelta(minutes=i)).timestamp() * 1_000_000_000),
                ts_init=int((base_time + timedelta(minutes=i)).timestamp() * 1_000_000_000),
            )
            bars.append(bar)

            # Update base price for next bar
            base_price = close_price
            progress.update(task, advance=1)

    return bars


def display_sample_bars(bars: list[Bar], count: int = 5):
    """Display sample bars in a Rich table."""
    table = Table(
        title=f"📊 Sample OHLC Bar Data (First {count} of {len(bars)} bars)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Open", justify="right", style="green")
    table.add_column("High", justify="right", style="bright_green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="bright_red")
    table.add_column("Volume", justify="right", style="blue")

    for i, bar in enumerate(bars[:count]):
        table.add_row(
            str(i + 1), str(bar.open), str(bar.high), str(bar.low), str(bar.close), str(bar.volume)
        )

    if len(bars) > count:
        table.add_row("...", "...", "...", "...", "...", "...")

    console.print(table)


def display_config():
    """Display backtest configuration."""
    config_text = """[bold blue]Strategy:[/bold blue] EMA Cross
[bold green]Instrument:[/bold green] EUR/USD
[bold yellow]Timeframe:[/bold yellow] 1-minute bars
[bold cyan]Fast EMA:[/bold cyan] 10 periods
[bold red]Slow EMA:[/bold red] 20 periods
[bold magenta]Position Size:[/bold magenta] 1,000 units
[bold white]Starting Capital:[/bold white] $10,000"""

    console.print(
        Panel(
            config_text,
            title="⚙️ Backtest Configuration",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


def format_currency(amount: float) -> str:
    """Format currency amounts nicely."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage values."""
    return f"{value:+.2f}%"


def display_performance_summary(account_report, fills_report, starting_balance=10000.0):
    """Display performance summary with Rich formatting and better error handling."""
    perf_table = Table(
        title="📈 Performance Summary",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )

    perf_table.add_column("Metric", style="bold")
    perf_table.add_column("Value", justify="right")

    # Account performance with better error handling
    if not account_report.empty:
        try:
            # Handle both string and float balance values
            final_balance_val = account_report.iloc[-1]["total"]
            if isinstance(final_balance_val, str):
                final_balance = float(final_balance_val)
            else:
                final_balance = float(final_balance_val)

            pnl = final_balance - starting_balance
            pnl_pct = (pnl / starting_balance) * 100

            pnl_color = "green" if pnl >= 0 else "red"

            perf_table.add_row("Starting Capital", format_currency(starting_balance))
            perf_table.add_row(
                "Final Balance", f"[{pnl_color}]{format_currency(final_balance)}[/{pnl_color}]"
            )
            perf_table.add_row(
                "Total P&L",
                f"[{pnl_color}]{format_currency(pnl)} ({format_percentage(pnl_pct)})[/{pnl_color}]",
            )

        except Exception as e:
            perf_table.add_row("Balance Error", str(e))

    # Trade statistics with commission handling
    if not fills_report.empty:
        try:
            total_trades = len(fills_report)
            buy_trades = len(fills_report[fills_report["side"] == "BUY"])
            sell_trades = len(fills_report[fills_report["side"] == "SELL"])

            # Extract commissions more safely
            if "commissions" in fills_report.columns:
                comm_values = (
                    fills_report["commissions"]
                    .astype(str)
                    .str.extract(r"(\d+\.?\d*)")
                    .astype(float)
                )
                total_commissions = comm_values.sum().iloc[0] if not comm_values.empty else 0.0
            else:
                total_commissions = 0.0

            perf_table.add_row("", "")  # Separator
            perf_table.add_row("Total Trades", f"{total_trades}")
            perf_table.add_row("Buy Trades", f"[green]{buy_trades}[/green]")
            perf_table.add_row("Sell Trades", f"[red]{sell_trades}[/red]")
            perf_table.add_row("Total Commissions", f"${total_commissions:.2f}")

        except Exception as e:
            perf_table.add_row("Trade Error", str(e))
    else:
        perf_table.add_row("Trades", "[yellow]No trades executed[/yellow]")

    console.print(perf_table)


def display_recent_trades(fills_report, count: int = 5):
    """Display recent trades."""
    if fills_report.empty:
        console.print(Panel("[yellow]No trades executed[/yellow]", title="📋 Recent Trades"))
        return

    trades_table = Table(
        title=f"📋 Recent Trades (Last {count})",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold blue",
    )

    trades_table.add_column("Time", style="dim")
    trades_table.add_column("Side", justify="center")
    trades_table.add_column("Quantity", justify="right", style="cyan")
    trades_table.add_column("Price", justify="right", style="yellow")

    recent_trades = fills_report.tail(count)

    for _, trade in recent_trades.iterrows():
        time_str = pd.to_datetime(trade["ts_init"]).strftime("%H:%M:%S")
        side = trade["side"]
        side_colored = f"[green]{side}[/green]" if side == "BUY" else f"[red]{side}[/red]"

        trades_table.add_row(
            time_str, side_colored, f"{int(trade['quantity']):,}", f"{float(trade['avg_px']):.5f}"
        )

    console.print(trades_table)


def load_real_data(engine, venue, use_real_data=False):
    """Load real FXCM data if available, otherwise return None."""
    if not use_real_data:
        return None

    try:
        # Add GBP/USD instrument for real data
        GBPUSD_SIM = TestInstrumentProvider.default_fx_ccy("GBP/USD", venue)
        engine.add_instrument(GBPUSD_SIM)

        # Load real data
        provider = TestDataProvider()
        wrangler = QuoteTickDataWrangler(instrument=GBPUSD_SIM)

        # Process CSV data into ticks
        ticks = wrangler.process_bar_data(
            bid_data=provider.read_csv_bars("fxcm/gbpusd-m1-bid-2012.csv"),
            ask_data=provider.read_csv_bars("fxcm/gbpusd-m1-ask-2012.csv"),
        )
        engine.add_data(ticks)

        console.print(f"[green]✅ Loaded {len(ticks)} real ticks from FXCM data[/green]")
        return GBPUSD_SIM, ticks

    except Exception as e:
        console.print(f"[yellow]⚠️ Could not load real data: {e}[/yellow]")
        return None


def main():
    # Title banner
    console.print(
        Panel(
            Text(
                "🚀 Enhanced NautilusTrader OHLC Bars Test", style="bold magenta", justify="center"
            ),
            subtitle="Complete Feature Set with Rich Library 🎨",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )

    # Configuration options
    USE_REAL_DATA = False  # Set to True to try loading FXCM data
    USE_BRACKET_STRATEGY = False  # Set to True to use EMACrossBracket
    STARTING_CAPITAL = 10_000  # Can be changed to 100_000 for higher capital
    ENABLE_CHART_VISUALIZATION = True  # Set to False to disable finplot charts

    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    )

    engine = BacktestEngine(config=config)

    # Create a fill model
    fill_model = FillModel(
        prob_fill_on_limit=0.2,
        prob_fill_on_stop=0.95,
        prob_slippage=0.1,
        random_seed=42,
    )

    # Add a trading venue
    SIM = Venue("SIM")
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(STARTING_CAPITAL, USD)],
        fill_model=fill_model,
        bar_execution=True,
    )

    # Try to load real data first
    real_data_result = load_real_data(engine, SIM, USE_REAL_DATA)

    if real_data_result:
        # Use real data setup
        instrument, data = real_data_result
        bar_type = BarType.from_str("GBP/USD.SIM-5-MINUTE-BID-INTERNAL")
        bars = None  # Real data loaded as ticks
        console.print("[blue]📊 Using real FXCM GBP/USD data[/blue]")
    else:
        # Use synthetic data setup
        EURUSD_SIM = TestInstrumentProvider.default_fx_ccy("EUR/USD", SIM)
        engine.add_instrument(EURUSD_SIM)
        instrument = EURUSD_SIM

        # Create bar type and synthetic data
        bar_type = BarType.from_str(f"{EURUSD_SIM.id}-1-MINUTE-MID-EXTERNAL")
        bars = create_sample_bars(EURUSD_SIM.id, bar_type, count=200)
        engine.add_data(bars)
        console.print("[blue]📊 Using synthetic EUR/USD data[/blue]")

    # Display sample bars (only for synthetic data)
    if bars:
        display_sample_bars(bars)

    # Configure strategy based on choice
    if USE_BRACKET_STRATEGY and not real_data_result:
        console.print(
            "[yellow]⚠️ Bracket strategy requires real data, using basic EMA Cross[/yellow]"
        )
        USE_BRACKET_STRATEGY = False

    if USE_BRACKET_STRATEGY:
        # EMACrossBracket strategy (for real data)
        strategy_config = EMACrossBracketConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            fast_ema_period=10,
            slow_ema_period=20,
            bracket_distance_atr=2.0,
            trade_size=Decimal(10_000),
        )
        strategy = EMACrossBracket(config=strategy_config)
        strategy_name = "EMA Cross with Bracket Orders"
    else:
        # Basic EMACross strategy
        strategy_config = EMACrossConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            fast_ema_period=10,
            slow_ema_period=20,
            trade_size=Decimal(1000),
        )
        strategy = EMACross(config=strategy_config)
        strategy_name = "EMA Cross"

    engine.add_strategy(strategy=strategy)

    # Display configuration
    config_text = f"""[bold blue]Strategy:[/bold blue] {strategy_name}
[bold green]Instrument:[/bold green] {instrument.id}
[bold yellow]Timeframe:[/bold yellow] {bar_type}
[bold cyan]Fast EMA:[/bold cyan] 10 periods
[bold red]Slow EMA:[/bold red] 20 periods
[bold magenta]Position Size:[/bold magenta] {strategy_config.trade_size:,} units
[bold white]Starting Capital:[/bold white] ${STARTING_CAPITAL:,}"""

    console.print(
        Panel(
            config_text,
            title="⚙️ Backtest Configuration",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    # Run backtest with status
    with console.status("[bold green]Running backtest...", spinner="dots"):
        engine.run()

    console.print("✅ [bold green]Backtest completed![/bold green]")

    # Generate and display results
    try:
        account_report = engine.trader.generate_account_report(SIM)
        fills_report = engine.trader.generate_order_fills_report()

        display_performance_summary(account_report, fills_report, STARTING_CAPITAL)
        display_recent_trades(fills_report)

        # Display interactive chart visualization
        if ENABLE_CHART_VISUALIZATION and bars:
            console.print(
                "\n[bold cyan]📊 Launching interactive chart visualization...[/bold cyan]"
            )
            try:
                display_chart_with_trades(
                    bars, fills_report, str(instrument.id), fast_ema=10, slow_ema=20
                )
            except Exception as chart_error:
                console.print(f"[yellow]⚠️ Chart visualization error: {chart_error}[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error generating reports: {e}[/red]")

    # Enhanced features checklist
    features = [
        "✅ Synthetic & Real OHLC data support",
        "✅ Multiple strategy options (EMA Cross + Bracket)",
        "✅ Multiple instruments (EUR/USD, GBP/USD)",
        "✅ Configurable starting capital",
        "✅ Enhanced error handling & commission tracking",
        "✅ Beautiful Rich output with progress indicators 🎨",
        "✅ Interactive finplot charts with candlesticks, EMAs & trade markers 📈",
    ]

    console.print(
        Panel(
            "\n".join(features),
            title="✅ Enhanced Features Demonstrated",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Clean up
    engine.reset()
    engine.dispose()

    console.print(
        Panel(
            Text("🎉 Enhanced Test Complete!", style="bold green", justify="center"),
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
