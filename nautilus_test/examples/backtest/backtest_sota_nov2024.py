#!/usr/bin/env python3
"""
Backtest SOTA Momentum Strategy - November 2024
===============================================

NautilusTrader native-style backtest script for testing SOTA momentum strategy
during pre-holiday market conditions with our custom enhancements.

This script follows NT conventions while preserving all our custom features:
- Real Binance specifications via API
- DSM data integration for historical data
- Realistic position sizing (2% risk, not 1 BTC)
- Funding rate integration
- Enhanced visualization with finplot
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Import finplot for enhanced visualization
import finplot as fplt
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Standard NautilusTrader imports
# Import our NT-native strategy
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from rich.console import Console
from rich.panel import Panel
from strategies.sota.sota_momentum import SOTAMomentum, create_sota_momentum_config

from nautilus_test.actors import FinplotActor, FundingActor

# Import our NT-native components (preserve all functionality)
from nautilus_test.providers import (
    BinanceSpecificationManager,
    EnhancedModernBarDataProvider,
    RealisticPositionSizer,
)

# Optional funding integration
try:
    from nautilus_test.funding import BacktestFundingIntegrator

    FUNDING_AVAILABLE = True
except ImportError:
    FUNDING_AVAILABLE = False

console = Console()  # Rich console for beautiful terminal output


# Enhanced Finplot Visualization Functions
def prepare_bars_dataframe(bars):
    """Convert NautilusTrader Bar objects to DataFrame for visualization."""
    data = []
    for bar in bars:
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


def create_enhanced_candlestick_chart(
    df: pd.DataFrame, title: str = "Enhanced OHLC Chart with Real Specs"
):
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
                    timestamp = pd.Timestamp(timestamp_val)
            except (ValueError, TypeError):
                continue  # Skip invalid timestamps

            try:
                # Ensure we have a proper Timestamp object
                if not isinstance(timestamp, pd.Timestamp):
                    timestamp = pd.Timestamp(timestamp)
                trade_time = timestamp.floor("min")

                if trade_time in df.index:
                    bar_row = df.loc[trade_time]
                else:
                    nearest_idx = df.index.get_indexer([trade_time], method="nearest")[0]
                    bar_row = df.iloc[nearest_idx]

                bar_high = float(bar_row["high"])
                bar_low = float(bar_row["low"])

                if fill["order_side"] == "BUY":
                    buy_times.append(timestamp)
                    buy_prices.append(bar_low - (bar_high - bar_low) * 0.05)
                else:
                    sell_times.append(timestamp)
                    sell_prices.append(bar_high + (bar_high - bar_low) * 0.05)

            except (IndexError, KeyError, TypeError):
                # Fallback: use last_px or a reasonable default
                try:
                    price = float(fill.get("last_px", 50000))
                except (ValueError, TypeError):
                    price = 50000  # Reasonable BTC price fallback
                price_offset = price * 0.001

                if fill["order_side"] == "BUY":
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
        fplt.plot(
            sell_df, ax=ax, style="v", color="#f85149", width=4, legend="Sell (Realistic Size)"
        )


def display_enhanced_chart(
    bars, fills_report: pd.DataFrame, instrument_id: str, specs: dict, position_calc: dict
):
    """Display ultimate chart with real specs + realistic positions + rich visualization."""
    # Convert bars to DataFrame
    df = prepare_bars_dataframe(bars)

    # Create enhanced chart
    chart_title = f"{instrument_id} - Real Binance Specs + Realistic Positions + Rich Visualization"
    ax, _ = create_enhanced_candlestick_chart(df, chart_title)

    # Add indicators
    add_enhanced_indicators(df, ax, fast_period=10, slow_period=21)

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
        bars, fills_report, "BTCUSDT-PERP Pre-Holiday Data", specs, position_calc
    )


async def main():
    """
    Run SOTA Momentum Strategy backtest for November 2024 period.

    This follows NautilusTrader's native pattern while preserving all our
    custom enhancements for real-world trading accuracy.
    """

    # Display startup banner (NT-style)
    console.print("=" * 80)
    console.print(
        Panel.fit(
            "[bold blue]🍂 SOTA Momentum Strategy - November 2024 Backtest[/bold blue]\\n"
            "[cyan]NautilusTrader Native Implementation[/cyan]\\n"
            "[yellow]Period: 2024-11-20 10:00 to 2024-11-22 10:00 (48 hours)[/yellow]",
            title="📊 NT-NATIVE PRE-HOLIDAY BACKTEST",
            border_style="green",
        )
    )
    console.print("=" * 80)

    try:
        # STEP 1: Fetch Real Binance Specifications (preserve custom feature)
        console.print("\\n🎯 STEP 1: Real Specification Management")
        specs_manager = BinanceSpecificationManager()
        if not specs_manager.fetch_btcusdt_perpetual_specs():
            raise RuntimeError("Failed to fetch Binance specifications")

        # STEP 2: Calculate Realistic Position Size (preserve safety feature)
        console.print("\\n🎯 STEP 2: Realistic Position Sizing")
        position_sizer = RealisticPositionSizer(specs_manager.specs)
        position_calc = position_sizer.display_position_analysis()

        # STEP 3: Create BacktestEngine (NT-native pattern)
        console.print("\\n🎯 STEP 3: Enhanced Backtesting Engine")
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("SOTA-TRADER-003"),
                logging=LoggingConfig(log_level="ERROR"),
                risk_engine=RiskEngineConfig(bypass=True),
            )
        )

        # STEP 4: Add Venue with Real Specifications (NT-native + our specs)
        console.print("\\n🎯 STEP 4: Venue Configuration with Real Specs")
        engine.add_venue(
            venue=Venue("SIM"),
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(10_000, USDT)],
            fill_model=FillModel(),
            fee_model=MakerTakerFeeModel(),  # Fees configured in instrument
        )

        # STEP 5: Create Instrument with Real Specs (preserve custom feature)
        console.print("\\n🎯 STEP 5: Real Instrument Configuration")
        instrument = specs_manager.create_nautilus_instrument()
        engine.add_instrument(instrument)

        # STEP 6: Load Market Data via DSM (preserve custom data source)
        console.print("\\n🎯 STEP 6: Enhanced Data Pipeline")
        bar_type = BarType.from_str("BTCUSDT-PERP.SIM-1-MINUTE-LAST-EXTERNAL")
        console.print(f"🔧 Creating bar_type: {bar_type}")

        data_provider = EnhancedModernBarDataProvider(specs_manager)
        bars = data_provider.fetch_real_market_bars(
            instrument=instrument,
            bar_type=bar_type,
            symbol="BTCUSDT",
            limit=2880,  # 48 hours * 60 minutes
            start_time=datetime(2024, 11, 20, 10, 0, 0),
            end_time=datetime(2024, 11, 22, 10, 0, 0),
        )

        if not bars:
            raise RuntimeError("No market data fetched")

        engine.add_data(bars)
        console.print(f"📊 Loaded {len(bars)} bars for backtesting")

        # STEP 7: Setup Funding Integration (preserve custom feature)
        funding_integrator = None
        if FUNDING_AVAILABLE:
            console.print("\\n🎯 STEP 7: Production Funding Integration")
            try:
                funding_integrator = BacktestFundingIntegrator(
                    cache_dir="data_cache/production_funding"
                )
                funding_actor = FundingActor()
                engine.add_actor(funding_actor)
                console.print("🎉 PRODUCTION funding integration: SUCCESS")
            except Exception as e:
                console.print(f"[yellow]⚠️ Funding integration warning: {e}[/yellow]")
                funding_integrator = None
        else:
            console.print("\\n[yellow]⚠️ Funding system not available[/yellow]")

        # STEP 8: Configure Strategy (NT-native config pattern)
        console.print("\\n🎯 STEP 8: Strategy Configuration")
        strategy_config = create_sota_momentum_config(
            instrument_id=instrument.id,  # Use InstrumentId object
            bar_type=bar_type,
            trade_size=Decimal(str(position_calc["position_size_btc"])),
        )

        strategy = SOTAMomentum(strategy_config)
        engine.add_strategy(strategy)

        # STEP 9: Add Visualization Actor (preserve custom feature)
        finplot_actor = FinplotActor()
        engine.add_actor(finplot_actor)
        console.print("✅ Native FinplotActor integrated - charts ready")

        # STEP 10: Run Backtest (NT-native execution)
        console.print("\\n🎯 STEP 9: Backtest Execution")
        console.print("🔍 Starting backtest execution...")
        engine.run()
        console.print("✅ Backtest execution completed!")

        # STEP 11: Generate Results (preserve custom funding calculations)
        console.print("\\n🎯 STEP 10: Results & Analysis")

        # Get account and fills reports (NT-native)
        account_report = engine.trader.generate_account_report(Venue("SIM"))
        fills_report = engine.trader.generate_fills_report()

        # Extract P&L using proper account balance method
        starting_balance = 10000.0
        try:
            if not account_report.empty:
                final_balance = float(account_report.iloc[-1]["total"])
                original_pnl = final_balance - starting_balance
            else:
                original_pnl = 0.0
        except (ValueError, AttributeError, IndexError):
            original_pnl = 0.0

        # Calculate funding-adjusted P&L (preserve custom feature)
        funding_cost = 0.0
        if funding_integrator:
            funding_cost = getattr(funding_integrator, "total_funding_cost", 0.0)

        funding_adjusted_pnl = original_pnl - funding_cost

        # Display Results (preserve our enhanced tables)
        _display_results_summary(
            specs_manager,
            position_calc,
            original_pnl,
            funding_cost,
            funding_adjusted_pnl,
            len(bars),
        )

        # STEP 12: Launch Visualization (preserve custom feature)
        console.print("\\n📊 Launching Enhanced Interactive Chart...")
        try:
            create_post_backtest_chart(bars, fills_report, specs_manager.specs, position_calc)
            console.print("✅ Enhanced finplot chart displayed successfully")
        except Exception as chart_error:
            console.print(f"[yellow]⚠️ Chart display failed: {chart_error}[/yellow]")
            console.print("📊 Chart Info: Real Specs + Realistic Position + Pre-Holiday Data")

        console.print(
            "\\n[bold green]🍂 NT-Native SOTA Pre-Holiday Backtest Complete![/bold green]"
        )
        console.print(f"[yellow]📊 Final P&L: ${funding_adjusted_pnl:+.2f}[/yellow]")

        return {
            "original_pnl": original_pnl,
            "funding_cost": funding_cost,
            "funding_adjusted_pnl": funding_adjusted_pnl,
            "total_bars": len(bars),
            "fills_report": fills_report,
        }

    except Exception as e:
        console.print(f"[red]❌ Pre-holiday backtest failed: {e}[/red]")
        raise
    finally:
        # Cleanup (NT-native)
        if "engine" in locals():
            engine.reset()
            engine.dispose()


def _display_results_summary(
    specs_manager, position_calc, original_pnl, funding_cost, funding_adjusted_pnl, total_bars
):
    """Display comprehensive results summary with our enhanced formatting."""
    from rich.table import Table

    table = Table(title="🍂 SOTA Momentum Strategy - November 2024 Pre-Holiday Performance")
    table.add_column("Category", style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add comprehensive metrics (preserve our detailed analysis)
    metrics = [
        ("📊 Real Specifications", "Price Precision", str(specs_manager.specs["price_precision"])),
        ("", "Size Precision", str(specs_manager.specs["quantity_precision"])),
        ("", "Tick Size", specs_manager.specs["tick_size"]),
        ("", "Step Size", specs_manager.specs["step_size"]),
        ("", "Min Notional", f"${specs_manager.specs['min_notional']}"),
        ("", "", ""),
        (
            "💰 Realistic Positions",
            "Position Size",
            f"{position_calc['position_size_btc']:.3f} BTC",
        ),
        ("", "Trade Value", f"${position_calc['notional_value']:.2f}"),
        ("", "Account Risk", f"{position_calc['risk_percentage']:.1f}%"),
        ("", "", ""),
        ("📈 Trading Performance", "Starting Balance", "$10,000.00"),
        ("", "Final Balance", f"${10000 + funding_adjusted_pnl:.2f}"),
        ("", "P&L", f"{funding_adjusted_pnl:+.2f} ({funding_adjusted_pnl / 10000 * 100:+.2f}%)"),
        ("", "Funding Cost", f"${funding_cost:+.2f}"),
        ("", "", ""),
        ("⏰ Time Period", "Period", "2024-11-20 to 2024-11-22"),
        ("", "Duration", "48 hours"),
        ("", "Total Bars", str(total_bars)),
        ("", "Description", "Pre-holiday test (Nov 2024)"),
    ]

    for category, metric, value in metrics:
        table.add_row(category, metric, value)

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
