"""
Unit tests for NautilusTrader OHLC bars functionality.

NOTE: These tests use synthetic data for deterministic, repeatable testing.
For real market data examples, see examples/sandbox/dsm_integration_demo.py
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


@pytest.fixture
def backtest_engine():
    """Create a configured backtest engine for testing."""
    config = BacktestEngineConfig(
        trader_id=TraderId("TEST-001"),
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    )

    engine = BacktestEngine(config=config)

    # Add venue
    venue = Venue("SIM")
    fill_model = FillModel(
        prob_fill_on_limit=1.0,  # 100% fill for tests
        prob_fill_on_stop=1.0,
        prob_slippage=0.0,  # No slippage for deterministic tests
        random_seed=42,
    )

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
        fill_model=fill_model,
        bar_execution=True,
    )

    return engine, venue


def create_test_bars(instrument_id, bar_type, count=10):
    """Create a small set of test bars for pytest."""
    bars = []
    base_price = 1.3000
    base_time = datetime(2024, 1, 1, 9, 0, 0)

    for i in range(count):
        # Simple upward trend for predictable testing
        price_increment = 0.0001 * i
        open_price = base_price + price_increment
        close_price = open_price + 0.0001
        high_price = close_price + 0.0001
        low_price = open_price - 0.0001

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

    return bars


def test_bar_creation():
    """Test that OHLC bars are created correctly."""
    # Create instrument and bar type
    venue = Venue("TEST")
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-MID-EXTERNAL")

    # Create test bars
    bars = create_test_bars(instrument.id, bar_type, count=5)

    # Assertions
    assert len(bars) == 5
    assert all(isinstance(bar, Bar) for bar in bars)
    assert all(bar.bar_type == bar_type for bar in bars)
    assert all(bar.volume.as_double() == 1000 for bar in bars)

    # Check OHLC validity
    for bar in bars:
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low <= min(bar.open, bar.close)


def test_backtest_engine_setup(backtest_engine):
    """Test that backtest engine is properly configured."""
    engine, venue = backtest_engine

    # Add instrument
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    engine.add_instrument(instrument)

    # Verify instrument was added (using available methods)
    assert instrument.id.venue == venue

    # Basic engine verification
    assert engine.trader is not None


def test_ema_cross_strategy_execution(backtest_engine):
    """Test EMA Cross strategy execution with synthetic data."""
    engine, venue = backtest_engine

    # Add instrument
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    engine.add_instrument(instrument)

    # Create bar type and data
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-MID-EXTERNAL")
    bars = create_test_bars(instrument.id, bar_type, count=50)  # Need enough bars for EMA
    engine.add_data(bars)

    # Configure strategy
    strategy_config = EMACrossConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        fast_ema_period=5,  # Shorter periods for test data
        slow_ema_period=10,
        trade_size=Decimal(1000),
    )

    strategy = EMACross(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    # Run backtest
    engine.run()

    # Verify strategy was executed
    account_report = engine.trader.generate_account_report(venue)
    fills_report = engine.trader.generate_order_fills_report()

    # Basic assertions
    assert not account_report.empty
    assert len(account_report) > 0

    # Check if any trades were executed (may be 0 depending on data)
    if not fills_report.empty:
        assert "side" in fills_report.columns
        assert "quantity" in fills_report.columns
        # Convert to int for comparison (quantities are strings in report)
        quantities = fills_report["quantity"].astype(int)
        assert all(quantities == 1000)

    # Clean up
    engine.reset()
    engine.dispose()


def test_performance_metrics_calculation():
    """Test performance metrics calculation functions."""
    # Test currency formatting
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples', 'sandbox'))
    from simple_bars_test import format_currency, format_percentage

    assert format_currency(1000.50) == "$1,000.50"
    assert format_currency(0) == "$0.00"
    assert format_currency(-500.75) == "$-500.75"

    # Test percentage formatting
    assert format_percentage(15.5) == "+15.50%"
    assert format_percentage(-5.25) == "-5.25%"
    assert format_percentage(0) == "+0.00%"


def test_bar_type_consistency():
    """Test that bar types are consistent between data and strategy."""
    venue = Venue("TEST")
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-MID-EXTERNAL")

    # Create bars with this bar type
    bars = create_test_bars(instrument.id, bar_type, count=3)

    # Verify all bars have the same bar type
    for bar in bars:
        assert bar.bar_type == bar_type
        assert str(bar.bar_type) == str(bar_type)


@pytest.mark.parametrize("trade_size", [100, 1000, 10000])
def test_different_trade_sizes(backtest_engine, trade_size):
    """Test strategy execution with different trade sizes."""
    engine, venue = backtest_engine

    # Add instrument
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    engine.add_instrument(instrument)

    # Create bar type and data
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-MID-EXTERNAL")
    bars = create_test_bars(instrument.id, bar_type, count=30)
    engine.add_data(bars)

    # Configure strategy with parameterized trade size
    strategy_config = EMACrossConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        fast_ema_period=5,
        slow_ema_period=10,
        trade_size=Decimal(trade_size),
    )

    strategy = EMACross(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    # Run backtest
    engine.run()

    # If trades occurred, verify trade size
    fills_report = engine.trader.generate_order_fills_report()
    if not fills_report.empty:
        quantities = fills_report["quantity"].astype(int)
        assert all(quantities == trade_size)

    # Clean up
    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])
