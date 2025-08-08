#!/usr/bin/env python3
"""
Independent test of NautilusTrader simulated exchange functionality.
Minimal test to verify order execution works without our complex setup.
"""

import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import TraderId, Symbol, InstrumentId, Venue
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.core.datetime import dt_to_unix_nanos

from rich.console import Console

console = Console()

class SimpleTestStrategy(Strategy):
    """Minimal test strategy that places one market order."""
    
    def __init__(self, config=None):
        super().__init__()
        self.order_placed = False
    
    def on_start(self):
        console.print("🔄 SimpleTestStrategy started")
    
    def on_bar(self, bar):
        if not self.order_placed:
            console.print(f"📊 Received bar: {bar.close} at {bar.ts_init}")
            
            # Get instrument
            instrument = self.cache.instrument(bar.bar_type.instrument_id)
            if instrument:
                console.print(f"📋 Found instrument: {instrument}")
                
                # Place one simple market order
                order = self.order_factory.market(
                    instrument_id=bar.bar_type.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_str("0.001"),  # Small test order
                )
                
                console.print(f"📤 Submitting test order: {order}")
                self.submit_order(order)
                self.order_placed = True
    
    def on_order_filled(self, order, fill):
        console.print(f"✅ ORDER FILLED! {order.side} {fill.quantity} @ {fill.price}")
        console.print(f"   Order ID: {order.id}")
        console.print(f"   Fill ID: {fill.id}")

def create_simple_test_data():
    """Create minimal test data - just 5 bars."""
    console.print("📊 Creating simple test data...")
    
    # Create simple OHLCV data - CRITICAL: Match precision requirements
    base_price = 60000.0
    data = []
    start_time = datetime(2024, 1, 1, 9, 0)
    
    for i in range(5):
        timestamp = start_time + timedelta(minutes=i)
        price = base_price + (i * 100)  # Price increases each bar
        
        data.append({
            'timestamp': timestamp,
            'open': price,
            'high': price + 50,
            'low': price - 50, 
            'close': price + 25,
            'volume': 0.100000  # CRITICAL: Use 6 decimal precision to match instrument
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    console.print(f"   Created {len(df)} test bars")
    console.print(f"   Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    
    return df

def convert_to_nautilus_bars(df, instrument_id):
    """Convert DataFrame to NT Bar objects."""
    console.print("🔄 Converting to NT bars...")
    
    bar_spec = BarSpecification(
        step=1,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST
    )
    
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL
    )
    
    bars = []
    for timestamp, row in df.iterrows():
        ts_ns = dt_to_unix_nanos(pd.Timestamp(timestamp, tz='UTC'))
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['open']:.2f}"),
            high=Price.from_str(f"{row['high']:.2f}"),
            low=Price.from_str(f"{row['low']:.2f}"),
            close=Price.from_str(f"{row['close']:.2f}"),
            volume=Quantity.from_str(f"{row['volume']:.6f}"),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    
    console.print(f"   Converted {len(bars)} bars")
    return bars

def test_simulated_exchange():
    """Test simulated exchange with minimal setup."""
    console.print("🧪 TESTING SIMULATED EXCHANGE")
    console.print("=" * 50)
    
    # 1. Create engine
    config = BacktestEngineConfig(trader_id=TraderId("TEST-001"))
    engine = BacktestEngine(config=config)
    console.print("✅ Created BacktestEngine")
    
    # 2. Add venue
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(10000, USDT)],
        bar_execution=True,
        trade_execution=True,
    )
    console.print("✅ Added venue")
    
    # 3. Create instrument
    instrument_id = InstrumentId(Symbol("BTCUSDT"), venue)
    instrument = CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=6,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.000001"),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.05"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        ts_event=0,
        ts_init=0,
    )
    
    engine.add_instrument(instrument)
    console.print("✅ Added instrument")
    
    # 4. Create simple test data
    df = create_simple_test_data()
    bars = convert_to_nautilus_bars(df, instrument_id)
    
    # 5. Add data
    engine.add_data(bars)
    console.print("✅ Added bar data")
    
    # 6. Add strategy
    strategy = SimpleTestStrategy()
    engine.add_strategy(strategy)
    console.print("✅ Added strategy")
    
    # 7. Run backtest
    console.print("🚀 Running test backtest...")
    engine.run()
    console.print("✅ Backtest completed")
    
    # 8. Check results
    console.print("\n📊 RESULTS:")
    console.print("-" * 30)
    
    cache = engine.trader._cache
    orders = cache.orders()
    positions = cache.positions()
    
    console.print(f"Orders placed: {len(orders)}")
    console.print(f"Positions created: {len(positions)}")
    
    if orders:
        order = orders[0]
        console.print(f"\nOrder details:")
        console.print(f"  Status: {order.status}")
        console.print(f"  Filled qty: {order.filled_qty}")
        console.print(f"  Leaves qty: {order.leaves_qty}")
        
    # Check account balance changes
    account = engine.trader._cache.account_for_venue(venue)
    if account:
        balance = account.balance_total(USDT)
        console.print(f"\nAccount balance: {balance}")
        if balance.as_double() != 10000.0:
            console.print("✅ BALANCE CHANGED - Orders were filled!")
        else:
            console.print("❌ No balance change - Orders not filled")
    
    return len(orders) > 0 and len(positions) > 0

if __name__ == "__main__":
    try:
        success = test_simulated_exchange()
        
        if success:
            console.print("\n🎉 SUCCESS: Simulated exchange is working!")
        else:
            console.print("\n❌ FAILURE: Simulated exchange not filling orders")
            
    except Exception as e:
        console.print(f"\n💥 ERROR: {e}")
        import traceback
        console.print(traceback.format_exc())