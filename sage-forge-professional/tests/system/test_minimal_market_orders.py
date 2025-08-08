#!/usr/bin/env python3
"""
Minimal test to verify MARKET orders execute with bar data in NautilusTrader.
Based on working examples from NT documentation.
"""

import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import TraderId, Symbol, InstrumentId, Venue
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, PositionSide
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.backtest.models import FillModel

from rich.console import Console

console = Console()

class MinimalMarketOrderStrategy(Strategy):
    """Minimal strategy that places MARKET orders on every bar."""
    
    def __init__(self, config=None):
        super().__init__()
        self.orders_placed = 0
        self.max_orders = 3  # Only place 3 orders for testing
    
    def on_start(self):
        console.print("🚀 MinimalMarketOrderStrategy started")
        
        # CRITICAL: Subscribe to bar data
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        from nautilus_trader.model.data import BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
        
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
        
        console.print(f"📡 Subscribing to bar data: {bar_type}")
        self.subscribe_bars(bar_type)
        console.print("✅ Subscription completed")
    
    def on_bar(self, bar):
        if self.orders_placed >= self.max_orders:
            return
            
        console.print(f"📊 Received bar #{self.orders_placed + 1}: {bar.close} at {bar.ts_init}")
        
        # Get instrument
        instrument = self.cache.instrument(bar.bar_type.instrument_id)
        if not instrument:
            console.print("❌ No instrument found")
            return
            
        console.print(f"📋 Found instrument: {instrument}")
        
        # Alternate between BUY and SELL orders
        order_side = OrderSide.BUY if self.orders_placed % 2 == 0 else OrderSide.SELL
        
        # Place MARKET order with fixed quantity
        try:
            console.print(f"📤 [BEFORE ORDER] Placing {order_side} MARKET order...")
            
            order = self.submit_order(
                self.order_factory.market(
                    instrument_id=bar.bar_type.instrument_id,
                    order_side=order_side,
                    quantity=Quantity.from_str("0.001"),  # Small test order
                )
            )
            
            self.orders_placed += 1
            console.print(f"✅ [AFTER ORDER] Order #{self.orders_placed} submitted: {order}")
            
        except Exception as e:
            console.print(f"❌ Order failed: {e}")
    
    def on_order_filled(self, fill):
        console.print(f"🎉 ORDER FILLED! {fill.client_order_id} - {fill.order_side} {fill.last_qty} @ {fill.last_px}")
        console.print(f"   Fill timestamp: {fill.ts_init}")

def create_simple_test_bars():
    """Create minimal test bars with known price movements."""
    console.print("📊 Creating test bars...")
    
    # Create simple price progression
    prices = [60000.0, 60100.0, 60050.0]  # Price movements
    data = []
    start_time = datetime(2024, 1, 1, 9, 0)
    
    for i, price in enumerate(prices):
        timestamp = start_time + timedelta(minutes=i)
        
        data.append({
            'timestamp': timestamp,
            'open': price,
            'high': price + 25,
            'low': price - 25,
            'close': price + 10,  # Close slightly higher
            'volume': 1.0  # Simple volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    console.print(f"   Created {len(df)} bars")
    console.print(f"   Price progression: {[f'${p:,.0f}' for p in prices]}")
    
    return df

def convert_to_nt_bars(df, instrument_id):
    """Convert DataFrame to NT Bar objects with proper precision."""
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
            open=Price.from_str(f"{row['open']:.2f}"),     # Match price_precision=2
            high=Price.from_str(f"{row['high']:.2f}"),     # Match price_precision=2
            low=Price.from_str(f"{row['low']:.2f}"),       # Match price_precision=2
            close=Price.from_str(f"{row['close']:.2f}"),   # Match price_precision=2
            volume=Quantity.from_str(f"{row['volume']:.3f}"), # Match size_precision=3
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    
    console.print(f"   Converted {len(bars)} bars")
    return bars

def test_minimal_market_orders():
    """Test minimal MARKET order execution with bar data."""
    console.print("🧪 TESTING MINIMAL MARKET ORDER EXECUTION")
    console.print("=" * 60)
    
    # 1. Create engine with FillModel
    fill_model = FillModel(
        prob_fill_on_limit=1.0,  # Always fill limit orders
        prob_slippage=0.0,       # No slippage
        random_seed=42,          # Reproducible
    )
    
    config = BacktestEngineConfig(trader_id=TraderId("TEST-001"))
    engine = BacktestEngine(config=config)
    console.print("✅ Created BacktestEngine")
    
    # 2. Add venue with FillModel
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(10000, USDT)],
        bar_execution=True,        # CRITICAL: Enable bar execution
        trade_execution=False,     # Only bar execution
        fill_model=fill_model,     # CRITICAL: Add FillModel
    )
    console.print("✅ Added venue with bar execution")
    
    # 3. Create instrument with matching precision
    instrument_id = InstrumentId(Symbol("BTCUSDT"), venue)
    instrument = CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,   # Match bar price precision
        size_precision=3,    # Match bar volume precision  
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"),
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
    df = create_simple_test_bars()
    bars = convert_to_nt_bars(df, instrument_id)
    
    # 5. Add bars to engine
    engine.add_data(bars)
    console.print("✅ Added bar data")
    
    # 6. Add strategy
    strategy = MinimalMarketOrderStrategy()
    engine.add_strategy(strategy)
    console.print("✅ Added strategy")
    
    # 7. Run backtest
    console.print("🚀 Running minimal test...")
    console.print("-" * 40)
    engine.run()
    console.print("-" * 40)
    console.print("✅ Backtest completed")
    
    # 8. Check results
    console.print("\\n📊 RESULTS:")
    console.print("-" * 30)
    
    cache = engine.trader._cache
    orders = cache.orders()
    positions = cache.positions()
    
    console.print(f"Total orders placed: {len(orders)}")
    console.print(f"Total positions created: {len(positions)}")
    
    if orders:
        for i, order in enumerate(orders, 1):
            console.print(f"\\nOrder {i}:")
            console.print(f"  Status: {order.status}")
            console.print(f"  Side: {order.side}")
            console.print(f"  Quantity: {order.quantity}")
            console.print(f"  Filled qty: {order.filled_qty}")
            console.print(f"  Remaining: {order.leaves_qty}")
    
    # Check for fills
    success = len(orders) > 0 and any(order.filled_qty > 0 for order in orders)
    
    if success:
        console.print("\\n🎉 SUCCESS: Market orders were filled!")
    else:
        console.print("\\n❌ FAILURE: Market orders not filled")
    
    return success

if __name__ == "__main__":
    try:
        success = test_minimal_market_orders()
        exit(0 if success else 1)
    except Exception as e:
        console.print(f"\\n💥 ERROR: {e}")
        import traceback
        console.print(traceback.format_exc())
        exit(1)