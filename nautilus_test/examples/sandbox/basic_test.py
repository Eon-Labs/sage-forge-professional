import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def main():
    print("Testing NautilusTrader functionality...")

    # Test 1: Create a simple instrument
    print("\n1. Creating test instrument:")
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD")
    print(f"   Instrument: {instrument.id}")
    print(f"   Symbol: {instrument.id.symbol}")
    print(f"   Venue: {instrument.id.venue}")

    # Test 2: Create price and quantity objects
    print("\n2. Creating price and quantity objects:")
    price = Price.from_str("1.08250")
    quantity = Quantity.from_str("100000")
    print(f"   Price: {price}")
    print(f"   Quantity: {quantity}")

    # Test 3: Create a quote tick
    print("\n3. Creating quote tick:")
    quote_tick = QuoteTick(
        instrument_id=instrument.id,
        bid_price=Price.from_str("1.08240"),
        ask_price=Price.from_str("1.08250"),
        bid_size=Quantity.from_str("1000000"),
        ask_size=Quantity.from_str("1000000"),
        ts_event=pd.Timestamp.now(tz="UTC").value,
        ts_init=pd.Timestamp.now(tz="UTC").value,
    )
    print(f"   Quote: {quote_tick}")

    print("\n✅ NautilusTrader basic functionality test completed successfully!")


if __name__ == "__main__":
    main()
