#!/usr/bin/env python3
"""
Controlled test to verify TiRex can actually produce trading signals.
We'll test with different approaches to force signal generation.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.strategies.tirex_sage_strategy import TiRexSageStrategy
from sage_forge.strategies.tirex_sage_config import TiRexSageStrategyConfig
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.core.datetime import dt_to_unix_nanos
from datetime import datetime
from rich.console import Console

console = Console()

def create_synthetic_trending_bars():
    """Create synthetic bars with a clear upward trend to trigger TiRex signals."""
    
    # Create bar type
    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bar_spec = BarSpecification(
        step=15,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST
    )
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL
    )
    
    # Create trending bars - clear upward trend
    base_price = 96000.0
    bars = []
    
    for i in range(20):
        # Strong upward trend with some volatility
        trend_factor = i * 200  # $200 increase per bar
        volatility = 50  # $50 volatility
        
        open_price = base_price + trend_factor
        high_price = open_price + volatility + 30
        low_price = open_price - volatility + 20  
        close_price = open_price + trend_factor * 0.8  # Trending up
        
        ts = dt_to_unix_nanos(datetime(2024, 12, 1, 0, i * 15))
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high_price:.2f}"),
            low=Price.from_str(f"{low_price:.2f}"),
            close=Price.from_str(f"{close_price:.2f}"),
            volume=Quantity.from_str("1000"),
            ts_event=ts,
            ts_init=ts,
        )
        bars.append(bar)
        
    return bars

def test_tirex_signal_capability():
    """Test if TiRex can produce signals with different configurations."""
    console.print("🧪 Testing TiRex Signal Generation Capability")
    console.print("=" * 60)
    
    # Test 1: Low confidence threshold
    console.print("\n📊 Test 1: Low Confidence Threshold (30%)")
    config1 = TiRexSageStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        min_confidence=0.3,  # Very low threshold
        max_position_size=0.1,
        risk_per_trade=0.02,
        model_name="NX-AI/TiRex",
        device="cuda"
    )
    
    try:
        strategy1 = TiRexSageStrategy(config1)
        console.print("✅ Strategy created with 30% confidence threshold")
        
        # Test with synthetic trending data
        trending_bars = create_synthetic_trending_bars()
        console.print(f"📈 Created {len(trending_bars)} synthetic trending bars")
        
        # Display the trend
        first_bar = trending_bars[0]
        last_bar = trending_bars[-1]
        trend_gain = float(last_bar.close) - float(first_bar.open)
        console.print(f"💹 Trend: ${float(first_bar.open):.2f} → ${float(last_bar.close):.2f} (+${trend_gain:.2f})")
        
        # Test TiRex prediction directly
        console.print("🔍 Testing TiRex prediction on trending data...")
        
        # This would need access to the strategy's internal TiRex model
        # Let's check if we can access the model directly
        if hasattr(strategy1, 'tirex_model') and strategy1.tirex_model:
            console.print("✅ TiRex model accessible")
            
            # Try to get a prediction
            try:
                # Convert bars to format TiRex expects
                ohlcv_data = []
                for bar in trending_bars:
                    ohlcv_data.append([
                        float(bar.open),
                        float(bar.high), 
                        float(bar.low),
                        float(bar.close),
                        float(bar.volume)
                    ])
                
                console.print(f"📊 Prepared {len(ohlcv_data)} OHLCV records for TiRex")
                
                # This is where we'd test the actual TiRex model prediction
                # But we need to simulate bar reception to trigger the strategy
                console.print("⚠️ Direct model testing requires strategy simulation")
                
            except Exception as e:
                console.print(f"❌ Model prediction test failed: {e}")
                
        else:
            console.print("❌ TiRex model not accessible directly")
            
    except Exception as e:
        console.print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Different market periods 
    console.print("\n📊 Test 2: Different Time Periods")
    
    test_periods = [
        ("2024-11-01", "2024-11-03"),  # Different period
        ("2024-10-15", "2024-10-17"),  # Another period
        ("2024-09-01", "2024-09-03"),  # Earlier period
    ]
    
    for start_date, end_date in test_periods:
        console.print(f"🗓️ Testing period: {start_date} to {end_date}")
        
        try:
            from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
            
            engine = TiRexBacktestEngine()
            success = engine.setup_backtest(
                symbol="BTCUSDT",
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000.0,
                timeframe="15m"
            )
            
            if success:
                bars_count = len(engine.market_bars)
                console.print(f"   📈 {bars_count} bars loaded for {start_date}")
                
                if bars_count > 0:
                    first_bar = engine.market_bars[0]
                    last_bar = engine.market_bars[-1]
                    price_change = float(last_bar.close) - float(first_bar.open)
                    change_pct = (price_change / float(first_bar.open)) * 100
                    console.print(f"   💹 Price change: {price_change:+.2f} ({change_pct:+.2f}%)")
                    
                    # Only test periods with significant movement
                    if abs(change_pct) > 2.0:
                        console.print(f"   🎯 Significant movement detected - testing TiRex...")
                        
                        # Quick test run
                        try:
                            results = engine.run_backtest()
                            if results:
                                console.print(f"   ✅ Backtest completed for {start_date}")
                                # Check for any orders/signals
                                if hasattr(results, 'orders') and results.orders:
                                    console.print(f"   🚨 SIGNALS FOUND: {len(results.orders)} orders!")
                                    return True
                                else:
                                    console.print(f"   📊 No signals in {start_date}")
                            else:
                                console.print(f"   ❌ No results for {start_date}")
                        except Exception as e:
                            console.print(f"   ⚠️ Backtest error for {start_date}: {str(e)[:50]}...")
                    else:
                        console.print(f"   📊 Low volatility period ({change_pct:+.1f}%)")
                else:
                    console.print(f"   ❌ No data for {start_date}")
            else:
                console.print(f"   ❌ Setup failed for {start_date}")
                
        except Exception as e:
            console.print(f"   ❌ Period test failed: {str(e)[:50]}...")
    
    console.print("\n🔍 CONCLUSION:")
    console.print("To definitively test TiRex signal generation, we need:")
    console.print("1. Access to TiRex model's internal prediction method")
    console.print("2. Known market periods with strong trends") 
    console.print("3. Ability to mock/inject synthetic data directly into strategy")
    console.print("4. Lower confidence thresholds for testing")
    
    return False

if __name__ == "__main__":
    found_signals = test_tirex_signal_capability()
    if not found_signals:
        console.print("\n⚠️ No signals found in any test - need deeper investigation")
    else:
        console.print("\n✅ TiRex signal generation confirmed!")