#!/usr/bin/env python3
"""
Debug the format of NT Bar objects we're creating to see if they have correct data.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from rich.console import Console

console = Console()

def debug_bar_format():
    """Debug the NT Bar objects we're creating."""
    console.print("🔍 Debugging NT Bar object format")
    console.print("="*50)
    
    try:
        # Create backtest engine
        engine = TiRexBacktestEngine()
        
        # Setup with short timespan 
        success = engine.setup_backtest(
            symbol="BTCUSDT",
            start_date="2024-12-01",
            end_date="2024-12-03", 
            initial_balance=10000.0,
            timeframe="15m"
        )
        
        if not success or not engine.market_bars:
            console.print("❌ Failed to create bars")
            return
        
        console.print(f"✅ Created {len(engine.market_bars)} bars from DSM")
        
        # Examine the first few bars
        console.print("\n📊 Bar object analysis:")
        for i in range(min(3, len(engine.market_bars))):
            bar = engine.market_bars[i]
            console.print(f"\n   Bar {i+1}:")
            console.print(f"     bar_type: {bar.bar_type}")
            console.print(f"     open: {bar.open} (type: {type(bar.open)})")
            console.print(f"     high: {bar.high} (type: {type(bar.high)})")
            console.print(f"     low: {bar.low} (type: {type(bar.low)})")
            console.print(f"     close: {bar.close} (type: {type(bar.close)})")
            console.print(f"     volume: {bar.volume} (type: {type(bar.volume)})")
            console.print(f"     ts_event: {bar.ts_event} (type: {type(bar.ts_event)})")
            console.print(f"     ts_init: {bar.ts_init} (type: {type(bar.ts_init)})")
        
        # Test writing a single bar to see what happens
        console.print("\n🧪 Testing single bar serialization:")
        
        import tempfile
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        
        temp_dir = Path(tempfile.mkdtemp(prefix="single_bar_test_"))
        catalog = ParquetDataCatalog(str(temp_dir))
        
        # Write just one bar
        console.print("   Writing single bar to catalog...")
        catalog.write_data([engine.market_bars[0]])
        
        # Check what was created
        console.print(f"   Files created:")
        for root, dirs, files in temp_dir.walk():
            level = len(root.parts) - len(temp_dir.parts)
            indent = "    " * level
            console.print(f"{indent}{root.name}/")
            for f in files:
                console.print(f"{indent}  {f}")
        
        # Try to read back from catalog
        console.print("\n   Trying to read back:")
        try:
            from nautilus_trader.model.data import Bar
            
            bars_back = catalog.query(
                data_cls=Bar,
                identifiers=[str(engine.market_bars[0].bar_type)],
                start=None,
                end=None
            )
            console.print(f"   Successfully read back: {len(bars_back) if bars_back else 0} bars")
            
            if bars_back:
                console.print(f"   First bar back: {bars_back[0]}")
            
        except Exception as e:
            console.print(f"   ❌ Failed to read back: {e}")
            import traceback
            traceback.print_exc()
        
        return temp_dir
        
    except Exception as e:
        console.print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_bar_format()