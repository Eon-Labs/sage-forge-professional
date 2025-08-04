#!/usr/bin/env python3
"""
Debug NT catalog data flow to identify why bars aren't reaching the strategy.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from rich.console import Console

console = Console()

def debug_catalog_data():
    """Debug the catalog data structure and content."""
    console.print("🔍 Debugging NT catalog data flow")
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
        
        if not success:
            console.print("❌ Failed to setup backtest")
            return
        
        console.print(f"✅ Created {len(engine.market_bars)} bars from DSM")
        
        # Create just the catalog part
        import tempfile
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        from nautilus_trader.model.instruments import CryptoPerpetual
        from nautilus_trader.model.objects import Price, Money, Quantity  
        from nautilus_trader.model.identifiers import Symbol, Venue, InstrumentId
        from nautilus_trader.model.enums import AssetClass, InstrumentClass
        from nautilus_trader.model.currencies import USDT, BTC
        from decimal import Decimal
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="debug_catalog_"))
        
        # Create catalog
        catalog = ParquetDataCatalog(str(temp_dir))
        console.print(f"📁 Created catalog at: {temp_dir}")
        
        # Create instrument
        instrument = CryptoPerpetual(
            instrument_id=engine.instrument_id,
            raw_symbol=Symbol("BTCUSDT"),
            base_currency=BTC,
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=2,
            size_precision=3,
            price_increment=Price.from_str("0.01"),
            size_increment=Quantity.from_str("0.001"),
            margin_init=Decimal("0.10"),
            margin_maint=Decimal("0.05"),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0004"),
            ts_event=0,
            ts_init=0,
        )
        
        # Write instrument and bars
        catalog.write_data([instrument])
        catalog.write_data(engine.market_bars)
        
        console.print("📊 Examining catalog structure:")
        
        # Check directory structure
        for root, dirs, files in temp_dir.walk():
            level = len(root.parts) - len(temp_dir.parts)
            indent = "  " * level
            console.print(f"{indent}{root.name}/")
            for f in files:
                file_path = root / f
                size = file_path.stat().st_size
                console.print(f"{indent}  {f} ({size:,} bytes)")
        
        # Try to query the catalog directly
        console.print("\n🔍 Querying catalog directly:")
        
        try:
            # Query bars from catalog
            from nautilus_trader.model.data import Bar
            
            # Get the bar type from our bars
            if engine.market_bars:
                bar_type = engine.market_bars[0].bar_type
                console.print(f"   Bar type: {bar_type}")
                
                # Try to query with different methods
                console.print("\n📋 Testing catalog queries:")
                
                # Method 1: Query by data class and instrument
                try:
                    bars1 = catalog.query(
                        data_cls=Bar,
                        identifiers=[str(engine.instrument_id)],
                        start=datetime(2024, 12, 1),
                        end=datetime(2024, 12, 3)
                    )
                    console.print(f"   Method 1 (by instrument): {len(bars1) if bars1 else 0} bars")
                except Exception as e:
                    console.print(f"   Method 1 failed: {e}")
                
                # Method 2: Query by bar type string
                try:
                    bars2 = catalog.query(
                        data_cls=Bar,
                        identifiers=[str(bar_type)],
                        start=datetime(2024, 12, 1),  
                        end=datetime(2024, 12, 3)
                    )
                    console.print(f"   Method 2 (by bar type): {len(bars2) if bars2 else 0} bars")
                except Exception as e:
                    console.print(f"   Method 2 failed: {e}")
                
                # Method 3: List all data in catalog
                try:
                    all_data = catalog.list_data_types()
                    console.print(f"   All data types in catalog: {all_data}")
                except Exception as e:
                    console.print(f"   List data types failed: {e}")
                
                # Method 4: Try to read parquet directly
                import polars as pl
                parquet_files = list(temp_dir.rglob("*.parquet"))
                console.print(f"\n📄 Direct parquet file analysis:")
                for pf in parquet_files:
                    try:
                        df = pl.read_parquet(pf)
                        console.print(f"   {pf.name}: {len(df)} rows, columns: {df.columns}")
                        if len(df) > 0:
                            console.print(f"      Sample row: {df.row(0)}")
                    except Exception as e:
                        console.print(f"   Failed to read {pf.name}: {e}")
        
        except Exception as e:
            console.print(f"❌ Catalog query failed: {e}")
            import traceback
            traceback.print_exc()
        
        return temp_dir
        
    except Exception as e:
        console.print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_catalog_data()