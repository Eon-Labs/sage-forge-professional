#!/usr/bin/env python3
"""
Test TiRex backtesting with fixed strategy bar subscription.
Short timespan test to verify data flow works correctly.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from rich.console import Console

console = Console()

def test_short_tirex_backtest():
    """Test TiRex backtesting with short timespan to verify fixes."""
    console.print("🧪 Testing TiRex backtesting with fixed strategy subscription")
    console.print("="*60)
    
    try:
        # Create backtest engine
        engine = TiRexBacktestEngine()
        
        # Setup with very short timespan for quick testing
        success = engine.setup_backtest(
            symbol="BTCUSDT",
            start_date="2024-12-01",  # 3 days only
            end_date="2024-12-03", 
            initial_balance=10000.0,  # Smaller balance for testing
            timeframe="15m"  # 15-minute bars
        )
        
        if not success:
            console.print("❌ Failed to setup backtest")
            return False
        
        console.print(f"✅ Backtest setup completed")
        console.print(f"   Data points: {len(engine.market_bars)}")
        
        # Create configuration
        config = engine.create_backtest_config()
        console.print("✅ Backtest configuration created")
        
        # Run the backtest
        console.print("🚀 Starting backtest execution...")
        results = engine.run_backtest(config)
        
        console.print("✅ Backtest completed successfully!")
        console.print("📊 Results summary:")
        for key, value in results["performance_summary"].items():
            console.print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_short_tirex_backtest()
    if success:
        console.print("🎉 TiRex backtesting test completed successfully!")
    else:
        console.print("💥 TiRex backtesting test failed")
        sys.exit(1)